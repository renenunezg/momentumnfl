begin;

-- A verified starter substitution may revise an open pick before kickoff.
-- Keep the complete original price, forecast and source receipts for audit.
create table nfl.recommendation_revisions (
  game_id text not null,
  market text not null,
  published_at timestamptz not null,
  superseded_at timestamptz not null default clock_timestamp(),
  reason text not null default 'expected_qb_changed'
    check (reason = 'expected_qb_changed'),
  previous_record jsonb not null,
  primary key (game_id, market, published_at)
);
alter table nfl.recommendation_revisions enable row level security;
grant select, insert on nfl.recommendation_revisions to service_role;
create function nfl.protect_recommendation_revision() returns trigger
language plpgsql set search_path = pg_catalog, nfl as $$
begin
  raise exception 'Archived recommendation revisions are immutable';
end $$;
create trigger immutable_recommendation_revisions
before update or delete or truncate on nfl.recommendation_revisions
for each statement execute function nfl.protect_recommendation_revision();

-- Shared by the publisher and trigger so an upsert cannot bypass the gate.
create function nfl.can_refresh_qb_pick(
  previous nfl.recommendations, revised nfl.recommendations
) returns boolean language sql volatile set search_path = pg_catalog, nfl as $$
  select coalesce(
    previous.status = 'recommended'
    and previous.outcome = 'pending' and revised.outcome = 'pending'
    and previous.start_date > clock_timestamp()
    and (previous.game_id, previous.market, previous.season, previous.week,
         previous.start_date, previous.home_team, previous.away_team)
      = (revised.game_id, revised.market, revised.season, revised.week,
         revised.start_date, revised.home_team, revised.away_team)
    and revised.forecast_as_of > previous.forecast_as_of
    and revised.decision_at > previous.decision_at
    and not exists (
      select 1 from unnest(array['home','away']) as sides(side)
      where not (
        revised.data_flags ->> (side || '_expected_qb') <> ''
        and (revised.data_flags ->> (side || '_qb_source_at'))::timestamptz
          between clock_timestamp() - interval '48 hours' and revised.forecast_as_of
      ) is true
    )
    and not exists (
      select 1 from unnest(array['depth_charts','qb_overrides']) as sources(name)
      where not (
        revised.source_timestamps -> name ->> 'sha256' <> ''
        and (revised.source_timestamps -> name ->> 'observed_at')::timestamptz
          <= revised.forecast_as_of
      ) is true
    )
    and exists (
      select 1 from unnest(array['home','away']) as sides(side)
      where previous.data_flags ->> (side || '_expected_qb') <> ''
        and revised.data_flags ->> (side || '_expected_qb')
          <> previous.data_flags ->> (side || '_expected_qb')
        and (revised.data_flags ->> (side || '_qb_source_at'))::timestamptz
          > (previous.data_flags ->> (side || '_qb_source_at'))::timestamptz
    ), false);
$$;

CREATE OR REPLACE FUNCTION nfl.protect_recommendation()
 RETURNS trigger
 LANGUAGE plpgsql
 SET search_path TO 'pg_catalog', 'nfl'
AS $function$
declare
  balance float8;
  expected_outcome text;
  expected_profit float8;
  source_name text;
  observed timestamptz;
  result nfl.recommendation_schedule;
begin
  if TG_OP = 'DELETE' then
    raise exception 'Recommendation history cannot be deleted';
  end if;
  if TG_OP = 'INSERT' then
    -- Caller-supplied historical timestamps must never admit a past pick.
    new.published_at := clock_timestamp();
    if exists (select 1 from nfl.game_results where game_id = new.game_id)
       or exists (select 1 from nfl.forecast_snapshots
                  where game_id = new.game_id and start_date <= clock_timestamp()) then
      raise exception 'Cannot insert a decision for a previously started game';
    end if;
    if new.outcome <> 'pending' then
      raise exception 'New recommendations must be pending';
    end if;
  end if;
  if TG_OP = 'UPDATE' then
    if old.outcome <> 'pending' and new is distinct from old then
      raise exception 'Settled recommendations are immutable';
    end if;
    if old.status = 'recommended' or old.start_date <= clock_timestamp() then
      if (to_jsonb(new) - array['outcome','home_points','away_points','profit_units','graded_at','settlement_reason','result_source_at'])
        is distinct from
         (to_jsonb(old) - array['outcome','home_points','away_points','profit_units','graded_at','settlement_reason','result_source_at']) then
        if not nfl.can_refresh_qb_pick(old, new) then
          raise exception 'Published picks and started game decisions are frozen';
        end if;
        insert into nfl.recommendation_revisions
          (game_id, market, published_at, previous_record)
        values (old.game_id, old.market, old.published_at, to_jsonb(old));
        new.published_at := clock_timestamp();
      end if;
    end if;
    if old.status = 'no_play' and old.outcome = 'pending' and new.outcome = 'pending'
       and new is distinct from old then
      new.published_at := clock_timestamp();
    end if;
  end if;
  if new.outcome = 'pending' then
    if new.home_points is not null or new.away_points is not null then
      raise exception 'Pending recommendations cannot contain final scores';
    end if;
  else
    if not (new.graded_at >= new.published_at and new.graded_at <= clock_timestamp()) then
      raise exception 'Invalid settlement timestamp';
    end if;
    select * into result from nfl.recommendation_schedule where game_id = new.game_id;
    if not (result.observed_at = new.result_source_at and result.observed_at <= new.graded_at
       and result.home_team = new.home_team and result.away_team = new.away_team) is true then
      raise exception 'Settlement requires a matching confirmed schedule observation';
    end if;
    if new.outcome = 'void' then
      if not ((new.settlement_reason = 'schedule_change' and
          (result.start_date <> new.start_date or result.game_status in ('canceled','cancelled','postponed')))
        or (new.settlement_reason = 'moneyline_tie' and new.market = 'h2h'
          and result.completed and result.home_points = result.away_points
          and result.start_date = new.start_date and new.graded_at >= new.start_date)
        or (new.settlement_reason = 'policy_withdrawn' and new.status = 'recommended'
          and old.outcome = 'pending' and new.graded_at < new.start_date
          and result.start_date = new.start_date)) is true then
        raise exception 'Void requires a schedule change, a confirmed moneyline tie, or a pregame policy withdrawal';
      end if;
      expected_profit := 0;
    elsif new.outcome = 'no_play' and new.status = 'no_play' then
      if new.graded_at < new.start_date then
        raise exception 'Cannot settle before kickoff';
      end if;
      expected_profit := 0;
    elsif new.status = 'recommended' and new.outcome in ('win','loss','push') then
      if not (new.graded_at >= new.start_date and new.home_points >= 0
              and new.away_points >= 0) is true then
        raise exception 'Settlement requires a started game and final scores';
      end if;
      if not (result.completed and result.start_date = new.start_date
         and result.home_points = new.home_points and result.away_points = new.away_points
         and new.settlement_reason = 'confirmed_final') is true then
        raise exception 'Scores must match a confirmed final schedule observation';
      end if;
      balance := case when new.market = 'h2h' then
        (new.home_points::float8 - new.away_points) * case new.side when 'home' then 1 else -1 end
        when new.market = 'spreads' then
        (new.home_points::float8 - new.away_points) * case new.side when 'home' then 1 else -1 end + new.point
        else (new.home_points::float8 + new.away_points - new.point) * case new.side when 'over' then 1 else -1 end end;
      if new.market = 'h2h' and balance = 0 then
        raise exception 'A tied moneyline must be void';
      end if;
      expected_outcome := case when balance > 0 then 'win' when balance < 0 then 'loss' else 'push' end;
      if new.outcome <> expected_outcome then
        raise exception 'Outcome disagrees with the recorded line and final score';
      end if;
      expected_profit := case new.outcome when 'win' then
        case when new.price > 0 then new.price / 100 else 100 / abs(new.price) end
        when 'loss' then -1 else 0 end;
    else
      raise exception 'Outcome is incompatible with the recorded decision';
    end if;
    if not (abs(new.profit_units - expected_profit) < 1e-10) is true then
      raise exception 'Profit disagrees with the recorded price and outcome';
    end if;
  end if;
  if new.outcome = 'pending' and new.status = 'recommended' then
    foreach source_name in array array['schedule','depth_charts','qb_overrides','win_totals','win_total_sources','preseason_qbs'] loop
      observed := (new.source_timestamps -> source_name ->> 'observed_at')::timestamptz;
      if not (observed <= new.forecast_as_of and
          new.source_timestamps -> source_name ->> 'sha256' is not null) is true then
        raise exception 'Missing or future source receipt: %', source_name;
      end if;
      if (source_name = 'schedule' and new.published_at - observed > interval '24 hours')
        or (source_name = 'depth_charts' and new.published_at - observed > interval '48 hours') then
        raise exception 'Stale availability inputs';
      end if;
    end loop;
    foreach source_name in array array['home','away'] loop
      observed := (new.data_flags ->> (source_name || '_qb_source_at'))::timestamptz;
      if not (new.data_flags ->> (source_name || '_expected_qb') <> ''
         and observed <= new.forecast_as_of
         and new.published_at - observed <= interval '48 hours') is true then
        raise exception 'Missing or stale expected QB';
      end if;
    end loop;
    expected_profit := case when new.price > 0 then new.price / 100 else 100 / abs(new.price) end;
    if not (abs(new.expected_value_per_unit - (new.win_probability * expected_profit
          - (1 - new.win_probability - new.push_probability))) < 1e-10
      and abs(new.probability_edge - (new.win_probability / (1 - new.push_probability)
          - 1 / (1 + expected_profit))) < 1e-10) is true then
      raise exception 'EV and edge must match recorded probability and price';
    end if;
  end if;
  return new;
end
$function$;

notify pgrst, 'reload schema';
commit;
