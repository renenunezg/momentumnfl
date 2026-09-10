begin;

-- nfl-picks-v3 prices sides and totals with the empirical dispersion around
-- the blended line instead of the engine's wider predictive spread, and gates
-- on points of edge beyond the price's break-even line, the same yardstick
-- for favourites and underdogs. margin_sd and total_sd now record the
-- dispersion the pick was priced with. A pick withdrawn before kickoff after
-- a policy defect is settled void with settlement_reason 'policy_withdrawn'.
alter table nfl.recommendations add column edge_points double precision;
alter table nfl.recommendations drop constraint recommendations_check2;
alter table nfl.recommendations add constraint recommendation_eligibility_v3
check ((status = 'no_play' and stake_units = 0) or
    (status = 'recommended' and stake_units = 1 and execution_eligibility_verified and
     (home_missing_input_count = 0 and away_missing_input_count = 0) and
     match_score >= 0.95 and match_score <= 1 and
     selection is not null and side is not null and
     ((market in ('spreads','h2h') and side in ('home','away')) or
      (market = 'totals' and side in ('over','under'))) and
     selection = case side when 'home' then home_team when 'away' then away_team
                           when 'over' then 'Over' else 'Under' end and
     ((market = 'h2h' and point is null) or
      (market <> 'h2h' and point is not null and abs(point) < 'Infinity'::float8
       and point * 2 = round(point * 2))) and
     abs(price) >= 100 and abs(price) < 'Infinity'::float8 and
     provider is not null and provider_key is not null and
     odds_api_event_id is not null and provider_start_date = start_date and
     provider_last_update <= market_fetched_at and market_fetched_at <= decision_at and
     published_at - provider_last_update <= interval '1 hour' and
     published_at - forecast_as_of <= interval '7 days' and
     abs(model_home_margin) < 'Infinity'::float8 and
     model_total >= 0 and model_total < 'Infinity'::float8 and
     margin_sd > 0 and margin_sd < 'Infinity'::float8 and
     total_sd > 0 and total_sd < 'Infinity'::float8 and
     (degrees_of_freedom is null or degrees_of_freedom > 2) and
     win_probability > 0 and win_probability < 1 and
     push_probability >= 0 and win_probability + push_probability <= 1 and
     probability_edge > -1 and probability_edge < 1 and
     ((policy_version in ('nfl-picks-v1', 'nfl-picks-v2') and probability_edge >= 0.045) or
      (policy_version = 'nfl-picks-v3' and edge_points >= 2 and edge_points < 'Infinity'::float8)) and
     expected_value_per_unit > 0 and expected_value_per_unit < 'Infinity'::float8) is true);

create or replace view nfl.recommendation_performance with (security_invoker = true) as
  with filtered as (
    select * from nfl.recommendations
    where (null::integer is null or season = null::integer)
      and ('all'::text = 'all' or market = 'all'::text)
      and (null::timestamptz is null or decision_at >= null::timestamptz)
  ), segments as (
    select r.*, s.kind, s.label
    from filtered r
    cross join lateral (
      select * from (values
        ('overall', 'All picks'), ('market', r.market),
        ('week', r.season::text || ' Week ' || r.week::text),
        ('policy', r.policy_version),
        ('edge', case
           when r.edge_points is null then case
             when r.probability_edge < 0.075 then '4.5-7.5 pp'
             when r.probability_edge < 0.10 then '7.5-10 pp' else '10+ pp' end
           when r.edge_points < 4 then '2-4 pts'
           when r.edge_points < 7 then '4-7 pts' else '7+ pts' end)
      ) v(kind,label)
      union all
      select 'side', r.market || ':' || case
        when r.market = 'totals' then r.side
        when r.market = 'spreads' then case when r.point < 0 then 'favorite'
          when r.point > 0 then 'underdog' else 'pickem' end
        else case when r.price < -100 then 'favorite' when r.price > 100 then 'underdog' else 'even' end
      end
      where r.status = 'recommended'
    ) s(kind,label)
  ), aggregates as (
    select null::integer as season, kind as segment_kind, label as segment,
      count(distinct game_id) filter (where status = 'recommended') as unique_games,
      count(*) filter (where status = 'recommended') as picks,
      count(*) filter (where status = 'no_play') as no_plays,
      count(*) filter (where status = 'recommended' and outcome = 'pending') as pending,
      count(*) filter (where outcome = 'win') as wins,
      count(*) filter (where outcome = 'loss') as losses,
      count(*) filter (where outcome = 'push') as pushes,
      count(*) filter (where status = 'recommended' and outcome = 'void') as voids,
      coalesce(sum(stake_units) filter (where outcome in ('win','loss','push')), 0) as staked_units,
      coalesce(sum(profit_units) filter (where outcome in ('win','loss','push')), 0) as profit_units,
      avg(expected_value_per_unit) filter (where status = 'recommended') as average_ev,
      min(decision_at) as first_decision_at,
      max(decision_at) as last_decision_at,
      max(graded_at) as last_graded_at
    from segments group by kind, label
  )
  select *, profit_units / nullif(staked_units, 0) as roi,
    wins::float8 / nullif(wins + losses, 0) as win_rate,
    wins + losses + pushes < 30 as thin_sample
  from aggregates;

-- Summaries and history share the same decision-date, season and market filters.
-- Returning only aggregates keeps request size independent of ledger growth.
create or replace function nfl.recommendation_summary(
  p_season integer default null,
  p_market text default 'all',
  p_from timestamptz default null
) returns setof nfl.recommendation_performance
language sql stable security invoker set search_path = pg_catalog, nfl as $$
  with filtered as (
    select * from nfl.recommendations
    where (p_season is null or season = p_season)
      and (p_market = 'all' or market = p_market)
      and (p_from is null or decision_at >= p_from)
  ), segments as (
    select r.*, s.kind, s.label
    from filtered r
    cross join lateral (
      select * from (values
        ('overall', 'All picks'), ('market', r.market),
        ('week', r.season::text || ' Week ' || r.week::text),
        ('policy', r.policy_version),
        ('edge', case
           when r.edge_points is null then case
             when r.probability_edge < 0.075 then '4.5-7.5 pp'
             when r.probability_edge < 0.10 then '7.5-10 pp' else '10+ pp' end
           when r.edge_points < 4 then '2-4 pts'
           when r.edge_points < 7 then '4-7 pts' else '7+ pts' end)
      ) v(kind,label)
      union all
      select 'side', r.market || ':' || case
        when r.market = 'totals' then r.side
        when r.market = 'spreads' then case when r.point < 0 then 'favorite'
          when r.point > 0 then 'underdog' else 'pickem' end
        else case when r.price < -100 then 'favorite' when r.price > 100 then 'underdog' else 'even' end
      end
      where r.status = 'recommended'
    ) s(kind,label)
  ), aggregates as (
    select p_season as season, kind as segment_kind, label as segment,
      count(distinct game_id) filter (where status = 'recommended') as unique_games,
      count(*) filter (where status = 'recommended') as picks,
      count(*) filter (where status = 'no_play') as no_plays,
      count(*) filter (where status = 'recommended' and outcome = 'pending') as pending,
      count(*) filter (where outcome = 'win') as wins,
      count(*) filter (where outcome = 'loss') as losses,
      count(*) filter (where outcome = 'push') as pushes,
      count(*) filter (where status = 'recommended' and outcome = 'void') as voids,
      coalesce(sum(stake_units) filter (where outcome in ('win','loss','push')), 0) as staked_units,
      coalesce(sum(profit_units) filter (where outcome in ('win','loss','push')), 0) as profit_units,
      avg(expected_value_per_unit) filter (where status = 'recommended') as average_ev,
      min(decision_at) as first_decision_at,
      max(decision_at) as last_decision_at,
      max(graded_at) as last_graded_at
    from segments group by kind, label
  )
  select *, profit_units / nullif(staked_units, 0) as roi,
    wins::float8 / nullif(wins + losses, 0) as win_rate,
    wins + losses + pushes < 30 as thin_sample
  from aggregates;
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
        raise exception 'Published picks and started game decisions are frozen';
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
