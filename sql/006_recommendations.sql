begin;

create table nfl.recommendation_schedule (
  game_id text primary key, season integer not null, start_date timestamptz,
  home_team text not null, away_team text not null,
  game_status text not null, completed boolean not null,
  home_points integer, away_points integer,
  observed_at timestamptz not null check (observed_at <= clock_timestamp()),
  check (not completed or (home_points >= 0 and away_points >= 0 and start_date < observed_at) is true)
);
alter table nfl.recommendation_schedule enable row level security;
grant select, insert, update on nfl.recommendation_schedule to service_role;

create table if not exists nfl.recommendations (
  game_id text not null,
  market text not null check (market in ('h2h', 'spreads', 'totals')),
  season integer not null,
  week integer not null,
  start_date timestamptz not null,
  home_team text not null,
  away_team text not null,
  model_version text not null,
  forecast_as_of timestamptz not null,
  home_missing_input_count integer,
  away_missing_input_count integer,
  policy_version text not null,
  decision_at timestamptz not null,
  published_at timestamptz not null default clock_timestamp(),
  status text not null check (status in ('recommended', 'no_play')),
  reason text not null,
  selection text,
  side text check (side in ('home', 'away', 'over', 'under')),
  point double precision,
  price double precision,
  provider text,
  provider_key text,
  market_fetched_at timestamptz,
  odds_api_event_id text,
  provider_start_date timestamptz,
  provider_last_update timestamptz,
  match_score double precision,
  execution_eligibility_verified boolean not null default false,
  win_probability double precision,
  push_probability double precision,
  probability_edge double precision,
  expected_value_per_unit double precision,
  stake_units double precision not null,
  model_home_margin double precision not null,
  model_total double precision not null,
  margin_sd double precision not null,
  total_sd double precision not null,
  degrees_of_freedom double precision,
  outcome text not null default 'pending'
    check (outcome in ('pending', 'win', 'loss', 'push', 'void', 'no_play')),
  home_points integer,
  away_points integer,
  profit_units double precision,
  graded_at timestamptz,
  source_timestamps jsonb not null,
  data_flags jsonb not null,
  pricing_weights jsonb not null,
  settlement_reason text,
  result_source_at timestamptz,
  primary key (game_id, market),
  check (forecast_as_of <= decision_at and decision_at <= published_at),
  check (published_at < start_date),
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
     probability_edge >= 0.045 and probability_edge < 1 and
     expected_value_per_unit > 0 and expected_value_per_unit < 'Infinity'::float8) is true),
  check ((outcome = 'pending' and graded_at is null and profit_units is null) or
    (outcome <> 'pending' and graded_at is not null and profit_units is not null))
);

create or replace function nfl.protect_recommendation() returns trigger
language plpgsql set search_path = pg_catalog, nfl as $$
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
          and result.start_date = new.start_date and new.graded_at >= new.start_date)) is true then
        raise exception 'Void requires a schedule change or confirmed moneyline tie';
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
$$;



create trigger protect_recommendation before insert or update or delete on nfl.recommendations
  for each row execute function nfl.protect_recommendation();
create function nfl.reject_recommendation_truncate() returns trigger language plpgsql as $$
begin raise exception 'Recommendation history cannot be truncated'; end $$;
create trigger reject_recommendation_truncate before truncate on nfl.recommendations
  for each statement execute function nfl.reject_recommendation_truncate();
create function nfl.check_recommendation_commit() returns trigger language plpgsql as $$
begin
  if new.published_at is distinct from old.published_at or TG_OP = 'INSERT' then
    if new.start_date <= clock_timestamp() then raise exception 'Publication crossed kickoff'; end if;
  end if;
  return new;
end $$;
create constraint trigger recommendation_commit after insert or update on nfl.recommendations
  deferrable initially deferred for each row execute function nfl.check_recommendation_commit();

create index if not exists recommendations_decision_history
  on nfl.recommendations (decision_at desc, game_id, market);
create index if not exists recommendations_season_decision_history
  on nfl.recommendations (season, decision_at desc, game_id, market);

create index recommendations_market_decision on nfl.recommendations (market, decision_at desc, game_id);
create index recommendations_season_market_decision on nfl.recommendations (season, market, decision_at desc, game_id);
create view nfl.recommendation_performance with (security_invoker = true) as
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
        ('edge', case when r.probability_edge < 0.075 then '4.5-7.5 pp'
                      when r.probability_edge < 0.10 then '7.5-10 pp' else '10+ pp' end)
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
        ('edge', case when r.probability_edge < 0.075 then '4.5-7.5 pp'
                      when r.probability_edge < 0.10 then '7.5-10 pp' else '10+ pp' end)
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

alter table nfl.recommendations enable row level security;
drop policy if exists public_read on nfl.recommendations;
create policy public_read on nfl.recommendations for select to anon, authenticated using (true);
grant usage on schema nfl to anon, authenticated, service_role;
grant select on nfl.recommendations, nfl.recommendation_performance to anon, authenticated;
revoke all on nfl.recommendations from service_role;
grant select, insert, update on nfl.recommendations to service_role;
grant select on nfl.recommendation_performance to service_role;

create function nfl.recommendation_history(
  p_season integer default null, p_market text default 'all',
  p_from timestamptz default null, p_page integer default 1
) returns jsonb language sql stable security invoker set search_path = pg_catalog, nfl as $$
  with metrics as materialized (
    select * from nfl.recommendation_summary(p_season,p_market,p_from)
  ), counts as (
    select coalesce(max(picks + no_plays) filter (where segment_kind = 'overall'), 0) as count from metrics
  ), totals as (
    select count, greatest(1, ceil(count / 50.0))::integer as pages from counts
  ), paging as (
    select *, least(greatest(coalesce(p_page, 1), 1), pages) as page from totals
  ) select jsonb_build_object(
    'count', count, 'page', page, 'total_pages', pages,
    'metrics', (select coalesce(jsonb_agg(to_jsonb(m)), '[]'::jsonb) from metrics m),
    'rows', (select coalesce(jsonb_agg(to_jsonb(r) - 'pricing_weights' order by r.decision_at desc, r.game_id, r.market), '[]'::jsonb)
             from (select * from nfl.recommendations
                   where (p_season is null or season = p_season)
                     and (p_market = 'all' or market = p_market)
                     and (p_from is null or decision_at >= p_from)
                   order by decision_at desc, game_id, market
                   limit 50 offset (page - 1) * 50) r),
    'seasons', (select coalesce(jsonb_agg(season order by season desc), '[]'::jsonb)
               from (select distinct season from nfl.recommendations) s)
  ) from paging;
$$;
create function nfl.recommendation_dashboard(
  p_season integer default null, p_market text default 'all', p_from timestamptz default null
) returns jsonb language sql stable security invoker set search_path = pg_catalog, nfl as $$
  select jsonb_build_object(
    'metrics', (select coalesce(jsonb_agg(to_jsonb(m)), '[]'::jsonb) from nfl.recommendation_summary(p_season,p_market,p_from) m),
    'seasons', (select coalesce(jsonb_agg(season order by season desc), '[]'::jsonb) from (select distinct season from nfl.recommendations) s)
  );
$$;
revoke all on function nfl.recommendation_dashboard(integer,text,timestamptz) from public;
grant execute on function nfl.recommendation_dashboard(integer,text,timestamptz) to anon, authenticated, service_role;
revoke all on function nfl.recommendation_summary(integer,text,timestamptz) from public;
revoke all on function nfl.recommendation_history(integer,text,timestamptz,integer) from public;
grant execute on function nfl.recommendation_summary(integer,text,timestamptz),
 nfl.recommendation_history(integer,text,timestamptz,integer) to anon, authenticated, service_role;
-- Accuracy reads the selected source only and returns paired aggregates.
create function nfl.forecast_accuracy(p_source text default 'live') returns jsonb
language sql stable security invoker set search_path = pg_catalog, nfl as $$
  with selected as materialized (
    select season, model_margin, pure_model_margin, closing_spread, actual_margin
    from nfl.live_predictions where p_source = 'live'
    union all
    select season, model_margin, pure_model_margin, closing_spread, actual_margin
    from nfl.backtest_predictions where p_source = 'backtest'
  ), paired as (
    select * from selected where model_margin is not null and pure_model_margin is not null
      and closing_spread is not null and actual_margin is not null
  ), metrics as (
    select season, grouping(season) as is_overall,
      coalesce(season::text, 'All seasons') as label, count(*) as games,
      avg(abs(model_margin - actual_margin)) as "modelMae",
      avg(abs(pure_model_margin - actual_margin)) as "pureMae",
      avg(abs(-closing_spread - actual_margin)) as "marketMae",
      avg((abs(model_margin - actual_margin) < abs(-closing_spread - actual_margin))::int) as "modelBeatsMarket",
      avg(model_margin - actual_margin) as bias
    from paired group by grouping sets ((season), ()) having count(*) > 0
  ) select jsonb_build_object(
    'overall', (select to_jsonb(m) from metrics m where is_overall = 1),
    'bySeason', (select coalesce(jsonb_agg(to_jsonb(m) order by season), '[]'::jsonb) from metrics m where is_overall = 0),
    'completed', (select count(*) from selected),
    'missingForecast', (select count(*) from selected where model_margin is null or pure_model_margin is null),
    'missingClose', (select count(*) from selected where closing_spread is null)
  );
$$;
revoke all on function nfl.forecast_accuracy(text) from public;
grant execute on function nfl.forecast_accuracy(text) to anon, authenticated, service_role;
notify pgrst, 'reload schema';
commit;
