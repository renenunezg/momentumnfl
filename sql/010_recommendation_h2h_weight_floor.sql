begin;

-- nfl-picks-v4 prices moneylines from the pre-decision market margin moved
-- 0.2 toward the pure model (the 0.5 blend priced moneyline picks 10 points
-- above their realised win rate on 2019 through 2025) and promotes the
-- highest-edge positive-EV offers up to a weekly volume floor, recorded with
-- reason volume_floor. Spreads and totals are unchanged. Floor picks carry
-- edge_points between 0 and 2, so the eligibility gate for v4 requires
-- positive edge and positive EV; v3 rows keep the 2-point gate.
alter table nfl.recommendations drop constraint recommendation_eligibility_v3;
alter table nfl.recommendations add constraint recommendation_eligibility_v4
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
      (policy_version = 'nfl-picks-v3' and edge_points >= 2 and edge_points < 'Infinity'::float8) or
      (policy_version = 'nfl-picks-v4' and edge_points > 0 and edge_points < 'Infinity'::float8)) and
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
           when r.edge_points < 2 then 'under 2 pts (floor)'
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
           when r.edge_points < 2 then 'under 2 pts (floor)'
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

notify pgrst, 'reload schema';
commit;
