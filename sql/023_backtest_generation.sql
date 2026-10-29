begin;

-- The historical backtest is a regenerated artifact, not a record of past
-- publications: every row now names the model version and the time the
-- production artifacts that produced it were generated. The backend replaces
-- the table only when that generation changes. Rows published before this
-- migration keep nulls until the next production-artifact generation replaces
-- them.
alter table nfl.backtest_predictions
  add column model_version text,
  add column generated_at timestamptz;

-- The backtest's published margin was blended 50/50 with the closing line it
-- is graded against, so its primary metrics are now the pure model's. The
-- closing-conditional blend stays visible as closingBlendMae, and the result
-- names its primary source and, for the backtest, its generation. Live
-- metrics are unchanged: the published line at forecast time is primary.
create or replace function nfl.forecast_accuracy(p_source text default 'live') returns jsonb
language sql stable security invoker set search_path = pg_catalog, nfl as $$
  with selected as materialized (
    select season, model_margin, pure_model_margin, closing_spread, actual_margin,
      model_margin as primary_margin
    from nfl.live_predictions where p_source = 'live'
    union all
    select season, model_margin, pure_model_margin, closing_spread, actual_margin,
      pure_model_margin as primary_margin
    from nfl.backtest_predictions where p_source = 'backtest'
  ), paired as (
    select * from selected where model_margin is not null and pure_model_margin is not null
      and closing_spread is not null and actual_margin is not null
  ), metrics as (
    select season, grouping(season) as is_overall,
      coalesce(season::text, 'All seasons') as label, count(*) as games,
      avg(abs(primary_margin - actual_margin)) as "modelMae",
      avg(abs(pure_model_margin - actual_margin)) as "pureMae",
      case when p_source = 'backtest'
        then avg(abs(model_margin - actual_margin)) end as "closingBlendMae",
      avg(abs(-closing_spread - actual_margin)) as "marketMae",
      avg((abs(primary_margin - actual_margin) < abs(-closing_spread - actual_margin))::int) as "modelBeatsMarket",
      avg(primary_margin - actual_margin) as bias
    from paired group by grouping sets ((season), ()) having count(*) > 0
  ) select jsonb_build_object(
    'primarySource', case when p_source = 'backtest' then 'pure_model' else 'published' end,
    'overall', (select to_jsonb(m) from metrics m where is_overall = 1),
    'bySeason', (select coalesce(jsonb_agg(to_jsonb(m) order by season), '[]'::jsonb) from metrics m where is_overall = 0),
    'completed', (select count(*) from selected),
    'missingForecast', (select count(*) from selected where model_margin is null or pure_model_margin is null),
    'missingClose', (select count(*) from selected where closing_spread is null),
    'modelVersion', case when p_source = 'backtest' then
      (select string_agg(distinct model_version, ',') from nfl.backtest_predictions) end,
    'generatedAt', case when p_source = 'backtest' then
      (select max(generated_at) from nfl.backtest_predictions) end
  );
$$;

notify pgrst, 'reload schema';

commit;
