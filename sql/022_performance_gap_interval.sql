begin;

-- Paired bootstrap 95% interval for the model-minus-market MAE gap (games
-- resampled with each model error paired to its closing-line error; 4,000
-- resamples, fixed seed, computed by backend/performance.py). The closing
-- market row now also carries probability_games, brier_score and log_loss,
-- scored with the same Student-t and pricing dispersion as the published
-- line on the same games. Ties count as half a home win in every source.
alter table nfl.performance_metrics
  add column model_minus_market_mae_ci_low double precision,
  add column model_minus_market_mae_ci_high double precision,
  add constraint performance_metrics_gap_interval check (
    model_minus_market_mae_ci_low is null
    or model_minus_market_mae_ci_low <= model_minus_market_mae_ci_high
  );

notify pgrst, 'reload schema';

commit;
