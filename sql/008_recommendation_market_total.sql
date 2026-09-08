begin;

-- nfl-picks-v2 prices totals from the model total blended toward the median
-- posted total across the decision-time offers, the same shrinkage the sides
-- already apply through the published margin. Record the market total each
-- pick was priced against; model_total stores the blended number.
alter table nfl.recommendations add column market_total double precision;
alter table nfl.recommendations add constraint recommendation_market_total_finite
  check (market_total is null or (market_total >= 0 and market_total < 'Infinity'::float8));
commit;
