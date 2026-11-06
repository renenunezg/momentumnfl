begin;

-- nfl-picks-v7 prices spreads and moneylines at the devigged market number
-- with no model weight, and ranked leans take whichever side the best price
-- favours instead of the model's side. Its eligibility rule is v6's.
alter table nfl.recommendations drop constraint recommendation_eligibility_v6;
alter table nfl.recommendations add constraint recommendation_eligibility_v7
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
     ((((policy_version in ('nfl-picks-v1', 'nfl-picks-v2') and probability_edge >= 0.045) or
        (policy_version = 'nfl-picks-v3' and edge_points >= 2 and edge_points < 'Infinity'::float8) or
        (policy_version in ('nfl-picks-v4', 'nfl-picks-v5') and edge_points > 0 and edge_points < 'Infinity'::float8)) and
       expected_value_per_unit > 0) or
      (policy_version in ('nfl-picks-v6', 'nfl-picks-v7') and abs(edge_points) < 'Infinity'::float8 and
       ((reason = 'positive_ev' and expected_value_per_unit > 0) or
        (reason = 'ranked_lean' and market in ('spreads', 'h2h') and expected_value_per_unit > -1)))) and
     expected_value_per_unit < 'Infinity'::float8) is true);

notify pgrst, 'reload schema';
commit;
