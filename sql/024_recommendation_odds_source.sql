begin;

-- Which feed priced each decision's offer: The Odds API ('odds_api', with the
-- book's own quote timestamp) or the ESPN scoreboard fallback ('espn', one
-- book and no quote timestamp, so provider_last_update is the fetch time).
-- A decision with no offer has no source; a recommendation always has one.
alter table nfl.recommendations add column odds_source text;

-- Backfill from the event id the ESPN client stamps ('espn-<event id>'). The
-- ledger trigger rejects edits to settled rows, so it is paused for this one
-- statement inside the migration's transaction and restored before commit.
-- The kickoff check runs immediately (published_at is untouched, so it
-- passes) so no deferred events block the table changes that follow.
set constraints nfl.recommendation_commit immediate;
alter table nfl.recommendations disable trigger protect_recommendation;
update nfl.recommendations
set odds_source = case
  when odds_api_event_id like 'espn-%' then 'espn'
  else 'odds_api'
end
where odds_api_event_id is not null;
alter table nfl.recommendations enable trigger protect_recommendation;

alter table nfl.recommendations
  add constraint recommendations_odds_source check (
    odds_source in ('odds_api', 'espn')
    and (odds_api_event_id like 'espn-%') = (odds_source = 'espn')
    or odds_source is null and odds_api_event_id is null
  ),
  add constraint recommendations_recommended_odds_source check (
    status <> 'recommended' or odds_source is not null
  );

notify pgrst, 'reload schema';

commit;
