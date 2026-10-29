begin;

-- sql/024's odds_source check could evaluate to NULL (an 'espn' source with
-- no event id, or no source with an event id), and CHECK accepts NULL. Every
-- branch below is a definite true or false: no offer has neither, an Odds API
-- offer has a plain event id, an ESPN offer has an 'espn-' event id.
alter table nfl.recommendations
  drop constraint recommendations_odds_source,
  add constraint recommendations_odds_source check (
    case
      when odds_api_event_id is null then odds_source is null
      else odds_source is not null
        and odds_source = case
          when odds_api_event_id like 'espn-%' then 'espn'
          else 'odds_api'
        end
    end
  );

notify pgrst, 'reload schema';

commit;
