begin;

-- The position trait history kept every trait and the display columns of
-- every player each week, tens of thousands of rows a week with no reader.
-- Keep only the headline rows a season-end grade needs (grade and war) and
-- drop the name and headshot, which nfl.position_traits already serves. The
-- seeded rows are trimmed the same way; the append-only trigger is paused
-- for that one delete inside this transaction.
alter table nfl.position_traits_history disable trigger append_only;
delete from nfl.position_traits_history where trait not in ('grade', 'war');
alter table nfl.position_traits_history enable trigger append_only;
alter table nfl.position_traits_history
  drop column athlete_name,
  drop column headshot_url,
  add constraint position_traits_history_headline check (trait in ('grade', 'war'));

-- No reader yet: the histories are private to the backend until one exists.
do $$
declare
  history text;
begin
  foreach history in array array[
    'season_win_total_history', 'playoff_forecast_history', 'position_traits_history'
  ] loop
    execute format('drop policy "anon read" on nfl.%I', history);
    execute format('revoke select on nfl.%I from anon, authenticated', history);
  end loop;
end $$;

notify pgrst, 'reload schema';

commit;
