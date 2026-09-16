begin;

-- Season-to-date counting stats for the published stat line on /nfl/awards.
-- Players carry their cumulative nflverse totals at the cutoff; coaches carry
-- the record and margin their tenure produced. The model reads only the
-- projections, so this column never changes a forecast.
alter table nfl.award_boards add column season_stats jsonb not null default '{}'::jsonb;

notify pgrst, 'reload schema';
commit;
