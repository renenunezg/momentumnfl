begin;

-- Published ratings are shifted so rating difference plus home field equals
-- the week's published line. The shift is stored so the market-free fitted
-- rating stays recoverable: fitted power rating is power_rating minus this
-- value, and offense and defense each carry half. Earlier rows keep NULL.
alter table nfl.team_ratings add column forecast_alignment_points double precision;

commit;
