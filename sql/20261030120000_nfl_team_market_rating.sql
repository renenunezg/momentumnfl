begin;

-- Market-implied strength from earlier games' closing spreads, published beside
-- the rating so the site can show where the fitted model and the market
-- disagree. Descriptive only. Earlier rows keep NULL.
alter table nfl.team_ratings
  add column market_rating double precision,
  add column market_rating_games integer;

commit;
