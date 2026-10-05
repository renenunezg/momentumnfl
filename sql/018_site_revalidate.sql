-- Shared function is owned by momentumweb/sql/001_site_revalidate.sql.
-- Football live probability tables intentionally use timed polling, not callbacks.
DROP TRIGGER IF EXISTS site_revalidate ON nfl.team_ratings;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON nfl.team_ratings FOR EACH STATEMENT EXECUTE FUNCTION site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON nfl.game_projections;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON nfl.game_projections FOR EACH STATEMENT EXECUTE FUNCTION site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON nfl.market_comparisons;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON nfl.market_comparisons FOR EACH STATEMENT EXECUTE FUNCTION site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON nfl.backtest_predictions;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON nfl.backtest_predictions FOR EACH STATEMENT EXECUTE FUNCTION site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON nfl.forecast_snapshots;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON nfl.forecast_snapshots FOR EACH STATEMENT EXECUTE FUNCTION site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON nfl.game_results;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON nfl.game_results FOR EACH STATEMENT EXECUTE FUNCTION site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON nfl.season_win_totals;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON nfl.season_win_totals FOR EACH STATEMENT EXECUTE FUNCTION site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON nfl.award_boards;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON nfl.award_boards FOR EACH STATEMENT EXECUTE FUNCTION site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON nfl.award_model_meta;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON nfl.award_model_meta FOR EACH STATEMENT EXECUTE FUNCTION site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON nfl.team_unit_ratings;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON nfl.team_unit_ratings FOR EACH STATEMENT EXECUTE FUNCTION site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON nfl.teams;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON nfl.teams FOR EACH STATEMENT EXECUTE FUNCTION site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON nfl.recommendations;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON nfl.recommendations FOR EACH STATEMENT EXECUTE FUNCTION site_revalidate();
