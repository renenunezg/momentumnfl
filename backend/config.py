from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

DATA_DIR = REPO_ROOT / "backend" / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STATIC_DIR = REPO_ROOT / "backend" / "data_static"

# First season with nflverse play-by-play the model trains on; everything
# earlier is ignored so the 32-team map is continuous.
HISTORY_START_SEASON = 2015
SEASONS = list(range(HISTORY_START_SEASON, 2027))
DEVELOPMENT_SEASONS = tuple(range(2016, 2022))
HOLDOUT_SEASONS = tuple(range(2022, 2026))
BACKTEST_PUBLISH_FLOOR = 2016
