from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

DATA_DIR = REPO_ROOT / "backend" / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STATIC_DIR = REPO_ROOT / "backend" / "data_static"

SEASONS = list(range(2015, 2027))
EVAL_SEASONS = list(range(2016, 2027))
DEVELOPMENT_SEASONS = tuple(range(2016, 2022))
HOLDOUT_SEASONS = tuple(range(2022, 2026))
BACKTEST_PUBLISH_FLOOR = 2016
