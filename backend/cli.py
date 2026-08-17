"""momentumnfl command line. Heavy imports stay inside each subcommand so
startup is instant and a broken optional dependency only breaks its own
command. Commands collect problems and exit 1 at the end rather than
aborting on the first."""

import argparse

from backend.config import SEASONS


def main() -> None:
    parser = argparse.ArgumentParser(prog="backend")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="pull raw nflverse data")
    ingest_parser.add_argument("--seasons", nargs="+", type=int, default=SEASONS)

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingest(args)


def run_ingest(args) -> None:
    from backend.etl import ingest

    problems: list[str] = []
    seasons = sorted(args.seasons)
    ingest.ingest_shared(seasons)
    print(f"ingested shared (schedules, teams, ngs) for {seasons[0]}-{seasons[-1]}")
    for season in seasons:
        try:
            ingest.ingest_season(season)
            print(f"ingested {season}")
        except Exception as error:  # noqa: BLE001 - report and continue
            problems.append(f"{season}: {error}")
    for problem in problems:
        print(f"PROBLEM: {problem}")
    if problems:
        raise SystemExit(1)
