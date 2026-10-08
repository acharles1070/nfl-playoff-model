"""Rebuild every processed table, in dependency order.

Usage:
    python -m src.pipeline                 # full rebuild
    python -m src.pipeline --from ratings  # resume from one stage
    python -m src.pipeline --stages schedules,epa,snapshots,matchups,opponent_adjusted,opponent_matchups,qb

Stages are run as modules so each keeps its own audit printout. Completed
seasons come from the local play-by-play cache (data/raw/pbp); delete a
season's file there to force a fresh download.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


STAGES = [
    ("schedules", "src.data.schedules"),
    ("epa", "src.features.epa"),
    ("snapshots", "src.features.snapshots"),
    ("matchups", "src.features.matchups"),
    ("opponent_adjusted", "src.features.opponent_adjusted"),
    ("opponent_matchups", "src.features.opponent_matchups"),
    ("qb", "src.features.qb"),
    ("injuries", "src.features.injuries_build"),
    ("rosters", "src.features.rosters_build"),
    ("ratings", "src.features.ratings"),
]


# What the live bracket needs (no injuries, rosters or the ratings audit stage):
BRACKET_STAGES = ["schedules", "epa", "snapshots", "matchups", "opponent_adjusted", "opponent_matchups", "qb"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from",
        dest="start",
        choices=[name for name, _ in STAGES],
        default=STAGES[0][0],
    )
    parser.add_argument(
        "--stages",
        help="comma-separated subset to run, in pipeline order (e.g. the bracket needs: " + ",".join(BRACKET_STAGES) + ")",
    )
    args = parser.parse_args()

    names = [name for name, _ in STAGES]
    todo = STAGES[names.index(args.start):]
    if args.stages:
        wanted = {x.strip() for x in args.stages.split(",")}
        unknown = wanted - set(names)
        if unknown:
            raise SystemExit(f"unknown stages: {sorted(unknown)}")
        todo = [(n, m) for n, m in STAGES if n in wanted]

    started = time.time()

    for name, module in todo:
        print(f"\n{'=' * 70}\n[{name}]  python -m {module}\n{'=' * 70}")
        t = time.time()

        result = subprocess.run([sys.executable, "-m", module])

        if result.returncode != 0:
            raise SystemExit(f"Stage '{name}' failed (exit {result.returncode}).")

        print(f"[{name}] done in {time.time() - t:.1f}s")

    print(f"\nPipeline complete in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
