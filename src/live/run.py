"""Log pre-game predictions for the upcoming week.

Usage:
    python -m src.live.run              # refresh data, predict, append to ledger
    python -m src.live.run --dry-run    # show predictions, write nothing
    python -m src.live.run --no-refresh # skip the data refresh

Safeguards:
  * refuses to log from a working tree with uncommitted code changes (the
    ledger records the git commit, so it must describe real committed code);
  * never logs a game whose kickoff has passed;
  * never logs the same (model, game) twice.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

from src.live import ledger
from src.live.predict import (
    MODELS,
    git_state,
    load_inputs,
    predict_upcoming,
    to_ledger_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--models", nargs="+", default=None,
                        help="model ids to log (default: all). Log the injury challenger "
                             "only after Friday injury reports.")
    parser.add_argument("--commit", action="store_true",
                        help="git-commit the ledger after a successful run")
    args = parser.parse_args()

    sha, dirty = git_state()

    if dirty and not args.allow_dirty and not args.dry_run:
        raise SystemExit(
            "Uncommitted code changes: commit first so the ledger's git sha "
            "describes the code that produced these predictions "
            "(or pass --allow-dirty)."
        )

    if not args.no_refresh:
        print("Refreshing data ...")
        result = subprocess.run([sys.executable, "-m", "src.pipeline"],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"Pipeline failed:\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}")

    now = datetime.now(timezone.utc)
    inputs = load_inputs()

    all_rows = []

    chosen = args.models or list(MODELS)
    unknown = set(chosen) - set(MODELS)

    if unknown:
        raise SystemExit(f"Unknown model ids {sorted(unknown)}; choose from {sorted(MODELS)}")

    for spec in (MODELS[m] for m in chosen):
        predictions = predict_upcoming(spec, inputs, now)

        if predictions.empty:
            print(f"[{spec.model_id}] no upcoming games with a future kickoff.")
            continue

        show = predictions[
            ["kickoff_utc", "away_team", "home_team", "p_home_win", "p_market"]
        ].copy()
        show["kickoff_utc"] = show["kickoff_utc"].dt.strftime("%a %m-%d %H:%M")
        show["model vs mkt"] = show["p_home_win"] - show["p_market"]

        print(f"\n[{spec.model_id}]  week {int(predictions['week'].iloc[0])}, "
              f"{len(predictions)} games (P = home win)")
        print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        all_rows += to_ledger_rows(predictions, logged_at=now, model_sha=sha)

    if args.dry_run:
        print("\nDry run: nothing written.")
        return

    added = ledger.append(all_rows)
    ok, message = ledger.verify()

    print(f"\nLogged {added} new predictions. {message}")

    if args.commit and added and ok:
        subprocess.run(["git", "add", "ledger/predictions.csv"], check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m",
             f"Ledger: log {added} pre-game predictions ({now:%Y-%m-%d %H:%M} UTC)"],
            check=True,
        )
        print("Committed ledger.")


if __name__ == "__main__":
    main()
