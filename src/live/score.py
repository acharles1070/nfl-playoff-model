"""Score the ledger against results as games finish.

Model vs market compares like with like: the market probability is the line
that existed WHEN the prediction was logged, not the closing line.

Usage:
    python -m src.live.score
"""

from __future__ import annotations

import nflreadpy as nfl
import numpy as np
import pandas as pd

from src.data.schedules import live_season
from src.live import ledger
from src.models.compare import paired_bootstrap_delta, per_game_log_loss
from src.models.market import market_probabilities


def scored_ledger() -> pd.DataFrame:
    rows = pd.DataFrame(ledger.read_rows())

    if rows.empty:
        return rows

    for col in ("p_home_win", "market_p_home"):
        rows[col] = pd.to_numeric(rows[col], errors="coerce")

    rows["week"] = rows["week"].astype(int)

    schedule = nfl.load_schedules([live_season()]).to_pandas()
    played = schedule.loc[schedule["home_score"].notna()]

    # once a game is played, nflverse's line columns hold the CLOSING line
    closing = market_probabilities(played).rename(columns={"p_market": "market_p_close"})
    results = played[["game_id", "home_score", "away_score"]].merge(
        closing[["game_id", "market_p_close"]], on="game_id", how="left"
    )

    out = rows.merge(results, on="game_id", how="inner")
    out["home_win"] = (out["home_score"] > out["away_score"]).astype(int)

    return out


def main() -> None:
    ok, message = ledger.verify()
    print(message)

    if not ok:
        raise SystemExit("Ledger integrity failure: results below are NOT trustworthy.")

    done = scored_ledger()
    total = len(ledger.read_rows())

    if done.empty:
        print(f"{total} predictions logged, none finished yet.")
        return

    done["model_ll"] = per_game_log_loss(done["home_win"], done["p_home_win"])
    done["market_ll"] = per_game_log_loss(done["home_win"], done["market_p_home"])
    done["close_ll"] = per_game_log_loss(done["home_win"], done["market_p_close"])
    done["model_correct"] = (done["p_home_win"] >= 0.5) == done["home_win"].astype(bool)
    done["market_correct"] = (done["market_p_home"] >= 0.5) == done["home_win"].astype(bool)

    print(f"{len(done)} of {total} logged predictions have finished.\n")

    by = (
        done.groupby(["model_id", "week"])
        .agg(games=("game_id", "size"), model_ll=("model_ll", "mean"),
             market_ll=("market_ll", "mean"), closing_ll=("close_ll", "mean"),
             model_acc=("model_correct", "mean"), market_acc=("market_correct", "mean"))
        .round(4)
    )
    print(by.to_string())

    print("\nCumulative (market = line when I logged | closing line = the harder test):")
    for model_id, g in done.groupby("model_id"):
        y, p = g["home_win"].to_numpy(), g["p_home_win"].to_numpy()
        note = "   (too few games to conclude anything)" if len(g) < 100 else ""
        print(f"  {model_id}: n={len(g)}  model {g['model_ll'].mean():.4f}")
        for label, col in (("market at logging", "market_p_home"), ("closing line   ", "market_p_close")):
            q = g[col].to_numpy()
            delta, lo, hi = paired_bootstrap_delta(y, p, q)
            print(f"     vs {label}: market {per_game_log_loss(y, q).mean():.4f}  "
                  f"delta {delta:+.4f} [{lo:+.4f}, {hi:+.4f}]{note}")


if __name__ == "__main__":
    main()
