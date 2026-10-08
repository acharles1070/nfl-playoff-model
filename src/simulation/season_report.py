"""Summarize the season-simulator backtest (python -m src.simulation.season_report)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import configured_path


def season_bootstrap(frame: pd.DataFrame, col: str, n_boot: int = 5000, seed: int = 0):
    """Mean and 95% CI of a per-snapshot difference, resampling SEASONS (clusters)."""
    by_season = frame.groupby("season")[col].mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(by_season), size=(n_boot, len(by_season)))
    boot = by_season.to_numpy()[idx].mean(axis=1)
    return by_season.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def main() -> None:
    out = configured_path("outputs")
    title = pd.read_csv(out / "season_backtest_title.csv")
    title["ls_diff"] = title["logscore_model"] - title["logscore_market"]

    rows = []
    slices = [("ALL snapshots", title)] + [(f"prior to week {w}", g) for w, g in title.groupby("week")]

    for label, frame in slices:
        for era, e in (("2011-2016", frame[frame.season <= 2016]), ("2017-2025", frame[frame.season >= 2017])):
            d, lo, hi = season_bootstrap(e, "ls_diff")
            rows.append(
                {
                    "slice": label, "seasons": e.season.nunique(), "era": era,
                    "model": e.logscore_model.mean(), "market": e.logscore_market.mean(),
                    "diff": d, "ci_low": lo, "ci_high": hi,
                }
            )

    print("Super Bowl title odds: -ln P(actual champion). diff < 0 means the model beats the market.")
    print("95% CI resamples SEASONS (snapshots within a season share one champion)\n")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    uniform = title.groupby("week").apply(lambda g: np.log(g["teams"]).mean(), include_groups=False)
    print("\nuniform-guess log score by week:", uniform.round(2).to_dict())

    po = pd.read_csv(out / "season_backtest_playoffs.csv")
    print(f"\nPlayoff qualification, {len(po)} team-snapshots: "
          f"mean predicted {po.p_playoffs.mean():.3f} vs actual {po.made.mean():.3f}")

    for w, g in po.groupby("week"):
        p = g["p_playoffs"].clip(1e-4, 1 - 1e-4)
        ll = -np.mean(g["made"] * np.log(p) + (1 - g["made"]) * np.log(1 - p))
        bins = pd.cut(g["p_playoffs"], [-0.001, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.001])
        t = g.groupby(bins, observed=True).agg(
            n=("made", "size"), predicted=("p_playoffs", "mean"), actual=("made", "mean")
        )
        print(f"\nprior to week {w}: log loss {ll:.3f}")
        print(t.round(3).T.to_string())


if __name__ == "__main__":
    main()
