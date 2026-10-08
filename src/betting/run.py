"""Backtest: bet the model's disagreements with the CLOSING moneyline.

Selection: threshold chosen on the dev window (2007-2016); holdout 2017-2025 scored once.
Honesty devices: real two-sided prices (vig included), a naive-baseline panel, a
bootstrap interval on ROI, max drawdown, and the Deflated Sharpe Ratio over every
strategy tried.

python -m src.betting.run
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.betting.sim import decimal_odds, place_bets
from src.betting.stats import deflated_sharpe, expected_max_sharpe, max_drawdown, roi_ci, sharpe
from src.config import configured_path
from src.features.ratings import RatingParams
from src.models.ratings_experiment import PRIMARY_FEATURES, Context
from src.models.walk_forward import run_walk_forward

THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08)
MODES = ("flat", "kelly")
DEV, HOLD = (2007, 2016), (2017, 2025)


def load() -> pd.DataFrame:
    ctx = Context()
    data = ctx.featurize(RatingParams(), use_qb=True)
    pred = run_walk_forward(data, feature_set_name="m", feature_columns=PRIMARY_FEATURES)[["game_id", "p_home_win"]]
    g = ctx.games.loc[ctx.games["is_played"] & ctx.games["home_moneyline"].notna() & ctx.games["away_moneyline"].notna(),
                      ["game_id", "season", "week", "home_moneyline", "away_moneyline", "home_win"]]
    g = g.merge(pred, on="game_id").rename(columns={"p_home_win": "p_home"})
    g["home_win"] = g["home_win"].astype(int)
    return g


def summarize(bets: pd.DataFrame) -> dict:
    if bets.empty:
        return {"bets": 0}
    roi, lo, hi = roi_ci(bets["ret"].to_numpy())
    equity = np.r_[100.0, bets["bankroll"].to_numpy()]
    cum = np.r_[0.0, bets["pnl"].cumsum().to_numpy()]
    return {"bets": len(bets), "ROI": roi, "ROI_lo": lo, "ROI_hi": hi, "sharpe_per_bet": sharpe(bets["ret"].to_numpy()),
            "total_pnl": bets["pnl"].sum(),
            "worst_peak_to_trough_units": float((cum - np.maximum.accumulate(cum)).min()),
            "final_bankroll": float(equity[-1]),
            "max_drawdown": max_drawdown(equity) if equity.min() > 0 else -1.0}


def main() -> None:
    g = load()
    dev = g[g.season.between(*DEV)]
    hold = g[g.season.between(*HOLD)]
    print(f"games with both moneylines: dev {len(dev)} (2007-16), holdout {len(hold)} (2017-25)")
    print(f"mean overround (vig) in the data: {np.mean(1 / decimal_odds(hold.home_moneyline) + 1 / decimal_odds(hold.away_moneyline)):.4f}\n")

    # ---- naive baselines (flat 1 unit, holdout): what the vig alone does
    dh, da = decimal_odds(hold.home_moneyline), decimal_odds(hold.away_moneyline)
    base = {
        "bet every home team": np.where(hold.home_win == 1, dh - 1, -1.0),
        "bet every away team": np.where(hold.home_win == 0, da - 1, -1.0),
        "bet every favorite": np.where(np.where(dh < da, hold.home_win == 1, hold.home_win == 0), np.where(dh < da, dh, da) - 1, -1.0),
        "bet every underdog": np.where(np.where(dh < da, hold.home_win == 0, hold.home_win == 1), np.where(dh < da, da, dh) - 1, -1.0),
    }
    print("NAIVE BASELINES (flat, holdout): ROI per bet [95% CI]")
    for k, r in base.items():
        m, lo, hi = roi_ci(r)
        print(f"   {k:<22} n={len(r):<5} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    rows, series = [], {}
    for mode in MODES:
        for th in THRESHOLDS:
            for name, frame in (("dev", dev), ("holdout", hold)):
                b = place_bets(frame, threshold=th, mode=mode)
                rows.append({"mode": mode, "threshold": th, "window": name, **summarize(b)})
                if name == "holdout":
                    series[(mode, th)] = b["ret"].to_numpy() if len(b) else np.array([])

    res = pd.DataFrame(rows)
    print("\nSTRATEGY GRID (ROI per bet; n = bets placed)")
    wide = res.pivot_table(index=["mode", "threshold"], columns="window", values=["bets", "ROI"])
    print(wide.round(3).to_string())

    # ---- select on dev, report holdout
    print("\nSELECTION ON DEV, SCORED ON HOLDOUT")
    sr = np.array([sharpe(v) for v in series.values() if len(v) > 5])
    n_trials, var_sr = len(sr), float(np.var(sr))
    for mode in MODES:
        d = res[(res["mode"] == mode) & (res.window == "dev") & (res.bets > 50)].sort_values("ROI", ascending=False)
        best = d.iloc[0]
        h = res[(res["mode"] == mode) & (res.window == "holdout") & (res.threshold == best.threshold)].iloc[0]
        r = series[(mode, best.threshold)]
        dsr = deflated_sharpe(r, n_trials, var_sr)
        risk = (f"total {h.total_pnl:+.0f} units, worst peak-to-trough {h.worst_peak_to_trough_units:.0f} units" if mode == "flat"
                else f"bankroll 100 -> {h.final_bankroll:.1f} (max drawdown {h.max_drawdown:.0%})")
        print(f"   {mode:<6} best-on-dev threshold {best.threshold:.2f}: dev ROI {best.ROI:+.3f}  ->  HOLDOUT bets {int(h.bets)}, "
              f"ROI {h.ROI:+.3f} [{h.ROI_lo:+.3f}, {h.ROI_hi:+.3f}], Sharpe/bet {h.sharpe_per_bet:+.3f}, {risk}, DSR {dsr:.2f}")
    print(f"\n   Deflated Sharpe uses {n_trials} strategies tried; luck-only expected best Sharpe/bet = {expected_max_sharpe(n_trials, var_sr):+.3f}")

    # ---- how big are the model's 'edges' at all?
    p = hold["p_home"].to_numpy()
    imp = (1 / dh) / (1 / dh + 1 / da)
    print(f"\nModel vs de-vigged market, holdout: mean |p_model - p_market| = {np.mean(np.abs(p - imp)):.3f};  share of games where either side shows EV > 0 at the closing price: "
          f"{np.mean(np.maximum(p * (dh - 1) - (1 - p), (1 - p) * (da - 1) - p) > 0):.1%}")
    res.to_csv(configured_path("outputs") / "betting_grid.csv", index=False)


if __name__ == "__main__":
    main()
