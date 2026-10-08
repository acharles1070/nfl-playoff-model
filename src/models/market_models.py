"""Market-input models (SEPARATE from the fundamentals model; never its input).

1. De-vig methods: how the bookmaker margin is removed changes the benchmark.
2. Calibrated market: logistic recalibration of the de-vigged price, fit
   walk-forward (only earlier seasons). Tests the favorite-longshot bias.
3. Stacked: market logit + my fundamentals logit.

Usage:
    python -m src.models.market_models
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

from src.features.ratings import RatingParams
from src.models.compare import paired_bootstrap_delta, per_game_log_loss
from src.models.market import moneyline_to_implied
from src.models.ratings_experiment import PRIMARY_FEATURES, Context
from src.models.walk_forward import run_walk_forward


def devig_multiplicative(qh, qa):
    return qh / (qh + qa)


def devig_additive(qh, qa):
    return qh - (qh + qa - 1.0) / 2.0


def devig_power(qh, qa):
    """Find k with qh^k + qa^k = 1; p = qh^k."""
    out = np.empty(len(qh))
    for i, (h, a) in enumerate(zip(qh, qa)):
        k = brentq(lambda k: h ** k + a ** k - 1.0, 0.5, 3.0)
        out[i] = h ** k
    return out


def devig_shin(qh, qa):
    """Shin (1993) for two outcomes: the margin is attributed to insider trading z."""
    out = np.empty(len(qh))
    for i, (h, a) in enumerate(zip(qh, qa)):
        s = h + a

        def total(z):
            p = [(np.sqrt(z * z + 4 * (1 - z) * q * q / s) - z) / (2 * (1 - z)) for q in (h, a)]
            return sum(p) - 1.0

        z = brentq(total, 1e-9, 0.99) if total(1e-9) > 0 else 0.0
        out[i] = (np.sqrt(z * z + 4 * (1 - z) * h * h / s) - z) / (2 * (1 - z))
    return out


DEVIG = {
    "multiplicative (what I used)": devig_multiplicative,
    "additive": devig_additive,
    "power": devig_power,
    "Shin": devig_shin,
}


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-3):
    Xa = np.column_stack([np.ones(len(X)), X])

    def nll(w):
        z = Xa @ w
        return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * l2 * np.sum(w[1:] ** 2)

    return minimize(nll, np.zeros(Xa.shape[1]), method="BFGS").x


def predict_logistic(w, X):
    return 1.0 / (1.0 + np.exp(-(w[0] + X @ w[1:])))


def walk_forward_recalibrate(frame: pd.DataFrame, cols: list[str], first_test: int) -> pd.Series:
    """Fit on earlier seasons only, predict each season in turn."""
    out = pd.Series(np.nan, index=frame.index)

    for season in sorted(frame["season"].unique()):
        if season < first_test:
            continue
        train = frame[frame["season"] < season]
        test = frame[frame["season"] == season]
        w = fit_logistic(train[cols].to_numpy(), train["home_win"].to_numpy(float))
        out.loc[test.index] = predict_logistic(w, test[cols].to_numpy())

    return out


def main() -> None:
    ctx = Context()
    g = ctx.games.loc[ctx.games["is_played"] & ctx.games["home_moneyline"].notna() & ctx.games["away_moneyline"].notna()].copy()
    g = g.loc[g["season"] >= 2006].reset_index(drop=True)

    qh, qa = moneyline_to_implied(g["home_moneyline"]).to_numpy(), moneyline_to_implied(g["away_moneyline"]).to_numpy()
    y = g["home_win"].astype(int).to_numpy()

    print(f"games with both moneylines, 2006-2025: {len(g)}  | mean overround {np.mean(qh + qa):.4f}\n")
    print("1) DE-VIG METHOD (log loss on all games; lower is better)")
    for name, fn in DEVIG.items():
        p = fn(qh, qa)
        g[f"p_{name}"] = p
        hold = (g["season"] >= 2017).to_numpy()
        print(f"   {name:<30} all {per_game_log_loss(y, p).mean():.5f} | 2017-25 {per_game_log_loss(y[hold], p[hold]).mean():.5f}")

    g["p_mkt"] = g["p_multiplicative (what I used)"]
    g["lm"] = logit(g["p_mkt"])

    # favorite-longshot check
    g["bin"] = pd.cut(g["p_mkt"], [0, .25, .4, .5, .6, .75, 1.0])
    t = g.groupby("bin", observed=True).agg(n=("home_win", "size"), priced=("p_mkt", "mean"), actual=("home_win", "mean"))
    print("\n2) FAVORITE-LONGSHOT CHECK (de-vigged price vs actual home win rate, 2006-2025)")
    print(t.round(3).T.to_string())

    # model predictions to stack
    data = ctx.featurize(RatingParams(), use_qb=True)
    pred = run_walk_forward(data, feature_set_name="m", feature_columns=PRIMARY_FEATURES)[["game_id", "p_home_win"]]
    g = g.merge(pred, on="game_id", how="left")
    g["lo"] = logit(g["p_home_win"])

    first = 2012
    g["p_cal"] = walk_forward_recalibrate(g, ["lm"], first)
    s = g.dropna(subset=["lo"]).copy()
    s["p_stack"] = walk_forward_recalibrate(s, ["lm", "lo"], first)

    hold = g[(g["season"] >= 2017) & g["p_cal"].notna()]
    dev = g[(g["season"].between(first, 2016)) & g["p_cal"].notna()]
    sh = s[(s["season"] >= 2017) & s["p_stack"].notna()]
    sd = s[(s["season"].between(first, 2016)) & s["p_stack"].notna()]

    print("\n3) MODELS vs the raw market (paired bootstrap; negative = better than the market)")
    print(f"   {'model':<34}| dev 2012-16            | holdout 2017-25")
    for label, d_, h_, col in (
        ("calibrated market (slope+intercept)", dev, hold, "p_cal"),
        ("stacked: market + my fundamentals", sd, sh, "p_stack"),
    ):
        cells = []
        for f in (d_, h_):
            yy = f["home_win"].astype(int).to_numpy()
            dl, lo_, hi_ = paired_bootstrap_delta(yy, f[col].to_numpy(), f["p_mkt"].to_numpy())
            cells.append(f"{dl:+.4f} [{lo_:+.4f},{hi_:+.4f}]")
        print(f"   {label:<34}| {cells[0]} | {cells[1]}")

    w = fit_logistic(g[["lm"]].to_numpy(), g["home_win"].to_numpy(float))
    print(f"\n   fitted on all data: logit(P) = {w[0]:+.3f} + {w[1]:.3f} * logit(market)   (slope > 1 => favorites win more than priced)")


if __name__ == "__main__":
    main()
