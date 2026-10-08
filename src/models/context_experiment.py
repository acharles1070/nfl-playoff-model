"""Do rest / bye / neutral-site / division context features improve the game model?

Variants are CHOSEN on the tuning window (2011-2016); 2017-2025 is reported.
"""

from __future__ import annotations

import pandas as pd

from src.features.context import attach_context
from src.features.ratings import RatingParams
from src.models.compare import paired_bootstrap_delta, per_game_log_loss
from src.models.ratings_experiment import (
    HOLDOUT_FIRST_SEASON, PRIMARY_FEATURES, TUNE_SEASONS, Context,
)
from src.models.walk_forward import run_walk_forward


VARIANTS = {
    "current (ratings + QB)": [],
    "+ rest_diff": ["rest_diff"],
    "+ bye flags": ["home_bye", "away_bye"],
    "+ neutral": ["neutral"],
    "+ bye flags + neutral": ["home_bye", "away_bye", "neutral"],
    "+ rest_diff + bye + neutral": ["rest_diff", "home_bye", "away_bye", "neutral"],
    "+ all context": ["rest_diff", "home_bye", "away_bye", "short_week", "neutral", "is_div_game", "dome"],
}


def ll(pred, seasons):
    p = pred.loc[pred["season"].isin(list(seasons))]
    return float(per_game_log_loss(p["home_win"].astype(int).to_numpy(), p["p_home_win"].to_numpy()).mean())


def main() -> None:
    ctx = Context()
    data = attach_context(ctx.featurize(RatingParams(), use_qb=True), ctx.games)

    preds = {
        name: run_walk_forward(data, feature_set_name=name, feature_columns=[*PRIMARY_FEATURES, *extra])
        for name, extra in VARIANTS.items()
    }

    ref = preds["current (ratings + QB)"].set_index("game_id")
    seasons_hold = range(HOLDOUT_FIRST_SEASON, 2026)

    print(f"{'variant':<30}| tune LL  | holdout LL | holdout vs current [95% CI]        | postseason vs current")
    for name, pred in preds.items():
        h = pred.loc[pred["season"] >= HOLDOUT_FIRST_SEASON].set_index("game_id")
        y = h["home_win"].astype(int).to_numpy()
        d, lo, hi = paired_bootstrap_delta(y, h["p_home_win"].to_numpy(), ref.loc[h.index, "p_home_win"].to_numpy())

        hp = h.loc[h["is_postseason"]]
        yp = hp["home_win"].astype(int).to_numpy()
        dp, _, _ = paired_bootstrap_delta(yp, hp["p_home_win"].to_numpy(), ref.loc[hp.index, "p_home_win"].to_numpy())

        print(f"{name:<30}| {ll(pred, TUNE_SEASONS):.4f}   | {ll(pred, seasons_hold):.4f}     | {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  | {dp:+.4f} (n={len(yp)})")

    # what the model learned about context (fit on everything through 2025)
    from src.models.backtest import build_logistic_model
    cols = [*PRIMARY_FEATURES, "rest_diff", "home_bye", "away_bye", "neutral"]
    train = data.loc[data["season"] <= 2025]
    m = build_logistic_model(cols).fit(train, train["home_win"].astype(int).to_numpy())
    scaler = m.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scaler"]
    coef = m.named_steps["model"].coef_[0] / scaler.scale_
    print("\nlogit effect per raw unit (fit on all data):", {c: round(float(v), 3) for c, v in zip(cols, coef)})
    print("intercept (home field):", round(float(m.named_steps['model'].intercept_[0]), 3))
    print("bye games in data: home_bye =", int(train.home_bye.sum()), " away_bye =", int(train.away_bye.sum()), " neutral =", int(train.neutral.sum()))


if __name__ == "__main__":
    main()
