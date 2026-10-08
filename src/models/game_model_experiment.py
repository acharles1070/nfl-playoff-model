"""Compare game-outcome model forms on identical games.

Selection rule: variants are CHOSEN on the tuning window (test seasons
2011-2016). The 2017-2025 holdout is reported for the record.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.compare import paired_bootstrap_delta, per_game_log_loss
from src.models.game_models import factory
from src.models.ratings_experiment import (
    PRIMARY_FEATURES,
    TUNE_SEASONS,
    HOLDOUT_FIRST_SEASON,
    Context,
)
from src.features.ratings import RatingParams
from src.models.walk_forward import run_walk_forward


def run(data: pd.DataFrame, variants: dict) -> pd.DataFrame:
    frames = []

    for name, (kind, features, kwargs) in variants.items():
        pred = run_walk_forward(
            data,
            feature_set_name=name,
            feature_columns=list(features),
            model_factory=factory(kind, **kwargs),
        )
        frames.append(pred)

    return pd.concat(frames, ignore_index=True)


def summarize(pred: pd.DataFrame, reference: str) -> pd.DataFrame:
    rows = []

    ref = pred.loc[pred["feature_set"].eq(reference)].set_index("game_id")

    for name, g in pred.groupby("feature_set", sort=False):
        g = g.set_index("game_id")
        y = g["home_win"].astype(int).to_numpy()
        ll = per_game_log_loss(y, g["p_home_win"].to_numpy()).mean()

        if name == reference:
            delta, lo, hi = 0.0, 0.0, 0.0
        else:
            aligned = ref.loc[g.index]
            delta, lo, hi = paired_bootstrap_delta(
                y, g["p_home_win"].to_numpy(), aligned["p_home_win"].to_numpy()
            )

        rows.append(
            {
                "variant": name,
                "games": len(g),
                "log_loss": ll,
                f"vs_{reference}": delta,
                "ci_low": lo,
                "ci_high": hi,
            }
        )

    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


def main() -> None:
    ctx = Context()
    data = ctx.featurize(RatingParams(), use_qb=True)

    base = tuple(PRIMARY_FEATURES)

    variants = {
        "logistic": ("logistic", base, {}),
        "margin_normal": ("margin_normal", base, {}),
        "margin_hetero": ("margin_hetero", base, {}),
        "margin_hetero alpha=10": ("margin_hetero", base, {"alpha": 10.0}),
    }

    pred = run(data, variants)

    tune = pred.loc[pred["season"].isin(list(TUNE_SEASONS))]
    hold = pred.loc[pred["season"] >= HOLDOUT_FIRST_SEASON]

    fmt = lambda x: f"{x:.4f}"

    print("=== TUNING WINDOW 2011-2016 (selection data) ===")
    print(summarize(tune, "logistic").to_string(index=False, float_format=fmt))

    print("\n=== HOLDOUT 2017-2025 (reported, not used to choose) ===")
    print(summarize(hold, "logistic").to_string(index=False, float_format=fmt))

    post = hold.loc[hold["is_postseason"]]
    print("\n=== HOLDOUT postseason only ===")
    print(summarize(post, "logistic").to_string(index=False, float_format=fmt))

    # what did the heteroscedastic model learn?
    from src.models.game_models import MarginHetero
    train = data.loc[data["season"] <= 2025]
    m = MarginHetero(features=base).fit(train)
    print(f"\nfitted on all data: s0={m.s0_:.1f} (pts^2, base game variance), "
          f"s1={m.s1_:.1f} (pts^2 per median-unit of rating variance)")
    print(f"  => extra variance from rating uncertainty, median game: "
          f"{m.s1_/(m.s0_+m.s1_):.1%} of total")


if __name__ == "__main__":
    main()
