"""Encompassing test: does the model carry information the market lacks?

Fit   P(home win) = sigmoid(a + b_mkt * logit(p_market) + b_model * logit(p_model))
on out-of-sample predictions. If b_model is indistinguishable from 0 the model
adds nothing beyond the market; if b_model > 0 it carries extra information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.data.schedules import last_complete_season
from src.features.ratings import RatingParams
from src.models.compare import attach_market
from src.models.ratings_experiment import HOLDOUT_FIRST_SEASON, PRIMARY_FEATURES, Context
from src.models.walk_forward import run_walk_forward


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def encompassing(frame: pd.DataFrame) -> dict:
    x = pd.DataFrame(
        {"market": _logit(frame["p_market"]), "model": _logit(frame["p_home_win"])}
    )
    fit = sm.Logit(frame["home_win"].astype(int), sm.add_constant(x)).fit(disp=0)
    ci = fit.conf_int()

    return {
        "n": len(frame),
        "market_coef": fit.params["market"],
        "market_ci": (ci.loc["market", 0], ci.loc["market", 1]),
        "model_coef": fit.params["model"],
        "model_ci": (ci.loc["model", 0], ci.loc["model", 1]),
        "model_p_value": fit.pvalues["model"],
        "corr": float(np.corrcoef(x["market"], x["model"])[0, 1]),
    }


def main() -> None:
    ctx = Context()
    data = ctx.featurize(RatingParams(), use_qb=True)
    pred = run_walk_forward(data, feature_set_name="m", feature_columns=PRIMARY_FEATURES)
    pred = attach_market(pred, ctx.games)
    pred = pred.loc[pred["season"].between(HOLDOUT_FIRST_SEASON, last_complete_season())]

    for label, frame in (
        ("all games", pred),
        ("regular season", pred.loc[~pred["is_postseason"]]),
        ("postseason", pred.loc[pred["is_postseason"]]),
    ):
        r = encompassing(frame)
        print(
            f"{label:<15} n={r['n']:<5} market coef {r['market_coef']:.2f} "
            f"[{r['market_ci'][0]:.2f}, {r['market_ci'][1]:.2f}] | "
            f"model coef {r['model_coef']:+.2f} [{r['model_ci'][0]:+.2f}, {r['model_ci'][1]:+.2f}] "
            f"p={r['model_p_value']:.3f} | corr {r['corr']:.2f}"
        )


if __name__ == "__main__":
    main()
