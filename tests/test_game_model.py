"""The game model's neutral-site probabilities must be symmetric: P(i beats j) + P(j beats i) = 1."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation.backtest import FEATURES, GameModel, build_table


def _fit(offset: float) -> GameModel:
    rng = np.random.default_rng(0)
    n = 4000
    # non-zero feature means (home teams are, on average, a bit better) so the scaler's centering matters
    x = rng.normal(offset, 0.1, size=(n, 2))
    logit = 0.25 + 4.0 * x[:, 0] - 3.0 * x[:, 1]
    frame = pd.DataFrame(x, columns=FEATURES)
    frame["home_win"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return GameModel(frame)


def _snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team": ["AAA", "BBB", "CCC"],
            "pregame_off": [0.10, -0.05, 0.02], "pregame_qb": [0.05, 0.00, -0.03], "pregame_def": [-0.04, 0.06, 0.01],
            "pregame_off_var": [0.004] * 3, "pregame_qb_var": [0.002] * 3, "pregame_def_var": [0.004] * 3,
        }
    )


def test_neutral_site_probabilities_are_symmetric_even_with_nonzero_feature_means():
    table = build_table(_snapshot(), _fit(offset=0.08))
    for i in table.teams:
        for j in table.teams:
            if i != j:
                assert table.m_neutral[(i, j)] + table.m_neutral[(j, i)] == np.float64(0) or abs(table.m_neutral[(i, j)] + table.m_neutral[(j, i)]) < 1e-9


def test_home_field_constant_accounts_for_feature_centering():
    model = _fit(offset=0.08)
    bare = model.intercept
    assert abs(model.home_logit_constant - bare) > 0.01          # the correction is not negligible here
    zero = pd.DataFrame([[0.0, 0.0]], columns=FEATURES)
    assert abs(model.home_logit(zero)[0] - model.home_logit_constant) < 1e-9   # logit at equal teams = home-field term
