"""De-vig methods must return proper probabilities that preserve the ordering."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.market_models import (
    devig_additive, devig_multiplicative, devig_power, devig_shin, fit_logistic, predict_logistic,
)

QH = np.array([0.60, 0.52, 0.80, 0.30])
QA = np.array([0.44, 0.53, 0.24, 0.76])      # each pair sums to > 1 (the vig)


@pytest.mark.parametrize("fn", [devig_multiplicative, devig_additive, devig_power, devig_shin])
def test_devig_gives_valid_probabilities_close_to_each_other(fn):
    p = fn(QH, QA)
    assert ((p > 0) & (p < 1)).all()
    assert np.allclose(p, devig_multiplicative(QH, QA), atol=0.025)    # methods differ most at heavy favorites
    assert (np.argsort(p) == np.argsort(QH / (QH + QA))).all()         # same ordering


def test_power_and_shin_remove_the_full_margin():
    for fn in (devig_power, devig_shin):
        p_home = fn(QH, QA)
        p_away = fn(QA, QH)
        assert np.allclose(p_home + p_away, 1.0, atol=1e-6)


def test_logistic_recalibration_recovers_a_known_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 40000)
    y = (rng.random(40000) < 1 / (1 + np.exp(-(0.3 + 1.4 * x)))).astype(float)
    w = fit_logistic(x[:, None], y)
    assert w[0] == pytest.approx(0.3, abs=0.05)
    assert w[1] == pytest.approx(1.4, abs=0.06)
    assert predict_logistic(w, np.array([[0.0]]))[0] == pytest.approx(1 / (1 + np.exp(-0.3)), abs=0.01)
