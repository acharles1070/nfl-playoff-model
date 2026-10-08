"""Performance statistics for a bet series, including the Deflated Sharpe Ratio."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER_GAMMA = 0.5772156649015329


def sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, float)
    return float(r.mean() / r.std(ddof=1)) if len(r) > 2 and r.std(ddof=1) > 0 else 0.0


def max_drawdown(equity: np.ndarray) -> float:
    e = np.asarray(equity, float)
    peak = np.maximum.accumulate(e)
    return float(((e - peak) / peak).min())


def roi_ci(returns: np.ndarray, n_boot: int = 4000, seed: int = 0):
    r = np.asarray(returns, float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(r, size=(n_boot, len(r))).mean(axis=1)
    return float(r.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def expected_max_sharpe(n_trials: int, sharpe_var: float) -> float:
    """Expected best Sharpe among n independent trials with zero true skill (Bailey & Lopez de Prado)."""
    if n_trials < 2:
        return 0.0
    return float(np.sqrt(sharpe_var) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1.0 / n_trials)
                                        + EULER_GAMMA * norm.ppf(1 - 1.0 / (n_trials * np.e))))


def deflated_sharpe(returns: np.ndarray, n_trials: int, sharpe_var: float) -> float:
    """Probability that the true Sharpe exceeds what luck alone would produce across n_trials."""
    r = np.asarray(returns, float)
    sr, T = sharpe(r), len(r)
    sr0 = expected_max_sharpe(n_trials, sharpe_var)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    denom = np.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom))
