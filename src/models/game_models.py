"""Game-outcome models: from rating features to a win probability.

All estimators expose fit(DataFrame, y) / predict_proba(DataFrame) so they plug
into src.models.walk_forward.run_walk_forward. Margin-based models read
home_score / away_score from the TRAINING frame only; predict_proba never
touches scores.

Models
------
logistic           sigmoid(a + w.x)                        (the v1 model)
margin_normal      margin ~ N(a + w.x, sigma^2),  P = Phi(mu / sigma)
margin_hetero      margin ~ N(a + w.x, s0^2 + s1 * V),    P = Phi(mu / sigma_i)
                   where V is the filter's total rating variance for THIS game:
                   a posterior-predictive win probability that widens when the
                   ratings are uncertain (early season, unknown quarterbacks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin

from src.models.backtest import build_logistic_model


EPS = 1e-6


def rating_variance(frame: pd.DataFrame) -> np.ndarray:
    """Total posterior variance of the four team components and two QBs."""
    total = (
        frame["home_pregame_off_var"]
        + frame["home_pregame_def_var"]
        + frame["away_pregame_off_var"]
        + frame["away_pregame_def_var"]
    )

    for side in ("home", "away"):
        col = f"{side}_pregame_qb_var"
        if col in frame.columns:
            total = total + frame[col].fillna(frame[col].median())

    return total.to_numpy(dtype=float)


class _MarginBase(BaseEstimator, ClassifierMixin):
    """Shared plumbing: standardize features, hold coefficients."""

    def __init__(self, features: tuple[str, ...] = (), alpha: float = 1.0):
        self.features = features
        self.alpha = alpha

    def _design(self, frame: pd.DataFrame, fit: bool = False) -> np.ndarray:
        x = frame[list(self.features)].to_numpy(dtype=float)

        if fit:
            self.fill_ = np.nanmedian(x, axis=0)
        x = np.where(np.isnan(x), self.fill_, x)

        if fit:
            self.mean_ = x.mean(axis=0)
            self.scale_ = x.std(axis=0)
            self.scale_[self.scale_ == 0] = 1.0

        return (x - self.mean_) / self.scale_

    @staticmethod
    def _margin(frame: pd.DataFrame) -> np.ndarray:
        return (frame["home_score"] - frame["away_score"]).to_numpy(dtype=float)

    def _proba(self, home_win: np.ndarray) -> np.ndarray:
        p = np.clip(home_win, EPS, 1 - EPS)
        return np.column_stack([1 - p, p])


class MarginNormal(_MarginBase):
    """Ridge regression on margin; P(home win) = Phi(mu / sigma)."""

    def fit(self, frame: pd.DataFrame, y=None):
        x = self._design(frame, fit=True)
        m = self._margin(frame)

        n, k = x.shape
        xa = np.column_stack([np.ones(n), x])

        penalty = np.diag([0.0] + [self.alpha] * k)
        self.beta_ = np.linalg.solve(xa.T @ xa + penalty, xa.T @ m)

        resid = m - xa @ self.beta_
        self.sigma_ = float(np.sqrt(resid.var(ddof=k + 1)))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        x = self._design(frame)
        mu = self.beta_[0] + x @ self.beta_[1:]
        return self._proba(norm.cdf(mu / self.sigma_))


class MarginHetero(_MarginBase):
    """Margin ~ N(mu_i, s0^2 + s1 * V_i) fit by maximum likelihood."""

    def fit(self, frame: pd.DataFrame, y=None):
        x = self._design(frame, fit=True)
        m = self._margin(frame)
        v = rating_variance(frame)

        self.v_scale_ = float(np.median(v))
        vs = v / self.v_scale_

        n, k = x.shape
        xa = np.column_stack([np.ones(n), x])

        start_beta = np.linalg.lstsq(xa, m, rcond=None)[0]
        start_var = float(np.var(m - xa @ start_beta))

        def nll(theta):
            beta = theta[: k + 1]
            s0 = np.exp(theta[k + 1])
            s1 = np.exp(theta[k + 2])
            mu = xa @ beta
            var = s0 + s1 * vs
            return 0.5 * np.sum(np.log(var) + (m - mu) ** 2 / var) + (
                0.5 * self.alpha * np.sum(beta[1:] ** 2)
            )

        theta0 = np.concatenate([start_beta, [np.log(start_var * 0.9), np.log(start_var * 0.1)]])
        fit = minimize(nll, theta0, method="L-BFGS-B")

        self.beta_ = fit.x[: k + 1]
        self.s0_ = float(np.exp(fit.x[k + 1]))
        self.s1_ = float(np.exp(fit.x[k + 2]))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        x = self._design(frame)
        mu = self.beta_[0] + x @ self.beta_[1:]
        var = self.s0_ + self.s1_ * rating_variance(frame) / self.v_scale_
        return self._proba(norm.cdf(mu / np.sqrt(var)))


def factory(kind: str, **kwargs):
    """Return a model_factory(feature_columns) for run_walk_forward."""
    if kind == "logistic":
        return build_logistic_model
    if kind == "margin_normal":
        return lambda cols: MarginNormal(features=tuple(cols), **kwargs)
    if kind == "margin_hetero":
        return lambda cols: MarginHetero(features=tuple(cols), **kwargs)
    raise ValueError(kind)
