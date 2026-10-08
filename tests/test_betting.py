"""Payout, Kelly and Deflated-Sharpe math must be exactly right."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.betting.sim import decimal_odds, expected_value, kelly_fraction, place_bets
from src.betting.stats import deflated_sharpe, expected_max_sharpe, max_drawdown, sharpe


def test_decimal_odds_conversion():
    assert decimal_odds([+100, -200, +300, -110]) == pytest.approx([2.0, 1.5, 4.0, 1 + 100 / 110])


def test_expected_value_and_kelly_known_values():
    assert expected_value(np.array([0.6]), np.array([2.0]))[0] == pytest.approx(0.2)
    assert kelly_fraction(np.array([0.6]), np.array([2.0]))[0] == pytest.approx(0.2)       # (1.2-1)/1
    assert kelly_fraction(np.array([0.4]), np.array([2.0]))[0] == 0.0                      # no edge, no bet
    assert expected_value(np.array([0.5]), np.array([1.909]))[0] < 0                       # -110 both sides is a losing price


def test_a_fair_coin_at_standard_vig_loses_about_the_vig():
    rng = np.random.default_rng(0)
    n = 20000
    g = pd.DataFrame({"season": 2020, "week": np.arange(n) % 17 + 1, "game_id": [f"g{i}" for i in range(n)],
                      "p_home": 0.5, "home_moneyline": -110.0, "away_moneyline": -110.0,
                      "home_win": rng.integers(0, 2, n)})
    bets = place_bets(g.assign(p_home=0.5 + 0.1), threshold=-1.0)       # bet everything
    # betting the home side at -110 every game on fair coins: ROI = -(vig)/2 ~ -4.5%
    assert bets["ret"].mean() == pytest.approx(-0.0455, abs=0.02)


def test_flat_pnl_is_exact_for_one_bet():
    g = pd.DataFrame({"season": [2020], "week": [1], "game_id": ["a"], "p_home": [0.7],
                      "home_moneyline": [150.0], "away_moneyline": [-170.0], "home_win": [1]})
    b = place_bets(g, threshold=0.0)
    assert b["side"].iloc[0] == "home" and b["pnl"].iloc[0] == pytest.approx(1.5)


def test_drawdown_and_sharpe():
    assert max_drawdown(np.array([100, 120, 90, 130])) == pytest.approx(-0.25)
    assert sharpe(np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])) == pytest.approx(0.0, abs=1e-9)


def test_deflated_sharpe_penalizes_many_trials():
    rng = np.random.default_rng(1)
    r = rng.normal(0.05, 1.0, 400)                      # a modest positive Sharpe
    few = deflated_sharpe(r, n_trials=1, sharpe_var=0.003)
    many = deflated_sharpe(r, n_trials=200, sharpe_var=0.003)
    assert many < few                                     # searching more makes luck more likely
    assert expected_max_sharpe(200, 0.003) > expected_max_sharpe(5, 0.003) > 0
