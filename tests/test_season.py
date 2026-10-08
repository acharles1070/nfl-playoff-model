"""Season simulator: must agree with the exact bracket solver and basic identities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.simulation.bracket import MatchupTable, simulate_bracket
from src.simulation.divisions import DIVISIONS
from src.simulation.season import SeasonState, simulate_season


ALL_TEAMS = sorted(t for d in DIVISIONS.values() for ts in d.values() for t in ts)


def _table(strength, home_edge=0.25, tau2=0.03):
    m_home, m_neutral = {}, {}
    for i in ALL_TEAMS:
        for j in ALL_TEAMS:
            if i != j:
                d = strength[i] - strength[j]
                m_home[(i, j)] = home_edge + d
                m_neutral[(i, j)] = d
    return MatchupTable(ALL_TEAMS, m_home, m_neutral, {t: tau2 for t in ALL_TEAMS})


def _rng_strength(seed=0, scale=0.4):
    rng = np.random.default_rng(seed)
    return {t: rng.normal(0, scale) for t in ALL_TEAMS}


def _no_games():
    return pd.DataFrame(columns=["home", "away", "neutral"])


def _fixed_wins():
    """Distinct records so every seed is deterministic (gaps of 0.5 > tiebreak noise)."""
    wins = {}
    for conf, divs in DIVISIONS.items():
        for k, teams in enumerate(divs.values()):
            wins[teams[0]] = 14 - k              # division winners: 14, 13, 12, 11
            wins[teams[1]] = 10.5 - 0.5 * k      # best runners-up: 10.5, 10, 9.5, 9
            wins[teams[2]] = 6.0
            wins[teams[3]] = 3.0
    return wins


def _expected_fields(wins):
    fields = {}
    for conf, divs in DIVISIONS.items():
        winners = sorted((ts[0] for ts in divs.values()), key=lambda t: -wins[t])
        runners = sorted((ts[1] for ts in divs.values()), key=lambda t: -wins[t])[:3]
        fields[conf] = winners + runners
    return fields


@pytest.mark.parametrize("n_seeds", [7, 6])
def test_season_sim_matches_exact_bracket_on_a_fixed_field(n_seeds):
    wins = _fixed_wins()
    table = _table(_rng_strength(3))
    fields = _expected_fields(wins)
    if n_seeds == 6:
        fields = {c: f[:6] for c, f in fields.items()}

    sim = simulate_season(SeasonState(wins, _no_games(), n_seeds), table, n_sims=65536, seed=4).set_index("team")
    exact = simulate_bracket(fields, table, n_draws=8192, seed=1).set_index("team")

    for team in exact.index:
        assert sim.loc[team, "p_win_sb"] == pytest.approx(exact.loc[team, "p_win_sb"], abs=0.012), team
        assert sim.loc[team, "p_win_conf"] == pytest.approx(exact.loc[team, "p_win_conf"], abs=0.015), team


@pytest.mark.parametrize("n_seeds", [7, 6])
def test_identities_hold_with_a_real_remaining_schedule(n_seeds):
    rng = np.random.default_rng(8)
    games = pd.DataFrame(
        [(h, a, False) for h, a in (rng.choice(ALL_TEAMS, 2, replace=False) for _ in range(180))],
        columns=["home", "away", "neutral"],
    )
    state = SeasonState({t: float(rng.integers(0, 6)) for t in ALL_TEAMS}, games, n_seeds)
    r = simulate_season(state, _table(_rng_strength(5)), n_sims=16384, seed=2)

    assert r["p_win_sb"].sum() == pytest.approx(1.0, abs=1e-9)
    assert r["p_playoffs"].sum() == pytest.approx(2 * n_seeds, abs=1e-9)
    assert r["p_division"].sum() == pytest.approx(8.0, abs=1e-9)
    assert r["p_first_seed"].sum() == pytest.approx(2.0, abs=1e-9)
    assert r["p_win_conf"].sum() == pytest.approx(2.0, abs=1e-9)
    assert (r["p_win_sb"] <= r["p_win_conf"] + 1e-12).all()
    assert (r["p_win_conf"] <= r["p_reach_div_round"] + 1e-12).all()
    assert (r["p_division"] <= r["p_playoffs"] + 1e-12).all()


def test_a_stronger_team_wins_more_and_the_game_marginal_is_preserved():
    games = pd.DataFrame([("BUF", "MIA", False)] * 1, columns=["home", "away", "neutral"])
    strength = _rng_strength(2, 0.0)
    table = _table(strength, tau2=0.06)
    r = simulate_season(SeasonState({t: 0.0 for t in ALL_TEAMS}, games), table, n_sims=65536, seed=7).set_index("team")

    # one game, so BUF's expected wins is exactly the model's home-win probability
    assert r.loc["BUF", "expected_wins"] == pytest.approx(norm.cdf(table.m_home[("BUF", "MIA")]), abs=0.006)
    assert r.loc["BUF", "expected_wins"] + r.loc["MIA", "expected_wins"] == pytest.approx(1.0, abs=1e-9)

    boosted = dict(strength, KC=1.5)
    r2 = simulate_season(SeasonState({t: 0.0 for t in ALL_TEAMS}, _no_games()), _table(boosted), n_sims=32768, seed=1).set_index("team")
    assert r2.loc["KC", "p_win_sb"] > 5 * (1 / 32)
