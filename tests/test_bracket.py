"""Bracket simulator: exact enumeration must agree with brute-force sampling."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from src.simulation.bracket import (
    MatchupTable,
    draw_shocks,
    game_prob,
    simulate_bracket,
    wild_card_pairs,
)


def _table(fields, strength, home_edge=0.25, tau2=0.03):
    teams = [t for seeds in fields.values() for t in seeds]
    m_home, m_neutral = {}, {}
    for i in teams:
        for j in teams:
            if i != j:
                d = strength[i] - strength[j]
                m_home[(i, j)] = home_edge + d
                m_neutral[(i, j)] = d
    return MatchupTable(teams, m_home, m_neutral, {t: tau2 for t in teams})


def _fields(n):
    names = {"A": [f"A{k}" for k in range(1, n + 1)], "B": [f"B{k}" for k in range(1, n + 1)]}
    return names


def _strengths(fields, rng):
    return {t: rng.normal(0, 0.5) for seeds in fields.values() for t in seeds}


def brute_force(fields, table, n_sims, seed):
    """Independent implementation: sample shocks, then sample every game."""
    rng = np.random.default_rng(seed)
    teams = table.teams
    titles = dict.fromkeys(teams, 0)
    conf_wins = dict.fromkeys(teams, 0)

    def play(h, a, shock, neutral=False):
        m = (table.m_neutral if neutral else table.m_home)[(h, a)]
        s = np.sqrt(1 - table.tau2[h] - table.tau2[a])
        p = norm.cdf((m + shock[h] - shock[a]) / s)
        return h if rng.random() < p else a

    for _ in range(n_sims):
        shock = {t: rng.normal(0, np.sqrt(table.tau2[t])) for t in teams}
        champs = []
        for seeds in fields.values():
            n = len(seeds)
            seed_of = {t: i + 1 for i, t in enumerate(seeds)}
            byes = 1 if n == 7 else 2
            alive = set(seeds[:byes])
            for h, a in wild_card_pairs(seeds):
                alive.add(play(h, a, shock))
            o = sorted(alive, key=seed_of.get)
            div = [play(o[0], o[3], shock), play(o[1], o[2], shock)]
            hi, lo = sorted(div, key=seed_of.get)
            champ = play(hi, lo, shock)
            conf_wins[champ] += 1
            champs.append(champ)
        a, b = champs
        titles[play(a, b, shock, neutral=True)] += 1

    return (
        {t: titles[t] / n_sims for t in teams},
        {t: conf_wins[t] / n_sims for t in teams},
    )


@pytest.mark.parametrize("n_teams", [7, 6])
def test_exact_matches_brute_force(n_teams):
    fields = _fields(n_teams)
    rng = np.random.default_rng(11)
    table = _table(fields, _strengths(fields, rng))

    exact = simulate_bracket(fields, table, n_draws=8192, seed=3).set_index("team")
    titles, conf = brute_force(fields, table, n_sims=60000, seed=5)

    for team in table.teams:
        assert exact.loc[team, "p_win_sb"] == pytest.approx(titles[team], abs=0.012), team
        assert exact.loc[team, "p_win_conf"] == pytest.approx(conf[team], abs=0.015), team


@pytest.mark.parametrize("n_teams", [7, 6])
def test_probabilities_are_a_proper_distribution(n_teams):
    fields = _fields(n_teams)
    table = _table(fields, _strengths(fields, np.random.default_rng(2)))
    r = simulate_bracket(fields, table, n_draws=2048, seed=1)

    assert r["p_win_sb"].sum() == pytest.approx(1.0, abs=1e-9)
    for label in ("A", "B"):
        c = r[r["conference"].eq(label)]
        assert c["p_win_conf"].sum() == pytest.approx(1.0, abs=1e-9)
        assert c["p_reach_div"].sum() == pytest.approx(4.0, abs=1e-9)
        assert c["p_reach_con"].sum() == pytest.approx(2.0, abs=1e-9)
    # reaching a later round can never be likelier than an earlier one
    assert (r["p_win_sb"] <= r["p_win_conf"] + 1e-12).all()
    assert (r["p_win_conf"] <= r["p_reach_con"] + 1e-12).all()


def test_marginal_preservation_of_single_game_probability():
    """Averaging the shock-conditional probability recovers the model's p."""
    fields = _fields(7)
    table = _table(fields, _strengths(fields, np.random.default_rng(4)), tau2=0.06)
    shocks = draw_shocks(table.teams, table.tau2, 16384, seed=9)

    for h, a in (("A2", "A7"), ("B3", "B6"), ("A1", "B1")):
        expected = norm.cdf(table.m_home[(h, a)])
        assert game_prob(table, shocks, h, a).mean() == pytest.approx(expected, abs=0.004)


def test_stronger_team_has_better_title_odds_and_seed_one_bye_helps():
    fields = _fields(7)
    equal = {t: 0.0 for seeds in fields.values() for t in seeds}
    r = simulate_bracket(fields, _table(fields, equal, home_edge=0.3), n_draws=4096, seed=0)
    r = r.set_index("team")

    # identical strength: the bye seed must be the favourite in its conference
    assert r.loc["A1", "p_win_conf"] > r.loc["A4", "p_win_conf"] > r.loc["A7", "p_win_conf"]

    boosted = dict(equal, A5=1.5)
    r2 = simulate_bracket(fields, _table(fields, boosted, home_edge=0.3), n_draws=4096, seed=0).set_index("team")
    assert r2.loc["A5", "p_win_sb"] > r.loc["A5", "p_win_sb"] * 2


def test_sobol_antithetic_draws_have_zero_mean_and_right_variance():
    shocks = draw_shocks(["X", "Y"], {"X": 0.04, "Y": 0.09}, 4096, seed=1)
    assert abs(shocks["X"].mean()) < 1e-9
    assert shocks["X"].var() == pytest.approx(0.04, rel=0.03)
    assert shocks["Y"].var() == pytest.approx(0.09, rel=0.03)
