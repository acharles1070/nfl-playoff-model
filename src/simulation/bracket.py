"""Playoff bracket simulator: exact enumeration x quasi-Monte Carlo over team strength.

Model
-----
Every team carries a latent strength shock  eps_t ~ N(0, tau_t^2)  on the probit
scale, drawn ONCE per simulation and shared by all of that team's games. That
shared draw is what correlates a team's fortunes across rounds. tau_t^2 comes
from the rating filter's own posterior variance (offense, defense, quarterback).

A game between i and j, given the draws, is won by i with probability

    p(eps) = Phi( (m_ij + eps_i - eps_j) / s_ij ),   s_ij^2 = 1 - tau_i^2 - tau_j^2

where m_ij = Phi^-1(p_ij) is the probit index of the fitted game model's
single-game probability. Averaging p(eps) over eps returns p_ij (marginal
preservation), so single-game predictions stay calibrated while multi-round
outcomes get the right dependence.

Given the draws, the bracket is solved EXACTLY (Rao-Blackwellization): every
combination of game outcomes, with NFL reseeding, is enumerated, weighted, and
collapsed by surviving-team set. The only Monte Carlo noise left is in eps, and
it is reduced with scrambled Sobol points plus antithetic pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc


MIN_S = 0.30          # floor on s_ij; keeps p(eps) sane if tau is ever huge


@dataclass
class MatchupTable:
    """Everything the simulator needs about a set of teams."""

    teams: list[str]
    m_home: dict[tuple[str, str], float]      # probit index, i at home vs j
    m_neutral: dict[tuple[str, str], float]   # probit index, neutral site
    tau2: dict[str, float]                    # strength-shock variance (probit scale)


def probit_index(p: float) -> float:
    return float(norm.ppf(np.clip(p, 1e-9, 1 - 1e-9)))


def draw_shocks(
    teams: list[str],
    tau2: dict[str, float],
    n_draws: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Scrambled-Sobol normal draws, doubled with antithetic (-z) pairs."""
    half = max(n_draws // 2, 2)
    power = int(np.ceil(np.log2(half)))

    sobol = qmc.Sobol(d=len(teams), scramble=True, seed=seed)
    u = sobol.random_base2(power)[:half]
    z = norm.ppf(np.clip(u, 1e-9, 1 - 1e-9))
    z = np.vstack([z, -z])

    return {t: np.sqrt(tau2[t]) * z[:, k] for k, t in enumerate(teams)}


def game_prob(
    table: MatchupTable,
    shocks: dict[str, np.ndarray],
    home: str,
    away: str,
    *,
    neutral: bool = False,
) -> np.ndarray:
    """P(home beats away) for every draw (vector)."""
    m = (table.m_neutral if neutral else table.m_home)[(home, away)]
    s = np.sqrt(max(1.0 - table.tau2[home] - table.tau2[away], MIN_S ** 2))

    return norm.cdf((m + shocks[home] - shocks[away]) / s)


# ---------------------------------------------------------------------------
# Bracket structure
# ---------------------------------------------------------------------------

def wild_card_pairs(seeds: list[str]) -> list[tuple[str, str]]:
    """(home, away) for the Wild Card round: 2v7,3v6,4v5 or 3v6,4v5."""
    n = len(seeds)
    byes = 1 if n == 7 else 2

    first_host = byes + 1
    count = (n - byes) // 2

    return [
        (seeds[first_host - 1 + i], seeds[n - 1 - i])
        for i in range(count)
    ]


def reseed_pairs(alive: frozenset, seed_of: dict[str, int]) -> list[tuple[str, str]]:
    """Four survivors: best seed hosts the worst, next hosts the other."""
    ordered = sorted(alive, key=lambda t: seed_of[t])
    return [(ordered[0], ordered[3]), (ordered[1], ordered[2])]


def _play_round(
    states: dict[frozenset, np.ndarray],
    pairs_for,
    table: MatchupTable,
    shocks: dict[str, np.ndarray],
) -> dict[frozenset, np.ndarray]:
    """Advance every state through one round, enumerating all game outcomes."""
    out: dict[frozenset, np.ndarray] = {}

    for alive, weight in states.items():
        branches = [(weight, alive)]

        for home, away in pairs_for(alive):
            p = game_prob(table, shocks, home, away)

            expanded = []
            for w, a in branches:
                expanded.append((w * p, a - {away}))
                expanded.append((w * (1.0 - p), a - {home}))
            branches = expanded

        for w, a in branches:
            out[a] = out.get(a, 0.0) + w

    return out


def _reach(states: dict[frozenset, np.ndarray]) -> dict[str, np.ndarray]:
    reach: dict[str, np.ndarray] = {}

    for alive, weight in states.items():
        for team in alive:
            reach[team] = reach.get(team, 0.0) + weight

    return reach


def conference_distribution(
    seeds: list[str],
    table: MatchupTable,
    shocks: dict[str, np.ndarray],
    *,
    start: str = "WC",
    alive: list[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-team, per-draw probabilities of reaching each round of one conference."""
    seed_of = {t: i + 1 for i, t in enumerate(seeds)}
    n_draws = len(next(iter(shocks.values())))
    ones = np.ones(n_draws)

    result: dict[str, dict[str, np.ndarray]] = {}

    def record(round_name: str, states):
        for team, w in _reach(states).items():
            result.setdefault(team, {})[round_name] = w

    if start == "WC":
        states = {frozenset(seeds): ones}
        record("WC", states)

        pairs = wild_card_pairs(seeds)
        states = _play_round(states, lambda a: [p for p in pairs if p[0] in a], table, shocks)
    elif start == "DIV":
        states = {frozenset(alive): ones}
    elif start == "CON":
        states = {frozenset(alive): ones}
    else:
        raise ValueError(start)

    if start in ("WC", "DIV"):
        record("DIV", states)
        states = _play_round(states, lambda a: reseed_pairs(a, seed_of), table, shocks)

    record("CON", states)

    def final(a):
        hi, lo = sorted(a, key=lambda t: seed_of[t])
        return [(hi, lo)]

    states = _play_round(states, final, table, shocks)
    record("WIN_CONF", states)

    return result


def simulate_bracket(
    fields: dict[str, list[str]],
    table: MatchupTable,
    *,
    n_draws: int = 4096,
    seed: int = 0,
    start: str = "WC",
    alive: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Round-by-round advancement and Super Bowl probabilities per team."""
    teams = [t for seeds in fields.values() for t in seeds]
    shocks = draw_shocks(teams, table.tau2, n_draws, seed)

    per_conf = {
        label: conference_distribution(
            seeds,
            table,
            shocks,
            start=start,
            alive=(alive or {}).get(label),
        )
        for label, seeds in fields.items()
    }

    labels = list(fields)
    a, b = per_conf[labels[0]], per_conf[labels[1]]

    win_a = {t: v["WIN_CONF"] for t, v in a.items() if "WIN_CONF" in v}
    win_b = {t: v["WIN_CONF"] for t, v in b.items() if "WIN_CONF" in v}

    sb_win: dict[str, np.ndarray] = {}

    for ta, wa in win_a.items():
        for tb, wb in win_b.items():
            p = game_prob(table, shocks, ta, tb, neutral=True)
            joint = wa * wb
            sb_win[ta] = sb_win.get(ta, 0.0) + joint * p
            sb_win[tb] = sb_win.get(tb, 0.0) + joint * (1.0 - p)

    rows = []

    for label, dist in per_conf.items():
        seed_of = {t: i + 1 for i, t in enumerate(fields[label])}

        for team, rounds in dist.items():
            row = {
                "conference": label,
                "seed": seed_of[team],
                "team": team,
                "p_reach_div": rounds["DIV"].mean() if "DIV" in rounds else np.nan,
                "p_reach_con": rounds["CON"].mean(),
                "p_win_conf": rounds["WIN_CONF"].mean(),
                "p_win_sb": sb_win[team].mean() if team in sb_win else 0.0,
                "sb_mc_se": sb_win[team].std(ddof=1) / np.sqrt(n_draws) if team in sb_win else 0.0,
            }
            rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("p_win_sb", ascending=False)
        .reset_index(drop=True)
    )
