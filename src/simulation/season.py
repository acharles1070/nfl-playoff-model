"""Season simulator: remaining schedule -> division winners -> seeds -> playoffs.

For every simulated season (one draw of per-team latent strength shocks, shared
by every game that team plays, regular season and playoffs):

  1. every remaining regular-season game is played with
        p = Phi((m + eps_home - eps_away) / s)            (see bracket.py)
  2. division winners and wild cards are resolved from final records
     (ties broken at random: the NFL's real tiebreakers are far more elaborate,
     so this is the main approximation),
  3. the playoff bracket is played with NFL reseeding, Super Bowl on a neutral
     field.

Everything is vectorized across simulations. Shocks use scrambled Sobol points
with antithetic pairs, so single-game marginals stay calibrated and shared
uncertainty is correlated across the whole season.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.simulation.bracket import MIN_S, MatchupTable, draw_shocks
from src.simulation.divisions import DIVISIONS


@dataclass
class SeasonState:
    """The league as of some point in a season."""

    wins: dict[str, float]                 # current wins (ties count 0.5)
    remaining: pd.DataFrame                # columns: home, away, neutral
    n_seeds: int = 7                       # 7 from 2020, 6 before


def _matrices(table: MatchupTable, teams: list[str]):
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    m_home = np.zeros((n, n))
    m_neutral = np.zeros((n, n))

    for (a, b), v in table.m_home.items():
        m_home[idx[a], idx[b]] = v
    for (a, b), v in table.m_neutral.items():
        m_neutral[idx[a], idx[b]] = v

    tau2 = np.array([table.tau2[t] for t in teams])
    return idx, m_home, m_neutral, tau2


def simulate_season(
    state: SeasonState,
    table: MatchupTable,
    *,
    n_sims: int = 32768,
    seed: int = 0,
    seed_probs: bool = False,
) -> pd.DataFrame:
    """Simulate the rest of the season and the playoffs.

    seed_probs=True adds p_seed_1..p_seed_N (probability of finishing as each playoff seed). The default
    leaves the output columns, and so the logged title-odds ledger, exactly as before.
    """
    teams = sorted(state.wins)
    n = len(teams)
    idx, m_home, m_neutral, tau2 = _matrices(table, teams)

    shocks = draw_shocks(teams, table.tau2, n_sims, seed)
    eps = np.vstack([shocks[t] for t in teams])          # (n, S)
    S = eps.shape[1]
    rng = np.random.default_rng(seed + 1)

    def prob(h, a, neutral=False):
        """P(h beats a) for every simulation; h, a are team indices (scalars)."""
        m = (m_neutral if neutral else m_home)[h, a]
        s = np.sqrt(max(1.0 - tau2[h] - tau2[a], MIN_S ** 2))
        return norm.cdf((m + eps[h] - eps[a]) / s)

    # ---- 1) remaining regular season -------------------------------------
    wins = np.tile(np.array([state.wins[t] for t in teams])[:, None], (1, S)).astype(float)

    for g in state.remaining.itertuples(index=False):
        h, a = idx[g.home], idx[g.away]
        home_wins = rng.random(S) < prob(h, a, bool(g.neutral))
        wins[h] += home_wins
        wins[a] += ~home_wins

    score = wins + rng.random(wins.shape) * 0.4          # random tiebreak (< one win)

    # ---- 2) divisions, seeds ------------------------------------------------
    seeds = {}
    made_playoffs = np.zeros((n, S), dtype=bool)
    won_division = np.zeros((n, S), dtype=bool)
    first_seed = np.zeros((n, S), dtype=bool)
    seed_counts = np.zeros((n, state.n_seeds))

    for conf, divs in DIVISIONS.items():
        winners, ranks = [], []
        members = [idx[t] for teams_ in divs.values() for t in teams_]

        for div_teams in divs.values():
            ids = np.array([idx[t] for t in div_teams])
            best = ids[np.argmax(score[ids], axis=0)]                 # (S,)
            winners.append(best)
            won_division[best, np.arange(S)] = True

        winners = np.vstack(winners)                                   # (4, S)
        win_scores = np.take_along_axis(score, winners, axis=0)
        order = np.argsort(-win_scores, axis=0)
        seeded_winners = np.take_along_axis(winners, order, axis=0)    # seeds 1-4

        masked = score[members].copy()
        position = np.full(n, -1)
        position[members] = np.arange(len(members))      # team index -> row in `masked`
        for w in winners:
            masked[position[w], np.arange(S)] = -np.inf
        n_wild = state.n_seeds - 4
        wild_order = np.argsort(-masked, axis=0)[:n_wild]
        wild_cards = np.array(members)[wild_order]                      # (n_wild, S)

        conf_seeds = np.vstack([seeded_winners, wild_cards]).T          # (S, n_seeds)
        seeds[conf] = conf_seeds

        for k in range(state.n_seeds):
            made_playoffs[conf_seeds[:, k], np.arange(S)] = True
            seed_counts[:, k] += np.bincount(conf_seeds[:, k], minlength=n)
        first_seed[conf_seeds[:, 0], np.arange(S)] = True

    # ---- 3) playoffs ------------------------------------------------------------
    rows_ = np.arange(S)

    def play(h_idx, a_idx, neutral=False):
        m = (m_neutral if neutral else m_home)[h_idx, a_idx]
        s = np.sqrt(np.maximum(1.0 - tau2[h_idx] - tau2[a_idx], MIN_S ** 2))
        p = norm.cdf((m + eps[h_idx, rows_] - eps[a_idx, rows_]) / s)
        return rng.random(S) < p

    champions, conf_wins, reached = {}, np.zeros((n, S), dtype=bool), np.zeros((n, S), dtype=bool)

    for conf, sd in seeds.items():
        n_seeds = state.n_seeds
        byes = 1 if n_seeds == 7 else 2

        survivors = [np.full(S, k) for k in range(1, byes + 1)]          # seed numbers

        first_host = byes + 1
        for i in range((n_seeds - byes) // 2):
            hs, as_ = first_host + i, n_seeds - i
            h, a = sd[:, hs - 1], sd[:, as_ - 1]
            win = play(h, a)
            survivors.append(np.where(win, hs, as_))

        surv = np.sort(np.vstack(survivors).T, axis=1)                   # (S, 4) seed numbers
        team = lambda seed_nums: sd[rows_, seed_nums - 1]

        for k in range(4):
            reached[team(surv[:, k]), rows_] = True

        # divisional round: 1 v 4, 2 v 3
        div_winner_seed = []
        for hi, lo in ((0, 3), (1, 2)):
            h, a = team(surv[:, hi]), team(surv[:, lo])
            win = play(h, a)
            div_winner_seed.append(np.where(win, surv[:, hi], surv[:, lo]))

        finalists = np.sort(np.vstack(div_winner_seed).T, axis=1)
        h, a = team(finalists[:, 0]), team(finalists[:, 1])
        win = play(h, a)
        champ = np.where(win, h, a)
        champions[conf] = champ
        conf_wins[champ, rows_] = True

    conferences = list(champions)
    a, b = champions[conferences[0]], champions[conferences[1]]
    win_a = play(a, b, neutral=True)
    sb_champ = np.where(win_a, a, b)

    title = np.zeros((n, S), dtype=bool)
    title[sb_champ, rows_] = True

    out = pd.DataFrame(
        {
            "team": teams,
            "expected_wins": wins.mean(axis=1),
            "p_playoffs": made_playoffs.mean(axis=1),
            "p_division": won_division.mean(axis=1),
            "p_first_seed": first_seed.mean(axis=1),
            "p_reach_div_round": reached.mean(axis=1),
            "p_win_conf": conf_wins.mean(axis=1),
            "p_win_sb": title.mean(axis=1),
        }
    )
    out["p_win_sb_se"] = np.sqrt(out["p_win_sb"] * (1 - out["p_win_sb"]) / S)
    if seed_probs:
        for k in range(state.n_seeds):
            out[f"p_seed_{k + 1}"] = seed_counts[:, k] / S
    return out.sort_values("p_win_sb", ascending=False).reset_index(drop=True)
