"""Opponent forecasts: horizon shrinkage and bands, who-plays-whom tracking, the planner's chalk bracket, the report.

The data-dependent checks at the bottom skip when data/processed is absent, e.g. on CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import configured_path, project_path
from src.opponents.forecast import HORIZON_SHRINK_A, revision_band, shrink_logit, sigmoid
from src.opponents.paths import meeting_matrices, opponents, reach_probability, wins_distribution
from src.opponents.paths_backtest import chalk_matchups
from src.opponents.report import ordinal, swap_qb
from src.simulation.divisions import DIVISIONS
from src.simulation.season import SeasonState, simulate_season

from test_season import _no_games, _rng_strength, _table  # noqa: E402  (tests/ is on sys.path)


# ---------------------------------------------------------------------------
# shrinkage and bands
# ---------------------------------------------------------------------------

def test_shrink_is_identity_at_zero_weeks_and_pulls_toward_even_odds():
    z = np.array([-2.0, -0.5, 0.0, 0.7, 3.0])
    assert np.allclose(shrink_logit(z, 0), z)
    far = shrink_logit(z, 10)
    assert np.all(np.abs(far) <= np.abs(z) + 1e-12) and np.all(np.sign(far) == np.sign(z))
    assert np.allclose(shrink_logit(z, 10), z / (1 + HORIZON_SHRINK_A * 10))


def test_shrink_never_makes_a_closer_game_further_away():
    z = np.linspace(-3, 3, 13)
    for h in range(0, 18):
        assert np.all(np.abs(shrink_logit(z, h + 1)) <= np.abs(shrink_logit(z, h)) + 1e-12)


def test_revision_band_is_centered_on_the_logit_and_widens_with_sd():
    z = np.array([-1.0, 0.0, 1.5])
    lo1, hi1 = revision_band(z, 0.2)
    lo2, hi2 = revision_band(z, 0.5)
    assert np.all(lo1 < sigmoid(z)) and np.all(sigmoid(z) < hi1)
    assert np.all(lo2 < lo1) and np.all(hi2 > hi1)
    # symmetric on the logit scale
    assert np.allclose(np.log(hi1 / (1 - hi1)) - z, z - np.log(lo1 / (1 - lo1)))


def test_revision_band_at_80_percent_uses_the_normal_quantile():
    lo, hi = revision_band(np.array([0.0]), 1.0, 0.80)
    assert np.isclose(np.log(hi[0] / (1 - hi[0])), 1.2815515655446004)


# ---------------------------------------------------------------------------
# who-plays-whom tracking in the season simulator
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tracked():
    from src.simulation.season_backtest import league_state  # noqa: F401  (import check only)

    state = SeasonState({t: 0.0 for d in DIVISIONS.values() for ts in d.values() for t in ts}, _no_games(), 7)
    table = _table(_rng_strength(5))
    detail: dict = {}
    with_detail = simulate_season(state, table, n_sims=4096, seed=11, seed_probs=True, detail=detail)
    without = simulate_season(state, table, n_sims=4096, seed=11, seed_probs=True)
    return with_detail, without, detail


def test_tracking_does_not_change_the_simulation(tracked):
    with_detail, without, _ = tracked
    pd.testing.assert_frame_equal(with_detail, without)


def test_every_round_has_the_right_number_of_games(tracked):
    _, _, d = tracked
    for rnd, games in {"WC": 6, "DIV": 4, "CON": 2, "SB": 1}.items():
        assert np.isclose(d["hosted"][rnd].sum() / d["n_sims"], games)


def test_meeting_probabilities_are_symmetric_and_reach_sums_to_two_per_game(tracked):
    _, _, d = tracked
    for rnd, games in {"WC": 6, "DIV": 4, "CON": 2, "SB": 1}.items():
        p, w = meeting_matrices(d, rnd)
        assert np.allclose(p, p.T)
        assert np.isclose(reach_probability(d, rnd).sum(), 2 * games)
        both = ~np.isnan(w) & ~np.isnan(w.T)
        assert np.allclose((w + w.T)[both], 1.0)
        assert np.allclose(np.diag(p), 0)


def test_teams_only_meet_inside_their_own_conference_until_the_super_bowl(tracked):
    from src.simulation.divisions import CONFERENCE_OF

    _, _, d = tracked
    teams = d["teams"]
    for rnd in ("WC", "DIV", "CON"):
        p, _ = meeting_matrices(d, rnd)
        for i, j in zip(*np.nonzero(p)):
            assert CONFERENCE_OF[teams[i]] == CONFERENCE_OF[teams[j]]
    p, _ = meeting_matrices(d, "SB")
    for i, j in zip(*np.nonzero(p)):
        assert CONFERENCE_OF[teams[i]] != CONFERENCE_OF[teams[j]]


def test_reach_matches_the_simulator_summary(tracked):
    with_detail, _, d = tracked
    s = with_detail.set_index("team")
    teams = d["teams"]
    assert np.allclose(reach_probability(d, "SB"), [s.loc[t, "p_win_conf"] for t in teams])
    assert np.allclose(reach_probability(d, "DIV"), [s.loc[t, "p_reach_div_round"] for t in teams])


def test_opponent_table_is_a_distribution(tracked):
    _, _, d = tracked
    team = d["teams"][0]
    o = opponents(d, team, "DIV")
    assert np.isclose(o["p_given_reach"].sum(), 1.0) and np.isclose(o["p_meet"].sum(), reach_probability(d, "DIV")[0])
    assert o["p_meet"].is_monotonic_decreasing


def test_wins_distribution_sums_to_one_and_matches_expected_wins(tracked):
    with_detail, _, d = tracked
    w = wins_distribution(d, d["teams"][3])
    assert np.isclose(w["p"].sum(), 1.0)
    assert np.isclose((w["wins"] * w["p"]).sum(), with_detail.set_index("team").loc[d["teams"][3], "expected_wins"])


# ---------------------------------------------------------------------------
# the planner's chalk bracket
# ---------------------------------------------------------------------------

def _field(n):
    return {"AFC": [f"A{i}" for i in range(1, n + 1)], "NFC": [f"N{i}" for i in range(1, n + 1)]}


def test_chalk_bracket_with_seven_seeds():
    c = chalk_matchups(_field(7))
    assert c["WC"]["A2"] == "A7" and c["WC"]["A3"] == "A6" and c["WC"]["A4"] == "A5" and "A1" not in c["WC"]
    assert c["DIV"]["A1"] == "A4" and c["DIV"]["A2"] == "A3"
    assert c["CON"]["A1"] == "A2" and c["CON"]["N1"] == "N2"
    assert c["SB"]["A1"] == "N1"


def test_chalk_bracket_with_six_seeds_gives_two_byes():
    c = chalk_matchups(_field(6))
    assert "A1" not in c["WC"] and "A2" not in c["WC"] and c["WC"]["A3"] == "A6" and c["WC"]["A4"] == "A5"
    assert c["DIV"]["A1"] == "A4" and c["DIV"]["A2"] == "A3"


# ---------------------------------------------------------------------------
# report helpers
# ---------------------------------------------------------------------------

def test_swap_qb_replaces_only_that_quarterback():
    frame = pd.DataFrame({"home_pregame_off": [0.05], "away_pregame_off": [-0.02], "home_pregame_qb": [0.06], "away_pregame_qb": [0.01],
                          "home_pregame_off_total": [0.11], "away_pregame_off_total": [-0.01], "diff_rating_off_total": [0.12]})
    out = swap_qb(frame, "away", -0.01)
    assert out["away_pregame_qb"].iloc[0] == -0.01 and out["home_pregame_qb"].iloc[0] == 0.06
    assert np.isclose(out["away_pregame_off_total"].iloc[0], -0.02 - 0.01)
    assert np.isclose(out["diff_rating_off_total"].iloc[0], 0.11 - (-0.03))
    assert frame["diff_rating_off_total"].iloc[0] == 0.12                    # the input is not modified


@pytest.mark.parametrize("n,s", [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (12, "12th"), (13, "13th"), (21, "21st"), (22, "22nd"), (32, "32nd")])
def test_ordinal(n, s):
    assert ordinal(n) == s


# ---------------------------------------------------------------------------
# data-dependent checks
# ---------------------------------------------------------------------------

needs_data = pytest.mark.skipif(not (configured_path("processed") / "games.parquet").exists(), reason="processed data not built")


@needs_data
def test_revision_sd_table_grows_with_the_horizon_and_the_code_constant_matches_the_fit():
    path = project_path("docs", "opponent_revision_sd.csv")
    assert path.exists()
    sd = pd.read_csv(path).set_index("h")
    assert sd["sd_all"].diff().dropna().gt(-0.02).all() and sd["sd_all"].iloc[-1] > 3 * sd["sd_all"].iloc[0]     # grows with distance, up to noise
    assert sd["sd_qb_changed"].dropna().gt(sd["sd_same_qb"].dropna()).all()
    fit = project_path("docs", "opponent_shrink.json")
    if fit.exists():
        import json

        assert abs(json.loads(fit.read_text())["a"] - HORIZON_SHRINK_A) < 5e-4, "re-fit the shrinkage and update HORIZON_SHRINK_A"


@needs_data
def test_report_for_a_finished_season_uses_only_games_before_the_as_of_week():
    from src.opponents.report import build_report
    from src.simulation.live_bracket import load_inputs

    inp = load_inputs()
    r = build_report(inp, "KC", 2024, 10, n_sims=4096)
    games = r["schedule"]["games"]
    assert games and min(g["week"] for g in games) >= 10
    assert r["data_through"] < str(inp.games.loc[inp.games["season"].eq(2024) & inp.games["week"].eq(10), "gameday"].min())[:10]
    for g in games:
        assert 0 < g["p_win"] < 1 and g["band_lo"] <= g["p_win"] <= g["band_hi"]
        assert g["band_if_qb_change_lo"] <= g["band_lo"] + 1e-9 and g["band_if_qb_change_hi"] >= g["band_hi"] - 1e-9
    o = r["outlook"]
    assert np.isclose(sum(x["p"] for x in o["wins_distribution"]), 1.0, atol=1e-3)
    assert [p["round"] for p in r["playoff_path"]] == ["WC", "DIV", "CON", "SB"]
    # a first seed skips the Wild Card round, so "plays in it" can be smaller than "plays in the Divisional round"
    assert r["playoff_path"][1]["p_reach"] >= r["playoff_path"][2]["p_reach"] >= r["playoff_path"][3]["p_reach"]


# ---------------------------------------------------------------------------
# as-of state of the league
# ---------------------------------------------------------------------------

def _mini_games(played_through: int, straggler: bool = False) -> pd.DataFrame:
    rows = []
    for week in (1, 2, 3):
        for home, away in (("KC", "DEN"), ("BUF", "MIA")):
            played = week <= played_through
            rows.append({"season": 2030, "game_type": "REG", "week": week, "home_team": home, "away_team": away, "location": "Home",
                         "home_score": 24.0 if played else np.nan, "away_score": 17.0 if played else np.nan, "is_played": played})
    g = pd.DataFrame(rows)
    if straggler:
        g.loc[(g["week"] == 2) & (g["home_team"] == "BUF"), ["home_score", "away_score", "is_played"]] = [np.nan, np.nan, False]
    return g


def test_state_before_agrees_with_league_state_when_everything_is_played():
    from src.opponents.paths import state_before
    from src.simulation.season_backtest import league_state

    g = _mini_games(3)
    for week in (1, 2, 3):
        w1, r1 = state_before(g, 2030, week)
        w2, r2, _ = league_state(g, 2030, week)
        assert w1 == w2 and r1.equals(r2)


def test_state_before_keeps_an_earlier_unplayed_game_in_the_remaining_schedule():
    from src.opponents.paths import state_before

    g = _mini_games(3, straggler=True).assign(is_played=lambda d: d["is_played"] & (d["week"] <= 2))
    g.loc[g["week"] == 3, "is_played"] = False
    wins, remaining = state_before(g, 2030, 3)                    # as of before week 3; BUF-MIA of week 2 never happened
    assert wins["KC"] == 2 and wins["BUF"] == 1 and wins["MIA"] == 0
    assert len(remaining) == 3 and {("BUF", "MIA")} <= set(zip(remaining["home"], remaining["away"]))
