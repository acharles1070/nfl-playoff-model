"""Bracket projection: field reconstruction, bracket building and the replay scoring, on synthetic data.

The data-dependent checks at the bottom (real seasons) skip when data/processed is absent, e.g. on CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.simulation.bracket_replay import field_score, seed_chalk, standings_field, truncate
from src.simulation.divisions import CONFERENCE_OF, DIVISIONS
from src.simulation.live_bracket import (
    ROUNDS, build_bracket, field_from_wildcard_round, make_game, phase_info, projected_field,
)

SEASON = 2030
AFC, NFC = list(DIVISIONS["AFC"].values()), list(DIVISIONS["NFC"].values())


def _season(wc_pairs: dict[str, list[tuple[str, str]]], records: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """A season with just what the field builder reads: each team's record (as one home win/loss pair of games) plus WC rows."""
    rows = []
    for i, (team, (wins, pd_)) in enumerate(records.items()):
        # one synthetic regular-season game per team against a fixed dummy opponent encodes win pct and point differential
        n = 17
        w = int(round(wins * 2))                     # wins in half-win units: 0.5 = tie
        for k in range(n):
            won = k < wins
            rows.append({"game_id": f"{SEASON}_{k:02d}_{team}", "season": SEASON, "week": k + 1, "game_type": "REG", "home_team": team,
                         "away_team": f"OPP{team}", "home_score": 20 + (pd_ if won else 0) / 17 + (1 if won else -1), "away_score": 20,
                         "is_postseason": False, "is_played": True})
    for conf, pairs in wc_pairs.items():
        for j, (h, a) in enumerate(pairs):
            rows.append({"game_id": f"{SEASON}_19_{a}_{h}", "season": SEASON, "week": 19, "game_type": "WC", "home_team": h, "away_team": a,
                         "home_score": np.nan, "away_score": np.nan, "is_postseason": True, "is_played": False})
    return pd.DataFrame(rows)


# -- a tie that the records cannot break (two division winners at 11-6), as in the 2025 NFC ----------------------

def _tied_field_games() -> tuple[pd.DataFrame, dict]:
    wins = {t: 4.0 for ts in DIVISIONS["AFC"].values() for t in ts}
    wins.update({t: 4.0 for ts in DIVISIONS["NFC"].values() for t in ts})
    # AFC: East BUF 14, North BAL 13, South JAX 12, West DEN 11; wild cards KC 11, LAC 10, MIA 10
    afc = {"BUF": 14, "BAL": 13, "JAX": 12, "DEN": 11, "KC": 11, "LAC": 10, "MIA": 10}
    # NFC: West SEA 14; two division winners tied at 11 (PHI East, CHI North); CAR South 9; wild cards LAR 12, SF 12, GB 11
    nfc = {"SEA": 14, "PHI": 11, "CHI": 11, "CAR": 9, "LAR": 12, "SF": 12, "GB": 11}
    wins.update({**afc, **nfc})
    recs = {t: (w, 0) for t, w in wins.items()}
    # true seeds: AFC BUF BAL JAX DEN | KC LAC MIA ; NFC SEA CHI PHI CAR | LAR SF GB  (note CHI is the 2 seed, PHI the 3)
    wc = {"AFC": [("BAL", "MIA"), ("JAX", "LAC"), ("DEN", "KC")], "NFC": [("CHI", "GB"), ("PHI", "SF"), ("CAR", "LAR")]}
    return _season(wc, recs), {"AFC": ["BUF", "BAL", "JAX", "DEN", "KC", "LAC", "MIA"], "NFC": ["SEA", "CHI", "PHI", "CAR", "LAR", "SF", "GB"]}


def test_wild_card_field_uses_pairing_consistency_to_resolve_record_ties():
    games, expected = _tied_field_games()
    got = field_from_wildcard_round(games, SEASON)
    # BUF/SEA (the divisions with no host) are the byes; visitors' seeds follow from who they play
    assert got["AFC"][0] == "BUF" and got["NFC"][0] == "SEA"
    assert got["NFC"][4:] == ["LAR", "SF", "GB"], "visitor seeds must come from the pairing, not from a guess at host order"
    assert got["NFC"][1:4] == ["CHI", "PHI", "CAR"]
    assert got == expected


def test_each_visitor_is_paired_with_the_right_host():
    games, _ = _tied_field_games()
    field = field_from_wildcard_round(games, SEASON)
    wc = games[games.game_type == "WC"].set_index("home_team")["away_team"]
    for conf, seeds in field.items():
        for host_seed, visitor_seed in ((2, 7), (3, 6), (4, 5)):
            assert wc[seeds[host_seed - 1]] == seeds[visitor_seed - 1]


# -- phases -----------------------------------------------------------------------------------------------

def _post(rows: list[tuple[str, str, str, bool]]) -> pd.DataFrame:
    cols = ["season", "is_postseason", "game_type", "home_team", "away_team", "is_played", "game_id"]
    return pd.DataFrame([{"season": SEASON, "is_postseason": True, "game_type": t, "home_team": h, "away_team": a, "is_played": p,
                          "game_id": f"{SEASON}_{t}_{h}_{a}"} for t, h, a, p in rows], columns=cols)


def test_phase_detection_follows_the_calendar():
    reg = pd.DataFrame({"season": [SEASON], "is_postseason": [False], "game_type": ["REG"], "home_team": ["X"], "away_team": ["Y"], "is_played": [True], "game_id": ["r"]})
    wc = [("WC", f"H{i}", f"V{i}", False) for i in range(6)]
    assert phase_info(reg, SEASON) == {"phase": "regular_season", "round_now": None}
    assert phase_info(pd.concat([reg, _post(wc[:3])]), SEASON)["phase"] == "regular_season"          # schedule still incomplete
    assert phase_info(pd.concat([reg, _post(wc)]), SEASON) == {"phase": "playoffs", "round_now": "WC"}
    played = [("WC", f"H{i}", f"V{i}", True) for i in range(6)]
    assert phase_info(pd.concat([reg, _post(played)]), SEASON)["round_now"] == "DIV"                 # next round not listed yet
    full = played + [("DIV", f"D{i}", f"E{i}", True) for i in range(4)] + [("CON", "A", "B", True), ("CON", "C", "D", True), ("SB", "A", "C", True)]
    assert phase_info(pd.concat([reg, _post(full)]), SEASON) == {"phase": "complete", "round_now": None}


# -- the bracket ------------------------------------------------------------------------------------------

class StubProbs:
    """P(home wins) from a fixed strength per team (logistic on the difference); neutral site drops the home edge."""

    def __init__(self, strength: dict[str, float], edge: float = 0.2):
        self.s, self.edge = strength, edge

    def game(self, home, away, *, game_id=None, neutral=False):
        return float(1 / (1 + np.exp(-(self.s[home] - self.s[away] + (0 if neutral else self.edge)))))


FIELD = {"AFC": ["A1", "A2", "A3", "A4", "A5", "A6", "A7"], "NFC": ["N1", "N2", "N3", "N4", "N5", "N6", "N7"]}
STRENGTH = {t: -i * 0.3 for seeds in FIELD.values() for i, t in enumerate(seeds)}         # better seed = stronger


def test_chalk_bracket_reseeds_and_hosts_by_seed():
    b = build_bracket(FIELD, _post([]).iloc[0:0], StubProbs(STRENGTH))
    assert [(g["home"], g["away"]) for g in b["rounds"]["AFC"]["WC"]] == [("A2", "A7"), ("A3", "A6"), ("A4", "A5")]
    assert b["alive"]["AFC"]["DIV"] == ["A1", "A2", "A3", "A4"]
    assert [(g["home"], g["away"]) for g in b["rounds"]["AFC"]["DIV"]] == [("A1", "A4"), ("A2", "A3")]
    assert (b["rounds"]["AFC"]["CON"][0]["home"], b["rounds"]["AFC"]["CON"][0]["away"]) == ("A1", "A2")
    assert b["SB"]["home"] == "A1" and b["champion"] in ("A1", "N1")
    assert all(g["status"] == "projected" for c in b["rounds"].values() for r in c.values() for g in r)


def test_real_results_override_the_projection_and_flag_upsets():
    upset = {"season": SEASON, "is_postseason": True, "game_type": "WC", "home_team": "A2", "away_team": "A7", "is_played": True, "game_id": "g1",
             "home_score": 10.0, "away_score": 17.0, "gameday": "2031-01-10", "gametime": "16:30"}
    post = pd.DataFrame([upset])
    b = build_bracket(FIELD, post, StubProbs(STRENGTH))
    g = b["rounds"]["AFC"]["WC"][0]
    assert (g["status"], g["winner"], g["favorite"], g["upset"]) == ("final", "A7", "A2", True)
    assert b["alive"]["AFC"]["DIV"] == ["A1", "A3", "A4", "A7"]                    # the 7 seed advances; reseeding follows
    assert [(x["home"], x["away"]) for x in b["rounds"]["AFC"]["DIV"]] == [("A1", "A7"), ("A3", "A4")]
    assert g["kickoff"] == "2031-01-10 16:30"


def test_super_bowl_is_neutral_site():
    p = StubProbs(STRENGTH, edge=5.0)                                              # a huge home edge must not apply to the Super Bowl
    g = make_game("SB", None, "A1", "N1", {"A1": 1, "N1": 1}, _post([]).iloc[0:0], p)
    assert abs(g["p_home"] - 0.5) < 1e-9


# -- projected field and baselines --------------------------------------------------------------------------

def test_projected_field_maximizes_expected_correct_seeds():
    teams = [t for ts in DIVISIONS["AFC"].values() for t in ts] + [t for ts in DIVISIONS["NFC"].values() for t in ts]
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(8), size=len(teams))[:, :7]
    sim = pd.DataFrame(P, columns=[f"p_seed_{k}" for k in range(1, 8)]).assign(team=teams)
    field = projected_field(sim)
    for conf, seeds in field.items():
        assert len(set(seeds)) == 7 and all(CONFERENCE_OF[t] == conf for t in seeds)
        sub = sim[sim.team.map(CONFERENCE_OF) == conf].set_index("team")
        chosen = sum(sub.loc[t, f"p_seed_{k + 1}"] for k, t in enumerate(seeds))
        greedy = sum(sub[f"p_seed_{k}"].max() for k in range(1, 8))                # an unconstrained upper bound
        assert chosen <= greedy + 1e-9


def test_field_score_and_seed_chalk():
    actual = {"AFC": ["A1", "A2", "A3", "A4", "A5", "A6", "A7"], "NFC": ["N1", "N2", "N3", "N4", "N5", "N6", "N7"]}
    projected = {"AFC": ["A1", "A3", "A2", "A4", "X", "A6", "Y"], "NFC": actual["NFC"]}
    assert field_score(projected, actual) == {"in_field": 12, "exact_seed": 3 + 7}      # AFC seeds 1, 4, 6 match; NFC all 7
    picks = seed_chalk(actual, _post([]), {t: (10, 7, 0) for s in actual.values() for t in s})
    assert picks["WC"] == ["A2", "A3", "A4", "N2", "N3", "N4"]                       # hosts beat visitors
    assert picks["CON"] == ["A1", "N1"] and picks["DIV"] == ["A1", "A2", "N1", "N2"]


def test_truncate_hides_later_playoff_rounds_only():
    g = pd.concat([_post([("WC", "a", "b", True), ("DIV", "c", "d", True), ("SB", "e", "f", True)]),
                   pd.DataFrame([{"season": SEASON, "is_postseason": False, "game_type": "REG", "home_team": "x", "away_team": "y", "is_played": True, "game_id": "r"}])])
    assert set(truncate(g, SEASON, None).game_type) == {"REG"}
    assert set(truncate(g, SEASON, "WC").game_type) == {"REG", "WC"}
    assert set(truncate(g, SEASON, "SB").game_type) == {"REG", "WC", "DIV", "SB"}


# -- real seasons (skipped when the processed data is absent) --------------------------------------------------

@pytest.fixture(scope="module")
def real_games():
    from src.config import configured_path
    path = configured_path("processed") / "games.parquet"
    if not path.exists():
        pytest.skip("processed data not built (python -m src.pipeline)")
    return pd.read_parquet(path)


@pytest.mark.parametrize("season", range(2020, 2026))
def test_wild_card_only_field_matches_the_validated_reconstruction(real_games, season):
    from src.simulation.seeding import reconstruct_field
    raw = reconstruct_field(real_games, season)
    expected = {("AFC" if CONFERENCE_OF[v[0]] == "AFC" else "NFC"): v for v in raw.values()}
    assert field_from_wildcard_round(real_games, season) == expected


def test_replay_inputs_cannot_see_the_future(real_games):
    """Truncating the schedule must truncate play-by-play, quarterback and matchup rows too."""
    from src.simulation.live_bracket import load_inputs
    inp = load_inputs(truncate(real_games, 2025, "WC"))
    later = set(real_games.loc[real_games.season.eq(2025) & real_games.game_type.isin(["DIV", "CON", "SB"]), "game_id"])
    for table in (inp.team_game, inp.qb_game, inp.matchups):
        assert not later & set(table["game_id"])


def test_pre_playoff_ratings_do_not_depend_on_playoff_results(real_games):
    """The leak found while building the replay: ratings 'before week 19' must be the same whether or not the playoffs
    (already played, in a replay) are in the schedule. Also a replayed regular-season week must not use later games."""
    from src.simulation.live_bracket import latest_snapshot, load_inputs
    full = load_inputs(real_games)
    before = load_inputs(truncate(real_games, 2025, None))
    a = latest_snapshot(full, 2025, 19).set_index("team").sort_index()
    b = latest_snapshot(before, 2025, 19).set_index("team").sort_index()
    for col in ("pregame_off", "pregame_def", "pregame_qb"):
        assert np.allclose(a[col].to_numpy(float), b[col].to_numpy(float), equal_nan=True), f"{col} differs: playoff results leaked"
    # a mid-season week of a finished season (every game flagged played) must also ignore the rest of the season
    mid_full = latest_snapshot(full, 2025, 9).set_index("team").sort_index()
    mid_cut = latest_snapshot(load_inputs(real_games.loc[~(real_games.season.eq(2025) & real_games.week.ge(9))]), 2025, 9)
    assert mid_cut.empty or np.allclose(mid_full["pregame_off"].to_numpy(float), mid_cut.set_index("team").sort_index()["pregame_off"].to_numpy(float), equal_nan=True)
