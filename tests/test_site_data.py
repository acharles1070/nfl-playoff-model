"""The public site is built from committed JSON; these tests keep that data honest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.live import ledger

DATA = Path(__file__).resolve().parents[1] / "site" / "data"
NAMES = ["scorecard", "quiz", "calibration", "betting", "registry", "leakage", "encompassing", "power", "replication", "bracket_replay", "kalman", "bracket", "ledger", "meta"]


def load(name: str):
    path = DATA / f"{name}.json"
    if not path.exists():
        pytest.skip("site data not generated (python -m src.report.site_data)")

    def forbid(token):
        raise ValueError(f"{name}.json contains non-JSON constant {token}")

    return json.loads(path.read_text(), parse_constant=forbid)


@pytest.mark.parametrize("name", NAMES)
def test_json_is_valid_and_has_no_nan(name):
    assert load(name)


def test_scorecard_orders_models_and_market_is_best():
    models = {m["key"]: m for m in load("scorecard")["models"]}
    order = ["p_home_only", "p_std_epa", "p_kalman", "p_champion", "p_market"]
    losses = [models[k]["log_loss"] for k in order]
    assert losses == sorted(losses, reverse=True)
    assert models["p_champion"]["gap_lo"] > 0          # the gap to the market is significant
    assert 0.0 < min(losses) < max(losses) < 0.7       # nothing is worse than a coin flip's 0.693


def test_quiz_games_are_valid():
    games = load("quiz")
    assert len(games) == 30
    for g in games:
        assert 0 < g["p_model"] < 1 and 0 < g["p_market"] < 1
        assert g["home_win"] in (0, 1)
        assert (g["home_score"] > g["away_score"]) == bool(g["home_win"])
        assert 2017 <= g["season"] <= 2025                # holdout games only


def test_calibration_bins_are_valid():
    cal = load("calibration")
    for key in ("p_champion", "p_market"):
        for b in cal[key]["bins"]:
            assert 0 <= b["lo"] <= b["obs"] <= b["hi"] <= 1 and b["n"] >= 5


def test_betting_curves_are_consistent():
    d = load("betting")
    for s in d["series"].values():
        assert len(s["flat"]) == s["bets"] + 1
        assert s["flat"][-1] == pytest.approx(s["roi"] * s["bets"], abs=1.0)
    assert d["overround"] > 1.0                          # bookmakers always take a margin


def test_registry_counts_are_consistent():
    reg = load("registry")
    assert reg["n"] == len(reg["rows"])
    assert reg["holm"] <= reg["raw_sig"] <= reg["n"]
    assert all(r["lo"] <= r["delta"] <= r["hi"] for r in reg["rows"])


def test_bracket_tables_are_well_formed():
    b = load("bracket")
    n = len(b["teams"])
    assert n == 14 and {len(v) for v in b["conferences"].values()} == {7}
    for key in ("m_home", "m_neutral"):
        m = np.array(b[key])
        assert m.shape == (n, n)
    neutral = np.array(b["m_neutral"])
    off_diag = ~np.eye(n, dtype=bool)
    assert np.allclose((neutral + neutral.T)[off_diag], 0, atol=1e-4)   # neutral-site odds are antisymmetric
    assert all(0 < t < 1 for t in b["tau2"])
    assert b["champion"] in b["teams"]


def test_exported_ledger_chain_verifies():
    d = load("ledger")
    prev = d["genesis"]
    for i, row in enumerate(d["rows"], start=1):
        assert row["entry_id"] == str(i)
        assert row["prev_hash"] == prev
        assert ledger._row_hash(prev, row, d["fields"]) == row["row_hash"]
        prev = row["row_hash"]


def test_no_market_futures_prices_are_published():
    d = load("ledger")
    assert all("market" not in key for team in d["title"]["teams"] for key in team)
    text = (DATA / "ledger.json").read_text() + (DATA / "bracket.json").read_text()
    assert "p_market_sb" not in text and "BetMGM" not in text


# -- bracket replay (January 2026 playoffs) and the live bracket snapshot ------------------------------------------

BRACKET_JSON = DATA.parents[1] / "bracket" / "data" / "bracket.json"


def test_bracket_replay_is_internally_consistent():
    r = load("bracket_replay")
    g, summary, scores = r["games"], r["summary"], r["bracket_scores"]
    assert len(g) == summary["n_games"] == 13
    assert summary["model_right"] == sum(x["model_right"] for x in g)
    assert summary["seed_right"] == sum(x["seed_right"] for x in g)
    assert scores["model"]["total"] == sum(scores["model"][k] for k in ("WC", "DIV", "CON", "SB"))
    assert all(scores["model"][k] <= scores["games"][k] for k in scores["games"])
    assert len(r["weekly"]) == 19 and all(0 <= w["model"]["in_field"] <= 14 and 0 <= w["standings"]["in_field"] <= 14 for w in r["weekly"])
    assert [x["winner"] for x in g if x["round"] == "SB"] == [r["champion"]]
    assert all(0 < x["p_model"] < 1 for x in g)
    # the pre-playoff title odds are a probability distribution over the 14 teams
    assert len(r["preplayoff_title_odds"]) == 14
    assert abs(sum(x["p_win_sb"] for x in r["preplayoff_title_odds"]) - 1) < 0.01


def test_live_bracket_snapshot_is_well_formed():
    if not BRACKET_JSON.exists():
        pytest.skip("bracket snapshot not generated (python -m src.simulation.live_bracket)")
    d = json.loads(BRACKET_JSON.read_text(), parse_constant=lambda t: (_ for _ in ()).throw(ValueError(t)))
    assert d["phase"] in ("regular_season", "playoffs", "complete") and d["playoffs_year"] == d["season"] + 1
    assert {len(d["field"][c]) for c in ("AFC", "NFC")} == {7}
    teams = d["teams"]
    assert len(teams) == 32
    assert abs(sum(t["p_win_sb"] for t in teams) - 1) < 0.01                       # exactly one champion
    for c in ("AFC", "NFC"):
        assert abs(sum(t["p_win_conf"] for t in teams if t["conf"] == c) - 1) < 0.01
        assert 3 == len(d["bracket"][c]["WC"]) and 2 == len(d["bracket"][c]["DIV"]) and 1 == len(d["bracket"][c]["CON"])
    games = [g for c in ("AFC", "NFC") for r in ("WC", "DIV", "CON") for g in d["bracket"][c][r]] + [d["bracket"]["SB"]]
    assert len(games) == 13 and all(0 < g["p_home"] < 1 for g in games)
    assert d["bracket"]["champion"] == d["bracket"]["SB"]["winner"]
    if d["phase"] == "regular_season":
        assert all(g["status"] == "projected" for g in games)                       # nothing is scheduled yet
        assert all(abs(sum(t["p_seed"]) - t["p_playoffs"]) < 0.01 for t in teams)  # seed odds add up to the playoff odds
