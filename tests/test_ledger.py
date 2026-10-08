"""The ledger must detect any after-the-fact edit, deletion, or reordering."""

from __future__ import annotations

import csv

import pytest

from src.live import ledger


def _row(game_id: str, p: str, model: str = "m1") -> dict:
    return {
        "logged_at_utc": "2026-10-01T12:00:00Z",
        "kickoff_utc": "2026-10-02T00:15:00Z",
        "season": "2026",
        "week": "4",
        "game_id": game_id,
        "home_team": "CLE",
        "away_team": "PIT",
        "model_id": model,
        "p_home_win": p,
    }


@pytest.fixture()
def path(tmp_path):
    p = tmp_path / "predictions.csv"
    ledger.append([_row("g1", "0.60"), _row("g2", "0.55"), _row("g3", "0.41")], p)
    return p


def _rewrite(path, mutate):
    rows = ledger.read_rows(path)
    mutate(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ledger.FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_clean_ledger_verifies(path):
    ok, message = ledger.verify(path)
    assert ok, message
    assert ledger.verify(tmp := path)[1].startswith("ledger OK: 3 entries")


def test_duplicate_model_game_pairs_are_skipped(path):
    assert ledger.append([_row("g1", "0.99")], path) == 0           # same model+game
    assert ledger.append([_row("g1", "0.99", model="m2")], path) == 1  # new model ok
    assert ledger.verify(path)[0]


def test_editing_a_probability_is_detected(path):
    _rewrite(path, lambda rows: rows[0].update(p_home_win="0.99"))
    ok, message = ledger.verify(path)
    assert not ok and "modified" in message


def test_deleting_a_row_is_detected(path):
    _rewrite(path, lambda rows: rows.pop(1))
    assert not ledger.verify(path)[0]


def test_reordering_rows_is_detected(path):
    _rewrite(path, lambda rows: rows.reverse())
    assert not ledger.verify(path)[0]


def test_rewriting_a_row_and_its_hash_still_breaks_the_chain(path):
    """An attacker who recomputes one row's hash still breaks the NEXT row."""
    def forge(rows):
        rows[0]["p_home_win"] = "0.99"
        rows[0]["row_hash"] = ledger._row_hash(rows[0]["prev_hash"], rows[0])
    _rewrite(path, forge)
    ok, message = ledger.verify(path)
    assert not ok and "chain broken" in message


def test_champion_identity_is_pinned():
    """Changing the champion's definition must be a conscious act: create a NEW
    model_id instead. This hash is what the ledger recorded for its first rows."""
    from src.live.predict import MODELS, params_sha

    assert params_sha(MODELS["ratings_qb_logit_v1"]) == "d7760e41a9d7"
    assert MODELS["ratings_qb_injury_logit_v1"].use_injuries is True


# ---------------------------------------------------------------------------
# Title-odds ledger reuses the same chain with its own fields and key
# ---------------------------------------------------------------------------

def _title_row(team, p, week="5", model="season_sim_v1"):
    return {"logged_at_utc": "2026-10-07T00:00:00Z", "season": "2026", "week_prior": week,
            "team": team, "model_id": model, "p_win_sb": p}


def test_title_ledger_chain_dedup_and_tamper_detection(tmp_path):
    from src.simulation.title_odds import FIELDS, KEY

    path = tmp_path / "title.csv"
    n = ledger.append([_title_row("BUF", "0.11"), _title_row("KC", "0.09")], path, fields=FIELDS, key=KEY)
    assert n == 2
    assert ledger.verify(path, FIELDS)[0]

    # same (model, season, week, team) is never logged twice; a new week is fine
    assert ledger.append([_title_row("BUF", "0.50")], path, fields=FIELDS, key=KEY) == 0
    assert ledger.append([_title_row("BUF", "0.12", week="6")], path, fields=FIELDS, key=KEY) == 1
    assert ledger.verify(path, FIELDS)[0]

    rows = ledger.read_rows(path)
    rows[0]["p_win_sb"] = "0.99"
    with path.open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    ok, message = ledger.verify(path, FIELDS)
    assert not ok and "modified" in message


def test_prediction_ledger_hashes_are_unchanged_by_the_refactor():
    """The first real entry must still verify: its chain was written before the refactor."""
    ok, message = ledger.verify()
    assert ok, message
