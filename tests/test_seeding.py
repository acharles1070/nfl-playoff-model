"""Playoff seed reconstruction must match known real brackets."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import configured_path
from src.simulation.seeding import reconstruct_field, validate_seeds


GAMES = configured_path("processed") / "games.parquet"

pytestmark = pytest.mark.skipif(not GAMES.exists(), reason="run the pipeline first")


@pytest.fixture(scope="module")
def games():
    return pd.read_parquet(GAMES)


def test_2025_bracket_matches_the_hand_entered_v1_notebook(games):
    fields = reconstruct_field(games, 2025)
    assert fields["A"] == ["DEN", "NE", "JAX", "PIT", "HOU", "BUF", "LAC"]
    assert fields["B"] == ["SEA", "CHI", "PHI", "CAR", "LAR", "SF", "GB"]


def test_2015_head_to_head_tiebreak_puts_denver_ahead_of_new_england(games):
    fields = reconstruct_field(games, 2015)
    afc = next(v for v in fields.values() if "DEN" in v)
    assert afc[:2] == ["DEN", "NE"]        # both 12-4; Denver won head-to-head


def test_every_season_reproduces_the_real_reseeded_matchups(games):
    for season in range(2009, 2026):
        fields = reconstruct_field(games, season)
        assert validate_seeds(games, season, fields) == [], season
        assert sorted(len(v) for v in fields.values()) == (
            [6, 6] if season < 2020 else [7, 7]
        )
