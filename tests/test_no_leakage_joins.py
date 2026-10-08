"""Perturbation tests for the builders that JOIN features onto games.

test_no_leakage.py covers the snapshot, rating and injury builders. These cover the layers on top:
schedule context, the matchup dataset and the opponent-adjusted matchup join. Same property: rewrite
one game's box score and result, and no pregame feature for that game or any earlier game may move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_no_leakage import _numeric_feature_cols, _perturb_game, _team_game

from src.features.context import CONTEXT_COLUMNS, build_context
from src.features.matchups import build_matchup_dataset
from src.features.opponent_adjusted import build_opponent_adjusted
from src.features.opponent_matchups import OA_FEATURES, build_opponent_matchups
from src.features.snapshots import build_pregame_snapshots


def _games_with_results(games: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = games.copy()
    g["game_type"] = "REG"
    g["home_score"] = rng.integers(7, 40, len(g))
    g["away_score"] = rng.integers(7, 40, len(g))
    g["home_win"] = (g["home_score"] > g["away_score"]).astype(int)
    g["is_postseason"] = False
    g["is_played"] = True
    return g


def _rewrite_result(games: pd.DataFrame, game_id: str) -> pd.DataFrame:
    g = games.copy()
    m = g["game_id"].eq(game_id)
    g.loc[m, "home_score"], g.loc[m, "away_score"] = 3, 45
    g.loc[m, "home_win"] = 0
    return g


def _assert_unchanged_through(base: pd.DataFrame, other: pd.DataFrame, week: int, cols: list[str], label: str) -> None:
    b = base.sort_values("game_id").reset_index(drop=True)
    o = other.sort_values("game_id").reset_index(drop=True)
    assert b["game_id"].equals(o["game_id"])
    keep = b["week"].le(week)
    for c in cols:
        assert np.allclose(b.loc[keep, c].to_numpy(float), o.loc[keep, c].to_numpy(float), equal_nan=True), (
            f"{label}: {c} moved for a game at/before week {week} after editing that week's data -> look-ahead")


def test_context_features_depend_only_on_the_schedule():
    _, games = _team_game()
    g = _games_with_results(games)
    g["home_rest"], g["away_rest"] = 7, 6
    g["location"], g["div_game"], g["roof"] = "Home", 0, "outdoors"

    base = build_context(g)
    changed = build_context(_rewrite_result(g, g["game_id"].iloc[3]))      # results rewritten

    assert list(base.columns) == ["game_id", *CONTEXT_COLUMNS]
    pd.testing.assert_frame_equal(base, changed)                          # context never reads a result


@pytest.mark.parametrize("target_week", [2, 4, 6])
def test_matchup_dataset_ignores_current_and_future_games(target_week):
    team_game, games = _team_game()
    g = _games_with_results(games)
    target = games.loc[games["week"].eq(target_week), "game_id"].iloc[0]

    base = build_matchup_dataset(g, build_pregame_snapshots(team_game))
    other = build_matchup_dataset(_rewrite_result(g, target), build_pregame_snapshots(_perturb_game(team_game, target)))

    cols = _numeric_feature_cols(base)
    assert cols, "matchup dataset exposes no pregame feature columns"
    _assert_unchanged_through(base, other, target_week, cols, "matchup dataset")


@pytest.mark.parametrize("target_week", [2, 4, 6])
def test_opponent_adjusted_matchups_ignore_current_and_future_games(target_week):
    team_game, games = _team_game()
    g = _games_with_results(games)
    target = games.loc[games["week"].eq(target_week), "game_id"].iloc[0]

    def build(tg, results):
        return build_opponent_matchups(build_matchup_dataset(results, build_pregame_snapshots(tg)), build_opponent_adjusted(tg, games))

    base = build(team_game, g)
    other = build(_perturb_game(team_game, target), _rewrite_result(g, target))

    present = [f"{side}_{c}" for side in ("home", "away") for c in OA_FEATURES if f"{side}_{c}" in base.columns]
    assert present, "opponent-adjusted matchups expose no OA feature columns"
    _assert_unchanged_through(base, other, target_week, present, "opponent-adjusted matchups")
