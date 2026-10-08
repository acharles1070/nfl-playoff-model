"""Leakage guards for every pregame feature builder.

The core property: a pregame feature for game G may depend only on games
played strictly BEFORE G. I verify it by perturbation: rewrite the box-score
numbers of one game and assert that no feature computed for that game, or for
any earlier game, moves. Features for later games are allowed to move.

These tests use small synthetic schedules, so they run in well under a second
and do not touch the network or data/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.opponent_adjusted import build_opponent_adjusted
from src.features.snapshots import (
    RATE_SPECS,
    build_pregame_snapshots,
    leakage_sanity_check,
)


N_TEAMS = 8
N_WEEKS = 7
SEASON = 2022


def _round_robin_games() -> pd.DataFrame:
    """Circle-method schedule: every team plays every week."""
    teams = [f"T{i}" for i in range(N_TEAMS)]
    rows = []
    rotating = teams[1:]
    for week in range(1, N_WEEKS + 1):
        order = [teams[0], *rotating]
        for i in range(N_TEAMS // 2):
            a, b = order[i], order[N_TEAMS - 1 - i]
            home, away = (a, b) if (week + i) % 2 == 0 else (b, a)
            rows.append(
                {
                    "season": SEASON,
                    "week": week,
                    "game_id": f"{SEASON}_{week:02d}_{away}_{home}",
                    "home_team": home,
                    "away_team": away,
                }
            )
        rotating = rotating[-1:] + rotating[:-1]
    return pd.DataFrame(rows)


def _team_game(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic team-game table with every column the builders require."""
    rng = np.random.default_rng(seed)
    games = _round_robin_games()

    rows = []
    for g in games.itertuples(index=False):
        for team in (g.home_team, g.away_team):
            off_plays = int(rng.integers(50, 75))
            pass_plays = int(off_plays * rng.uniform(0.45, 0.65))
            rush_plays = off_plays - pass_plays
            def_plays = int(rng.integers(50, 75))
            def_pass = int(def_plays * rng.uniform(0.45, 0.65))
            def_rush = def_plays - def_pass
            row = {
                "season": g.season,
                "season_type": "REG",
                "week": g.week,
                "game_id": g.game_id,
                "team": team,
                "off_plays": off_plays,
                "off_pass_plays": pass_plays,
                "off_rush_plays": rush_plays,
                "off_epa_sum": rng.normal(0, 6),
                "off_success_sum": rng.integers(15, 40),
                "off_pass_epa_sum": rng.normal(0, 5),
                "off_rush_epa_sum": rng.normal(0, 3),
                "off_explosive_passes": rng.integers(0, 6),
                "off_explosive_rushes": rng.integers(0, 4),
                "def_allowed_plays": def_plays,
                "def_allowed_pass_plays": def_pass,
                "def_allowed_rush_plays": def_rush,
                "def_allowed_epa_sum": rng.normal(0, 6),
                "def_allowed_success_sum": rng.integers(15, 40),
                "def_allowed_pass_epa_sum": rng.normal(0, 5),
                "def_allowed_rush_epa_sum": rng.normal(0, 3),
                "def_allowed_explosive_passes": rng.integers(0, 6),
                "def_allowed_explosive_rushes": rng.integers(0, 4),
            }
            row["off_epa_per_play"] = row["off_epa_sum"] / off_plays
            row["def_allowed_epa_per_play"] = row["def_allowed_epa_sum"] / def_plays
            rows.append(row)

    return pd.DataFrame(rows), games


def _numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c.endswith("_std") or "_last4" in c or "_last8" in c
    ]


def _perturb_game(team_game: pd.DataFrame, game_id: str) -> pd.DataFrame:
    """Rewrite every box-score number in one game (both teams)."""
    out = team_game.copy()
    mask = out["game_id"].eq(game_id)
    stat_cols = [
        c
        for c in out.columns
        if c.endswith(("_sum", "_passes", "_rushes", "_per_play"))
    ]
    out.loc[mask, stat_cols] = out.loc[mask, stat_cols] * 37.0 + 11.0
    return out


def _compare_before_and_at(
    base: pd.DataFrame,
    perturbed: pd.DataFrame,
    *,
    game_week: int,
    feature_cols: list[str],
    label: str,
) -> None:
    keys = ["game_id", "team"]
    b = base.sort_values(keys).reset_index(drop=True)
    p = perturbed.sort_values(keys).reset_index(drop=True)
    assert b[keys].equals(p[keys])

    not_after = b["week"].le(game_week)
    for col in feature_cols:
        left = b.loc[not_after, col].to_numpy(dtype=float)
        right = p.loc[not_after, col].to_numpy(dtype=float)
        assert np.allclose(left, right, equal_nan=True), (
            f"{label}: {col} changed for a game at/before week {game_week} "
            "after editing that week's results -> future information is leaking "
            "into a pregame feature"
        )


@pytest.mark.parametrize("target_week", [2, 4, 6])
def test_raw_epa_snapshots_ignore_current_and_future_games(target_week):
    team_game, games = _team_game()
    target_game = games.loc[games["week"].eq(target_week), "game_id"].iloc[0]

    base = build_pregame_snapshots(team_game)
    perturbed = build_pregame_snapshots(_perturb_game(team_game, target_game))

    _compare_before_and_at(
        base,
        perturbed,
        game_week=target_week,
        feature_cols=_numeric_feature_cols(base),
        label="raw EPA snapshots",
    )


@pytest.mark.parametrize("target_week", [2, 4, 6])
def test_opponent_adjusted_snapshots_ignore_current_and_future_games(target_week):
    team_game, games = _team_game()
    target_game = games.loc[games["week"].eq(target_week), "game_id"].iloc[0]

    base = build_opponent_adjusted(team_game, games)
    perturbed = build_opponent_adjusted(
        _perturb_game(team_game, target_game), games
    )

    feature_cols = _numeric_feature_cols(base) + [
        "opp_pregame_off_epa",
        "opp_pregame_def_allowed_epa",
        "pregame_off_epa",
        "pregame_def_allowed_epa",
    ]
    feature_cols = [c for c in feature_cols if c in base.columns]

    _compare_before_and_at(
        base,
        perturbed,
        game_week=target_week,
        feature_cols=feature_cols,
        label="opponent-adjusted snapshots",
    )


def test_first_game_of_season_has_no_history():
    team_game, _ = _team_game()
    snapshots = build_pregame_snapshots(team_game)
    leakage_sanity_check(snapshots)  # raises on failure

    first = snapshots.loc[snapshots["games_played_before"].eq(0)]
    assert len(first) == N_TEAMS
    assert first[_numeric_feature_cols(snapshots)].isna().all().all()


def test_snapshots_are_play_weighted_not_mean_of_means():
    """Season-to-date rates must weight by plays, not average game rates."""
    team_game, _ = _team_game()
    snapshots = build_pregame_snapshots(team_game)

    team = "T3"
    t = team_game.loc[team_game["team"].eq(team)].sort_values("week")
    snap = snapshots.loc[snapshots["team"].eq(team)].sort_values("week")

    k = 4  # features for the 5th game use the first four
    expected = t["off_epa_sum"].iloc[:k].sum() / t["off_plays"].iloc[:k].sum()
    got = snap["off_epa_per_play_std"].iloc[k]
    assert got == pytest.approx(expected)

    mean_of_means = (t["off_epa_sum"].iloc[:k] / t["off_plays"].iloc[:k]).mean()
    assert got != pytest.approx(mean_of_means)


def test_rate_specs_cover_offense_and_defense():
    offense = [r for r in RATE_SPECS if r.startswith("off_")]
    defense = [r for r in RATE_SPECS if r.startswith("def_allowed_")]
    assert len(offense) == len(defense) > 0


# ---------------------------------------------------------------------------
# Team ratings (Kalman filter)
# ---------------------------------------------------------------------------

from src.features.ratings import RatingParams, build_team_ratings  # noqa: E402


def _two_season_team_game():
    """Two seasons so cross-season carry-over is exercised."""
    frames_tg, frames_games = [], []
    for season, seed in ((2021, 1), (2022, 2)):
        tg, games = _team_game(seed)
        tg["season"] = season
        games = games.copy()
        games["season"] = season
        games["game_id"] = games["game_id"].str.replace("2022", str(season))
        tg["game_id"] = tg["game_id"].str.replace("2022", str(season))
        frames_tg.append(tg)
        frames_games.append(games)
    return pd.concat(frames_tg, ignore_index=True), pd.concat(
        frames_games, ignore_index=True
    )


@pytest.mark.parametrize("season,target_week", [(2021, 3), (2022, 2), (2022, 5)])
def test_ratings_ignore_current_and_future_games(season, target_week):
    team_game, games = _two_season_team_game()
    target = games.loc[
        games["season"].eq(season) & games["week"].eq(target_week),
        "game_id",
    ].iloc[0]

    base = build_team_ratings(team_game, games)
    perturbed = build_team_ratings(
        _perturb_game(team_game, target), games
    )

    cols = [
        "pregame_off",
        "pregame_def",
        "pregame_net",
        "pregame_off_var",
        "pregame_def_var",
    ]

    b = base.sort_values(["game_id", "team"]).reset_index(drop=True)
    p = perturbed.sort_values(["game_id", "team"]).reset_index(drop=True)

    not_after = b["season"].lt(season) | (
        b["season"].eq(season) & b["week"].le(target_week)
    )

    for col in cols:
        assert np.allclose(
            b.loc[not_after, col], p.loc[not_after, col]
        ), f"ratings leak: {col} moved for a game at/before the edited one"

    # ...and the edit must actually matter for the future (test has teeth)
    later = ~not_after
    assert not np.allclose(b.loc[later, "pregame_off"], p.loc[later, "pregame_off"])


def test_ratings_carry_over_and_regress_between_seasons():
    team_game, games = _two_season_team_game()
    r = build_team_ratings(
        team_game, games, RatingParams(carryover_off=0.5, carryover_def=0.5)
    )

    first_2022 = r.loc[r["season"].eq(2022) & r["week"].eq(1)]
    last_2021 = (
        r.loc[r["season"].eq(2021) & r["week"].eq(N_WEEKS)]
        .set_index("team")
    )

    # Week-1 2022 rating is non-trivial (not reset to 0) but shrunk vs 2021 end
    assert first_2022["pregame_off"].abs().sum() > 0
    shrunk = first_2022.set_index("team")["pregame_off"].abs()
    assert (shrunk.sum()) < last_2021["pregame_off"].abs().sum() * 1.5 + 1e-9
    assert first_2022["rating_games_seen"].eq(N_WEEKS).all()


# ---------------------------------------------------------------------------
# Quarterback layer
# ---------------------------------------------------------------------------

def _qb_game_for(team_game: pd.DataFrame, assign=None) -> pd.DataFrame:
    """One QB per team-game: 'QB_<team>' unless assign(game_id, team) says otherwise."""
    rows = []
    for r in team_game[["game_id", "team"]].itertuples(index=False):
        qb = assign(r.game_id, r.team) if assign else None
        rows.append(
            {
                "game_id": r.game_id,
                "posteam": r.team,
                "qb_player_id": qb or f"QB_{r.team}",
                "dropbacks": 30,
                "started": 1,
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("season,target_week", [(2021, 3), (2022, 2), (2022, 5)])
def test_qb_ratings_ignore_current_and_future_games(season, target_week):
    team_game, games = _two_season_team_game()
    qb_game = _qb_game_for(team_game)
    target = games.loc[
        games["season"].eq(season) & games["week"].eq(target_week),
        "game_id",
    ].iloc[0]

    base = build_team_ratings(team_game, games, qb_game=qb_game)
    perturbed = build_team_ratings(
        _perturb_game(team_game, target), games, qb_game=qb_game
    )

    b = base.sort_values(["game_id", "team"]).reset_index(drop=True)
    p = perturbed.sort_values(["game_id", "team"]).reset_index(drop=True)

    not_after = b["season"].lt(season) | (
        b["season"].eq(season) & b["week"].le(target_week)
    )

    for col in ["pregame_qb", "pregame_qb_var", "pregame_off_total", "pregame_net_total"]:
        assert np.allclose(
            b.loc[not_after, col], p.loc[not_after, col]
        ), f"QB ratings leak: {col} moved for a game at/before the edited one"

    assert not np.allclose(
        b.loc[~not_after, "pregame_qb"], p.loc[~not_after, "pregame_qb"]
    )


def test_quarterback_rating_follows_him_to_a_new_team():
    """A great QB at T0 keeps his rating when he starts for T1."""
    team_game, games = _team_game(seed=5)

    # flat, noise-free EPA so the only signal is the QB
    team_game["off_epa_per_play"] = 0.0
    team_game["off_plays"] = 63

    STAR = "QB_STAR"

    def assign(game_id, team):
        week = int(game_id.split("_")[1])
        if team == "T0" and week <= 4:
            return STAR
        if team == "T1" and week >= 5:
            return STAR
        return None

    qb_game = _qb_game_for(team_game, assign)

    star_games = qb_game.loc[qb_game["qb_player_id"].eq(STAR), ["game_id", "posteam"]]
    mask = team_game.set_index(["game_id", "team"]).index.isin(
        list(zip(star_games["game_id"], star_games["posteam"]))
    )
    team_game.loc[mask, "off_epa_per_play"] = 0.30   # STAR games are great

    r = build_team_ratings(team_game, games, qb_game=qb_game)

    star_pre = r.loc[r["pregame_qb_id"].eq(STAR)].sort_values("week")

    week5 = star_pre.loc[star_pre["week"].eq(5)].iloc[0]
    week1 = star_pre.loc[star_pre["week"].eq(1)].iloc[0]

    assert week5["team"] == "T1"
    assert week5["pregame_qb"] > week1["pregame_qb"] + 0.02        # learned he's good
    assert week5["pregame_off_total"] > week5["pregame_off"] + 0.02  # and it travels
    assert week5["pregame_qb_starts"] == 4


def test_ratings_without_qb_data_are_unchanged_by_the_qb_layer():
    team_game, games = _two_season_team_game()
    plain = build_team_ratings(team_game, games)

    assert plain["pregame_qb"].eq(0.0).all()
    assert np.allclose(plain["pregame_off_total"], plain["pregame_off"])


def test_future_games_are_snapshotted_without_updating_the_filter():
    """Appending unplayed games must not change any earlier snapshot, and each
    future game gets the team's latest starter (or the announced one)."""
    team_game, games = _two_season_team_game()
    qb_game = _qb_game_for(team_game)

    base = build_team_ratings(team_game, games, qb_game=qb_game)

    future = pd.DataFrame(
        [
            {
                "season": 2023,
                "week": 1,
                "game_id": "2023_01_T1_T0",
                "home_team": "T0",
                "away_team": "T1",
            },
            {
                "season": 2023,
                "week": 1,
                "game_id": "2023_01_T3_T2",
                "home_team": "T2",
                "away_team": "T3",
            },
        ]
    )

    extended = build_team_ratings(
        team_game,
        pd.concat([games, future], ignore_index=True),
        qb_game=qb_game,
        projected_starters={("2023_01_T3_T2", "T3"): "QB_ANNOUNCED"},
    )

    keys = ["game_id", "team"]
    old = extended.merge(base[keys], on=keys).sort_values(keys).reset_index(drop=True)
    assert np.allclose(
        old["pregame_off"], base.sort_values(keys).reset_index(drop=True)["pregame_off"]
    )

    fut = extended.loc[extended["game_id"].isin(future["game_id"])].set_index(keys)
    assert fut.loc[("2023_01_T1_T0", "T0"), "pregame_qb_id"] == "QB_T0"   # last starter
    assert fut.loc[("2023_01_T3_T2", "T3"), "pregame_qb_id"] == "QB_ANNOUNCED"
    assert len(extended) == len(base) + 4


# ---------------------------------------------------------------------------
# Point-margin ratings
# ---------------------------------------------------------------------------

from src.features.margin_ratings import build_margin_ratings  # noqa: E402


def _scored_games():
    _, games = _two_season_team_game()
    rng = np.random.default_rng(3)
    games = games.copy()
    games["home_score"] = rng.integers(10, 40, len(games)).astype(float)
    games["away_score"] = rng.integers(10, 40, len(games)).astype(float)
    return games


@pytest.mark.parametrize("season,target_week", [(2021, 3), (2022, 2), (2022, 5)])
def test_margin_ratings_ignore_current_and_future_games(season, target_week):
    games = _scored_games()
    target = games.loc[
        games["season"].eq(season) & games["week"].eq(target_week), "game_id"
    ].iloc[0]

    edited = games.copy()
    mask = edited["game_id"].eq(target)
    edited.loc[mask, "home_score"] += 50.0            # rewrite that game's result

    a = build_margin_ratings(games).merge(
        games[["game_id", "season", "week"]], on="game_id"
    )
    b = build_margin_ratings(edited)

    keys = ["game_id", "team"]
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)

    early = a["season"].lt(season) | (a["season"].eq(season) & a["week"].le(target_week))

    assert np.allclose(a.loc[early, "pregame_pts"], b.loc[early, "pregame_pts"])
    assert not np.allclose(a.loc[~early, "pregame_pts"], b.loc[~early, "pregame_pts"])


# ---------------------------------------------------------------------------
# Injury burden
# ---------------------------------------------------------------------------

from src.features.injuries import (  # noqa: E402
    measure_availability,
    prior_snap_share,
    team_week_burden,
)


def _injury_fixture():
    """One team, 3 seasons x 8 weeks: a starting OL (P1) and a DB (P2)."""
    snaps, reports = [], []
    for season in (2020, 2021, 2022):
        for week in range(1, 9):
            for pid, pos, off, dfn in (("P1", "T", 1.0, 0.0), ("P2", "CB", 0.0, 0.9)):
                snaps.append(
                    {"season": season, "week": week, "pfr_player_id": pid,
                     "position": pos, "team": "T0", "offense_pct": off, "defense_pct": dfn}
                )
            # P1 is Questionable every week; P2 is Out in week 5 only
            reports.append({"season": season, "week": week, "team": "T0", "gsis_id": "G1",
                            "position": "T", "report_status": "Questionable"})
            if week == 5:
                reports.append({"season": season, "week": week, "team": "T0", "gsis_id": "G2",
                                "position": "CB", "report_status": "Out"})
    id_map = pd.DataFrame({"gsis_id": ["G1", "G2"], "pfr_id": ["P1", "P2"]})
    return pd.DataFrame(reports), pd.DataFrame(snaps), id_map


def test_injury_burden_uses_only_prior_snap_history():
    reports, snaps, id_map = _injury_fixture()
    avail = {s: {"_overall": 0.4} for s in (2020, 2021, 2022)}

    base = team_week_burden(reports, snaps, id_map, avail)

    # rewrite P2's snap share in (2021, week 5) and every later week
    edited = snaps.copy()
    late = (edited["season"] > 2021) | ((edited["season"] == 2021) & (edited["week"] >= 5))
    edited.loc[late & edited["pfr_player_id"].eq("P2"), "defense_pct"] = 0.05
    changed = team_week_burden(reports, edited, id_map, avail)

    key = ["season", "week"]
    a = base.sort_values(key).reset_index(drop=True)
    b = changed.sort_values(key).reset_index(drop=True)

    same = (a["season"] < 2021) | ((a["season"] == 2021) & (a["week"] <= 5))
    assert np.allclose(a.loc[same, "lost_DB"], b.loc[same, "lost_DB"]), \
        "injury burden used a snap count from the same or a later week"
    assert not np.allclose(a.loc[~same, "lost_DB"], b.loc[~same, "lost_DB"])


def test_out_player_costs_his_full_prior_snap_share():
    reports, snaps, id_map = _injury_fixture()
    avail = {s: {"_overall": 0.4} for s in (2020, 2021, 2022)}
    burden = team_week_burden(reports, snaps, id_map, avail)

    wk5 = burden.loc[burden["season"].eq(2021) & burden["week"].eq(5)].iloc[0]
    assert wk5["lost_DB"] == pytest.approx(0.9)              # Out => p_out = 1
    assert wk5["lost_OL"] == pytest.approx(0.4 * 1.0)        # Questionable => 0.4


def test_availability_ignores_the_target_season_and_later():
    reports, snaps, id_map = _injury_fixture()
    drifted = snaps.loc[~(snaps["season"].eq(2022))]         # nobody plays in 2022
    a = measure_availability(reports, drifted, id_map, target_season=2022)
    b = measure_availability(reports, snaps, id_map, target_season=2022)
    assert a == b                                            # 2022 outcomes not used


def test_missing_report_is_unknown_not_zero():
    reports, snaps, id_map = _injury_fixture()
    avail = {s: {"_overall": 0.4} for s in (2020, 2021, 2022)}
    burden = team_week_burden(reports.loc[~reports["week"].eq(3)], snaps, id_map, avail)
    assert burden.loc[burden["week"].eq(3)].empty            # absent -> NaN downstream


# ---------------------------------------------------------------------------
# Roster burden (injured reserve + game-day inactives)
# ---------------------------------------------------------------------------

from src.features.rosters import roster_burden  # noqa: E402


def _roster_fixture():
    """One team, 2 seasons x 8 weeks. P1 (OL) is a starter; P2 (CB) too."""
    snaps, rosters = [], []
    for season in (2020, 2021):
        for week in range(1, 9):
            for pid, off, dfn in (("P1", 1.0, 0.0), ("P2", 0.0, 0.9)):
                if season == 2021 and pid == "P1" and week >= 5:
                    continue                    # no snaps while on injured reserve
                snaps.append({"season": season, "week": week, "pfr_player_id": pid,
                              "offense_pct": off, "defense_pct": dfn})
            for pid, pos in (("P1", "OL"), ("P2", "CB")):
                status = "ACT"
                if season == 2021 and pid == "P1" and week >= 5:
                    status = "RES"          # P1 lands on IR in week 5 and stays there
                if season == 2021 and pid == "P2" and week == 6:
                    status = "INA"          # P2 is a game-day inactive in week 6
                rosters.append({"season": season, "week": week, "team": "T0",
                                "pfr_id": pid, "position": pos, "status": status})
    return pd.DataFrame(rosters), pd.DataFrame(snaps)


def test_ir_status_applies_from_the_following_week_only():
    rosters, snaps = _roster_fixture()
    b = roster_burden(rosters, snaps).set_index(["season", "week"])

    assert b.loc[(2021, 5), "lost_ir_OL"] == 0.0          # RES is first seen in week 5
    assert b.loc[(2021, 6), "lost_ir_OL"] == pytest.approx(1.0)   # known the next week
    assert b.loc[(2021, 7), "lost_ir_OL"] == pytest.approx(1.0)   # and it persists


def test_game_day_inactive_applies_to_the_same_week_only():
    rosters, snaps = _roster_fixture()
    b = roster_burden(rosters, snaps).set_index(["season", "week"])

    assert b.loc[(2021, 6), "lost_ina_DB"] == pytest.approx(0.9)
    assert b.loc[(2021, 5), "lost_ina_DB"] == 0.0
    assert b.loc[(2021, 7), "lost_ina_DB"] == 0.0


def test_roster_burden_uses_only_prior_snap_history():
    rosters, snaps = _roster_fixture()
    base = roster_burden(rosters, snaps).set_index(["season", "week"])

    edited = snaps.copy()
    late = (edited["season"] == 2021) & (edited["week"] >= 6)
    edited.loc[late & edited["pfr_player_id"].eq("P2"), "defense_pct"] = 0.1
    changed = roster_burden(rosters, edited).set_index(["season", "week"])

    # P2's importance in his week-6 inactive must not see his week-6 (or later) snaps
    assert changed.loc[(2021, 6), "lost_ina_DB"] == pytest.approx(base.loc[(2021, 6), "lost_ina_DB"])
