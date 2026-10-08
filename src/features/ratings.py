"""Leakage-free team strength ratings (online Kalman filter on EPA).

Each team carries two latent strengths, in EPA/play units:

    O   offense: EPA/play the team generates above a league-average offense
        (with a QB layer, this is everything EXCEPT the quarterback)
    Dv  defense: EPA/play the team ALLOWS above a league-average defense
        (lower is better, so a team's net strength is O - Dv)

and, optionally, every quarterback carries a strength Q that travels with him
from team to team. A game is two independent observations:

    home offense EPA/play = mu + O_home + sum_k(s_k * Q_k) + Dv_away + noise
    away offense EPA/play = mu + O_away + sum_k(s_k * Q_k) + Dv_home + noise

where s_k is quarterback k's share of his team's dropbacks in the game. The
filter splits credit between team and QB in proportion to their uncertainty,
so a QB who changes teams, or a backup who plays, is handled correctly.

After a game each rating moves toward the observation in proportion to how
uncertain it was (the Kalman gain), so a team I know well moves less than a
team I know poorly. Between games uncertainty drifts up; between seasons
ratings regress toward league average (offense, defense and QBs persist at
different rates) and uncertainty jumps, so last year's strength is a prior
that this year's games gradually overwrite.

Every rating used to predict a game is computed from games in EARLIER
(season, week) groups only. The QB used for a pregame prediction is the
starter (known at kickoff); his rating is the one from before the game.
Teams play at most once per week, so updating a whole week together is
identical to updating game by game.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.config import configured_path


# Average offensive plays per team-game; observation noise scales with this.
REFERENCE_PLAYS = 62.7


@dataclass(frozen=True)
class RatingParams:
    """Hyperparameters of the rating filter (all in EPA/play units).

    Values were tuned by coordinate descent on test seasons 2011-2016 ONLY
    (src/models/ratings_experiment.py) and frozen before scoring 2017-2025.
    The loss surface is very flat, so none of these is finely sensitive.
    """

    carryover_off: float = 0.30   # team (non-QB) offense is the least persistent
    carryover_def: float = 0.70   # defense persists once QBs are split out
    season_var: float = 0.0025    # extra uncertainty injected each new season
    drift_var: float = 0.0004     # uncertainty added per game played
    obs_var: float = 0.050        # single-game EPA/play noise at REFERENCE_PLAYS
    prior_var: float = 0.008      # uncertainty of a never-seen team

    # Quarterback layer (ignored when no qb_game is supplied)
    carryover_qb: float = 0.85    # QB skill persists more than team context
    qb_season_var: float = 0.0005
    qb_drift_var: float = 0.0001
    qb_prior_var: float = 0.004   # uncertainty of a never-seen QB


def _week_order(games: pd.DataFrame) -> pd.DataFrame:
    return games.sort_values(
        ["season", "week", "game_id"]
    ).reset_index(drop=True)


def _qb_lookup(
    qb_game: pd.DataFrame | None,
) -> tuple[dict, dict]:
    """(game_id, team) -> starter id, and -> [(qb_id, dropback share)]."""

    if qb_game is None:
        return {}, {}

    required = {"game_id", "posteam", "qb_player_id", "dropbacks", "started"}
    missing = required - set(qb_game.columns)
    if missing:
        raise ValueError(f"qb_game missing columns: {sorted(missing)}")

    starters: dict[tuple, str] = {}
    shares: dict[tuple, list[tuple[str, float]]] = {}

    for (game_id, team), group in qb_game.groupby(
        ["game_id", "posteam"], sort=False
    ):
        total = float(group["dropbacks"].sum())

        if total <= 0:
            continue

        shares[(game_id, team)] = [
            (qb, float(n) / total)
            for qb, n in zip(group["qb_player_id"], group["dropbacks"])
        ]

        started = group.loc[group["started"].eq(1), "qb_player_id"]

        if len(started):
            starters[(game_id, team)] = started.iloc[0]

    return starters, shares


def build_team_ratings(
    team_game: pd.DataFrame,
    games: pd.DataFrame,
    params: RatingParams | None = None,
    qb_game: pd.DataFrame | None = None,
    projected_starters: dict[tuple, str] | None = None,
) -> pd.DataFrame:
    """Return one PREGAME rating row per team per game.

    Always: season, week, game_id, team, opponent, is_home,
            pregame_off, pregame_def, pregame_net,
            pregame_off_var, pregame_def_var, league_mu, rating_games_seen
    Games with no result yet (future games) are snapshotted but never update
    the filter. Their starter is projected_starters[(game_id, team)] when
    given, else that team's most recent starter.
    QB layer (NaN / equal to off-only when qb_game is None):
            pregame_qb_id, pregame_qb, pregame_qb_var, pregame_qb_starts,
            pregame_off_total, pregame_net_total
    """

    params = params or RatingParams()

    required_tg = {
        "season",
        "game_id",
        "team",
        "off_epa_per_play",
        "off_plays",
    }
    missing = required_tg - set(team_game.columns)
    if missing:
        raise ValueError(
            f"team_game missing columns: {sorted(missing)}"
        )

    required_games = {
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
    }
    missing = required_games - set(games.columns)
    if missing:
        raise ValueError(
            f"games missing columns: {sorted(missing)}"
        )

    schedule = _week_order(games)
    starters, shares = _qb_lookup(qb_game)
    projected_starters = projected_starters or {}
    last_starter: dict[str, str] = {}

    observed = (
        team_game
        .set_index(["game_id", "team"])[
            ["off_epa_per_play", "off_plays"]
        ]
    )

    if observed.index.duplicated().any():
        raise ValueError("Duplicate team-game rows in team_game.")

    # ---- filter state --------------------------------------------------
    off: dict[str, float] = {}
    dfn: dict[str, float] = {}
    off_var: dict[str, float] = {}
    def_var: dict[str, float] = {}
    last_season: dict[str, int] = {}
    seen: dict[str, int] = {}

    qb_rating: dict[str, float] = {}
    qb_var: dict[str, float] = {}
    qb_last_season: dict[str, int] = {}
    qb_starts: dict[str, int] = {}

    # league mean EPA/play, play-weighted, season to date
    league_num = 0.0
    league_den = 0.0
    previous_mu = 0.0
    current_season: int | None = None

    def touch(team: str, season: int) -> None:
        """Create a team on first sight; regress it on a new season."""
        if team not in off:
            off[team] = 0.0
            dfn[team] = 0.0
            off_var[team] = params.prior_var
            def_var[team] = params.prior_var
            seen[team] = 0
            last_season[team] = season
            return

        if last_season[team] != season:
            off[team] *= params.carryover_off
            dfn[team] *= params.carryover_def
            off_var[team] = (
                params.carryover_off ** 2 * off_var[team]
                + params.season_var
            )
            def_var[team] = (
                params.carryover_def ** 2 * def_var[team]
                + params.season_var
            )
            last_season[team] = season

    def touch_qb(qb: str, season: int) -> None:
        """Create a QB on first sight; regress him on a new season."""
        if qb not in qb_rating:
            qb_rating[qb] = 0.0
            qb_var[qb] = params.qb_prior_var
            qb_starts[qb] = 0
            qb_last_season[qb] = season
            return

        if qb_last_season[qb] != season:
            qb_rating[qb] *= params.carryover_qb
            qb_var[qb] = (
                params.carryover_qb ** 2 * qb_var[qb]
                + params.qb_season_var
            )
            qb_last_season[qb] = season

    rows: list[dict] = []

    for (season, week), week_games in schedule.groupby(
        ["season", "week"],
        sort=True,
    ):
        season = int(season)

        if season != current_season:
            if league_den > 0:
                previous_mu = league_num / league_den
            league_num = 0.0
            league_den = 0.0
            current_season = season

        mu = (
            league_num / league_den
            if league_den > 2000
            else previous_mu
        )

        # ---- 1) snapshot every rating BEFORE any update this week -------
        for g in week_games.itertuples(index=False):
            for team, opponent, is_home in (
                (g.home_team, g.away_team, 1),
                (g.away_team, g.home_team, 0),
            ):
                touch(team, season)
                touch(opponent, season)

                starter = starters.get((g.game_id, team))

                if starter is None and (g.game_id, team) not in observed.index:
                    # future game: announced starter, else last known starter
                    starter = projected_starters.get(
                        (g.game_id, team), last_starter.get(team)
                    )

                if starter is not None:
                    touch_qb(starter, season)
                    q = qb_rating[starter]
                    q_var = qb_var[starter]
                    q_starts = qb_starts[starter]
                else:
                    q, q_var, q_starts = 0.0, np.nan, 0

                rows.append(
                    {
                        "season": season,
                        "week": int(week),
                        "game_id": g.game_id,
                        "team": team,
                        "opponent": opponent,
                        "is_home": is_home,
                        "pregame_off": off[team],
                        "pregame_def": dfn[team],
                        "pregame_net": off[team] - dfn[team],
                        "pregame_off_var": off_var[team],
                        "pregame_def_var": def_var[team],
                        "league_mu": mu,
                        "rating_games_seen": seen[team],
                        "pregame_qb_id": starter,
                        "pregame_qb": q,
                        "pregame_qb_var": q_var,
                        "pregame_qb_starts": q_starts,
                        "pregame_off_total": off[team] + q,
                        "pregame_net_total": off[team] + q - dfn[team],
                    }
                )

        # ---- 2) update from this week's results -------------------------
        updates: list[tuple] = []
        week_num = 0.0
        week_den = 0.0

        for g in week_games.itertuples(index=False):
            for attacker, defender in (
                (g.home_team, g.away_team),
                (g.away_team, g.home_team),
            ):
                key = (g.game_id, attacker)

                if key not in observed.index:
                    continue

                y, plays = observed.loc[key]

                if pd.isna(y) or pd.isna(plays) or plays <= 0:
                    continue

                qbs = shares.get(key, [])

                for qb, _ in qbs:
                    touch_qb(qb, season)

                updates.append(
                    (attacker, defender, float(y), float(plays), qbs)
                )
                week_num += float(y) * float(plays)
                week_den += float(plays)

        # Compute every gain from the PRE-update state, then apply.
        pending = []

        for attacker, defender, y, plays, qbs in updates:
            q_term = sum(share * qb_rating[qb] for qb, share in qbs)

            innovation = (
                y - (mu + off[attacker] + q_term + dfn[defender])
            )

            noise = params.obs_var * REFERENCE_PLAYS / plays
            total = (
                off_var[attacker]
                + def_var[defender]
                + sum(share ** 2 * qb_var[qb] for qb, share in qbs)
                + noise
            )

            pending.append(
                (
                    attacker,
                    defender,
                    innovation,
                    off_var[attacker] / total,
                    def_var[defender] / total,
                    [
                        (qb, share, share * qb_var[qb] / total)
                        for qb, share in qbs
                    ],
                )
            )

        for attacker, defender, innovation, gain_off, gain_def, qb_gains in pending:
            off[attacker] += gain_off * innovation
            dfn[defender] += gain_def * innovation
            off_var[attacker] *= 1.0 - gain_off
            def_var[defender] *= 1.0 - gain_def

            for qb, share, gain_qb in qb_gains:
                qb_rating[qb] += gain_qb * innovation
                qb_var[qb] *= 1.0 - share * gain_qb

        played = {
            team
            for g in week_games.itertuples(index=False)
            for team in (g.home_team, g.away_team)
        }

        for team in played:
            off_var[team] += params.drift_var
            def_var[team] += params.drift_var
            seen[team] += 1

        for g in week_games.itertuples(index=False):
            for team in (g.home_team, g.away_team):
                for qb, _ in shares.get((g.game_id, team), []):
                    qb_var[qb] += params.qb_drift_var

                starter = starters.get((g.game_id, team))

                if starter is not None:
                    qb_starts[starter] += 1
                    last_starter[team] = starter

        league_num += week_num
        league_den += week_den

    out = pd.DataFrame(rows)

    if out.duplicated(["game_id", "team"]).any():
        raise ValueError("Duplicate rating rows.")

    return (
        out.sort_values(["season", "week", "game_id", "team"])
        .reset_index(drop=True)
    )


def attach_ratings_to_matchups(
    matchups: pd.DataFrame,
    ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Add home/away pregame ratings and their differences to each game."""

    cols = [
        "pregame_off",
        "pregame_def",
        "pregame_net",
        "pregame_off_var",
        "pregame_def_var",
        "rating_games_seen",
        "pregame_qb_id",
        "pregame_qb",
        "pregame_qb_var",
        "pregame_qb_starts",
        "pregame_off_total",
        "pregame_net_total",
    ]
    cols = [c for c in cols if c in ratings.columns]

    def side(prefix: str, team_col: str) -> pd.DataFrame:
        part = ratings[["game_id", "team", *cols]].rename(
            columns={
                "team": team_col,
                **{c: f"{prefix}_{c}" for c in cols},
            }
        )
        return part

    out = matchups.merge(
        side("home", "home_team"),
        on=["game_id", "home_team"],
        how="left",
        validate="one_to_one",
    )

    out = out.merge(
        side("away", "away_team"),
        on=["game_id", "away_team"],
        how="left",
        validate="one_to_one",
    )

    out["diff_rating_net"] = (
        out["home_pregame_net"] - out["away_pregame_net"]
    )
    out["diff_rating_off"] = (
        out["home_pregame_off"] - out["away_pregame_off"]
    )
    # positive = home defense better (allows less)
    out["diff_rating_def"] = (
        out["away_pregame_def"] - out["home_pregame_def"]
    )
    out["rating_uncertainty"] = np.sqrt(
        out["home_pregame_off_var"]
        + out["home_pregame_def_var"]
        + out["away_pregame_off_var"]
        + out["away_pregame_def_var"]
    )

    if "home_pregame_qb" in out.columns:
        out["diff_qb"] = out["home_pregame_qb"] - out["away_pregame_qb"]
        out["diff_rating_off_total"] = (
            out["home_pregame_off_total"] - out["away_pregame_off_total"]
        )
        out["diff_rating_net_total"] = (
            out["home_pregame_net_total"] - out["away_pregame_net_total"]
        )

    if len(out) != len(matchups):
        raise ValueError("Row count changed while attaching ratings.")

    return out


def save_ratings(ratings: pd.DataFrame) -> None:
    path = configured_path("processed") / "team_pregame_ratings.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(path, index=False)
    print(f"Saved: {path}")


if __name__ == "__main__":
    processed = configured_path("processed")

    team_game = pd.read_parquet(processed / "team_game_epa.parquet")
    games = pd.read_parquet(processed / "games.parquet")
    games = games.loc[games["is_played"]]

    qb_path = processed / "qb_game.parquet"
    qb_game = pd.read_parquet(qb_path) if qb_path.exists() else None

    params = RatingParams()
    print("Params:", asdict(params))
    print("QB layer:", "on" if qb_game is not None else "off (no qb_game.parquet)")

    ratings = build_team_ratings(team_game, games, params, qb_game)
    save_ratings(ratings)

    latest = ratings["season"].max()
    latest_week = ratings.loc[ratings["season"].eq(latest), "week"].max()
    snap = ratings.loc[
        ratings["season"].eq(latest) & ratings["week"].eq(latest_week)
    ]

    print()
    print(f"Ratings entering {latest} week {latest_week} "
          "(top 5 by net incl. starting QB):")
    cols = ["team", "pregame_qb_id", "pregame_off", "pregame_qb",
            "pregame_def", "pregame_net_total"]
    print(
        snap.sort_values("pregame_net_total", ascending=False)[cols]
        .head(5)
        .to_string(index=False)
    )
