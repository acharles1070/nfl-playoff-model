"""Pre-game predictions for the upcoming week.

Everything here is computed from games that have already finished. For each
registered model:

  1. rebuild the team/QB ratings over every played game;
  2. project each team's starting QB (the schedule's announced starter when
     available, else the team's last starter);
  3. fit the game model on all completed games (the same thing the walk-forward
     backtest does at this point in time);
  4. score the next unplayed week, skipping any game whose kickoff has passed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.config import configured_path
from src.features.injuries import attach_burden
from src.features.ratings import (
    RatingParams,
    attach_ratings_to_matchups,
    build_team_ratings,
)
from src.models.backtest import build_logistic_model
from src.models.market import market_probabilities


EASTERN = ZoneInfo("America/New_York")
FIRST_MODEL_SEASON = 2010          # 2009 is a cold start


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    features: tuple[str, ...]
    params: RatingParams = RatingParams()
    use_qb: bool = True
    use_injuries: bool = False


MODELS: dict[str, ModelSpec] = {
    # CHAMPION: frozen after the 2017-2025 holdout.
    "ratings_qb_logit_v1": ModelSpec(
        model_id="ratings_qb_logit_v1",
        features=("diff_rating_off_total", "diff_rating_def"),
    ),
    # CHALLENGER: adds measured injury burden. Historically a small, not
    # statistically significant gain (holdout -0.0014 log loss, CI includes 0),
    # so it is forward-tested here instead of adopted. Log it AFTER the Friday
    # injury reports so it sees real information.
    "ratings_qb_injury_logit_v1": ModelSpec(
        model_id="ratings_qb_injury_logit_v1",
        features=(
            "diff_rating_off_total",
            "diff_rating_def",
            "diff_lost_off_total",
            "diff_lost_def_total",
        ),
        use_injuries=True,
    ),
}


def git_state() -> tuple[str, bool]:
    """(HEAD sha, working tree has uncommitted tracked changes outside ledger/)."""
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    dirty = any(not line[3:].startswith("ledger/") for line in status)

    return sha, dirty


def params_sha(spec: ModelSpec) -> str:
    payload = {
        "model_id": spec.model_id,
        "features": list(spec.features),
        "params": asdict(spec.params),
        "use_qb": spec.use_qb,
    }

    # Only recorded when on, so models defined before this option existed keep
    # the identity already written to the ledger.
    if spec.use_injuries:
        payload["use_injuries"] = True

    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def kickoff_utc(games: pd.DataFrame) -> pd.Series:
    """nflverse gameday + gametime are Eastern."""
    local = pd.to_datetime(
        games["gameday"].astype(str) + " " + games["gametime"].astype(str),
        errors="coerce",
    )

    return local.dt.tz_localize(
        EASTERN, nonexistent="shift_forward", ambiguous="NaT"
    ).dt.tz_convert(timezone.utc)


def load_inputs() -> dict[str, pd.DataFrame]:
    processed = configured_path("processed")

    return {
        "team_game": pd.read_parquet(processed / "team_game_epa.parquet"),
        "games": pd.read_parquet(processed / "games.parquet"),
        "qb_game": pd.read_parquet(processed / "qb_game.parquet"),
        "matchups": pd.read_parquet(processed / "historical_matchups_oa.parquet"),
        "burden": pd.read_parquet(processed / "team_week_injury_burden.parquet"),
    }


def next_week_games(
    games: pd.DataFrame,
    now: datetime,
) -> pd.DataFrame:
    """Unplayed games in the earliest week that still has games to come."""
    pending = games.loc[~games["is_played"]].copy()
    pending["kickoff_utc"] = kickoff_utc(pending)

    pending = pending.loc[pending["kickoff_utc"].gt(now)]

    if pending.empty:
        return pending

    first = pending.sort_values(["season", "week"]).iloc[0]

    return pending.loc[
        pending["season"].eq(first["season"]) & pending["week"].eq(first["week"])
    ].sort_values("kickoff_utc")


def predict_upcoming(
    spec: ModelSpec,
    inputs: dict[str, pd.DataFrame],
    now: datetime,
) -> pd.DataFrame:
    games = inputs["games"]
    upcoming = next_week_games(games, now)

    if upcoming.empty:
        return upcoming

    # announced starters for games that have not happened yet
    announced: dict[tuple, str] = {}

    for side in ("home", "away"):
        col = f"{side}_qb_id"

        for g in upcoming.itertuples(index=False):
            qb = getattr(g, col)

            if pd.notna(qb):
                announced[(g.game_id, getattr(g, f"{side}_team"))] = qb

    ratings = build_team_ratings(
        inputs["team_game"],
        games,
        spec.params,
        inputs["qb_game"] if spec.use_qb else None,
        projected_starters=announced,
    )

    # ---- fit on every completed game ------------------------------------
    train = attach_ratings_to_matchups(inputs["matchups"], ratings)
    train = train.loc[train["season"].ge(FIRST_MODEL_SEASON)]

    if spec.use_injuries:
        train = attach_burden(train, inputs["burden"])

    features = list(spec.features)
    model = build_logistic_model(features)
    model.fit(train, train["home_win"].astype(int).to_numpy())

    # ---- score the upcoming week -----------------------------------------
    frame = upcoming[["game_id", "season", "week", "home_team", "away_team"]]
    scored = attach_ratings_to_matchups(frame, ratings)

    if spec.use_injuries:
        scored = attach_burden(scored, inputs["burden"])

    scored["p_home_win"] = model.predict_proba(scored)[:, 1]

    market = market_probabilities(upcoming)[["game_id", "p_market"]]

    out = (
        scored.merge(
            upcoming[
                [
                    "game_id", "kickoff_utc", "spread_line",
                    "home_moneyline", "away_moneyline",
                ]
            ],
            on="game_id",
        )
        .merge(market, on="game_id", how="left")
        .sort_values("kickoff_utc")
        .reset_index(drop=True)
    )

    out["model_id"] = spec.model_id
    out["params_sha"] = params_sha(spec)

    last_played = games.loc[games["is_played"]].sort_values(["season", "week"])
    out["data_asof"] = last_played["game_id"].iloc[-1]

    return out


def to_ledger_rows(
    predictions: pd.DataFrame,
    *,
    logged_at: datetime,
    model_sha: str,
) -> list[dict[str, str]]:
    def num(x, digits=8):
        return "" if pd.isna(x) else f"{float(x):.{digits}f}"

    rows = []

    for r in predictions.itertuples(index=False):
        rows.append(
            {
                "logged_at_utc": logged_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "kickoff_utc": r.kickoff_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "season": int(r.season),
                "week": int(r.week),
                "game_id": r.game_id,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "model_id": r.model_id,
                "model_sha": model_sha,
                "params_sha": r.params_sha,
                "p_home_win": num(r.p_home_win),
                "home_qb": r.home_pregame_qb_id,
                "away_qb": r.away_pregame_qb_id,
                "diff_rating_off_total": num(r.diff_rating_off_total),
                "diff_rating_def": num(r.diff_rating_def),
                "market_p_home": num(r.p_market),
                "spread_line": num(r.spread_line, 1),
                "home_moneyline": num(r.home_moneyline, 0),
                "away_moneyline": num(r.away_moneyline, 0),
                "data_asof": r.data_asof,
            }
        )

    return rows
