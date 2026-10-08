"""Super Bowl odds for the season in progress, with a forward-test ledger.

    python -m src.simulation.title_odds              # print current odds vs the market
    python -m src.simulation.title_odds --log        # also append to ledger/title_odds.csv
    python -m src.simulation.title_odds --log --commit

Each logged row is one team's simulated odds as of "prior to week W". At the end
of the season the ledger is scored against the champion and against the market's
price for the same week (src.simulation.title_score).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd

from src.config import configured_path, project_path
from src.data.schedules import live_season
from src.features.ratings import RatingParams, attach_ratings_to_matchups, build_team_ratings
from src.live import ledger
from src.live.predict import git_state
from src.simulation.backtest import FIRST_MODEL_SEASON, GameModel, build_table
from src.simulation.season import SeasonState, simulate_season
from src.simulation.season_backtest import league_state, snapshot_at


# v2: neutral-site (Super Bowl) probabilities fixed; v1 subtracted only the logistic intercept, ignoring the
# scaler's centering, which shifted every neutral-site logit by about -0.015 (~0.4 points at 50%).
MODEL_ID = "season_sim_v2"
TITLE_LEDGER = project_path("ledger", "title_odds.csv")
KEY = ("model_id", "season", "week_prior", "team")

FIELDS = [
    "entry_id", "logged_at_utc", "season", "week_prior", "team", "model_id",
    "model_sha", "params_sha", "p_win_sb", "p_win_conf", "p_playoffs", "p_division",
    "expected_wins", "n_sims", "data_asof", "prev_hash", "row_hash",
]


def params_sha(n_sims: int) -> str:
    payload = json.dumps(
        {"model": MODEL_ID, "ratings": asdict(RatingParams()), "features": ["diff_rating_off_total", "diff_rating_def"],
         "tiebreak": "random", "n_sims": n_sims, "first_model_season": FIRST_MODEL_SEASON},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def next_week(games: pd.DataFrame, season: int) -> int:
    reg = games.loc[games["season"].eq(season) & games["game_type"].eq("REG") & ~games["is_played"]]
    if reg.empty:
        raise SystemExit(f"No unplayed regular-season games left in {season}.")
    return int(reg["week"].min())


def current_odds(n_sims: int = 65536) -> tuple[int, pd.DataFrame, str]:
    processed = configured_path("processed")
    season = live_season()

    games = pd.read_parquet(processed / "games.parquet")
    team_game = pd.read_parquet(processed / "team_game_epa.parquet")
    qb_game = pd.read_parquet(processed / "qb_game.parquet")
    matchups = pd.read_parquet(processed / "historical_matchups_oa.parquet")

    week = next_week(games, season)
    params = RatingParams()

    full = build_team_ratings(team_game, games.loc[games["is_played"]], params, qb_game)
    train = attach_ratings_to_matchups(
        matchups.loc[matchups["season"].between(FIRST_MODEL_SEASON, season)], full
    )
    model = GameModel(train)

    wins, remaining, _ = league_state(games, season, week)
    snap = snapshot_at(team_game, qb_game, games, season, week, params)
    table = build_table(snap, model)

    sim = simulate_season(SeasonState(wins, remaining, 7), table, n_sims=n_sims, seed=season * 100 + week)

    played = games.loc[games["is_played"]].sort_values(["season", "week"])
    return week, sim, played["game_id"].iloc[-1]


def market_prices(week: int) -> tuple[pd.Series, int]:
    path = configured_path("processed") / "futures_weekly_sb.parquet"
    if not path.exists():
        return pd.Series(dtype=float), -1

    w = pd.read_parquet(path)
    w = w.loc[w["season"].eq(live_season()) & w["week_prior"].le(week)]
    if w.empty:
        return pd.Series(dtype=float), -1

    latest = int(w["week_prior"].max())
    return w.loc[w["week_prior"].eq(latest)].set_index("team")["p_market"], latest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--sims", type=int, default=65536)
    args = parser.parse_args()

    sha, dirty = git_state()
    if args.log and dirty:
        raise SystemExit("Uncommitted code changes: commit first so the ledger's git sha is meaningful.")

    if not args.no_refresh:
        print("Refreshing data ...")
        run = subprocess.run([sys.executable, "-m", "src.pipeline"], capture_output=True, text=True)
        if run.returncode != 0:
            raise SystemExit(run.stdout[-1500:] + run.stderr[-1500:])
        subprocess.run([sys.executable, "-m", "src.data.futures"], capture_output=True, text=True)

    week, sim, asof = current_odds(args.sims)
    prices, price_week = market_prices(week)

    out = sim.set_index("team")
    out["market_sb"] = prices.reindex(out.index)
    show = out[["expected_wins", "p_playoffs", "p_division", "p_first_seed", "p_win_conf", "p_win_sb", "market_sb"]]

    print(f"\n{live_season()} Super Bowl odds, prior to week {week} ({args.sims:,} simulated seasons)")
    print(f"market = BetMGM via Covers, de-vigged, prior to week {price_week}\n")
    print(show.head(16).round(3).to_string())
    print(f"\nmax Monte Carlo standard error on a title probability: {sim['p_win_sb_se'].max():.4f}")

    if not args.log:
        return

    now = datetime.now(timezone.utc)
    rows = [
        {
            "logged_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "season": live_season(), "week_prior": week,
            "team": r.team, "model_id": MODEL_ID, "model_sha": sha, "params_sha": params_sha(args.sims),
            "p_win_sb": f"{r.p_win_sb:.6f}", "p_win_conf": f"{r.p_win_conf:.6f}",
            "p_playoffs": f"{r.p_playoffs:.6f}", "p_division": f"{r.p_division:.6f}",
            "expected_wins": f"{r.expected_wins:.4f}", "n_sims": args.sims, "data_asof": asof,
        }
        for r in sim.itertuples()
    ]

    added = ledger.append(rows, TITLE_LEDGER, fields=FIELDS, key=KEY)
    ok, message = ledger.verify(TITLE_LEDGER, FIELDS)
    print(f"\nLogged {added} team rows for week {week}. {message}")

    if args.commit and added and ok:
        subprocess.run(["git", "add", "ledger/title_odds.csv"], check=True)
        subprocess.run(["git", "commit", "-q", "-m", f"Title ledger: week {week} odds ({now:%Y-%m-%d} UTC)"], check=True)
        print("Committed.")


if __name__ == "__main__":
    main()
