"""Export every dataset the public website and paper use, straight from pipeline outputs.

    python -m src.report.site_data            # writes site/data/*.json

No number on the site is typed by hand: each JSON file is recomputed here from the
walk-forward predictions, ledgers and registry, so the site can be rebuilt after every
season. Market prices
come from nflverse's schedule file (public); the Covers futures prices are NOT exported.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.betting.run import THRESHOLDS, load as load_betting
from src.betting.sim import decimal_odds, place_bets
from src.betting.stats import deflated_sharpe, roi_ci, sharpe
from src.config import configured_path, project_path
from src.models.compare import attach_market, paired_bootstrap_delta
from src.models.encompassing import encompassing
from src.models.registry import games_needed, paired_sigma
from src.data.schedules import last_complete_season
from src.features.ratings import RatingParams, attach_ratings_to_matchups, build_team_ratings
from src.live import ledger as live_ledger
from src.models.market import market_probabilities
from src.simulation.backtest import FIRST_MODEL_SEASON, GameModel, build_table, round_snapshot
from src.simulation.seeding import reconstruct_field

HOLDOUT = (2017, 2025)
AFC = {"BAL", "BUF", "CIN", "CLE", "DEN", "HOU", "IND", "JAX", "KC", "LAC", "LV", "MIA", "NE", "NYJ", "PIT", "TEN", "OAK", "SD"}
SITE_DATA = project_path("site", "data")
EPS = 1e-12

TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams", "LAR": "Los Angeles Rams", "LAC": "Los Angeles Chargers", "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints",
    "NYG": "New York Giants", "NYJ": "New York Jets", "OAK": "Oakland Raiders", "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers", "SD": "San Diego Chargers", "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers",
    "STL": "St. Louis Rams", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _clean(obj):
    """NaN/inf are not valid JSON; browsers refuse to parse them. Map them to null."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
        return None
    return obj


def write(name: str, payload) -> None:
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    path = SITE_DATA / f"{name}.json"
    path.write_text(json.dumps(_clean(payload), separators=(",", ":"), default=float, allow_nan=False))
    print(f"  {name}.json  {path.stat().st_size / 1024:.1f} KB")


def holdout_population() -> pd.DataFrame:
    """Holdout games with a market line: one row per game, one column per model."""
    out = configured_path("outputs")
    games = pd.read_parquet(configured_path("processed") / "games.parquet")

    pred = pd.read_parquet(out / "ratings_experiment_predictions.parquet")
    pred = pred[pred.season.between(*HOLDOUT)]
    wide = pred.pivot(index="game_id", columns="feature_set", values="p_home_win")

    base = pred[pred.feature_set == "home_only"][["game_id", "season", "week", "game_type", "is_postseason",
                                                  "home_team", "away_team", "home_win"]].drop_duplicates("game_id")
    pop = base.merge(wide, left_on="game_id", right_index=True)
    pop = attach_market(pop.rename(columns={"home_only": "p_home_only"}), games)
    return pop.rename(columns={
        "baseline std_epa": "p_std_epa",
        "ratings, no QB (previous model)": "p_kalman",
        "ratings + QB (tuned, frozen)": "p_champion",
    })


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

def scorecard(pop: pd.DataFrame) -> dict:
    y = pop["home_win"].astype(int).to_numpy()
    rows = []
    market = pop["p_market"].to_numpy()

    for key, label, blurb in (
        ("p_home_only", "Home-field only", "Always say the home team wins ~57% of the time."),
        ("p_std_epa", "Season-to-date EPA", "Rate teams by their average points-added-per-play so far this year."),
        ("p_kalman", "Kalman ratings", "Team strengths that update after every game and remember last season."),
        ("p_champion", "Kalman + quarterback", "Also tracks each quarterback separately. My champion model."),
        ("p_market", "Betting market", "The closing line, with the bookmaker's margin removed."),
    ):
        p = pop[key].to_numpy()
        ll = log_loss(y, p)
        d, lo, hi = (0.0, 0.0, 0.0) if key == "p_market" else paired_bootstrap_delta(y, p, market)
        rows.append({
            "key": key, "label": label, "blurb": blurb, "log_loss": ll,
            "brier": float(np.mean((p - y) ** 2)), "accuracy": float(np.mean((p > 0.5) == (y == 1))),
            "gap_vs_market": d, "gap_lo": lo, "gap_hi": hi,
        })

    post = pop[pop.is_postseason.astype(bool)]
    return {"games": int(len(pop)), "postseason_games": int(len(post)), "window": list(HOLDOUT), "models": rows}


def quiz(pop: pd.DataFrame, n: int = 30, seed: int = 7) -> list[dict]:
    games = pd.read_parquet(configured_path("processed") / "games.parquet")
    cols = ["game_id", "gameday", "home_score", "away_score", "spread_line"]
    sample = pop.sample(n=n, random_state=seed).merge(games[cols], on="game_id")
    sample = sample.sort_values("gameday")

    return [
        {
            "id": r.game_id, "season": int(r.season), "week": int(r.week), "type": r.game_type, "date": str(r.gameday)[:10],
            "home": r.home_team, "away": r.away_team, "home_score": int(r.home_score), "away_score": int(r.away_score),
            "spread": None if pd.isna(r.spread_line) else float(r.spread_line),
            "p_model": round(float(r.p_champion), 4), "p_market": round(float(r.p_market), 4), "home_win": int(r.home_win),
        }
        for r in sample.itertuples()
    ]


def calibration(pop: pd.DataFrame, bins: int = 10) -> dict:
    out = {}
    y = pop["home_win"].astype(int).to_numpy()
    edges = np.linspace(0, 1, bins + 1)

    for key in ("p_champion", "p_market"):
        p = pop[key].to_numpy()
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        rows = []
        for b in range(bins):
            m = idx == b
            n = int(m.sum())
            if n < 5:
                continue
            obs, pred = float(y[m].mean()), float(p[m].mean())
            half = 1.96 * np.sqrt(max(obs * (1 - obs), 0.02) / n)
            rows.append({"n": n, "pred": pred, "obs": obs, "lo": max(0.0, obs - half), "hi": min(1.0, obs + half)})
        # calibration slope: logistic regression of outcome on logit(p)
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        slope = float(LogisticRegression(C=1e6).fit(z.reshape(-1, 1), y).coef_[0][0])
        out[key] = {"bins": rows, "slope": slope}
    return out


def betting() -> dict:
    g = load_betting()
    hold = g[g.season.between(*HOLDOUT)]
    series = {}

    # Deflated Sharpe: every threshold (flat and Kelly place the same bets) counts as a trial
    all_sharpes = [sharpe(place_bets(hold, threshold=t, mode="flat").ret.to_numpy()) for t in THRESHOLDS] * 2
    n_trials, var_sr = len(all_sharpes), float(np.var(all_sharpes))

    for th in (0.0, 0.02, 0.04, 0.08):
        flat = place_bets(hold, threshold=th, mode="flat")
        kelly = place_bets(hold, threshold=th, mode="kelly")
        roi, roi_lo, roi_hi = roi_ci(flat.ret.to_numpy())
        series[str(th)] = {
            "roi_lo": float(roi_lo), "roi_hi": float(roi_hi), "sharpe": float(sharpe(flat.ret.to_numpy())),
            "dsr": float(deflated_sharpe(flat.ret.to_numpy(), n_trials, var_sr)), "n_trials": n_trials,
            "bets": int(len(flat)), "roi": float(flat.ret.mean()),
            "flat": np.round(np.r_[0.0, flat.pnl.cumsum().to_numpy()], 2).tolist(),
            "kelly": np.round(np.r_[100.0, kelly.bankroll.to_numpy()], 2).tolist(),
            "season_end": [int(i) for i in np.flatnonzero(np.diff(flat.season.to_numpy()) != 0) + 1],
            "seasons": sorted(int(s) for s in flat.season.unique()),
        }

    dh, da = decimal_odds(hold.home_moneyline), decimal_odds(hold.away_moneyline)
    fav_home = dh < da
    won_fav = np.where(fav_home, hold.home_win == 1, hold.home_win == 0)
    base = {
        "Bet every favorite": np.where(won_fav, np.where(fav_home, dh, da) - 1, -1.0),
        "Bet every home team": np.where(hold.home_win == 1, dh - 1, -1.0),
    }
    baselines = {k: {"roi": float(v.mean()), "curve": np.round(np.r_[0.0, np.cumsum(v)], 2).tolist(),
                     "roi_lo": float(roi_ci(v)[1]), "roi_hi": float(roi_ci(v)[2])} for k, v in base.items()}
    # model's picks vs. betting the favorite on the SAME games (paired, bootstrap over games)
    flat0 = place_bets(hold, threshold=0.0, mode="flat").set_index("game_id")
    fav_ret = pd.Series(base["Bet every favorite"], index=hold["game_id"].to_numpy())
    diff = flat0["ret"].to_numpy() - fav_ret.loc[flat0.index].to_numpy()
    boot = diff[np.random.default_rng(0).integers(0, len(diff), (5000, len(diff)))].mean(axis=1)
    favorite_is_home = pd.Series(dh < da, index=hold["game_id"].to_numpy()).loc[flat0.index]
    on_favorite = float(np.mean(flat0["side"].to_numpy() == np.where(favorite_is_home.to_numpy(), "home", "away")))
    vs_fav = {"diff": float(diff.mean()), "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
              "model_on_favorite": on_favorite}
    return {"series": series, "baselines": baselines, "vs_favorite": vs_fav, "overround": float(np.mean(1 / dh + 1 / da)), "games": int(len(hold))}


def registry() -> dict:
    df = pd.read_csv(project_path("docs", "experiment_registry.csv"))
    df = df.sort_values("id", key=lambda c: c.str[1:].astype(int)).reset_index(drop=True)
    rows = [
        {"id": r.id, "area": r.area, "name": r.experiment, "delta": r.delta_log_loss, "lo": r.ci_low, "hi": r.ci_high,
         "p": r.p_raw, "p_holm": r.p_holm, "raw_sig": bool(r.significant_raw), "survives": bool(r.survives_holm)}
        for r in df.itertuples()
    ]
    return {"rows": rows, "n": len(rows), "raw_sig": int(df.significant_raw.sum()), "holm": int(df.survives_holm.sum())}


def encompassing_export(pop: pd.DataFrame) -> dict:
    """Does the model add information beyond the closing market? (holdout, champion model)"""
    r = encompassing(pop.rename(columns={"p_champion": "p_home_win"}))
    return {"n": int(r["n"]), "model_coef": float(r["model_coef"]), "model_lo": float(r["model_ci"][0]), "model_hi": float(r["model_ci"][1]),
            "model_p": float(r["model_p_value"]), "market_coef": float(r["market_coef"]), "market_lo": float(r["market_ci"][0]),
            "market_hi": float(r["market_ci"][1]), "corr": float(r["corr"])}


def replication_export() -> dict:
    """E01 replication on seasons outside the design (docs/replication_e01.csv, from src.models.replication)."""
    df = pd.read_csv(project_path("docs", "replication_e01.csv"))
    return {"rows": df.to_dict("records")}


def power_export() -> dict:
    """Single source for the power calculations used by the site, paper and registry."""
    sigma = paired_sigma()
    return {"sigma": sigma, "games_002": games_needed(sigma, 0.002), "games_005": games_needed(sigma, 0.005)}


def leakage_demo(pop: pd.DataFrame) -> dict:
    """What 'peeking' does, using only EPA (no proprietary data).

    LEAKY: rate each team by its FULL-season net EPA/play (which already contains the game
    being predicted and every later game). HONEST: season-to-date EPA, games before kickoff.
    Both are fit walk-forward and scored on the same regular-season holdout games.
    This deliberately leaky code exists ONLY to illustrate the trap; no model uses it.
    """
    processed = configured_path("processed")
    tg = pd.read_parquet(processed / "team_game_epa.parquet")
    games = pd.read_parquet(processed / "games.parquet")

    reg = tg[tg.season_type == "REG"].copy()
    reg["net"] = reg["off_epa_per_play"] - reg["def_allowed_epa_per_play"]
    final = reg.groupby(["season", "team"])["net"].mean().rename("final_net").reset_index()

    g = games[games.is_played & (games.game_type == "REG")][["game_id", "season", "home_team", "away_team", "home_win"]]
    g = g.merge(final.rename(columns={"team": "home_team", "final_net": "h"}), on=["season", "home_team"])
    g = g.merge(final.rename(columns={"team": "away_team", "final_net": "a"}), on=["season", "away_team"])
    g["diff"] = g["h"] - g["a"]

    preds = []
    for season in range(HOLDOUT[0], HOLDOUT[1] + 1):
        train, test = g[(g.season < season) & (g.season >= 2002)], g[g.season == season]
        m = LogisticRegression(C=1e6).fit(train[["diff"]], train.home_win.astype(int))
        preds.append(pd.DataFrame({"game_id": test.game_id.to_numpy(), "p_leaky": m.predict_proba(test[["diff"]])[:, 1]}))
    leaky = pd.concat(preds)

    reg_pop = pop[~pop.is_postseason.astype(bool)].merge(leaky, on="game_id")
    y = reg_pop.home_win.astype(int).to_numpy()
    rows = [
        ("Peeking: full-season EPA", reg_pop.p_leaky, "peek"),
        ("Honest: season-to-date EPA", reg_pop.p_std_epa, "honest"),
        ("Honest: Kalman + QB", reg_pop.p_champion, "honest"),
        ("Betting market", reg_pop.p_market, "market"),
    ]
    return {"games": int(len(reg_pop)),
            "rows": [{"label": lab, "log_loss": log_loss(y, p.to_numpy()), "kind": kind} for lab, p, kind in rows]}


def kalman_demo() -> dict:
    """Per-game observations for a handful of teams in the last complete season, plus the
    full model's own pregame rating, so the browser can re-run a simplified 1-D filter."""
    processed = configured_path("processed")
    tg = pd.read_parquet(processed / "team_game_epa.parquet")
    games = pd.read_parquet(processed / "games.parquet")
    qb = pd.read_parquet(processed / "qb_game.parquet")
    season = last_complete_season()

    ratings = build_team_ratings(tg, games[games.is_played], RatingParams(), qb)
    r = ratings[(ratings.season == season)].merge(
        tg[["game_id", "team", "season_type", "off_epa_per_play", "def_allowed_epa_per_play"]], on=["game_id", "team"])
    r = r[r.season_type == "REG"].sort_values(["team", "week"])
    r["obs"] = r.off_epa_per_play - r.def_allowed_epa_per_play

    summary = r.groupby("team").agg(first=("pregame_net_total", "first"), last=("pregame_net_total", "last"), avg=("obs", "mean"))
    summary["move"] = summary["last"] - summary["first"]
    picks: dict[str, str] = {}
    for label, column, ascending in (("Best team", "avg", False), ("Worst team", "avg", True),
                                     ("Biggest riser", "move", False), ("Biggest faller", "move", True)):
        # the demo needs four DIFFERENT teams, so skip anyone already chosen
        picks[label] = next(t for t in summary[column].sort_values(ascending=ascending).index if t not in picks.values())
    out = []
    for label, team in picks.items():
        t = r[r.team == team]
        out.append({
            "label": label, "team": team, "season": season,
            "prior_mean": float(t.pregame_net_total.iloc[0]), "prior_sd": float(np.sqrt(t.pregame_off_var.iloc[0] + t.pregame_def_var.iloc[0])),
            "games": [{"week": int(w), "opp": o, "obs": float(x), "model": float(m)}
                      for w, o, x, m in zip(t.week, t.opponent, t.obs, t.pregame_net_total)],
        })
    return {"season": season, "teams": out, "league_game_sd": float(np.sqrt(0.1))}


def bracket_demo() -> dict:
    """The last complete season's playoff field with the fitted model's game probabilities."""
    processed = configured_path("processed")
    games = pd.read_parquet(processed / "games.parquet")
    tg = pd.read_parquet(processed / "team_game_epa.parquet")
    qb = pd.read_parquet(processed / "qb_game.parquet")
    matchups = pd.read_parquet(processed / "historical_matchups_oa.parquet")
    season = last_complete_season()

    ratings = build_team_ratings(tg, games[games.is_played], RatingParams(), qb)
    train = attach_ratings_to_matchups(matchups[matchups.season.between(FIRST_MODEL_SEASON, season - 1)], ratings)
    model = GameModel(train)

    post = games[(games.season == season) & games.is_postseason & games.is_played]
    fields = reconstruct_field(games, season)
    teams = [t for seeds in fields.values() for t in seeds]
    table = build_table(round_snapshot(ratings, post, "WC", teams), model)

    conf = {}
    for label, seeds in fields.items():
        conf["AFC" if set(seeds) & AFC else "NFC"] = seeds

    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    mh, mn = np.zeros((n, n)), np.zeros((n, n))
    for (i, j), v in table.m_home.items():
        mh[idx[i], idx[j]] = v
    for (i, j), v in table.m_neutral.items():
        mn[idx[i], idx[j]] = v

    results = [{"round": r.game_type, "home": r.home_team, "away": r.away_team, "home_score": int(r.home_score),
                "away_score": int(r.away_score)} for r in post.sort_values("week").itertuples()]
    sb = next(r for r in results if r["round"] == "SB")
    champion = sb["home"] if sb["home_score"] > sb["away_score"] else sb["away"]
    return {"season": season, "teams": teams, "conferences": conf, "m_home": np.round(mh, 5).tolist(),
            "m_neutral": np.round(mn, 5).tolist(), "tau2": [round(float(table.tau2[t]), 6) for t in teams],
            "results": results, "champion": champion}


def live_ledger_export() -> dict:
    games = pd.read_parquet(configured_path("processed") / "games.parquet")
    rows = live_ledger.read_rows()
    done = games[games.is_played].set_index("game_id")
    finished = {r["game_id"]: {"home_score": int(done.loc[r["game_id"], "home_score"]), "away_score": int(done.loc[r["game_id"], "away_score"])}
                for r in rows if r["game_id"] in done.index}

    title = pd.read_csv(project_path("ledger", "title_odds.csv"))
    latest = title[title.week_prior == title.week_prior.max()].sort_values("p_win_sb", ascending=False)
    return {
        "fields": live_ledger.FIELDS, "genesis": live_ledger.GENESIS, "rows": rows, "finished": finished,
        "title": {"season": int(latest.season.iloc[0]), "week": int(latest.week_prior.iloc[0]),
                  "logged": latest.logged_at_utc.iloc[0],
                  "teams": latest[["team", "p_win_sb", "p_win_conf", "p_playoffs", "p_division", "expected_wins"]].round(4).to_dict("records")},
    }


def meta() -> dict:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    games = pd.read_parquet(configured_path("processed") / "games.parquet")
    played = games[games.is_played].sort_values(["season", "week"])
    return {"built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "git_sha": sha,
            "last_game": played.game_id.iloc[-1], "teams": TEAM_NAMES}


def main() -> None:
    print("Exporting site data ...")
    pop = holdout_population()
    write("scorecard", scorecard(pop))
    write("quiz", quiz(pop))
    write("calibration", calibration(pop))
    write("betting", betting())
    write("registry", registry())
    write("leakage", leakage_demo(pop))
    write("encompassing", encompassing_export(pop))
    write("power", power_export())
    write("replication", replication_export())
    write("kalman", kalman_demo())
    write("bracket", bracket_demo())
    write("ledger", live_ledger_export())
    write("meta", meta())


if __name__ == "__main__":
    main()
