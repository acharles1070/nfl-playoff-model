"""How far ahead can an opponent be forecast?

    python -m src.opponents.horizon

Opponent planning is a forecast made weeks before kickoff, with ratings that will still move and a quarterback
who may change. For every test season and every as-of week w, this forecasts each remaining regular-season game
from ratings built from games BEFORE w only (ratings frozen, each team's quarterback = its last starter), then
scores it against what happened and against three references on the same games:

  oracle     the same game model with the pregame ratings and the actual starter on game day (horizon zero, the
             forecast the week of the game). Forecast minus oracle is the price of looking ahead.
  preseason  the same model with ratings frozen before week 1 (an offseason power ranking).
  home field the training-window home win rate, no team information at all.

Two things come out of it (both set on the development window 2011-2016 and then scored on 2017-2025):

  shrink     one parameter a: logit / (1 + a * weeks_ahead), because frozen ratings make the game model overconfident.
             This is a candidate change to the forecast, so it is in the experiment registry (E39).
  revision   the spread of how far the shrunk forecast moves by game week, by horizon: the 80% band shown in reports.

Everything else here is a diagnostic of one fixed model.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

from src.config import configured_path, project_path
from src.data.schedules import last_complete_season
from src.features.ratings import attach_ratings_to_matchups
from src.models.compare import per_game_log_loss
from src.opponents.forecast import HORIZON_SHRINK_A, forecast_frame, projected_ratings, revision_band, shrink_logit, sigmoid
from src.simulation.backtest import FIRST_MODEL_SEASON, GameModel
from src.simulation.live_bracket import Inputs, load_inputs

FIRST_TEST_SEASON = 2011
DEV_LAST = 2016
AS_OF_WEEKS = (1, 3, 6, 9, 12, 15)
BUCKETS = [("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5-8", 5, 8), ("9+", 9, 99)]
SD_CAP = 12                       # horizons of 12 or more weeks share one revision SD
N_BOOT = 2000


def _changed(a: pd.Series, b: pd.Series) -> np.ndarray:
    return ((a.to_numpy() != b.to_numpy()) & ~(a.isna().to_numpy() & b.isna().to_numpy()))


def season_rows(inp: Inputs, season: int, as_of_weeks=AS_OF_WEEKS) -> pd.DataFrame:
    """Forecast every remaining regular-season game of `season` from each as-of week."""
    mu = inp.matchups
    train = mu.loc[mu["season"].between(FIRST_MODEL_SEASON, season - 1)]
    model = GameModel(attach_ratings_to_matchups(train, inp.ratings))
    p_const = float(train.loc[train["game_type"].eq("REG"), "home_win"].mean())

    games = mu.loc[mu["season"].eq(season) & mu["game_type"].eq("REG")].reset_index(drop=True)
    neutral = games["location"].astype(str).str.lower().eq("neutral").to_numpy()
    oracle_frame = attach_ratings_to_matchups(games, inp.ratings)
    oracle = forecast_frame(model, oracle_frame, neutral)

    pre_frame = attach_ratings_to_matchups(games, projected_ratings(inp, season, 1))
    pre = forecast_frame(model, pre_frame, neutral)

    out = []
    for w in as_of_weeks:
        keep = games["week"].ge(w).to_numpy()
        if not keep.any():
            continue
        frozen_frame = pre_frame if w == 1 else attach_ratings_to_matchups(games, projected_ratings(inp, season, w))
        fz = forecast_frame(model, frozen_frame, neutral)
        sel = np.flatnonzero(keep)
        qb_h = _changed(frozen_frame["home_pregame_qb_id"], oracle_frame["home_pregame_qb_id"])
        qb_a = _changed(frozen_frame["away_pregame_qb_id"], oracle_frame["away_pregame_qb_id"])
        out.append(pd.DataFrame({
            "season": season, "as_of": w, "game_id": games.loc[sel, "game_id"].to_numpy(),
            "week": games.loc[sel, "week"].to_numpy(), "h": games.loc[sel, "week"].to_numpy() - w,
            "y": games.loc[sel, "home_win"].to_numpy(dtype=float),
            "logit_frozen": fz["logit"].to_numpy()[sel], "p_frozen": fz["p_home"].to_numpy()[sel],
            "logit_oracle": oracle["logit"].to_numpy()[sel], "p_oracle": oracle["p_home"].to_numpy()[sel],
            "p_pre": pre["p_home"].to_numpy()[sel], "p_const": p_const,
            "qb_changed": (qb_h | qb_a)[sel],
        }))
    return pd.concat(out, ignore_index=True)


def build_rows(inp: Inputs, seasons=None, as_of_weeks=AS_OF_WEEKS) -> pd.DataFrame:
    seasons = list(seasons or range(FIRST_TEST_SEASON, last_complete_season() + 1))
    frames = []
    for s in seasons:
        t = time.time()
        frames.append(season_rows(inp, s, as_of_weeks))
        print(f"  {s}: {len(frames[-1]):,} forecasts ({time.time() - t:.0f}s)", flush=True)
    rows = pd.concat(frames, ignore_index=True)
    rows["window"] = np.where(rows["season"].le(DEV_LAST), "dev", "holdout")
    return rows


def fit_shrink(rows: pd.DataFrame) -> float:
    """The one fitted parameter: a in logit / (1 + a h), by log loss on the development window only."""
    dev = rows.loc[rows["window"].eq("dev")]
    y, z, h = dev["y"].to_numpy(), dev["logit_frozen"].to_numpy(), dev["h"].to_numpy()
    res = minimize_scalar(lambda a: per_game_log_loss(y, sigmoid(shrink_logit(z, h, a))).mean(), bounds=(0.0, 1.0), method="bounded")
    return float(res.x)


def add_shrunk(rows: pd.DataFrame, a: float) -> pd.DataFrame:
    z = shrink_logit(rows["logit_frozen"].to_numpy(), rows["h"].to_numpy(), a)
    return rows.assign(logit_shrunk=z, p_shrunk=sigmoid(z))


def revision_sd(rows: pd.DataFrame) -> pd.DataFrame:
    """RMS of (oracle logit - shrunk logit) by horizon, on the development window only."""
    dev = rows.loc[rows["window"].eq("dev")].copy()
    dev["rev"] = dev["logit_oracle"] - dev["logit_shrunk"]
    dev["hh"] = dev["h"].clip(upper=SD_CAP)
    out = []
    for h, g in dev.groupby("hh"):
        rms = lambda x: float(np.sqrt((x ** 2).mean())) if len(x) else np.nan
        out.append({"h": int(h), "n": len(g), "sd_all": rms(g["rev"]), "sd_same_qb": rms(g.loc[~g["qb_changed"], "rev"]),
                    "sd_qb_changed": rms(g.loc[g["qb_changed"], "rev"])})
    return pd.DataFrame(out)


def attach_band(rows: pd.DataFrame, sd: pd.DataFrame) -> pd.DataFrame:
    table = sd.set_index("h")["sd_all"]
    s = rows["h"].clip(upper=SD_CAP).map(table).to_numpy()
    lo, hi = revision_band(rows["logit_shrunk"].to_numpy(), s)
    return rows.assign(lo_band=lo, hi_band=hi)


def _boot(diff_by_game: np.ndarray, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff_by_game), size=(N_BOOT, len(diff_by_game)))
    boot = diff_by_game[idx].mean(axis=1)
    return float(diff_by_game.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _slope(y: np.ndarray, logit: np.ndarray) -> float:
    return float(LogisticRegression(C=1e6, max_iter=1000).fit(logit.reshape(-1, 1), y.astype(int)).coef_[0, 0])


def _summarize(g: pd.DataFrame) -> dict:
    y = g["y"].to_numpy()
    ll = {k: per_game_log_loss(y, g[c].to_numpy()) for k, c in
          (("frozen", "p_frozen"), ("shrunk", "p_shrunk"), ("oracle", "p_oracle"), ("pre", "p_pre"), ("const", "p_const"))}
    row = {"n_forecasts": len(g), "n_games": int(g["game_id"].nunique())}
    for k, v in ll.items():
        row[f"ll_{k}"] = float(v.mean())
    ok = len(g) > 50 and 0 < y.mean() < 1
    row["slope_frozen"] = _slope(y, g["logit_frozen"].to_numpy()) if ok else np.nan
    row["slope_shrunk"] = _slope(y, g["logit_shrunk"].to_numpy()) if ok else np.nan
    for ref in ("frozen", "oracle", "pre", "const"):
        d = pd.Series(ll["shrunk"] - ll[ref]).groupby(g["game_id"].to_numpy()).mean().to_numpy()
        row[f"d_vs_{ref}"], row[f"d_vs_{ref}_lo"], row[f"d_vs_{ref}_hi"] = _boot(d)
    row["cover_band"] = float(((g["p_oracle"] >= g["lo_band"]) & (g["p_oracle"] <= g["hi_band"])).mean())
    row["mean_abs_revision_pts"] = float(100 * (g["p_oracle"] - g["p_shrunk"]).abs().mean())
    return row


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for window in ("dev", "holdout", "all"):
        w = rows if window == "all" else rows.loc[rows["window"].eq(window)]
        for qb in ("all", "same_qb", "qb_changed"):
            q = w if qb == "all" else w.loc[w["qb_changed"].eq(qb == "qb_changed")]
            for name, lo, hi in BUCKETS:
                g = q.loc[q["h"].between(lo, hi)]
                if len(g) >= 100:
                    out.append({"window": window, "qb": qb, "horizon": name, **_summarize(g)})
    return pd.DataFrame(out)


def shrink_result(rows: pd.DataFrame, a: float) -> dict:
    """The registry line: shrunk vs unshrunk on the holdout, every forecast made at least one week ahead."""
    ho = rows.loc[rows["window"].eq("holdout") & rows["h"].ge(1)]
    d = (per_game_log_loss(ho["y"].to_numpy(), ho["p_shrunk"].to_numpy()) - per_game_log_loss(ho["y"].to_numpy(), ho["p_frozen"].to_numpy()))
    est, lo, hi = _boot(pd.Series(d).groupby(ho["game_id"].to_numpy()).mean().to_numpy())
    return {"a": a, "a_frozen_in_code": HORIZON_SHRINK_A, "window": "2017-25", "horizon": ">=1 week", "n_games": int(ho["game_id"].nunique()),
            "delta_log_loss": est, "ci_low": lo, "ci_high": hi}


def finalize(rows: pd.DataFrame) -> pd.DataFrame:
    """From raw forecasts to the tracked summaries: fit the shrinkage, measure revisions, score, and write."""
    a = fit_shrink(rows)
    rows = add_shrunk(rows, a)
    sd = revision_sd(rows)
    rows = attach_band(rows, sd)
    summary = summarize(rows)
    shrink = shrink_result(rows, a)

    out, docs = configured_path("outputs"), project_path("docs")
    out.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out / "opponent_horizon_games.parquet", index=False)              # every forecast (not tracked)
    # small summaries are tracked: the report, the paper and the tests read these
    sd.round(4).to_csv(docs / "opponent_revision_sd.csv", index=False)
    summary.round(5).to_csv(docs / "opponent_horizon.csv", index=False)
    (docs / "opponent_shrink.json").write_text(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in shrink.items()}, indent=2) + "\n")

    show = summary.loc[summary["qb"].eq("all") & summary["window"].isin(["dev", "holdout"])]
    cols = ["window", "horizon", "n_games", "ll_frozen", "ll_shrunk", "ll_oracle", "ll_pre", "ll_const", "slope_frozen", "slope_shrunk", "cover_band"]
    print(show[cols].round(4).to_string(index=False))
    print(f"shrink a = {a:.4f} (frozen in code: {HORIZON_SHRINK_A}); holdout h>=1: {shrink['delta_log_loss']:+.4f} [{shrink['ci_low']:+.4f}, {shrink['ci_high']:+.4f}]")
    return rows


def main() -> None:
    t0 = time.time()
    inp = load_inputs()
    print(f"horizon backtest: seasons {FIRST_TEST_SEASON}-{last_complete_season()}, as-of weeks {AS_OF_WEEKS}")
    finalize(build_rows(inp))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
