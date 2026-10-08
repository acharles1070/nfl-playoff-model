"""Two-stage injury adjustment (efficient: uses continuous EPA residuals).

Stage 1 (per test season S, using seasons < S only): ridge-regress each
team-game's EPA/play residual (actual minus what the ratings predicted) on the
team's own and the opponent's lost starter-equivalents:
    offense residual ~ own lost OL / WR_TE / RB + opponent lost DL / LB / DB
    defense residual ~ own lost DL / LB / DB   + opponent lost OL / WR_TE / RB
Stage 2: shift every game's ratings by the fitted injury effects and feed the
adjusted differences to the same logistic game model.

Pre-registered rule: adopt only if the holdout improvement's paired-bootstrap
95% interval excludes zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.config import configured_path
from src.features.injuries import attach_burden
from src.features.ratings import RatingParams, build_team_ratings
from src.models.backtest import build_logistic_model
from src.models.compare import paired_bootstrap_delta, per_game_log_loss
from src.models.ratings_experiment import HOLDOUT_FIRST_SEASON, PRIMARY_FEATURES, TUNE_SEASONS, Context


OFF_OWN = ["OL", "WR_TE", "RB"]
DEF_OWN = ["DL", "LB", "DB"]
RIDGE_ALPHA = 30.0


class _Zero:
    coef_ = np.zeros(6)

    def predict(self, X):
        return np.zeros(len(X))


def team_game_frame(ctx: Context, burden: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game: residual targets + own/opponent burdens."""
    ratings = build_team_ratings(ctx.team_game, ctx.played, RatingParams(), ctx.qb_game)

    tg = ctx.team_game[["game_id", "team", "off_epa_per_play", "def_allowed_epa_per_play"]]
    x = ratings.merge(tg, on=["game_id", "team"], how="inner")

    opp = ratings[["game_id", "team", "pregame_off", "pregame_qb", "pregame_def"]].rename(
        columns={"team": "opponent", "pregame_off": "opp_off", "pregame_qb": "opp_qb", "pregame_def": "opp_def"}
    )
    x = x.merge(opp, on=["game_id", "opponent"], how="left")

    x["off_resid"] = x["off_epa_per_play"] - (x["league_mu"] + x["pregame_off"] + x["pregame_qb"] + x["opp_def"])
    x["def_resid"] = x["def_allowed_epa_per_play"] - (x["league_mu"] + x["opp_off"] + x["opp_qb"] + x["pregame_def"])

    lost = [c for c in burden.columns if c.startswith("lost_")]
    own = burden.rename(columns={c: f"own_{c}" for c in lost})
    oth = burden.rename(columns={"team": "opponent", **{c: f"opp_{c}" for c in lost}})
    x = x.merge(own, on=["season", "week", "team"], how="left").merge(oth, on=["season", "week", "opponent"], how="left")
    return x


def stage1_design(x: pd.DataFrame, side: str) -> tuple[list[str], list[str]]:
    own_groups, opp_groups = (OFF_OWN, DEF_OWN) if side == "off" else (DEF_OWN, OFF_OWN)
    return [f"own_lost_{g}" for g in own_groups], [f"opp_lost_{g}" for g in opp_groups]


def fit_stage1(x: pd.DataFrame, before: int) -> dict[str, tuple[list[str], Ridge]]:
    train = x.loc[x["season"].lt(before)].dropna(subset=["own_lost_OL", "opp_lost_OL", "off_resid", "def_resid"])
    models = {}
    for side, target in (("off", "off_resid"), ("def", "def_resid")):
        own, opp = stage1_design(x, side)
        cols = own + opp
        if len(train) < 200:                     # no injury history yet: no adjustment
            models[side] = (cols, _Zero())
        else:
            models[side] = (cols, Ridge(alpha=RIDGE_ALPHA).fit(train[cols], train[target]))
    return models


def predicted_shift(x: pd.DataFrame, models, side: str) -> np.ndarray:
    cols, model = models[side]
    return model.predict(x[cols].fillna(0.0))


def adjusted_matchups(data: pd.DataFrame, x: pd.DataFrame, models) -> pd.DataFrame:
    """Add injury-adjusted rating differences to the matchup frame."""
    x = x.copy()
    x["d_off"] = predicted_shift(x, models, "off")
    x["d_def"] = predicted_shift(x, models, "def")
    shifts = x[["game_id", "team", "d_off", "d_def"]]

    out = data
    for side in ("home", "away"):
        out = out.merge(
            shifts.rename(columns={"team": f"{side}_team", "d_off": f"{side}_d_off", "d_def": f"{side}_d_def"}),
            on=["game_id", f"{side}_team"], how="left",
        )
    for c in ("home_d_off", "home_d_def", "away_d_off", "away_d_def"):
        out[c] = out[c].fillna(0.0)

    # positive d_def = team ALLOWS more (worse defense); diff_rating_def = away_def - home_def
    out["diff_rating_off_total_adj"] = out["diff_rating_off_total"] + (out["home_d_off"] - out["away_d_off"])
    out["diff_rating_def_adj"] = out["diff_rating_def"] + (out["away_d_def"] - out["home_d_def"])
    return out


def main() -> None:
    ctx = Context()
    burden = pd.read_parquet(configured_path("processed") / "team_week_injury_burden.parquet")
    data = attach_burden(ctx.featurize(RatingParams(), use_qb=True), burden)
    x = team_game_frame(ctx, burden)

    base_features = list(PRIMARY_FEATURES)
    adj_features = ["diff_rating_off_total_adj", "diff_rating_def_adj"]

    rows = []
    seasons = sorted(data["season"].unique())

    for test in seasons:
        train_seasons = [s for s in seasons if s < test]
        if not train_seasons:
            continue

        models = fit_stage1(x, before=test)
        adj = adjusted_matchups(data, x, models)

        tr, te = adj.loc[adj["season"].isin(train_seasons)], adj.loc[adj["season"].eq(test)]
        y = tr["home_win"].astype(int).to_numpy()

        for name, feats in (("current", base_features), ("injury-adjusted (2-stage)", adj_features)):
            m = build_logistic_model(feats).fit(tr, y)
            p = m.predict_proba(te)[:, 1]
            rows.append(pd.DataFrame({
                "game_id": te["game_id"].to_numpy(), "season": test, "variant": name,
                "home_win": te["home_win"].astype(int).to_numpy(), "p": p,
                "is_postseason": te["is_postseason"].to_numpy(),
            }))

    pred = pd.concat(rows)
    cur = pred.loc[pred["variant"].eq("current")].set_index("game_id")
    inj = pred.loc[pred["variant"].ne("current")].set_index("game_id")

    def ll(frame, seasons_):
        f = frame.loc[frame["season"].isin(list(seasons_))]
        return per_game_log_loss(f["home_win"].to_numpy(), f["p"].to_numpy()).mean()

    print(f"tune window 2011-2016:  current {ll(cur, TUNE_SEASONS):.4f}  vs  injury-adjusted {ll(inj, TUNE_SEASONS):.4f}")

    for label, mask_fn in (("ALL holdout 2017-2025", lambda d: d), ("postseason only", lambda d: d.loc[d["is_postseason"]])):
        h_cur = mask_fn(cur.loc[cur["season"] >= HOLDOUT_FIRST_SEASON])
        h_inj = inj.loc[h_cur.index]
        d, lo, hi = paired_bootstrap_delta(h_cur["home_win"].to_numpy(), h_inj["p"].to_numpy(), h_cur["p"].to_numpy())
        verdict = "  <-- ADOPT (interval excludes 0)" if hi < 0 else ""
        print(f"{label:<24} n={len(h_cur):<5} current {per_game_log_loss(h_cur['home_win'].to_numpy(), h_cur['p'].to_numpy()).mean():.4f}  "
              f"injury-adjusted {per_game_log_loss(h_inj['home_win'].to_numpy(), h_inj['p'].to_numpy()).mean():.4f}  "
              f"delta {d:+.4f} [{lo:+.4f}, {hi:+.4f}]{verdict}")

    # the stage-1 effects fit on everything through 2025
    models = fit_stage1(x, before=2026)
    print("\nStage-1 effects, EPA/play per lost starter-equivalent (fit 2013-2025):")
    for side in ("off", "def"):
        cols, model = models[side]
        print("  " + ("offense" if side == "off" else "defense") + " residual:",
              {c.replace("own_lost_", "own ").replace("opp_lost_", "opp "): round(float(v), 4) for c, v in zip(cols, model.coef_)})


if __name__ == "__main__":
    main()
