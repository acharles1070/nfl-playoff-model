"""Do measured injury burdens improve the game model?

Pre-registered rule: adopt a variant only if its 2017-2025 holdout improvement
over the current model has a paired-bootstrap 95% interval that excludes zero.
"""

from __future__ import annotations

import pandas as pd

from src.config import configured_path
from src.features.injuries import BURDEN_GROUPS, attach_burden
from src.features.ratings import RatingParams
from src.models.compare import paired_bootstrap_delta, per_game_log_loss
from src.models.ratings_experiment import (
    HOLDOUT_FIRST_SEASON, PRIMARY_FEATURES, TUNE_SEASONS, Context,
)
from src.models.walk_forward import run_walk_forward


TOTALS = ["diff_lost_off_total", "diff_lost_def_total"]
GROUPED = [f"diff_lost_{g}" for g in BURDEN_GROUPS]

VARIANTS = {
    "current (ratings + QB)": [],
    "+ burden totals (off, def)": TOTALS,
    "+ burden by position group": GROUPED,
    "+ OL only": ["diff_lost_OL"],
    "+ DB only": ["diff_lost_DB"],
}


def ll(pred, seasons):
    p = pred.loc[pred["season"].isin(list(seasons))]
    return float(per_game_log_loss(p["home_win"].astype(int).to_numpy(), p["p_home_win"].to_numpy()).mean())


def main() -> None:
    ctx = Context()
    burden = pd.read_parquet(configured_path("processed") / "team_week_injury_burden.parquet")
    data = attach_burden(ctx.featurize(RatingParams(), use_qb=True), burden)

    have = data["home_lost_OL"].notna() & data["away_lost_OL"].notna()
    print(f"games with injury reports on both sides: {have.mean():.1%}  (2013+ only by design)\n")

    preds = {
        name: run_walk_forward(data, feature_set_name=name, feature_columns=[*PRIMARY_FEATURES, *extra])
        for name, extra in VARIANTS.items()
    }

    ref = preds["current (ratings + QB)"].set_index("game_id")

    print(f"{'variant':<30}| tune LL | holdout LL | holdout vs current [95% CI]      | postseason vs current")
    for name, pred in preds.items():
        h = pred.loc[pred["season"] >= HOLDOUT_FIRST_SEASON].set_index("game_id")
        y = h["home_win"].astype(int).to_numpy()
        d, lo, hi = paired_bootstrap_delta(y, h["p_home_win"].to_numpy(), ref.loc[h.index, "p_home_win"].to_numpy())
        hp = h.loc[h["is_postseason"]]
        dp, _, _ = paired_bootstrap_delta(
            hp["home_win"].astype(int).to_numpy(), hp["p_home_win"].to_numpy(), ref.loc[hp.index, "p_home_win"].to_numpy()
        )
        flag = "  <-- adopt" if hi < 0 and name != "current (ratings + QB)" else ""
        print(f"{name:<30}| {ll(pred, TUNE_SEASONS):.4f}  | {ll(pred, range(HOLDOUT_FIRST_SEASON, 2026)):.4f}     | {d:+.4f} [{lo:+.4f}, {hi:+.4f}]{flag:<11}| {dp:+.4f} (n={len(hp)})")

    from src.models.backtest import build_logistic_model
    cols = [*PRIMARY_FEATURES, *GROUPED]
    train = data.loc[data["season"].between(2013, 2025)]
    m = build_logistic_model(cols).fit(train, train["home_win"].astype(int).to_numpy())
    scaler = m.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scaler"]
    coef = m.named_steps["model"].coef_[0] / scaler.scale_
    print("\nlogit effect of one extra lost starter-equivalent (home minus away), fit 2013-2025:")
    for c, v in zip(cols, coef):
        print(f"   {c:<24} {v:+.3f}")


if __name__ == "__main__":
    main()
