"""Do injured-reserve and game-day-inactive burdens improve the game model?

Common population: seasons 2013+ (snap counts needed for importance), test
seasons 2014-2025. Pre-registered rule: keep a variant only if its pooled
paired-bootstrap 95% interval excludes zero AND its 2017-2025 estimate is
also negative.

IR burden is pregame-safe (previous week's status). Game-day inactive burden is
kickoff-time information (what the closing line knows) and cannot be used for
earlier predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import configured_path
from src.features.injuries import attach_burden
from src.features.ratings import RatingParams
from src.features.rosters import attach_roster_burden
from src.models.compare import attach_market, paired_bootstrap_delta, per_game_log_loss
from src.models.ratings_experiment import HOLDOUT_FIRST_SEASON, PRIMARY_FEATURES, Context
from src.models.walk_forward import run_walk_forward


IR = ["diff_lost_ir_off_total", "diff_lost_ir_def_total"]
INA = ["diff_lost_ina_off_total", "diff_lost_ina_def_total"]
REPORT = ["diff_lost_off_total", "diff_lost_def_total"]
GROUPS = ["OL", "WR_TE", "RB", "DL", "LB", "DB"]

VARIANTS = {
    "base (ratings + QB)": [],
    "+ injury-report burden (Fri status)": REPORT,
    "+ IR burden (pregame-safe)": IR,
    "+ game-day inactive burden": INA,
    "+ IR + inactives": IR + INA,
    "+ IR + inactives + report": IR + INA + REPORT,
    "+ IR + inactives, by position group": [f"diff_lost_{k}_{g}" for k in ("ir", "ina") for g in GROUPS],
}


def main() -> None:
    ctx = Context()
    processed = configured_path("processed")
    data = ctx.featurize(RatingParams(), use_qb=True)
    data = data.loc[data["season"] >= 2013]
    data = attach_burden(data, pd.read_parquet(processed / "team_week_injury_burden.parquet"))
    data = attach_roster_burden(data, pd.read_parquet(processed / "team_week_roster_burden.parquet"))

    preds = {n: run_walk_forward(data, feature_set_name=n, feature_columns=[*PRIMARY_FEATURES, *extra])
             for n, extra in VARIANTS.items()}

    ref = preds["base (ratings + QB)"].set_index("game_id")
    market = attach_market(preds["base (ratings + QB)"], ctx.games).set_index("game_id")["p_market"]

    def delta(pred, mask_fn):
        g = pred.set_index("game_id")
        g = g.loc[mask_fn(g)]
        y = g["home_win"].astype(int).to_numpy()
        return paired_bootstrap_delta(y, g["p_home_win"].to_numpy(), ref.loc[g.index, "p_home_win"].to_numpy())

    print(f"{'variant':<40}| pooled 2014-25 [95% CI]          | holdout 2017-25 | late season (wk 13+, pooled) | rule")
    for name, pred in preds.items():
        if name.startswith("base"):
            continue
        dp, lo, hi = delta(pred, lambda g: g["season"] >= 2014)
        dh, _, _ = delta(pred, lambda g: g["season"] >= HOLDOUT_FIRST_SEASON)
        dl, _, _ = delta(pred, lambda g: (g["week"] >= 13) & ~g["is_postseason"])
        ok = "KEEP" if (hi < 0 and dh < 0) else ""
        print(f"{name:<40}| {dp:+.4f} [{lo:+.4f},{hi:+.4f}]  | {dh:+.4f}         | {dl:+.4f}                      | {ok}")

    # how much of the market gap does the best kickoff-time variant close?
    best = preds["+ IR + inactives + report"].set_index("game_id")
    for label, p in (("base", ref), ("kickoff-time (IR+inactives+report)", best)):
        g = p.loc[p.index.intersection(market.index)]
        g = g.loc[g["season"] >= HOLDOUT_FIRST_SEASON]
        y = g["home_win"].astype(int).to_numpy()
        d, lo, hi = paired_bootstrap_delta(y, g["p_home_win"].to_numpy(), market.loc[g.index].to_numpy())
        print(f"gap to closing market 2017-25, {label:<36}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  (n={len(g)})")


if __name__ == "__main__":
    main()
