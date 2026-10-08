"""Replicate the registry's one survivor (E01) on seasons that played no part in its design.

E01 = Kalman team ratings (no QB layer, default params) vs season-to-date EPA, paired log loss.
It was found on 2017-2025, when that was the only test window (2026-09-30), so the holdout is not
a clean confirmation of the architecture. History starts in 1999 only since 2026-10-06, so test
seasons 2003-2009 were never looked at while the filter was designed, and its hyperparameters were
tuned on 2011-2016.

PRE-SPECIFIED before the numbers were seen (2026-10-07):
  primary   : test seasons 2003-2009 (7 seasons), paired bootstrap over games, same comparison as E01
  secondary : dev window 2002-2016 and the holdout 2017-2025 (descriptive; the holdout must reproduce E01)
Both models are walk-forward from 2000; early test seasons train on few seasons, equally for both.

    python -m src.models.replication
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import configured_path, project_path
from src.features.ratings import RatingParams
from src.models.compare import paired_bootstrap_delta
from src.models.ratings_experiment import NO_QB_FEATURES, Context
from src.models.walk_forward import run_walk_forward

WINDOWS = {"primary: 2003-2009 (untouched by design and tuning)": (2003, 2009),
           "secondary: dev window 2002-2016": (2002, 2016),
           "secondary: holdout 2017-2025 (= E01)": (2017, 2025)}
BASELINE = ["diff_off_epa_per_play_std", "diff_def_allowed_epa_per_play_std"]


def run() -> pd.DataFrame:
    ctx = Context()
    kalman = run_walk_forward(ctx.featurize(RatingParams(), use_qb=False), feature_set_name="kalman", feature_columns=NO_QB_FEATURES)
    base = run_walk_forward(ctx.featurize(RatingParams(), use_qb=True), feature_set_name="std_epa", feature_columns=BASELINE)
    k, b = kalman.set_index("game_id"), base.set_index("game_id")
    common = k.index.intersection(b.index)
    k, b = k.loc[common], b.loc[common]

    rows = []
    for label, (lo_s, hi_s) in WINDOWS.items():
        m = k["season"].between(lo_s, hi_s).to_numpy()
        y = k["home_win"].astype(int).to_numpy()[m]
        d, lo, hi = paired_bootstrap_delta(y, k["p_home_win"].to_numpy()[m], b["p_home_win"].to_numpy()[m])
        rows.append({"window": label, "seasons": f"{lo_s}-{hi_s}", "games": int(m.sum()), "delta_log_loss": d, "ci_low": lo, "ci_high": hi,
                     "per_season_better": int(sum(
                         (np.mean(-(y_s * np.log(np.clip(pk, 1e-9, 1)) - (1 - y_s) * np.log(np.clip(1 - pk, 1e-9, 1)))) <
                          np.mean(-(y_s * np.log(np.clip(pb, 1e-9, 1)) - (1 - y_s) * np.log(np.clip(1 - pb, 1e-9, 1)))))
                         for s in range(lo_s, hi_s + 1)
                         for y_s, pk, pb in [(k.loc[k.season.eq(s), "home_win"].astype(int).to_numpy(), k.loc[k.season.eq(s), "p_home_win"].to_numpy(),
                                              b.loc[k.season.eq(s), "p_home_win"].to_numpy())])), "n_seasons": hi_s - lo_s + 1})
    return pd.DataFrame(rows)


def main() -> None:
    out = run()
    out.to_csv(configured_path("outputs") / "replication_e01.csv", index=False)
    out.round(5).to_csv(project_path("docs") / "replication_e01.csv", index=False)       # tracked: the paper and site read this
    print("E01 replication: change in log loss, Kalman ratings vs season-to-date EPA (negative = Kalman better)\n")
    for r in out.itertuples():
        print(f"  {r.window:<58} {r.games:>5} games  {r.delta_log_loss:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]  better in {r.per_season_better}/{r.n_seasons} seasons")


if __name__ == "__main__":
    main()
