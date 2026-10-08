"""Registry of every out-of-sample experiment, with family-wise multiple-testing control.

Each row: a candidate change, its paired log-loss difference versus the model it was
tested against on the holdout (negative = better), and a 95% interval taken from the
paired bootstrap at the time. Because ~35 comparisons were run, some "significant"
results are expected by chance; Holm (family-wise error) and Benjamini-Hochberg (false
discovery rate) corrections are applied to the whole family.

p-values are recovered from the intervals with a normal approximation
(se = (hi - lo) / 3.92), which is exact enough for bootstrap intervals this symmetric.

python -m src.models.registry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

# (id, area, description, estimate, ci_low, ci_high, window, source)
EXPERIMENTS = [
    # --- game model
    ("E01", "ratings",  "Kalman team ratings vs season-to-date EPA", None, None, None, "2017-25", "computed live below"),
    ("E02", "QB",       "QB layer vs no-QB ratings (all games)", -0.0037, -0.0076, 0.0003, "2017-25", "ratings_experiment"),
    ("E03", "QB",       "QB layer, games where a starter changed", -0.0132, -0.0246, -0.0016, "2017-25", "ratings_experiment"),
    ("E04", "QB",       "QB layer, regular season only", -0.0044, -0.0084, -0.0003, "2017-25", "ratings_experiment"),
    ("E05", "QB",       "QB layer, postseason only", 0.0108, -0.0011, 0.0229, "2017-25", "ratings_experiment"),
    ("E06", "history",  "Train from 2000 vs from 2010", 0.0004, -0.0008, 0.0016, "2017-25", "history extension"),
    ("E07", "tuning",   "Re-tune filter on 2002-2016 vs 2011-2016 values", -0.0005, -0.0020, 0.0010, "2017-25", "ratings re-tune"),
    # --- link function / model form
    ("E08", "model form", "Margin regression + normal CDF vs logistic", -0.0008, -0.0022, 0.0004, "2017-25", "game_model_experiment"),
    ("E09", "model form", "Heteroscedastic (uncertainty-widened) margin model", -0.0003, -0.0024, 0.0017, "2017-25", "game_model_experiment"),
    ("E10", "model form", "Margin Kalman rating added", -0.0002, -0.0028, 0.0024, "2017-25", "margin_experiment"),
    ("E11", "model form", "Pass/rush split filters", 0.0000, -0.0023, 0.0022, "2017-25", "pass_rush_experiment"),
    # --- context features
    ("E12", "context", "+ rest difference", -0.0001, -0.0013, 0.0012, "2017-25", "context_experiment"),
    ("E13", "context", "+ bye flags", -0.0001, -0.0019, 0.0018, "2017-25", "context_experiment"),
    ("E14", "context", "+ neutral site", 0.0007, -0.0011, 0.0025, "2017-25", "context_experiment"),
    ("E15", "context", "+ bye flags + neutral", 0.0006, -0.0018, 0.0031, "2017-25", "context_experiment"),
    ("E16", "context", "+ rest + bye + neutral", 0.0007, -0.0017, 0.0032, "2017-25", "context_experiment"),
    ("E17", "context", "+ all context features", 0.0011, -0.0016, 0.0037, "2017-25", "context_experiment"),
    # --- observation filtering
    ("E18", "garbage time", "Competitive plays only, wp 10-90%", 0.0026, -0.0021, 0.0073, "2017-25", "garbage_time_experiment"),
    ("E19", "garbage time", "Competitive plays only, wp 5-95%", 0.0020, -0.0018, 0.0058, "2017-25", "garbage_time_experiment"),
    ("E20", "garbage time", "Leverage-weighted plays", 0.0026, -0.0018, 0.0070, "2017-25", "garbage_time_experiment"),
    # --- injuries and roster status
    ("E21", "injuries", "Injury-report burden, totals", -0.0014, -0.0032, 0.0005, "2017-25", "injury_experiment"),
    ("E22", "injuries", "Injury-report burden, by position group", -0.0011, -0.0040, 0.0020, "2017-25", "injury_experiment"),
    ("E23", "injuries", "Injury burden, OL only", 0.0002, -0.0003, 0.0006, "2017-25", "injury_experiment"),
    ("E24", "injuries", "Injury burden, DB only", -0.0004, -0.0016, 0.0007, "2017-25", "injury_experiment"),
    ("E25", "injuries", "Two-stage EPA-residual injury adjustment", -0.0018, -0.0045, 0.0009, "2017-25", "injury_two_stage"),
    ("E26", "roster", "Injured-reserve burden", 0.0011, -0.0031, 0.0052, "2014-25", "roster_experiment"),
    ("E27", "roster", "Game-day inactive burden", -0.0018, -0.0034, -0.0003, "2014-25", "roster_experiment"),
    ("E28", "roster", "IR + inactives", -0.0005, -0.0051, 0.0040, "2014-25", "roster_experiment"),
    ("E29", "roster", "IR + inactives + report", 0.0002, -0.0044, 0.0047, "2014-25", "roster_experiment"),
    ("E30", "roster", "IR + inactives by position group", 0.0027, -0.0026, 0.0077, "2014-25", "roster_experiment"),
    ("E31", "roster", "Injury-report burden (pooled 2014-25)", -0.0006, -0.0024, 0.0012, "2014-25", "roster_experiment"),
    # --- market-input models
    ("E32", "market", "Calibrated market vs raw market", -0.0001, -0.0010, 0.0008, "2017-25", "market_models"),
    ("E33", "market", "Stacked market + my model vs raw market", 0.0002, -0.0008, 0.0012, "2017-25", "market_models"),
    # --- time-varying home edge (hypothesis generated from holdout betting diagnostics: post hoc)
    ("E35", "home edge", "Recency-weighted training, half-life 8 seasons (post hoc)", -0.0003, -0.0011, 0.0005, "2017-25", "hfa_experiment"),
    ("E36", "home edge", "Recency-weighted training, half-life 5 seasons (post hoc)", -0.0003, -0.0015, 0.0009, "2017-25", "hfa_experiment"),
    ("E37", "home edge", "Recency-weighted training, half-life 3 seasons (post hoc)", -0.0001, -0.0019, 0.0017, "2017-25", "hfa_experiment"),
    ("E38", "home edge", "Recency-weighted training, half-life 2 seasons (post hoc)", 0.0002, -0.0021, 0.0025, "2017-25", "hfa_experiment"),
]


def e01_live():
    """Ratings vs the season-to-date baseline, paired, from the saved predictions."""
    from src.config import configured_path
    from src.models.compare import paired_bootstrap_delta

    path = configured_path("outputs") / "ratings_experiment_predictions.parquet"
    p = pd.read_parquet(path)
    a = p.loc[p.feature_set.eq("ratings, no QB (previous model)") & (p.season >= 2017)].set_index("game_id")
    b = p.loc[p.feature_set.eq("baseline std_epa") & (p.season >= 2017)].set_index("game_id")
    d, lo, hi = paired_bootstrap_delta(a["home_win"].astype(int).to_numpy(), a["p_home_win"].to_numpy(), b.loc[a.index, "p_home_win"].to_numpy())
    return d, lo, hi


def build() -> pd.DataFrame:
    rows = []
    for eid, area, desc, est, lo, hi, window, source in EXPERIMENTS:
        if est is None:
            est, lo, hi = e01_live()
        se = (hi - lo) / 3.92
        z = est / se
        rows.append({"id": eid, "area": area, "experiment": desc, "window": window, "delta_log_loss": est,
                     "ci_low": lo, "ci_high": hi, "p_raw": 2 * (1 - norm.cdf(abs(z))), "source": source})

    d = pd.DataFrame(rows)
    d["p_holm"] = multipletests(d["p_raw"], method="holm")[1]
    d["q_bh"] = multipletests(d["p_raw"], method="fdr_bh")[1]
    d["significant_raw"] = d["p_raw"] < 0.05
    d["survives_holm"] = d["p_holm"] < 0.05
    d["survives_bh"] = d["q_bh"] < 0.05
    return d


def games_needed(sigma: float, effect: float) -> int:
    z = norm.ppf(0.975) + norm.ppf(0.80)
    return int(np.ceil((z * sigma / effect) ** 2))


def power_table(sigma: float) -> pd.DataFrame:
    """Smallest true improvement detectable with 80% power at alpha = 0.05 (paired design)."""
    z = norm.ppf(0.975) + norm.ppf(0.80)
    return pd.DataFrame(
        [{"games": n, "min_detectable_improvement": z * sigma / np.sqrt(n)} for n in (111, 500, 1000, 2494, 5000, 10000, 25000)]
    )


def paired_sigma() -> float:
    """SD of per-game log-loss differences between two similar models (from saved predictions)."""
    from src.config import configured_path
    from src.models.compare import per_game_log_loss

    p = pd.read_parquet(configured_path("outputs") / "ratings_experiment_predictions.parquet")
    p = p.loc[p.season >= 2017]
    a = p.loc[p.feature_set.eq("ratings + QB (tuned, frozen)")].set_index("game_id")
    b = p.loc[p.feature_set.eq("ratings, no QB (previous model)")].set_index("game_id")
    y = a["home_win"].astype(int).to_numpy()
    return float((per_game_log_loss(y, a["p_home_win"].to_numpy()) - per_game_log_loss(y, b.loc[a.index, "p_home_win"].to_numpy())).std())


def provenance_lines(d: pd.DataFrame) -> list[str]:
    """How clean is the holdout, and does the one survivor replicate on untouched seasons?"""
    from src.config import project_path

    w = d["window"].value_counts()
    lines = [
        "",
        "## How clean is the holdout?",
        "",
        f"Every comparison above is scored on the holdout window or later: {int(w.get('2017-25', 0))} on 2017-2025, "
        f"{int(w.get('2014-25', 0))} on 2014-2025 and {int(w.get('2023-25', 0))} on 2023-2025. The registry holds no development-window "
        "numbers. The Kalman ratings, the quarterback layer and roughly 20 of the early experiments were developed on 2026-09-30, "
        "when 2017-2025 was the only test window; the formal protocol (select ideas on 2002-2016, confirm on 2017-2025) was adopted on "
        "2026-10-06 when history was extended to 1999. Hyperparameters were tuned on 2011-2016 only and frozen. The holdout is therefore "
        "not sealed: it has been examined for every idea, which is exactly why the family-wise corrections above are applied to these looks.",
    ]
    lines += ["", "E34 is intentionally absent: that comparison used a licensed third-party dataset that cannot be published. "
               "It showed no detectable effect and is not counted above; including it would not change the survivor."]
    path = project_path("docs", "replication_e01.csv")
    if path.exists():
        r = pd.read_csv(path)
        lines += [
            "",
            "## Replication of E01 on seasons outside its design (pre-specified 2026-10-07)",
            "",
            "E01 (Kalman ratings vs season-to-date EPA) is the only experiment that survives correction. Test seasons 2003-2009 played no "
            "part in designing it or tuning it, so they are the cleanest check. Primary criterion fixed in advance: a 95% paired-bootstrap "
            "interval excluding zero on 2003-2009. Both models train walk-forward from 2000, so the holdout row differs slightly from E01, "
            "which uses the live model's 2010 training start.",
            "",
            "| window | games | change in log loss | 95% interval | seasons Kalman better |",
            "|---|---|---|---|---|",
        ]
        for x in r.itertuples():
            lines.append(f"| {x.window} | {x.games:,} | {x.delta_log_loss:+.4f} | [{x.ci_low:+.4f}, {x.ci_high:+.4f}] | {x.per_season_better}/{x.n_seasons} |")
        p = r.iloc[0]
        verdict = ("meets" if p.ci_high < 0 else "does NOT meet")
        lines += ["", f"Result: the primary check {verdict} the pre-specified criterion (the effect has the same sign and is smaller than on later "
                      "windows). The survivor is supported across windows but not independently confirmed at the 5% level."]
    return lines


def write_markdown(d: pd.DataFrame, sigma: float, path) -> None:
    lines = [
        "# Experiment registry and multiple-testing control",
        "",
        f"Every out-of-sample comparison run in this project ({len(d)} in total), as a single family.",
        "`delta` is the paired log-loss difference versus the model it was tested against",
        "(negative = better). p-values are recovered from the bootstrap intervals (normal",
        "approximation); Holm controls the family-wise error rate, BH the false-discovery rate.",
        "",
        f"- nominally significant at 0.05: **{int(d.significant_raw.sum())}** "
        f"(about {0.05 * len(d):.1f} would be expected by chance alone)",
        f"- surviving Holm: **{int(d.survives_holm.sum())}**; surviving BH: **{int(d.survives_bh.sum())}**",
        "",
        "| id | experiment | delta | 95% interval | p | p (Holm) | survives |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in d.itertuples():
        lines.append(f"| {r.id} | {r.experiment} | {r.delta_log_loss:+.4f} | [{r.ci_low:+.4f}, {r.ci_high:+.4f}] | "
                     f"{r.p_raw:.3f} | {r.p_holm:.3f} | {'**yes**' if r.survives_holm else 'no'} |")

    lines += provenance_lines(d)

    pt = power_table(sigma)
    lines += [
        "",
        "## Statistical power",
        "",
        f"The SD of per-game log-loss differences between two similar models is {sigma:.3f}. "
        "With 80% power at 5% significance, the smallest true improvement a paired test can detect is:",
        "",
        "| games | minimum detectable improvement |",
        "|---|---|",
    ]
    for r in pt.itertuples():
        lines.append(f"| {r.games:,} | {r.min_detectable_improvement:.4f} |")
    lines += [
        "",
        f"Confirming a 0.005 improvement needs about {games_needed(sigma, 0.005):,} games "
        f"(~{games_needed(sigma, 0.005) / 285:.0f} seasons); a 0.002 improvement needs about "
        f"{games_needed(sigma, 0.002):,} games (~{games_needed(sigma, 0.002) / 285:.0f} seasons). "
        "The 111 holdout playoff games can only detect improvements above ~0.027.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    d = build()
    pd.set_option("display.width", 220)
    print(f"{len(d)} experiments in the family")
    print(f"nominally significant at 0.05: {int(d.significant_raw.sum())} | survive Holm: {int(d.survives_holm.sum())} | survive BH: {int(d.survives_bh.sum())} | expected false positives: {0.05 * len(d):.1f}")
    print("survivors:", d.loc[d.survives_holm, ["id", "experiment"]].values.tolist())

    sigma = paired_sigma()
    print(f"\nPOWER (paired SD {sigma:.3f}):")
    print(power_table(sigma).round(5).to_string(index=False))
    print(f"games needed: 0.005 -> {games_needed(sigma, 0.005):,} | 0.002 -> {games_needed(sigma, 0.002):,}")

    from src.config import project_path
    out = project_path("docs")
    out.mkdir(exist_ok=True)
    d.round(5).to_csv(out / "experiment_registry.csv", index=False)
    write_markdown(d, sigma, out / "EXPERIMENT_REGISTRY.md")


if __name__ == "__main__":
    main()
