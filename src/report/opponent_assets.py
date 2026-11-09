"""Figures, tables and paper variables for paper/opponent_forecasting.qmd.

Reads only tracked summaries (docs/opponent_*.csv, docs/opponent_shrink.json and the worked-example report in
docs/examples/), so the paper rebuilds without the raw data. Called from src.report.paper_assets.

    python -m src.report.paper_assets
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import project_path
from src.opponents.report import ordinal
from src.report.paper_assets import AMBER, BLUE, FIG, GRAY, GREEN, RED, TAB, VIOLET, plt, save, typo

DOCS = project_path("docs")
EXAMPLE = "WAS_2026_w10"
BUCKETS = ["0", "1", "2", "3-4", "5-8", "9+"]
BUCKET_LABELS = ["0", "1", "2", "3–4", "5–8", "9+"]          # tables
FIG_LABELS = ["0", "1", "2", "3-4", "5-8", "9+"]             # figures: the Computer Modern font has no en dash
ROUND_NAMES = {"WC": "Wild Card", "DIV": "Divisional", "CON": "Conference", "SB": "Super Bowl"}


def _horizon() -> pd.DataFrame:
    return pd.read_csv(DOCS / "opponent_horizon.csv")


def _pick(d: pd.DataFrame, window: str, qb: str = "all") -> pd.DataFrame:
    g = d.loc[d["window"].eq(window) & d["qb"].eq(qb)].set_index("horizon")
    return g.reindex([b for b in BUCKETS if b in g.index])


def example() -> dict:
    return json.loads((DOCS / "examples" / f"{EXAMPLE}.json").read_text())


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_opp_horizon() -> None:
    d = _horizon()
    g = _pick(d, "holdout")
    x = np.arange(len(g))
    labels = [FIG_LABELS[BUCKETS.index(b)] for b in g.index]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    for col, lab, c, ls in (("ll_const", "home field only", GRAY, ":"), ("ll_pre", "preseason ratings", AMBER, "--"),
                            ("ll_frozen", "ratings frozen, unadjusted", RED, "-"), ("ll_shrunk", "ratings frozen, shrunk", GREEN, "-"),
                            ("ll_oracle", "forecast made the week of the game", BLUE, "-")):
        ax.plot(x, g[col], color=c, ls=ls, lw=2 if col in ("ll_shrunk", "ll_oracle") else 1.5, marker="o", ms=3.5, label=lab)
    ax.set_xticks(x, labels)
    ax.set_xlabel("weeks between the forecast and the game")
    ax.set_ylabel("log loss (lower is better)")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    ax.set_title("Skill by horizon, 2017-2025", fontsize=9)
    ax = axes[1]
    ax.axhline(1.0, color="black", lw=1, ls=":")
    ax.plot(x, g["slope_frozen"], color=RED, lw=2, marker="o", ms=3.5, label="unadjusted")
    ax.plot(x, g["slope_shrunk"], color=GREEN, lw=2, marker="o", ms=3.5, label="shrunk")
    ax.set_xticks(x, labels)
    ax.set_xlabel("weeks between the forecast and the game")
    ax.set_ylabel("calibration slope (1 = calibrated)")
    ax.set_ylim(0.5, 1.4)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("Overconfidence grows with distance", fontsize=9)
    save(fig, "opp_horizon")


def fig_opp_qb() -> None:
    d = _horizon()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for qb, lab, c, off in (("same_qb", "both starters as assumed", GREEN, -0.08), ("qb_changed", "a starter changed", RED, 0.08)):
        g = _pick(d, "holdout", qb)
        x = np.array([BUCKETS.index(b) for b in g.index]) + off
        ax.errorbar(x, g["d_vs_oracle"], yerr=[g["d_vs_oracle"] - g["d_vs_oracle_lo"], g["d_vs_oracle_hi"] - g["d_vs_oracle"]],
                    color=c, lw=1.6, marker="o", ms=4, capsize=2, label=lab)
    ax.axhline(0, color="black", lw=1, ls=":")
    ax.set_xticks(range(len(BUCKETS)), FIG_LABELS)
    ax.set_xlabel("weeks between the forecast and the game")
    ax.set_ylabel("log loss cost of forecasting ahead")
    ax.legend(fontsize=7.5, loc="upper left")
    save(fig, "opp_qb")


def fig_opp_band() -> None:
    sd = pd.read_csv(DOCS / "opponent_revision_sd.csv")
    d = _horizon()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    ax.plot(sd["h"], sd["sd_qb_changed"], color=RED, lw=2, marker="o", ms=3, label="a starter changed")
    ax.plot(sd["h"], sd["sd_all"], color=BLUE, lw=2, marker="o", ms=3, label="all games")
    ax.plot(sd["h"], sd["sd_same_qb"], color=GREEN, lw=2, marker="o", ms=3, label="starters as assumed")
    ax.set_xlabel("weeks between the forecast and the game")
    ax.set_ylabel("typical revision (logit SD)")
    ax.set_xticks([0, 3, 6, 9, 12], ["0", "3", "6", "9", "12+"])
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("How far forecasts move", fontsize=9)
    ax = axes[1]
    ax.axhline(0.80, color="black", lw=1, ls=":")
    x = np.arange(len(BUCKETS))
    for window, qb, lab, c, ls in (("holdout", "all", "holdout, all games", BLUE, "-"), ("holdout", "same_qb", "holdout, starters as assumed", GREEN, "-"),
                                   ("holdout", "qb_changed", "holdout, a starter changed", RED, "-"), ("dev", "all", "development window (fitted)", GRAY, "--")):
        g = _pick(d, window, qb)
        ax.plot([BUCKETS.index(b) for b in g.index], g["cover_band"], color=c, ls=ls, lw=2, marker="o", ms=3, label=lab)
    ax.set_xticks(x, FIG_LABELS)
    ax.set_xlabel("weeks between the forecast and the game")
    ax.set_ylabel("share of later forecasts inside the 80% band")
    ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("Do the bands cover?", fontsize=9)
    save(fig, "opp_band")


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def _pct(x: float, digits: int = 0) -> str:
    return f"{100 * x:.{digits}f}%"


def write_tables() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    g = _pick(_horizon(), "holdout")
    lines = ["| Weeks ahead | Games | Home field | Preseason ratings | Unadjusted | Shrunk | Week of the game |", "|:--|--:|--:|--:|--:|--:|--:|"]
    for b, r in g.iterrows():
        lines.append(f"| {BUCKET_LABELS[BUCKETS.index(b)]} | {int(r.n_games):,} | {r.ll_const:.3f} | {r.ll_pre:.3f} | {r.ll_frozen:.3f} | {r.ll_shrunk:.3f} | {r.ll_oracle:.3f} |")
    (TAB / "opp_horizon.md").write_text(typo("\n".join(lines) + "\n"))

    p = pd.read_csv(DOCS / "opponent_paths.csv")
    p = p.loc[p["window"].eq("all")]
    name = {"6": "Before week 6", "10": "Before week 10", "14": "Before week 14", "end": "End of regular season"}
    lines = ["| Forecast made | Team-rounds | Top pick | Top 3 | Mean prob. | Standings |", "|:--|--:|--:|--:|--:|--:|"]
    for tag in ("6", "10", "14", "end"):
        r = p.loc[p["as_of"].astype(str).eq(tag) & p["round"].eq("ALL")].iloc[0]
        lines.append(f"| {name[tag]} | {int(r.n_team_rounds):,} | {_pct(r.model_top1)} | {_pct(r.model_top3)} | {_pct(r.model_mean_p)} | {_pct(r.standings_top1)} |")
    (TAB / "opp_paths.md").write_text(typo("\n".join(lines) + "\n"))

    lines = ["| Round | Team-rounds | Top pick | Standings | Mean prob. |", "|:--|--:|--:|--:|--:|"]
    for rnd in ("WC", "DIV", "CON", "SB"):
        r = p.loc[p["as_of"].astype(str).eq("10") & p["round"].eq(rnd)].iloc[0]
        lines.append(f"| {ROUND_NAMES[rnd]} | {int(r.n_team_rounds):,} | {_pct(r.model_top1)} | {_pct(r.standings_top1)} | {_pct(r.model_mean_p)} |")
    (TAB / "opp_paths_round.md").write_text(typo("\n".join(lines) + "\n"))

    ex = example()
    lines = ["| Wk | Opp | Site | Win | 80% band | If their QB is out | If ours is out | Opp rating (rank) | Opp QB (starts) |", "|--:|:--|:-:|--:|:--|--:|--:|:--|:--|"]
    for r in ex["schedule"]["games"]:
        qb = f'{r["opp_qb"]} ({r["opp_qb_starts"]})' if r["opp_qb"] else "n/a"
        lines.append(f'| {r["week"]} | {r["opponent"]} | {r["site"]} | {_pct(r["p_win"])} | {_pct(r["band_lo"])}–{_pct(r["band_hi"])} | '
                     f'{_pct(r["p_win_if_opp_qb_replaced"])} | {_pct(r["p_win_if_own_qb_replaced"])} | {r["opp_net_pts"]:+.1f} ({r["opp_rank"]}) | {qb} |')
    (TAB / "opp_example_schedule.md").write_text(typo("\n".join(lines) + "\n"))

    lines = ["| Round | Chance of playing it | If it gets there: most likely opponents (share; win chance) |", "|:--|--:|:--|"]
    for r in ex["playoff_path"]:
        opp = "; ".join(f'{o["team"]} {_pct(o["p_given_reach"])} ({_pct(o["p_win_if_meet"])})' for o in r["opponents"][:4]
                        if o["p_given_reach"] is not None and o["p_win_if_meet"] is not None)
        lines.append(f'| {r["round_name"]} | {_pct(r["p_reach"], 1)} | {opp or "n/a"} |')
    (TAB / "opp_example_path.md").write_text(typo("\n".join(lines) + "\n"))
    print("  tables/opp_*.md")


# ---------------------------------------------------------------------------
# variables
# ---------------------------------------------------------------------------

def variables() -> dict:
    d = _horizon()
    ho, dev = _pick(d, "holdout"), _pick(d, "dev")
    same, chg = _pick(d, "holdout", "same_qb"), _pick(d, "holdout", "qb_changed")
    sh = json.loads((DOCS / "opponent_shrink.json").read_text())
    sd = pd.read_csv(DOCS / "opponent_revision_sd.csv").set_index("h")
    f3 = lambda x: f"{x:.3f}"
    s = lambda x: f"{x:+.4f}"
    v = {
        "o_shrink_a": f'{sh["a"]:.4f}', "o_shrink_delta": s(sh["delta_log_loss"]), "o_shrink_lo": s(sh["ci_low"]), "o_shrink_hi": s(sh["ci_high"]),
        "o_shrink_est": f'{sh["delta_log_loss"]:+.4f} [{sh["ci_low"]:+.4f}, {sh["ci_high"]:+.4f}]', "o_shrink_n": f'{sh["n_games"]:,}',
        "o_shrink_gain": f'{-sh["delta_log_loss"]:.4f} [{-sh["ci_high"]:.4f}, {-sh["ci_low"]:.4f}]',   # improvement = minus the change
        "o_n_forecasts": f'{int(d.loc[d["window"].eq("all") & d["qb"].eq("all"), "n_forecasts"].sum()):,}',
        "o_n_fc_holdout": f'{int(ho["n_forecasts"].sum()):,}',
    }
    for b, tag in (("0", "0"), ("1", "1"), ("3-4", "34"), ("5-8", "58"), ("9+", "9")):
        r = ho.loc[b]
        v.update({f"o_ll_orc_{tag}": f3(r.ll_oracle), f"o_ll_shr_{tag}": f3(r.ll_shrunk), f"o_ll_raw_{tag}": f3(r.ll_frozen), f"o_ll_pre_{tag}": f3(r.ll_pre),
                  f"o_ll_const_{tag}": f3(r.ll_const), f"o_slope_raw_{tag}": f"{r.slope_frozen:.2f}", f"o_slope_shr_{tag}": f"{r.slope_shrunk:.2f}",
                  f"o_cost_{tag}": f"{r.d_vs_oracle:.3f} [{r.d_vs_oracle_lo:.3f}, {r.d_vs_oracle_hi:.3f}]",
                  f"o_gain_pre_{tag}": f"{-r.d_vs_pre:.3f} [{-r.d_vs_pre_hi:.3f}, {-r.d_vs_pre_lo:.3f}]",
                  f"o_gain_const_{tag}": f"{-r.d_vs_const:.3f} [{-r.d_vs_const_hi:.3f}, {-r.d_vs_const_lo:.3f}]",
                  f"o_rev_pts_{tag}": f"{r.mean_abs_revision_pts:.1f}"})
        v[f"o_cover_{tag}"] = _pct(r.cover_band)
        v[f"o_qbshare_{tag}"] = _pct(chg.loc[b, "n_forecasts"] / r.n_forecasts)
        v[f"o_cost_same_{tag}"] = f"{same.loc[b, 'd_vs_oracle']:.3f}"
        v[f"o_cost_chg_{tag}"] = f"{chg.loc[b, 'd_vs_oracle']:.3f}"
        v[f"o_cover_same_{tag}"] = _pct(same.loc[b, "cover_band"])
        v[f"o_cover_chg_{tag}"] = _pct(chg.loc[b, "cover_band"])
    ahead = ho.drop(index="0")
    v["o_cover_ahead"] = _pct(float((ahead["cover_band"] * ahead["n_forecasts"]).sum() / ahead["n_forecasts"].sum()))
    dahead = dev.drop(index="0")
    v["o_cover_dev"] = _pct(float((dahead["cover_band"] * dahead["n_forecasts"]).sum() / dahead["n_forecasts"].sum()))
    for h in (1, 4, 8, 12):
        v[f"o_sd_{h}"] = f3(sd.loc[h, "sd_all"])
    v["o_sd_qb_1"], v["o_sd_same_1"] = f3(sd.loc[1, "sd_qb_changed"]), f3(sd.loc[1, "sd_same_qb"])

    p = pd.read_csv(DOCS / "opponent_paths.csv")
    pa = p.loc[p["window"].eq("all") & p["round"].eq("ALL")].set_index(p.loc[p["window"].eq("all") & p["round"].eq("ALL"), "as_of"].astype(str))
    for tag in ("6", "10", "14", "end"):
        r = pa.loc[tag]
        v.update({f"o_pt_top1_{tag}": _pct(r.model_top1), f"o_pt_top3_{tag}": _pct(r.model_top3), f"o_pt_p_{tag}": _pct(r.model_mean_p), f"o_pt_std_{tag}": _pct(r.standings_top1)})
    v["o_pt_n"] = f'{int(pa.loc["10", "n_team_rounds"]):,}'
    v["o_pt_seasons"] = str(int(pa.loc["10", "n_seasons"]))

    ex = example()
    o, sch, rt, m, rec = ex["outlook"], ex["schedule"], ex["rating"], ex["model"], ex["record"]
    v.update({
        "o_ex_team": ex["name"], "o_ex_week": str(ex["as_of_week"]), "o_ex_season": str(ex["season"]), "o_ex_through": ex["data_through"],
        "o_ex_record": f'{rec["wins"]}–{rec["losses"]}' + (f'–{rec["ties"]}' if rec["ties"] else ""),
        "o_ex_rating": f'{rt["net_pts"]:+.1f}', "o_ex_rank": str(rt["rank"]), "o_ex_ew": f'{o["expected_wins"]:.1f}',
        "o_ex_w10": f'{o["wins_p10"]:g}', "o_ex_w90": f'{o["wins_p90"]:g}', "o_ex_playoffs": _pct(o["p_playoffs"]),
        "o_ex_remaining": str(sch["remaining"]), "o_ex_rem_ew": f'{sch["expected_wins"]:.1f}', "o_ex_sos": f'{sch["strength_pts"]:+.1f}',
        "o_ex_sos_rank": ordinal(sch["strength_rank_hardest"]), "o_ex_repl": f'{m["replacement_qb_pts"]:+.1f}', "o_ex_sims": f'{m["n_sims"]:,}',
        "o_ex_qb": rt["qb"], "o_ex_qb_pts": f'{rt["qb_pts"]:+.1f}',
    })
    swing = max(sch["games"], key=lambda g: g["p_win_if_opp_qb_replaced"] - g["p_win"])
    v["o_ex_swing"] = (f'week {swing["week"]} against {swing["opponent"]}: {_pct(swing["p_win"])} with {swing["opp_qb"] or "their projected starter"} '
                       f'starting, {_pct(swing["p_win_if_opp_qb_replaced"])} with a replacement')
    best = max(sch["games"], key=lambda g: g["p_win"])
    worst = min(sch["games"], key=lambda g: g["p_win"])
    v["o_ex_best"] = f'week {best["week"]} against {best["opponent"]} ({_pct(best["p_win"])})'
    v["o_ex_worst"] = f'week {worst["week"]} against {worst["opponent"]} ({_pct(worst["p_win"])})'
    v["o_ex_band_w"] = f'{_pct(sch["games"][0]["band_lo"])}\u2013{_pct(sch["games"][0]["band_hi"])}'
    v["o_ex_next"] = f'{sch["games"][0]["opponent"]} ({_pct(sch["games"][0]["p_win"])})'
    v["o_ex_next_week"] = str(sch["games"][0]["week"])
    return v


def make_figures() -> None:
    for fn in (fig_opp_horizon, fig_opp_qb, fig_opp_band):
        fn()
