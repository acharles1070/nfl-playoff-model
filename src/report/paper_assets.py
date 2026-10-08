"""Generate every figure, table and number the paper uses, from the same JSON as the website.

    python -m src.report.site_data      # first: (re)build site/data/*.json
    python -m src.report.paper_assets   # then: paper/figures/*, paper/tables/*.md, paper/_variables.yml

The paper text refers to numbers only through Quarto variables ({{< var name >}}), so the
prose cannot drift away from the pipeline's outputs.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import pandas as pd

from src.config import configured_path, project_path

DATA = project_path("site", "data")
PAPER = project_path("paper")
FIG = PAPER / "figures"
TAB = PAPER / "tables"

GREEN, AMBER, BLUE, GRAY, RED, VIOLET = "#2e9d63", "#d98a00", "#2f6fdb", "#7b8794", "#d64545", "#7a45d6"
plt.rcParams.update({
    "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linestyle": ":", "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
})


def load(name: str):
    return json.loads((DATA / f"{name}.json").read_text())


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  figures/{name}")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_scorecard() -> None:
    d = load("scorecard")
    models = d["models"]
    market = next(m for m in models if m["key"] == "p_market")
    colors = {"p_home_only": GRAY, "p_std_epa": BLUE, "p_kalman": GREEN, "p_champion": GREEN, "p_market": AMBER}

    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for i, m in enumerate(models):
        ax.barh(i, m["log_loss"], color=colors[m["key"]], alpha=1.0 if m["key"] == "p_champion" else 0.75, height=0.6)
        ax.text(max(m["log_loss"], market["log_loss"] + m["gap_hi"]) + 0.002, i, f'{m["log_loss"]:.4f}', va="center", fontsize=8.5)
        if m["key"] != "p_market":
            ax.errorbar(m["log_loss"], i, xerr=[[m["log_loss"] - (market["log_loss"] + m["gap_lo"])], [market["log_loss"] + m["gap_hi"] - m["log_loss"]]],
                        color="black", capsize=3, lw=1.2)
    ax.axvline(market["log_loss"], color=AMBER, ls="--", lw=1)
    ax.set_yticks(range(len(models)), [m["label"] for m in models])
    ax.invert_yaxis()
    ax.set_xlim(0.59, 0.72)
    ax.set_xlabel("log loss, holdout 2017-2025 (lower is better)")
    ax.grid(axis="y", visible=False)
    save(fig, "scorecard")


def fig_calibration() -> None:
    d = load("calibration")
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1)
    for key, color, label, dx in (("p_champion", GREEN, "Kalman + QB model", -0.006), ("p_market", AMBER, "Closing market", 0.006)):
        b = d[key]["bins"]
        x = np.array([r["pred"] for r in b]) + dx
        y = np.array([r["obs"] for r in b])
        lo = y - np.array([r["lo"] for r in b])
        hi = np.array([r["hi"] for r in b]) - y
        ax.errorbar(x, y, yerr=[lo, hi], fmt="o", color=color, capsize=2, ms=4, label=f'{label} (slope {d[key]["slope"]:.2f})')
    ax.set_xlabel("forecast probability of a home win")
    ax.set_ylabel("observed home-win frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "calibration")


def fig_registry() -> None:
    rows = load("registry")["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 7.4))
    for i, r in enumerate(rows):
        color = GREEN if r["survives"] else AMBER if r["raw_sig"] else GRAY
        ax.errorbar(r["delta"], i, xerr=[[r["delta"] - r["lo"]], [r["hi"] - r["delta"]]], fmt="o", color=color, capsize=0, ms=4, lw=1.6)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_yticks(range(len(rows)), [f'{r["id"]} {r["name"][:58]}' for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("change in log loss vs. the comparison model (negative = better)")
    ax.set_xlim(-0.03, 0.03)
    ax.grid(axis="y", visible=False)
    ax.plot([], [], "o", color=GREEN, label="survives Holm and BH")
    ax.plot([], [], "o", color=AMBER, label="nominally significant only")
    ax.plot([], [], "o", color=GRAY, label="no detectable effect")
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, "registry")


def fig_betting() -> None:
    d = load("betting")
    s = d["series"]["0.0"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.38})
    ax = axes[0]
    ax.plot(s["flat"], color=RED, lw=1.8, label="model's picks")
    for (name, v), color in zip(d["baselines"].items(), (AMBER, BLUE)):
        ax.plot(v["curve"], color=color, lw=1.2, ls="--", label=name.lower())
    for e in s["season_end"]:
        ax.axvline(e, color=GRAY, lw=0.5, alpha=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("bets placed (date order)")
    ax.set_ylabel("cumulative profit (units)")
    ax.set_title("flat 1-unit stakes", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower left")
    ax = axes[1]
    ax.semilogy(np.maximum(s["kelly"], 0.05), color=RED, lw=1.8)
    ax.axhline(100, color="black", lw=0.8)
    ax.set_xlabel("bets placed (date order)")
    ax.set_ylabel("bankroll (log scale, start 100)")
    ax.set_title("quarter-Kelly staking", fontsize=9)
    save(fig, "betting")


def fig_leakage() -> None:
    d = load("leakage")
    fig, ax = plt.subplots(figsize=(6.0, 2.3))
    kinds = {"peek": RED, "honest": GREEN, "market": AMBER}
    for i, r in enumerate(d["rows"]):
        ax.barh(i, r["log_loss"], color=kinds[r["kind"]], height=0.6)
        ax.text(r["log_loss"] + 0.002, i, f'{r["log_loss"]:.4f}', va="center", fontsize=8.5)
    ax.set_yticks(range(len(d["rows"])), [r["label"] for r in d["rows"]])
    ax.invert_yaxis()
    ax.set_xlim(0.55, 0.68)
    ax.set_xlabel(f'log loss on {d["games"]:,} regular-season holdout games (lower is better)')
    ax.grid(axis="y", visible=False)
    save(fig, "leakage")


def fig_power() -> None:
    sd = load("power")["sigma"]
    n = np.logspace(1.7, 4.5, 200)
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for delta, color in ((0.002, GRAY), (0.005, BLUE), (0.01, GREEN), (0.027, AMBER)):
        z = delta / (sd / np.sqrt(n))
        ax.semilogx(n, norm.cdf(z - 1.96) + norm.cdf(-z - 1.96), color=color, lw=1.8, label=f"true gain {delta:g}")
    ax.axhline(0.8, color="black", ls=":", lw=0.9)
    for x, label in ((111, "111 playoff\ngames"), (2494, "2,494 holdout\ngames")):
        ax.axvline(x, color=RED, ls="--", lw=0.9)
        ax.text(x * 1.06, 0.04, label, fontsize=7.5, color=RED)
    ax.set_xlabel("games in the test")
    ax.set_ylabel("power (5% two-sided test)")
    ax.legend(fontsize=7.5, loc="center left")
    save(fig, "power")


def fig_kalman() -> None:
    d = load("kalman")
    t = next(x for x in d["teams"] if x["label"] == "Best team")
    g = t["games"]
    obs_sd, drift_sd = d["league_game_sd"], 0.03
    x, P = t["prior_mean"], t["prior_sd"] ** 2
    path = []
    for k, q in enumerate(g):
        if k:
            P += drift_sd ** 2
        path.append((x, np.sqrt(P)))
        K = P / (P + obs_sd ** 2)
        x += K * (q["obs"] - x)
        P *= 1 - K
    wk = np.arange(1, len(g) + 1)
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.scatter(wk, [q["obs"] for q in g], color="black", s=18, zorder=3, label="single-game net EPA/play")
    ax.plot(wk, [p[0] for p in path], color=GREEN, lw=2, label="simplified filter (belief going in)")
    ax.fill_between(wk, [p[0] - p[1] for p in path], [p[0] + p[1] for p in path], color=GREEN, alpha=0.18)
    ax.plot(wk, [q["model"] for q in g], color=AMBER, lw=1.6, label="full model (off/def/QB, opponent-adjusted)")
    ax.plot(wk, [np.mean([q["obs"] for q in g[:k]]) if k else t["prior_mean"] for k in range(len(g))], color=BLUE, ls="--", lw=1.2, label="plain average so far")
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xlabel(f'{t["team"]}, {d["season"]} regular season: game number')
    ax.set_ylabel("net EPA / play")
    ax.set_ylim(-0.25, 0.98)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    save(fig, "kalman")


def fig_bracket_replay() -> None:
    r = load("bracket_replay")
    W = r["weekly"]
    x = np.arange(1, len(W) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.32})
    ax = axes[0]
    ax.plot(x, [w["model"]["in_field"] for w in W], color=GREEN, lw=2, label="model projection")
    ax.plot(x, [w["standings"]["in_field"] for w in W], color=BLUE, lw=2, label="standings baseline")
    ax.set_xlabel("week (projection made before that week)")
    ax.set_ylabel("of the 14 playoff teams in the field")
    ax.set_xticks([1, 4, 7, 10, 13, 16, len(W)], ["1", "4", "7", "10", "13", "16", "end"])
    ax.set_ylim(4, 14.5)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("Projected field", fontsize=9)
    qc = r["qualification_context"]
    ax = axes[1]
    if qc:
        weeks = [c["week"] for c in qc["checkpoints"]]
        ax.plot(weeks, [c["all_seasons"] for c in qc["checkpoints"]], color=GRAY, lw=2, marker="o", label=f'average, {qc["n_seasons"]} seasons')
        ax.plot(weeks, [c["this_season"] for c in qc["checkpoints"]], color=RED, lw=2, marker="o", label=f'{r["season"]}')
        ax.axhline(qc["base_rate_entropy"], color="black", ls=":", lw=1)
        ax.text(weeks[-1], qc["base_rate_entropy"] + 0.012, "no-skill guess", fontsize=7, ha="right")
        ax.set_xlabel("week")
        ax.set_ylabel("qualification log loss")
        ax.legend(fontsize=7.5, loc="lower left")
        ax.set_title("Playoff-qualification forecasts", fontsize=9)
    save(fig, "bracket_replay")


# ---------------------------------------------------------------------------
# variables and tables
# ---------------------------------------------------------------------------

def variables() -> dict:
    sc, bt, rg = load("scorecard"), load("betting"), load("registry")
    leak, cal, led, meta = load("leakage"), load("calibration"), load("ledger"), load("meta")
    m = {x["key"]: x for x in sc["models"]}
    s0 = bt["series"]["0.0"]
    fav, home = bt["baselines"]["Bet every favorite"], bt["baselines"]["Bet every home team"]
    peek = next(r for r in leak["rows"] if r["kind"] == "peek")
    std = next(r for r in leak["rows"] if "season-to-date" in r["label"])
    f4 = lambda x: f"{x:.4f}"
    pc = lambda x, d=1: f"{x * 100:.{d}f}"
    enc = load("encompassing")
    by_id = {r["id"]: r for r in rg["rows"]}
    est = lambda i: f'{by_id[i]["delta"]:+.4f} [{by_id[i]["lo"]:+.4f}, {by_id[i]["hi"]:+.4f}]'
    reg_vars = {f"e{i:02d}": est(f"E{i:02d}") for i in (1, 2, 3, 4, 5, 8, 21, 25, 27)}
    reg_vars.update({"e27_p": f'{by_id["E27"]["p"]:.3f}', "e03_p": f'{by_id["E03"]["p"]:.3f}', "e04_p": f'{by_id["E04"]["p"]:.3f}', "e02_p": f'{by_id["E02"]["p"]:.3f}'})
    rep_rows = load("replication")["rows"]
    pick = lambda needle: next(r for r in rep_rows if needle in r["window"])
    fmt = lambda r: f'{r["delta_log_loss"]:+.4f} [{r["ci_low"]:+.4f}, {r["ci_high"]:+.4f}]'
    rep_vars = {"rep_primary": fmt(pick("primary")), "rep_n": f'{pick("primary")["games"]:,}',
                "rep_dev": fmt(pick("dev window")), "rep_hold": fmt(pick("holdout"))}
    br = load("bracket_replay")
    bs, bsum, bw = br["bracket_scores"], br["summary"], br["weekly"]
    qc = br["qualification_context"] or {"checkpoints": [], "n_seasons": 0, "base_rate_entropy": float("nan"), "seasons": ""}
    cp = {c["week"]: c for c in qc["checkpoints"]}
    ahead = sum(w["model"]["in_field"] > w["standings"]["in_field"] for w in bw)
    behind = sum(w["model"]["in_field"] < w["standings"]["in_field"] for w in bw)
    def wilson(k: int, n: int) -> tuple[float, float]:
        z, ph = 1.96, k / n
        d = 1 + z * z / n
        c, m = (ph + z * z / (2 * n)) / d, z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
        return c - m, c + m

    w_lo, w_hi = wilson(bsum["model_right"], bsum["n_games"])
    br_vars = {
        "br_lo": f"{w_lo * 100:.0f}", "br_hi": f"{w_hi * 100:.0f}",
        "br_season": str(br["season"]), "br_year": str(br["playoffs_year"]), "br_champ": br["champion"],
        "br_games_model": str(bsum["model_right"]), "br_games_market": str(bsum["market_right"]), "br_games_seed": str(bsum["seed_right"]),
        "br_n": str(bsum["n_games"]), "br_ll_model": f'{bsum["log_loss_model"]:.3f}', "br_ll_market": f'{bsum["log_loss_market"]:.3f}',
        "br_picks_model": str(bs["model"]["total"]), "br_picks_seed": str(bs["higher_seed"]["total"]),
        "br_champ_odds": f'{bsum["champion_title_odds"] * 100:.1f}', "br_champ_rank": str(bsum["champion_title_rank"]),
        "br_pick": br["chalk_bracket"]["champion"], "br_first_model": str(bw[0]["model"]["in_field"]), "br_first_std": str(bw[0]["standings"]["in_field"]),
        "br_last_model": str(bw[-1]["model"]["in_field"]), "br_last_std": str(bw[-1]["standings"]["in_field"]),
        "br_weeks": str(len(bw)), "br_ahead": str(ahead), "br_behind": str(behind), "br_level": str(len(bw) - ahead - behind),
        "br_q1": f'{cp.get(1, {}).get("this_season", float("nan")):.2f}', "br_q1_all": f'{cp.get(1, {}).get("all_seasons", float("nan")):.2f}',
        "br_q9": f'{cp.get(9, {}).get("this_season", float("nan")):.2f}', "br_q9_all": f'{cp.get(9, {}).get("all_seasons", float("nan")):.2f}',
        "br_base": f'{qc["base_rate_entropy"]:.2f}', "br_nseasons": str(qc["n_seasons"]),
    }
    pw = load("power")
    z80 = 1.959964 + 0.841621
    power_vars = {"sigma": f'{pw["sigma"]:.3f}', "mde_post": f"{z80 * pw['sigma'] / np.sqrt(sc['postseason_games']):.3f}",
                  "mde_hold": f"{z80 * pw['sigma'] / np.sqrt(sc['games']):.4f}", "games_002": f"{pw['games_002']:,}",
                  "seasons_002": f"{pw['games_002'] / 285:.0f}", "games_005": f"{pw['games_005']:,}"}
    done = [(r, led["finished"][r["game_id"]]) for r in led["rows"] if r["game_id"] in led["finished"] and r["model_id"] == "ratings_qb_logit_v1"]
    ys = [1.0 if f["home_score"] > f["away_score"] else 0.0 for _, f in done]
    ll = lambda p, y: -(y * np.log(p) + (1 - y) * np.log(1 - p))
    led_model = np.mean([ll(float(r["p_home_win"]), y) for (r, _), y in zip(done, ys)]) if done else float("nan")
    led_market = np.mean([ll(float(r["market_p_home"]), y) for (r, _), y in zip(done, ys)]) if done else float("nan")
    return {
        **reg_vars, **power_vars, **rep_vars, **br_vars,
        "leak_gap": f4(std["log_loss"] - peek["log_loss"]),
        "enc_coef": f'{enc["model_coef"]:.2f}', "enc_lo": f'{enc["model_lo"]:.2f}', "enc_hi": f'{enc["model_hi"]:.2f}', "enc_p": f'{enc["model_p"]:.2f}',
        "enc_corr": f'{enc["corr"]:.2f}', "enc_market": f'{enc["market_coef"]:.2f}',
        "led_done": str(len(done)), "led_model": f"{led_model:.3f}", "led_market": f"{led_market:.3f}",
        "n_games": f'{sc["games"]:,}', "n_post": str(sc["postseason_games"]),
        "ll_home": f4(m["p_home_only"]["log_loss"]), "ll_epa": f4(m["p_std_epa"]["log_loss"]), "ll_kalman": f4(m["p_kalman"]["log_loss"]),
        "ll_champ": f4(m["p_champion"]["log_loss"]), "ll_market": f4(m["p_market"]["log_loss"]),
        "acc_champ": pc(m["p_champion"]["accuracy"]), "acc_market": pc(m["p_market"]["accuracy"]), "acc_home": pc(m["p_home_only"]["accuracy"]),
        "gap": f4(m["p_champion"]["gap_vs_market"]), "gap_lo": f4(m["p_champion"]["gap_lo"]), "gap_hi": f4(m["p_champion"]["gap_hi"]),
        "slope_model": f'{cal["p_champion"]["slope"]:.2f}', "slope_market": f'{cal["p_market"]["slope"]:.2f}',
        "n_tests": str(rg["n"]), "n_raw": str(rg["raw_sig"]), "n_holm": str(rg["holm"]), "exp_fp": f'{rg["n"] * 0.05:.1f}',
        "bets_n": f'{s0["bets"]:,}', "roi": pc(s0["roi"]), "roi_lo": pc(s0["roi_lo"]), "roi_hi": pc(s0["roi_hi"]),
        "roi_fav": pc(fav["roi"]), "roi_fav_lo": pc(fav["roi_lo"]), "roi_fav_hi": pc(fav["roi_hi"]), "roi_home": pc(home["roi"]),
        "vsfav": f'{bt["vs_favorite"]["diff"] * 100:.1f}', "vsfav_lo": f'{bt["vs_favorite"]["lo"] * 100:.1f}', "vsfav_hi": f'{bt["vs_favorite"]["hi"] * 100:+.1f}',
        "on_fav": f'{bt["vs_favorite"]["model_on_favorite"] * 100:.0f}', "sharpe": f'{s0["sharpe"]:.3f}', "dsr": f'{s0["dsr"]:.2f}', "kelly_end": f'{s0["kelly"][-1]:.1f}', "overround": pc(bt["overround"]),
        "units": f'{s0["flat"][-1]:.0f}', "n_trials": str(s0["n_trials"]),
        "ll_peek": f4(peek["log_loss"]), "ll_std_epa_reg": f4(std["log_loss"]), "n_leak": f'{leak["games"]:,}',
        "n_ledger": str(len(led["rows"])), "built": meta["built_utc"], "sha": meta["git_sha"],
    }


def write_variables(extra: dict | None = None) -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    v = {**variables(), **(extra or {})}
    (PAPER / "_variables.yml").write_text("".join(f'{k}: "{val}"\n' for k, val in v.items()))
    print(f"  _variables.yml ({len(v)} values)")


def write_tables() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    sc = load("scorecard")
    market = next(m for m in sc["models"] if m["key"] == "p_market")
    lines = ["| Model | Log loss | Brier | Accuracy | Gap to market [95% CI] |", "|:--|--:|--:|--:|--:|"]
    for m in sc["models"]:
        gap = "reference" if m["key"] == "p_market" else f'+{m["gap_vs_market"]:.4f} [+{m["gap_lo"]:.4f}, +{m["gap_hi"]:.4f}]'
        lines.append(f'| {m["label"]} | {m["log_loss"]:.4f} | {m["brier"]:.4f} | {m["accuracy"] * 100:.1f}% | {gap} |')
    (TAB / "scorecard.md").write_text("\n".join(lines) + "\n")

    rep = load("replication")["rows"]
    lines = ["| Test seasons | Games | Change in log loss [95% CI] | Seasons Kalman better |", "|:--|--:|--:|--:|"]
    lines += [f'| {r["window"].replace("primary: ", "").replace("secondary: ", "")} | {r["games"]:,} | {r["delta_log_loss"]:+.4f} [{r["ci_low"]:+.4f}, {r["ci_high"]:+.4f}] | {r["per_season_better"]}/{r["n_seasons"]} |' for r in rep]
    (TAB / "replication.md").write_text("\n".join(lines) + "\n")
    br = load("bracket_replay")
    pick = [1, 5, 9, 13, 17]
    rows = ["| Projection made | Model: in field | Model: exact seed | Standings: in field | Standings: exact seed | Qualification log loss |", "|:--|--:|--:|--:|--:|--:|"]
    for w in br["weekly"]:
        if w["week"] in pick or w["label"].startswith("End"):
            rows.append(f'| {w["label"].lower().capitalize()} | {w["model"]["in_field"]}/14 | {w["model"]["exact_seed"]}/14 | {w["standings"]["in_field"]}/14 | {w["standings"]["exact_seed"]}/14 | {w["qual_log_loss"]:.3f} |')
    (TAB / "bracket_replay_field.md").write_text("\n".join(rows) + "\n")
    sm, bs = br["summary"], br["bracket_scores"]
    games = ["| 13 playoff games | Winners picked right | Log loss |", "|:--|--:|--:|",
             f'| My model, real matchups | {sm["model_right"]}/{sm["n_games"]} | {sm["log_loss_model"]:.3f} |',
             f'| Closing betting line | {sm["market_right"]}/{sm["market_games"]} | {sm["log_loss_market"]:.3f} |',
             f'| Higher seed always wins | {sm["seed_right"]}/{sm["n_games"]} | n/a |',
             f'| Coin flip | 6.5/13 | {sm["log_loss_coin"]:.3f} |',
             f'| Full bracket picked before kickoff: model | {bs["model"]["total"]}/13 | n/a |',
             f'| Full bracket picked before kickoff: higher seed | {bs["higher_seed"]["total"]}/13 | n/a |']
    (TAB / "bracket_replay_games.md").write_text("\n".join(games) + "\n")
    print("  tables/*.md")


def write_backtest_tables() -> dict:
    """Bracket and title-odds scoring versus futures prices; skipped when outputs/ is absent.
    The generated tables are committed, so the paper builds without the raw data."""
    from src.simulation.season_report import season_bootstrap

    out, extra = configured_path("outputs"), {}
    if (out / "bracket_backtest.csv").exists():
        b = pd.read_csv(out / "bracket_backtest.csv")
        b = b[b.season >= 2017]
        g = b.groupby(["market", "round"]).agg(n=("season", "size"), model=("logscore_model", "mean"), market=("logscore_market", "mean"))
        g["diff"] = g.model - g.market
        name = {"afc": "AFC champion", "nfc": "NFC champion", "sb": "Super Bowl winner"}
        order = {"WC": 0, "DIV": 1, "CON": 2, "SB": 3}
        lines = ["| Market | Prices as of | Seasons | Model | Futures | Model - futures |", "|:--|:--|--:|--:|--:|--:|"]
        for (mk, rd), r in sorted(g.iterrows(), key=lambda kv: (["afc", "nfc", "sb"].index(kv[0][0]), order[kv[0][1]])):
            lines.append(f'| {name[mk]} | before {rd} | {int(r.n)} | {r.model:.3f} | {r.market:.3f} | {r["diff"]:+.3f} |')
        (TAB / "bracket.md").write_text("\n".join(lines) + "\n")
        extra.update({"br_cells": str(len(g)), "br_trail": str(int((g["diff"] > 0).sum()))})
    if (out / "season_backtest_title.csv").exists():
        t = pd.read_csv(out / "season_backtest_title.csv")
        t["d"] = t.logscore_model - t.logscore_market
        lines = ["| Prior to week | Model | Market | Model - market [95% CI, seasons resampled] |", "|--:|--:|--:|--:|"]
        for w, e in t[t.season >= 2017].groupby("week"):
            d, lo, hi = season_bootstrap(e, "d")
            lines.append(f"| {w} | {e.logscore_model.mean():.3f} | {e.logscore_market.mean():.3f} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
        (TAB / "title.md").write_text("\n".join(lines) + "\n")
        d, lo, hi = season_bootstrap(t[t.season >= 2017], "d")
        extra.update({"ti_diff": f"{d:+.2f}", "ti_lo": f"{lo:+.2f}", "ti_hi": f"{hi:+.2f}"})
    if (out / "season_backtest_playoffs.csv").exists():
        po = pd.read_csv(out / "season_backtest_playoffs.csv")
        extra.update({"po_n": f"{len(po):,}", "po_pred": f"{po.p_playoffs.mean():.3f}", "po_obs": f"{po.made.mean():.3f}"})
    print("  tables/bracket.md, title.md")
    return extra


def main() -> None:
    print("Generating paper assets ...")
    for fn in (fig_scorecard, fig_calibration, fig_registry, fig_betting, fig_leakage, fig_power, fig_kalman, fig_bracket_replay):
        fn()
    TAB.mkdir(parents=True, exist_ok=True)
    extra = write_backtest_tables()
    write_variables(extra)
    write_tables()


if __name__ == "__main__":
    main()
