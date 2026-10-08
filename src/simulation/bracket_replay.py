"""Replay the bracket projection for a completed season and score it against what happened.

    python -m src.simulation.bracket_replay --season 2025 --out site/data/bracket_replay.json

For every week of the regular season the projection uses only games before that week (ratings are walk-forward
and the game model is refit on earlier games), so nothing about the outcome leaks in. Scored three ways:

  field      Of the 14 teams that really made the playoffs, how many were projected, and in the right seed?
             Compared with a naive baseline: the standings at that moment.
  bracket    The full bracket picked before the Wild Card round (favorite in every game, with reseeding)
             versus the higher-seed baseline, scored by how many of the 13 winners each got right.
  games      Round by round, with the real matchups: model versus the closing line versus the higher seed,
             by log loss and picks.

There are only 13 playoff games a season, so these numbers describe one bracket; they are not evidence of skill.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import project_path
from src.data.schedules import last_complete_season
from src.models.market import market_probabilities
from src.simulation.divisions import CONFERENCE_OF, DIVISIONS
from src.simulation.live_bracket import (
    CONFS, ROUND_GAMES, ROUND_NAMES, ROUNDS, Inputs, Probs, build_bracket, fit_model, latest_snapshot, load_inputs,
    playoff_field, postseason, projected_field, season_odds, playoff_odds,
)
from src.simulation.seeding import regular_season_table
from src.teams import TEAM_NAMES

ALL_WEEKS = list(range(1, 19))


def truncate(games: pd.DataFrame, season: int, through: str | None) -> pd.DataFrame:
    """The schedule as it stood at a point in the calendar: playoff rounds after `through` do not exist yet."""
    if through is None:
        return games.loc[~(games["season"].eq(season) & games["is_postseason"])].copy()
    keep = ROUNDS[: ROUNDS.index(through) + 1]
    drop = games["season"].eq(season) & games["is_postseason"] & ~games["game_type"].isin(keep)
    return games.loc[~drop].copy()


def standings_field(games: pd.DataFrame, season: int, before_week: int) -> dict[str, list[str]]:
    """Naive baseline: the field if the season ended today (division leaders, then best records)."""
    reg = games.loc[games["season"].eq(season) & games["game_type"].eq("REG") & games["is_played"] & games["week"].lt(before_week)]
    if reg.empty:                                              # before week 1: last year's playoff field
        return playoff_field(games, season - 1)
    table = regular_season_table(reg.assign(is_postseason=False), season)
    key = table.assign(name=table.index).sort_values(["win_pct", "point_diff", "name"], ascending=[False, False, True])
    order = {t: i for i, t in enumerate(key.index)}
    field = {}
    for conf, divs in DIVISIONS.items():
        leaders = sorted((min(ts, key=lambda t: order.get(t, 99)) for ts in divs.values()), key=lambda t: order.get(t, 99))
        rest = sorted((t for ts in divs.values() for t in ts if t not in leaders), key=lambda t: order.get(t, 99))[:3]
        field[conf] = leaders + rest
    return field


def field_score(projected: dict, actual: dict) -> dict:
    ps = {t for s in projected.values() for t in s}
    ac = {t for s in actual.values() for t in s}
    exact = sum(1 for c in actual for i, t in enumerate(actual[c]) if projected[c][i] == t)
    return {"in_field": len(ps & ac), "exact_seed": exact}


def seed_chalk(field: dict, post: pd.DataFrame, records: dict) -> dict:
    """Baseline bracket: the higher seed wins every game (reseeding as the real bracket does)."""
    seed_of = {t: i + 1 for s in field.values() for i, t in enumerate(s)}
    picks = {"WC": [], "DIV": [], "CON": [], "SB": []}
    champs = {}
    for conf, seeds in field.items():
        s = {t: i + 1 for i, t in enumerate(seeds)}
        wc = [seeds[h] for h in (1, 2, 3)]                      # hosts (2, 3, 4) beat visitors (7, 6, 5)
        picks["WC"] += wc
        div_alive = sorted([seeds[0], *wc], key=s.get)
        div = [div_alive[0], div_alive[1]]
        picks["DIV"] += div
        champ = min(div, key=s.get)
        picks["CON"].append(champ)
        champs[conf] = champ
    wins = lambda t: records.get(t, (0, 0, 0))[0]
    picks["SB"].append(max(champs.values(), key=lambda t: (wins(t), -seed_of[t])))
    return picks


def picks_of(bracket: dict) -> dict:
    out = {"WC": [], "DIV": [], "CON": [], "SB": [bracket["SB"]["winner"]]}
    for conf in CONFS:
        for rnd in ("WC", "DIV", "CON"):
            out[rnd] += [g["winner"] for g in bracket["rounds"][conf][rnd]]
    return out


def qualification_context(season: int) -> dict | None:
    """How the model's playoff-qualification probabilities did across ALL seasons at the backtest checkpoints, and in `season`.

    Reads the multi-season backtest (outputs/season_backtest_playoffs.csv, 65,536 simulations); None if it was not built.
    The base rate entropy is the log loss of always saying the league-wide qualification rate (a no-skill reference).
    """
    from src.config import configured_path

    path = configured_path("outputs") / "season_backtest_playoffs.csv"
    if not path.exists():
        return None
    po = pd.read_csv(path)
    ll = lambda g: float(-np.mean(g["made"] * np.log(g["p_playoffs"].clip(1e-4, 1 - 1e-4)) + (1 - g["made"]) * np.log(1 - g["p_playoffs"].clip(1e-4, 1 - 1e-4))))
    rate = float(po["made"].mean())
    out = {"seasons": f"{int(po.season.min())}-{int(po.season.max())}", "n_seasons": int(po.season.nunique()),
           "base_rate_entropy": round(float(-(rate * np.log(rate) + (1 - rate) * np.log(1 - rate))), 4), "checkpoints": []}
    for week, g in po.groupby("week"):
        mine = g.loc[g["season"].eq(season)]
        out["checkpoints"].append({"week": int(week), "all_seasons": round(ll(g), 4), "this_season": round(ll(mine), 4) if len(mine) else None})
    return out


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def replay(inp: Inputs, season: int, n_sims: int = 16384, weeks: list[int] | None = None) -> dict:
    games = inp.games
    post = postseason(games, season)
    field = playoff_field(games, season)
    actual_set = {t for s in field.values() for t in s}
    rec_final = {}
    reg = games.loc[games["season"].eq(season) & games["game_type"].eq("REG") & games["is_played"]]
    for g in reg.itertuples(index=False):
        for team, mine, theirs in ((g.home_team, g.home_score, g.away_score), (g.away_team, g.away_score, g.home_score)):
            r = rec_final.setdefault(team, [0, 0, 0])
            r[0 if mine > theirs else 1 if mine < theirs else 2] += 1
    rec_final = {t: tuple(v) for t, v in rec_final.items()}

    # ---- weekly projections ------------------------------------------------------------
    weekly = []
    champion = None
    last_week = max(w for w in ALL_WEEKS if (reg["week"] == w).any())
    for w in (weeks or ALL_WEEKS + [last_week + 1]):
        sim, _, _ = season_odds(inp, season, w, n_sims)
        proj = projected_field(sim)
        base = standings_field(games, season, w)
        made = sim["team"].isin(actual_set).astype(float).to_numpy()
        p = sim["p_playoffs"].to_numpy()
        top = sim.sort_values("p_win_sb", ascending=False)
        weekly.append({
            "week": int(w), "label": f"Before week {w}" if w <= last_week else "End of regular season",
            "field": {c: proj[c] for c in CONFS}, "baseline_field": {c: base[c] for c in CONFS},
            "model": field_score(proj, field), "standings": field_score(base, field),
            "qual_log_loss": round(log_loss(p, made), 4), "qual_brier": round(float(np.mean((p - made) ** 2)), 4),
            "title_top": [{"team": r.team, "p": round(float(r.p_win_sb), 4)} for r in top.head(6).itertuples()],
            "p_seed": {r.team: [round(float(getattr(r, f"p_seed_{k}")), 3) for k in range(1, 8)] for r in sim.itertuples() if r.team in actual_set},
        })

    # ---- the bracket before the playoffs ---------------------------------------------------
    start_week = last_week + 1
    model = fit_model(inp, season, start_week)
    probs = Probs(inp, model, latest_snapshot(inp, season, start_week))
    actual = build_bracket(field, post, probs)                       # real results, pregame probabilities
    chalk = build_bracket(field, post.iloc[0:0], probs)              # nothing known: favorite everywhere
    champion = actual["champion"]
    title = playoff_odds(field, chalk, probs, "WC", 16384, season * 100 + 50)
    title_rank = int((title["p_win_sb"] > title.set_index("team").loc[champion, "p_win_sb"]).sum()) + 1

    won = picks_of(actual)
    model_picks = picks_of(chalk)
    seed_picks = seed_chalk(field, post, rec_final)
    bracket_scores = {"model": {r: len(set(model_picks[r]) & set(won[r])) for r in ROUNDS},
                      "higher_seed": {r: len(set(seed_picks[r]) & set(won[r])) for r in ROUNDS},
                      "games": dict(ROUND_GAMES)}
    for k in ("model", "higher_seed"):
        bracket_scores[k]["total"] = int(sum(bracket_scores[k][r] for r in ROUNDS))

    # ---- game by game with the real matchups ------------------------------------------------
    market = market_probabilities(games).set_index("game_id")["p_market"]
    rows = []
    for conf in CONFS:
        for rnd in ("WC", "DIV", "CON"):
            rows += actual["rounds"][conf][rnd]
    rows.append(actual["SB"])
    g_rows = []
    seed_of = {t: i + 1 for s in field.values() for i, t in enumerate(s)}
    for g in rows:
        pm = float(market.get(g["id"], np.nan))
        home_won = g["winner"] == g["home"]
        higher_home = (seed_of[g["home"]] < seed_of[g["away"]]) if g["round"] != "SB" else (rec_final.get(g["home"], (0,))[0] >= rec_final.get(g["away"], (0,))[0])
        g_rows.append({"round": g["round"], "conf": g["conf"], "home": g["home"], "away": g["away"], "p_model": g["p_home"],
                       "p_market": None if np.isnan(pm) else round(pm, 4), "home_won": bool(home_won), "winner": g["winner"],
                       "home_score": g["home_score"], "away_score": g["away_score"],
                       "model_right": bool((g["p_home"] >= 0.5) == home_won),
                       "market_right": None if np.isnan(pm) else bool((pm >= 0.5) == home_won),
                       "seed_right": bool(higher_home == home_won)})
    y = np.array([r["home_won"] for r in g_rows], float)
    pm = np.array([r["p_model"] for r in g_rows])
    pk = np.array([np.nan if r["p_market"] is None else r["p_market"] for r in g_rows])
    ok = ~np.isnan(pk)
    summary = {"n_games": len(g_rows), "model_right": int(sum(r["model_right"] for r in g_rows)),
               "market_right": int(sum(bool(r["market_right"]) for r in g_rows if r["market_right"] is not None)), "market_games": int(ok.sum()),
               "seed_right": int(sum(r["seed_right"] for r in g_rows)),
               "log_loss_model": round(log_loss(pm, y), 4), "log_loss_market": round(log_loss(pk[ok], y[ok]), 4) if ok.any() else None,
               "log_loss_model_same_games": round(log_loss(pm[ok], y[ok]), 4) if ok.any() else None, "log_loss_coin": round(float(np.log(2)), 4),
               "champion": champion, "champion_title_odds": round(float(title.set_index("team").loc[champion, "p_win_sb"]), 4),
               "champion_title_rank": title_rank, "title_field_size": 14}

    return {
        "season": season, "playoffs_year": season + 1, "champion": champion, "n_sims_weekly": n_sims,
        "actual_field": {c: [{"seed": i + 1, "team": t, "name": TEAM_NAMES[t]} for i, t in enumerate(field[c])] for c in CONFS},
        "actual_bracket": {"AFC": actual["rounds"]["AFC"], "NFC": actual["rounds"]["NFC"], "SB": actual["SB"]},
        "chalk_bracket": {"AFC": chalk["rounds"]["AFC"], "NFC": chalk["rounds"]["NFC"], "SB": chalk["SB"], "champion": chalk["champion"]},
        "preplayoff_title_odds": [{"team": r.team, "seed": int(r.seed), "p_win_conf": round(float(r.p_win_conf), 4), "p_win_sb": round(float(r.p_win_sb), 4)}
                                  for r in title.itertuples()],
        "weekly": weekly, "bracket_scores": bracket_scores, "games": g_rows, "summary": summary,
        "qualification_context": qualification_context(season),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=last_complete_season())
    parser.add_argument("--out", default="site/data/bracket_replay.json")
    parser.add_argument("--sims", type=int, default=16384)
    args = parser.parse_args()

    result = replay(load_inputs(), args.season, n_sims=args.sims)
    out = Path(args.out)
    if not out.is_absolute():
        out = project_path(*out.parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, separators=(",", ":")))
    s, b = result["summary"], result["bracket_scores"]
    print(f"{args.season} season (playoffs {args.season + 1}); champion {s['champion']} -> {out} ({out.stat().st_size / 1024:.0f} KB)")
    print(f"games: model {s['model_right']}/{s['n_games']}, market {s['market_right']}/{s['market_games']}, higher seed {s['seed_right']}/{s['n_games']}; "
          f"log loss model {s['log_loss_model']}, market {s['log_loss_market']} (model on same games {s['log_loss_model_same_games']}), coin {s['log_loss_coin']}")
    print(f"pre-playoff bracket winners right: model {b['model']['total']}/13, higher seed {b['higher_seed']['total']}/13; champion odds {s['champion_title_odds']:.1%}, rank {s['champion_title_rank']} of 14")
    for w in result["weekly"]:
        print(f"  {w['label']:<24} in-field {w['model']['in_field']:>2}/14 exact {w['model']['exact_seed']:>2}  | standings {w['standings']['in_field']:>2}/14 exact {w['standings']['exact_seed']:>2}  | qual log loss {w['qual_log_loss']:.3f}  | title top {w['title_top'][0]['team']} {w['title_top'][0]['p']:.1%}")


if __name__ == "__main__":
    main()
