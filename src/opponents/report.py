"""Opponent forecast report for one team.

    python -m src.opponents.report --team WAS
    python -m src.opponents.report --team WAS --season 2025 --week 10      # as it would have looked before week 10

For a team and an as-of week (default: the next week with games left), using only games before that week:

  schedule       every remaining opponent: win probability, an 80% band for where that forecast will stand when the
                 game arrives, the opponent's rating and rank, its quarterback, and what the probability becomes if
                 either quarterback is replaced by a first-start replacement
  outlook        distribution of final wins, playoff / division / seed odds (the logged title simulator)
  schedule       strength of the remaining schedule and its league rank
  playoff path   for each round, the chance of getting there and the opponents it is likely to meet there

Everything is derived from the rating model; no betting-market input is used. Writes JSON (for tools), a Markdown
brief (for people) and a CSV of the games to outputs/opponents/.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import configured_path, project_path
from src.data.schedules import live_season
from src.features.ratings import REFERENCE_PLAYS, attach_ratings_to_matchups
from src.opponents.forecast import (HORIZON_SHRINK_A, forecast_frame, projected_ratings, replacement_qb, revision_band,
                                    shrink_logit, sigmoid)
from src.opponents.horizon import SD_CAP
from src.opponents.paths import ROUNDS, opponents, reach_probability, season_detail, wins_distribution
from src.simulation.divisions import CONFERENCE_OF, DIVISIONS
from src.simulation.live_bracket import records
from src.teams import TEAM_NAMES

SCHEMA = 1
BAND_LEVEL = 0.80
ROUND_NAMES = {"WC": "Wild Card", "DIV": "Divisional", "CON": "Conference championship", "SB": "Super Bowl"}
SD_PATH = ("docs", "opponent_revision_sd.csv")


def pts(epa_per_play: float) -> float:
    """EPA per play -> rough points per game (x average offensive plays per team-game)."""
    return float(epa_per_play * REFERENCE_PLAYS)


def next_week(games: pd.DataFrame, season: int) -> int | None:
    """The week after the last one with a result; None once the regular season is over.

    Games from earlier weeks that are still unplayed (Monday night) stay on the schedule as remaining games.
    """
    reg = games.loc[games["season"].eq(season) & games["game_type"].eq("REG")]
    if reg["is_played"].all():
        return None
    return int(reg.loc[reg["is_played"], "week"].max()) + 1 if reg["is_played"].any() else 1


def revision_table() -> pd.DataFrame:
    path = project_path(*SD_PATH)
    if not path.exists():
        raise SystemExit(f"{path} is missing; build it with: python -m src.opponents.horizon")
    return pd.read_csv(path).set_index("h")


def qb_names(qb_game: pd.DataFrame) -> dict[str, str]:
    q = qb_game.sort_values(["season", "week"]).drop_duplicates("qb_player_id", keep="last")
    return dict(zip(q["qb_player_id"], q["qb_player_name"]))


def swap_qb(frame: pd.DataFrame, side: str, value: float) -> pd.DataFrame:
    """The frame with `side`'s quarterback rating replaced by `value` (offense total and the difference follow)."""
    f = frame.copy()
    f[f"{side}_pregame_qb"] = value
    f[f"{side}_pregame_off_total"] = f[f"{side}_pregame_off"] + value
    f["diff_rating_off_total"] = f["home_pregame_off_total"] - f["away_pregame_off_total"]
    return f


def team_rows(latest: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    """One row per team: ratings in EPA/play, and rank by overall strength (1 = best)."""
    t = latest.set_index("team")
    out = pd.DataFrame({
        "off": t["pregame_off"], "qb": t["pregame_qb"], "dfn": t["pregame_def"],
        "qb_id": t["pregame_qb_id"], "qb_starts": t["pregame_qb_starts"],
    })
    out["net"] = out["off"] + out["qb"] - out["dfn"]
    out["rank"] = out["net"].rank(ascending=False, method="first").astype(int)
    out["qb_name"] = out["qb_id"].map(names)
    return out


def build_report(inp, team: str, season: int, week: int, n_sims: int = 65536) -> dict:
    games = inp.games
    sim, detail, model, latest = season_detail(inp, season, week, n_sims)
    teams = team_rows(latest, qb_names(inp.qb_game))
    sd = revision_table()
    repl = replacement_qb(inp.ratings, season)

    # ---- the remaining games, forecast from ratings built before `week` -----------------------------------
    proj = projected_ratings(inp, season, week)
    reg = games.loc[games["season"].eq(season) & games["game_type"].eq("REG")]
    mine = reg.loc[(reg["week"].ge(week) | ~reg["is_played"]) & (reg["home_team"].eq(team) | reg["away_team"].eq(team))]
    mine = mine.sort_values(["week", "game_id"]).reset_index(drop=True)
    frame = attach_ratings_to_matchups(mine, proj)
    neutral = mine["location"].astype(str).str.lower().eq("neutral").to_numpy()
    h = np.maximum(mine["week"].to_numpy() - week, 0)
    is_home = mine["home_team"].eq(team).to_numpy()
    sign = np.where(is_home, 1.0, -1.0)

    def focal_logit(f: pd.DataFrame) -> np.ndarray:
        return sign * shrink_logit(forecast_frame(model, f, neutral)["logit"].to_numpy(), h)

    z = focal_logit(frame)
    sd_h = sd.loc[np.minimum(h, SD_CAP), "sd_all"].to_numpy()
    sd_qb = sd.loc[np.minimum(h, SD_CAP), "sd_qb_changed"].to_numpy()
    lo, hi = revision_band(z, sd_h, BAND_LEVEL)
    lo_q, hi_q = revision_band(z, sd_qb, BAND_LEVEL)
    opp_side = np.where(is_home, "away", "home")
    own_side = np.where(is_home, "home", "away")

    def what_if(k: int, side: str) -> float:
        """Focal logit for game k with `side`'s quarterback replaced."""
        f = swap_qb(frame.iloc[[k]], side, repl)
        return float(sign[k] * shrink_logit(forecast_frame(model, f, neutral[[k]])["logit"].to_numpy()[0], h[k]))

    z_opp_out = np.array([what_if(k, opp_side[k]) for k in range(len(mine))])
    z_own_out = np.array([what_if(k, own_side[k]) for k in range(len(mine))])
    names = qb_names(inp.qb_game)

    rows = []
    for k, g in mine.iterrows():
        opp = g["away_team"] if is_home[k] else g["home_team"]
        opp_side_k = opp_side[k]
        fr = frame.iloc[k]
        opp_qb = fr[f"{opp_side_k}_pregame_qb_id"]
        t = teams.loc[opp]
        opp_qb_id = None if pd.isna(opp_qb) else str(opp_qb)
        rows.append({
            "week": int(g["week"]), "date": str(g["gameday"])[:10], "weeks_ahead": int(h[k]), "opponent": opp, "opponent_name": TEAM_NAMES[opp],
            "site": "N" if neutral[k] else ("H" if is_home[k] else "A"),
            "rest_days": None if pd.isna(g.get("home_rest")) else int(g["home_rest"] if is_home[k] else g["away_rest"]),
            "opp_rest_days": None if pd.isna(g.get("home_rest")) else int(g["away_rest"] if is_home[k] else g["home_rest"]),
            "p_win": round(float(sigmoid(z[k])), 4), "band_lo": round(float(lo[k]), 4), "band_hi": round(float(hi[k]), 4),
            "band_if_qb_change_lo": round(float(lo_q[k]), 4), "band_if_qb_change_hi": round(float(hi_q[k]), 4),
            "p_win_if_opp_qb_replaced": round(float(sigmoid(z_opp_out[k])), 4), "p_win_if_own_qb_replaced": round(float(sigmoid(z_own_out[k])), 4),
            "opp_net_pts": round(pts(t["net"]), 2), "opp_rank": int(t["rank"]), "opp_off_pts": round(pts(t["off"] + t["qb"]), 2),
            "opp_def_pts": round(pts(-t["dfn"]), 2), "opp_qb": names.get(opp_qb_id), "opp_qb_id": opp_qb_id, "opp_qb_pts": round(pts(fr[f"{opp_side_k}_pregame_qb"]), 2),
            "opp_qb_starts": int(fr[f"{opp_side_k}_pregame_qb_starts"]) if pd.notna(fr[f"{opp_side_k}_pregame_qb_starts"]) else None,
        })

    # ---- strength of the remaining schedule, for every team (1 = hardest) --------------------------------------
    sched = proj.assign(opp_net=proj["opponent"].map(teams["net"])).groupby("team")["opp_net"].mean()
    sos = pts(sched.loc[team])
    sos_rank = int(sched.rank(ascending=False, method="first").loc[team])

    # ---- outlook and playoff path --------------------------------------------------------------------------------
    s = sim.set_index("team").loc[team]
    rec = records(games, season).get(team, (0, 0, 0))
    dist = wins_distribution(detail, team)
    cum = dist["p"].cumsum()
    q = lambda x: float(dist.loc[(cum >= x).idxmax(), "wins"])
    path = []
    reach = dict(zip(ROUNDS, (reach_probability(detail, r)[detail["teams"].index(team)] for r in ROUNDS)))
    for r in ROUNDS:
        opp = opponents(detail, team, r, top=5)
        path.append({"round": r, "round_name": ROUND_NAMES[r], "p_reach": round(float(reach[r]), 4),
                     "opponents": [{"team": o.opponent, "name": o.name, "p_meet": round(float(o.p_meet), 4),
                                    "p_given_reach": None if pd.isna(o.p_given_reach) else round(float(o.p_given_reach), 4),
                                    "p_win_if_meet": None if pd.isna(o.p_win_if_meet) else round(float(o.p_win_if_meet), 4)}
                                   for o in opp.itertuples()]})

    me = teams.loc[team]
    played = reg.loc[reg["is_played"] & reg["week"].lt(week)].sort_values(["gameday", "game_id"])      # what the forecast actually used
    conf = CONFERENCE_OF[team]
    div = next(d for d, ts in DIVISIONS[conf].items() if team in ts)
    return {
        "schema": SCHEMA, "team": team, "name": TEAM_NAMES[team], "conference": conf, "division": f"{conf} {div}",
        "season": season, "as_of_week": week, "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_through": str(played.iloc[-1]["gameday"])[:10] if len(played) else None,
        "model": {"game_model": "ratings_qb_logit_v1", "season_simulator": "season_sim_v2", "horizon_shrink_a": HORIZON_SHRINK_A,
                  "band_level": BAND_LEVEL, "n_sims": n_sims, "seed": season * 100 + week, "replacement_qb_epa": round(repl, 4),
                  "replacement_qb_pts": round(pts(repl), 2)},
        "record": {"wins": rec[0], "losses": rec[1], "ties": rec[2]},
        "rating": {"net_pts": round(pts(me["net"]), 2), "rank": int(me["rank"]), "offense_pts": round(pts(me["off"]), 2),
                   "qb_pts": round(pts(me["qb"]), 2), "defense_pts": round(pts(-me["dfn"]), 2), "qb": me["qb_name"],
                   "qb_starts": int(me["qb_starts"]) if pd.notna(me["qb_starts"]) else None},
        "outlook": {"expected_wins": round(float(s["expected_wins"]), 2), "wins_p10": q(0.10), "wins_p50": q(0.50), "wins_p90": q(0.90),
                    "p_playoffs": round(float(s["p_playoffs"]), 4), "p_division": round(float(s["p_division"]), 4),
                    "p_first_seed": round(float(s["p_first_seed"]), 4), "p_win_conf": round(float(s["p_win_conf"]), 4),
                    "p_win_sb": round(float(s["p_win_sb"]), 4),
                    "p_seed": [round(float(s[f"p_seed_{k}"]), 4) for k in range(1, 8)],
                    "wins_distribution": [{"wins": float(r.wins), "p": round(float(r.p), 4)} for r in dist.itertuples()]},
        "schedule": {"games": rows, "remaining": len(rows), "expected_wins": round(float(sum(r["p_win"] for r in rows)), 2),
                     "strength_pts": round(sos, 2), "strength_rank_hardest": sos_rank, "teams": int(len(sched))},
        "playoff_path": path,
    }


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def fmt_pct(x: float, digits: int = 0) -> str:
    return f"{100 * x:.{digits}f}%"


def markdown(r: dict) -> str:
    o, sch, rt, m = r["outlook"], r["schedule"], r["rating"], r["model"]
    rec = r["record"]
    record = f"{rec['wins']}-{rec['losses']}" + (f"-{rec['ties']}" if rec["ties"] else "")
    L = [f"# {r['name']}: opponent forecast, {r['season']} season, before week {r['as_of_week']}", "",
         f"Record {record}; data through {r['data_through']}. Rating model `{m['game_model']}`, simulator `{m['season_simulator']}` "
         f"({m['n_sims']:,} simulations, seed {m['seed']}). Generated {r['generated_utc']}.", "",
         "## Where the team stands", "",
         f"Overall rating {rt['net_pts']:+.1f} points per game against an average team, ranked {rt['rank']} of 32 "
         f"(offense incl. quarterback {rt['offense_pts'] + 0:+.1f}, defense {rt['defense_pts']:+.1f}; "
         f"quarterback {rt['qb']} contributes {rt['qb_pts']:+.1f}).", "",
         f"Expected final wins {o['expected_wins']:.1f} (80% range {o['wins_p10']:g} to {o['wins_p90']:g}); "
         f"playoffs {fmt_pct(o['p_playoffs'])}, division {fmt_pct(o['p_division'])}, first seed {fmt_pct(o['p_first_seed'], 1)}, "
         f"Super Bowl {fmt_pct(o['p_win_sb'], 1)}.", "",
         f"## Remaining schedule ({sch['remaining']} games)", "",
         "| Wk | Date | Opp | Site | Win % | 80% band | If opp QB out | If our QB out | Opp rating (rank) | Opp QB (starts) |",
         "|---:|:--|:--|:-:|---:|:--|---:|---:|:--|:--|"]
    for g in sch["games"]:
        qb = f"{g['opp_qb']} ({g['opp_qb_starts']})" if g["opp_qb"] else "n/a"
        L.append(f"| {g['week']} | {g['date']} | {g['opponent']} | {g['site']} | {fmt_pct(g['p_win'])} | "
                 f"{fmt_pct(g['band_lo'])}-{fmt_pct(g['band_hi'])} | {fmt_pct(g['p_win_if_opp_qb_replaced'])} | "
                 f"{fmt_pct(g['p_win_if_own_qb_replaced'])} | {g['opp_net_pts']:+.1f} ({g['opp_rank']}) | {qb} |")
    L += ["",
          f"Expected wins from the remaining games: {sch['expected_wins']:.1f} of {sch['remaining']}. "
          f"Remaining opponents average {sch['strength_pts']:+.1f} points per game; the {ordinal(sch['strength_rank_hardest'])}"
          f" hardest remaining schedule of {sch['teams']}.", "",
          "How to read it. *Win %* is the model's probability, pulled toward 50-50 for games further away (the model is "
          "overconfident on ratings frozen weeks earlier). *80% band* is where that probability will probably stand when the "
          "game arrives, from how far forecasts have moved in past seasons at this distance; it is not a confidence interval "
          "for the true probability. *If QB out* replaces that team's starter with a first-start replacement "
          f"({m['replacement_qb_pts']:+.1f} points per game against {rt['qb_pts']:+.1f} for this team's quarterback) and nothing else. "
          "Rest days and weather are not in the model. Ratings are in EPA per play x 62.7 plays: a rough points-per-game scale.", "",
          "## Playoff path (conditional on getting there)", "",
          "| Round | Chance of playing it | If we get there: most likely opponents (share of those cases; our win chance) |", "|:--|---:|:--|"]
    for p in r["playoff_path"]:
        opp = "; ".join(f"{x['team']} {fmt_pct(x['p_given_reach'])} ({fmt_pct(x['p_win_if_meet'])})" for x in p["opponents"][:4]
                        if x["p_win_if_meet"] is not None and x["p_given_reach"] is not None)
        L.append(f"| {p['round_name']} | {fmt_pct(p['p_reach'], 1)} | {opp or 'n/a'} |")
    L += ["", "Opponent odds are from the same simulations as the title odds (shared team-strength shocks, NFL reseeding, random tiebreaks "
          "as an approximation). They describe the model's uncertainty today and will change every week. A round the team rarely "
          "reaches has few simulated cases behind it, so its opponent shares are rough."]
    return "\n".join(L) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", required=True, help="team abbreviation, e.g. WAS")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None, help="as-of week (default: the next week with games left)")
    parser.add_argument("--sims", type=int, default=65536)
    parser.add_argument("--out", default=None, help="directory (default: outputs/opponents)")
    args = parser.parse_args()

    from src.simulation.live_bracket import load_inputs

    team = args.team.upper()
    if team not in TEAM_NAMES:
        raise SystemExit(f"unknown team {team}; use an abbreviation such as {', '.join(sorted(TEAM_NAMES)[:6])}")
    season = args.season or live_season()
    inp = load_inputs()
    week = args.week or next_week(inp.games, season)
    if week is None:
        raise SystemExit(f"{season}: no regular-season games left; the live bracket (bracket/) covers the playoffs")
    report = build_report(inp, team, season, week, args.sims)

    out = Path(args.out) if args.out else configured_path("outputs") / "opponents"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{team}_{season}_w{week:02d}"
    (out / f"{stem}.json").write_text(json.dumps(report, indent=1))
    (out / f"{stem}.md").write_text(markdown(report))
    pd.DataFrame(report["schedule"]["games"]).to_csv(out / f"{stem}_games.csv", index=False)
    print(markdown(report))
    print(f"wrote {out / stem}.json, .md, _games.csv")


if __name__ == "__main__":
    main()
