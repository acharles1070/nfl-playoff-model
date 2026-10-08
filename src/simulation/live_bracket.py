"""Playoff bracket projection: live for the season in progress, replayable for any past season.

    python -m src.simulation.live_bracket --out bracket/data/bracket.json

The output depends on how far the season is:

  regular_season  No playoff schedule exists yet. Simulate the rest of the season (the same 65,536-season
                  simulator whose odds are logged in ledger/title_odds.csv), report every team's chance at
                  each seed, and draw the projected field and the bracket the favorites would produce.
  playoffs        Wild Card matchups exist. The field is the real one, finished games carry their scores,
                  unplayed games carry the model's probability, and odds are re-simulated from the survivors.
  complete        The Super Bowl has been played.

Every game probability is the fitted game model's single-game probability from ratings that use only earlier
games, so a replay of a past season (src.simulation.bracket_replay) is leak-free by construction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm

from src.config import configured_path, project_path
from src.data.schedules import live_season
from src.features.ratings import RatingParams, attach_ratings_to_matchups, build_team_ratings
from src.simulation.backtest import FIRST_MODEL_SEASON, GameModel, build_table
from src.simulation.bracket import simulate_bracket
from src.simulation.divisions import CONFERENCE_OF, DIVISIONS
from src.simulation.season import SeasonState, simulate_season
from src.simulation.season_backtest import league_state, snapshot_at
from src.simulation.seeding import reconstruct_field, regular_season_table
from src.teams import TEAM_NAMES

ROUNDS = ["WC", "DIV", "CON", "SB"]
ROUND_GAMES = {"WC": 6, "DIV": 4, "CON": 2, "SB": 1}
ROUND_NAMES = {"WC": "Wild Card", "DIV": "Divisional", "CON": "Conference championship", "SB": "Super Bowl"}
CONFS = ("AFC", "NFC")
SCHEMA = 1


# ---------------------------------------------------------------------------
# inputs and model
# ---------------------------------------------------------------------------

@dataclass
class Inputs:
    games: pd.DataFrame
    team_game: pd.DataFrame
    qb_game: pd.DataFrame
    matchups: pd.DataFrame
    params: RatingParams
    ratings: pd.DataFrame          # pregame rating rows for every game, played or scheduled


def load_inputs(games: pd.DataFrame | None = None) -> Inputs:
    processed = configured_path("processed")
    games = pd.read_parquet(processed / "games.parquet") if games is None else games
    # every table is restricted to games flagged played, so a schedule edited to emulate an earlier date (tests,
    # replays) cannot leak later results through play-by-play, quarterback or matchup rows
    played = set(games.loc[games["is_played"], "game_id"])
    team_game = pd.read_parquet(processed / "team_game_epa.parquet")
    qb_game = pd.read_parquet(processed / "qb_game.parquet")
    matchups = pd.read_parquet(processed / "historical_matchups_oa.parquet")
    team_game, qb_game, matchups = (t.loc[t["game_id"].isin(played)] for t in (team_game, qb_game, matchups))
    params = RatingParams()
    ratings = build_team_ratings(team_game, games, params, qb_game)
    return Inputs(games, team_game, qb_game, matchups, params, ratings)


def fit_model(inp: Inputs, season: int, before_week: int) -> GameModel:
    """Game model trained on games strictly before (season, before_week)."""
    m = inp.matchups
    mask = (m["season"] >= FIRST_MODEL_SEASON) & ((m["season"] < season) | ((m["season"] == season) & (m["week"] < before_week)))
    return GameModel(attach_ratings_to_matchups(m.loc[mask], inp.ratings))


def latest_snapshot(inp: Inputs, season: int, week: int) -> pd.DataFrame:
    """Each team's rating state going into its next game, as of before `week`.

    When no regular-season games remain (late season, playoffs), a dummy week in which every team plays once
    supplies the state after every game before `week`.
    """
    games = inp.games
    # schedule rows from `week` on, played or not: in a replay of a finished season they are all flagged played
    has_future = (games["season"].eq(season) & games["game_type"].eq("REG") & games["week"].ge(week)).any()
    if not has_future:
        # no regular-season games left: a dummy game at `week` in which every team plays once supplies the state after
        # every game BEFORE `week`. Pass week=99 in the live playoffs (everything so far counts); a replay passes the
        # first playoff week so that nothing from the playoffs leaks into pre-playoff ratings.
        teams = sorted(CONFERENCE_OF)
        dummy = pd.DataFrame({"game_id": [f"{season}_{week:02d}_DUMMY_{i}" for i in range(len(teams) // 2)], "season": season, "week": week,
                              "game_type": "REG", "home_team": teams[0::2], "away_team": teams[1::2], "is_played": False,
                              "is_postseason": False, "location": "Home"})
        games = pd.concat([games, dummy], ignore_index=True)
    return snapshot_at(inp.team_game, inp.qb_game, games, season, week, inp.params)


class Probs:
    """Single-game win probabilities from the game model.

    A game with a schedule row uses that game's pregame ratings; a hypothetical matchup uses each team's
    latest state. Both are the marginal model probability Phi(m), before any strength shocks.
    """

    def __init__(self, inp: Inputs, model: GameModel, latest: pd.DataFrame):
        self.inp, self.model = inp, model
        self.table = build_table(latest, model)
        self._by_game = inp.ratings.set_index("game_id")
        self._cache: dict[str, tuple] = {}

    def _scheduled(self, gid: str, home: str, away: str, neutral: bool) -> float | None:
        if gid not in self._by_game.index:
            return None
        rows = self._by_game.loc[[gid]].reset_index()
        rows = rows[rows["team"].isin([home, away])]
        if len(rows) != 2:
            return None
        t = build_table(rows, self.model)
        return float(norm.cdf((t.m_neutral if neutral else t.m_home)[(home, away)]))

    def game(self, home: str, away: str, *, game_id: str | None = None, neutral: bool = False) -> float:
        if game_id:
            p = self._scheduled(game_id, home, away, neutral)
            if p is not None:
                return p
        return float(norm.cdf((self.table.m_neutral if neutral else self.table.m_home)[(home, away)]))


# ---------------------------------------------------------------------------
# phase and field
# ---------------------------------------------------------------------------

def postseason(games: pd.DataFrame, season: int) -> pd.DataFrame:
    return games.loc[games["season"].eq(season) & games["is_postseason"]]


def phase_info(games: pd.DataFrame, season: int) -> dict:
    """Which phase the season is in, and (for the playoffs) the first round with games left."""
    post = postseason(games, season)
    wc = post.loc[post["game_type"].eq("WC")]
    if len(wc) < ROUND_GAMES["WC"]:
        return {"phase": "regular_season", "round_now": None}
    for rnd in ROUNDS:
        g = post.loc[post["game_type"].eq(rnd)]
        if len(g) < ROUND_GAMES[rnd] or not g["is_played"].all():
            return {"phase": "playoffs", "round_now": rnd}
    return {"phase": "complete", "round_now": None}


def _division_of(team: str) -> tuple[str, str]:
    for conf, divs in DIVISIONS.items():
        for name, teams in divs.items():
            if team in teams:
                return conf, name
    raise KeyError(team)


def field_from_wildcard_round(games: pd.DataFrame, season: int) -> dict[str, list[str]]:
    """Seeds 1-7 per conference from the Wild Card schedule and the regular-season standings.

    The three hosts are the 2-4 seeds and the bye team is the winner of the one division with no host. A visitor's
    seed follows from the pairing (2 v 7, 3 v 6, 4 v 5), so the only unknown is the order of the hosts. The NFL
    tiebreakers are not implemented, so ties in record are resolved by consistency: of the six host orders, keep the one
    that violates the records least on BOTH sides (a higher seed should not have a strictly worse record than a lower
    one), then break remaining ties by point differential.
    """
    wc = postseason(games, season).query("game_type == 'WC'")
    table = regular_season_table(games, season)
    rank = table.sort_values(["win_pct", "point_diff"], ascending=False).index.tolist()
    order = {t: i for i, t in enumerate(rank)}
    pct = table["win_pct"].to_dict()
    field: dict[str, list[str]] = {}

    def violations(seeds: list[str]) -> int:
        """Pairs (higher seed, lower seed) within hosts and within visitors whose records are strictly reversed."""
        hosts, visitors = seeds[1:4], seeds[4:7]
        bad = 0
        for group in (hosts, visitors):
            bad += sum(1 for i in range(3) for j in range(i + 1, 3) if pct[group[i]] < pct[group[j]] - 1e-9)
        return bad

    for conf in CONFS:
        g = wc.loc[wc["home_team"].map(CONFERENCE_OF).eq(conf)]
        if len(g) != 3:
            raise ValueError(f"{season} {conf}: expected 3 Wild Card games, found {len(g)}")
        visitor_of = dict(zip(g["home_team"], g["away_team"]))
        host_divisions = {_division_of(h)[1] for h in visitor_of}
        bye_division = next(name for name in DIVISIONS[conf] if name not in host_divisions)
        bye = min(DIVISIONS[conf][bye_division], key=order.get)

        candidates = []
        for hosts in permutations(visitor_of):
            seeds = [bye, *hosts, visitor_of[hosts[2]], visitor_of[hosts[1]], visitor_of[hosts[0]]]
            candidates.append((violations(seeds), sum(order[t] * w for t, w in zip(seeds, range(7, 0, -1))), seeds))
        field[conf] = min(candidates, key=lambda c: (c[0], c[1]))[2]
    return field


def playoff_field(games: pd.DataFrame, season: int) -> dict[str, list[str]]:
    """The real field. Prefers the validated record-consistency reconstruction once Divisional games exist."""
    post = postseason(games, season)
    if post["game_type"].isin(["DIV"]).any() and post["game_type"].isin(["CON", "SB"]).any():
        try:
            raw = reconstruct_field(games, season)
            return {("AFC" if CONFERENCE_OF[seeds[0]] == "AFC" else "NFC"): seeds for seeds in raw.values()}
        except Exception:                                  # fall back to the Wild Card structure
            pass
    return field_from_wildcard_round(games, season)


def projected_field(sim: pd.DataFrame, n_seeds: int = 7) -> dict[str, list[str]]:
    """The lineup that maximizes the expected number of correct seeds (assignment problem), per conference."""
    out = {}
    for conf in CONFS:
        sub = sim.loc[sim["team"].map(CONFERENCE_OF).eq(conf)]
        P = sub[[f"p_seed_{k}" for k in range(1, n_seeds + 1)]].to_numpy()
        rows, cols = linear_sum_assignment(-P)
        seeds = [None] * n_seeds
        for r, c in zip(rows, cols):
            seeds[c] = sub["team"].iloc[r]
        out[conf] = seeds
    return out


# ---------------------------------------------------------------------------
# the bracket
# ---------------------------------------------------------------------------

def _kickoff(row) -> str | None:
    d, t = row.get("gameday"), row.get("gametime")
    return f"{str(d)[:10]} {t}" if pd.notna(d) and pd.notna(t) else None


def make_game(rnd: str, conf: str | None, a: str, b: str, seed_of: dict[str, int], post: pd.DataFrame, probs: Probs) -> dict:
    """One game: the real row if the schedule has it, otherwise a projection."""
    found = post.loc[post["game_type"].eq(rnd) & ((post["home_team"].eq(a) & post["away_team"].eq(b)) | (post["home_team"].eq(b) & post["away_team"].eq(a)))]
    if len(found):
        row = found.iloc[0]
        home, away, gid = row["home_team"], row["away_team"], row["game_id"]
    else:
        row, gid = None, None
        home, away = (a, b) if rnd == "SB" or seed_of.get(a, 99) <= seed_of.get(b, 99) else (b, a)
    neutral = rnd == "SB"
    p = probs.game(home, away, game_id=gid, neutral=neutral)
    favorite = home if p >= 0.5 else away
    game = {"id": gid, "round": rnd, "conf": conf, "home": home, "away": away,
            "home_seed": seed_of.get(home), "away_seed": seed_of.get(away), "p_home": round(p, 4), "favorite": favorite,
            "status": "projected", "kickoff": None, "home_score": None, "away_score": None, "winner": favorite}
    if row is not None:
        game["kickoff"] = _kickoff(row)
        game["status"] = "final" if bool(row["is_played"]) else "scheduled"
        if game["status"] == "final":
            game["home_score"], game["away_score"] = int(row["home_score"]), int(row["away_score"])
            game["winner"] = home if row["home_score"] > row["away_score"] else away
            game["upset"] = game["winner"] != favorite
    return game


def build_bracket(field: dict[str, list[str]], post: pd.DataFrame, probs: Probs) -> dict:
    """Fill the bracket with real results where they exist and the model's favorite elsewhere."""
    out: dict = {"rounds": {c: {"WC": [], "DIV": [], "CON": []} for c in field}, "alive": {c: {} for c in field}}
    champs: dict[str, str] = {}

    for conf, seeds in field.items():
        seed_of = {t: i + 1 for i, t in enumerate(seeds)}
        wc = [make_game("WC", conf, seeds[h], seeds[7 - h], seed_of, post, probs) for h in (1, 2, 3)]      # 2v7, 3v6, 4v5
        out["rounds"][conf]["WC"] = wc
        alive_div = sorted([seeds[0], *[g["winner"] for g in wc]], key=seed_of.get)
        out["alive"][conf]["DIV"] = alive_div

        div = [make_game("DIV", conf, alive_div[0], alive_div[3], seed_of, post, probs),
               make_game("DIV", conf, alive_div[1], alive_div[2], seed_of, post, probs)]
        out["rounds"][conf]["DIV"] = div
        alive_con = sorted([g["winner"] for g in div], key=seed_of.get)
        out["alive"][conf]["CON"] = alive_con

        con = make_game("CON", conf, alive_con[0], alive_con[1], seed_of, post, probs)
        out["rounds"][conf]["CON"] = [con]
        champs[conf] = con["winner"]

    seed_all = {t: i + 1 for seeds in field.values() for i, t in enumerate(seeds)}
    sb = make_game("SB", None, champs["AFC"], champs["NFC"], seed_all, post, probs)
    out["SB"] = sb
    out["champion"] = sb["winner"]
    return out


# ---------------------------------------------------------------------------
# odds
# ---------------------------------------------------------------------------

def records(games: pd.DataFrame, season: int) -> dict[str, tuple[int, int, int]]:
    reg = games.loc[games["season"].eq(season) & games["game_type"].eq("REG") & games["is_played"]]
    rec: dict[str, list[int]] = {}
    for g in reg.itertuples(index=False):
        h = (g.home_score > g.away_score) - (g.home_score < g.away_score)
        for team, res in ((g.home_team, h), (g.away_team, -h)):
            r = rec.setdefault(team, [0, 0, 0])
            r[0 if res > 0 else 1 if res < 0 else 2] += 1
    return {t: tuple(v) for t, v in rec.items()}


def season_odds(inp: Inputs, season: int, week: int, n_sims: int) -> tuple[pd.DataFrame, GameModel, pd.DataFrame]:
    """Season simulation as of before `week`, with seed probabilities. Same seed rule as the logged title odds."""
    model = fit_model(inp, season, week)
    latest = latest_snapshot(inp, season, week)
    wins, remaining, _ = league_state(inp.games, season, week)
    table = build_table(latest, model)
    sim = simulate_season(SeasonState(wins, remaining, 7), table, n_sims=n_sims, seed=season * 100 + week, seed_probs=True)
    return sim, model, latest


def playoff_odds(field, bracket, probs: Probs, round_now: str | None, n_draws: int, seed: int) -> pd.DataFrame:
    """Odds from the survivors of the real bracket (exact enumeration, shared strength shocks)."""
    if round_now in (None, "SB"):
        sb = bracket["SB"]
        rows = []
        for conf, seeds in field.items():
            for i, t in enumerate(seeds):
                p = (sb["p_home"] if t == sb["home"] else 1 - sb["p_home"]) if t in (sb["home"], sb["away"]) else 0.0
                done = sb["status"] == "final"
                if done:
                    p = 1.0 if t == sb["winner"] else 0.0
                rows.append({"conference": conf, "seed": i + 1, "team": t, "p_reach_div": 1.0 if t in bracket["alive"][conf]["DIV"] else 0.0,
                             "p_reach_con": 1.0 if t in bracket["alive"][conf]["CON"] else 0.0,
                             "p_win_conf": 1.0 if t in (sb["home"], sb["away"]) else 0.0, "p_win_sb": p})
        return pd.DataFrame(rows)
    start = round_now
    alive = None if start == "WC" else {c: bracket["alive"][c][start] for c in field}
    return simulate_bracket(field, probs.table, n_draws=n_draws, seed=seed, start=start, alive=alive)


# ---------------------------------------------------------------------------
# payload
# ---------------------------------------------------------------------------

def _base_row(team: str, rec: dict) -> dict:
    conf, div = _division_of(team)
    w, l, tie = rec.get(team, (0, 0, 0))
    return {"team": team, "name": TEAM_NAMES[team], "conf": conf, "division": f"{conf} {div}", "wins": w,
            "record": f"{w}-{l}" + (f"-{tie}" if tie else "")}


def _season_rows(games, season, sim, prev) -> list[dict]:
    """Regular season: every number comes from the season simulation."""
    rec = records(games, season)
    rows = []
    for r in sim.itertuples(index=False):
        row = _base_row(r.team, rec)
        row.update({"expected_wins": round(float(r.expected_wins), 2), "p_playoffs": round(float(r.p_playoffs), 4),
                    "p_division": round(float(r.p_division), 4), "p_first_seed": round(float(r.p_first_seed), 4),
                    "p_win_conf": round(float(r.p_win_conf), 4), "p_win_sb": round(float(r.p_win_sb), 4),
                    "p_seed": [round(float(getattr(r, f"p_seed_{k}")), 4) for k in range(1, 8)]})
        if prev is not None and r.team in prev.index:
            row["d_win_sb"] = round(float(r.p_win_sb - prev.loc[r.team, "p_win_sb"]), 4)
        rows.append(row)
    return rows


def _playoff_rows(games, season, field, odds, bracket, round_now) -> list[dict]:
    """Playoffs: the field is real, so playoff and seed probabilities are 0 or 1; odds come from the survivors.

    The odds table only lists teams still alive at the start of the current round; everyone else has been eliminated
    (or never made it) and shows 0.
    """
    rec = records(games, season)
    seed_of = {t: i + 1 for seeds in field.values() for i, t in enumerate(seeds)}
    po = odds.set_index("team")
    rows = []
    for team in sorted(CONFERENCE_OF):
        row = _base_row(team, rec)
        in_field = team in seed_of
        alive = in_field and team in po.index
        row.update({"in_field": in_field, "eliminated": in_field and not alive, "expected_wins": float(row["wins"]),
                    "p_playoffs": 1.0 if in_field else 0.0, "p_division": 1.0 if in_field and seed_of[team] <= 4 else 0.0,
                    "p_first_seed": 1.0 if seed_of.get(team) == 1 else 0.0,
                    "p_win_conf": round(float(po.loc[team, "p_win_conf"]), 4) if alive else 0.0,
                    "p_win_sb": round(float(po.loc[team, "p_win_sb"]), 4) if alive else 0.0,
                    "p_seed": [1.0 if seed_of.get(team) == k else 0.0 for k in range(1, 8)]})
        if in_field:
            row["seed"] = seed_of[team]
        rows.append(row)
    return rows


def build_payload(inp: Inputs, season: int, n_sims: int = 65536, n_draws: int = 8192) -> dict:
    games = inp.games
    info = phase_info(games, season)
    phase = info["phase"]
    played = games.loc[games["season"].eq(season) & games["is_played"]]
    last = played.sort_values(["gameday", "game_id"]).iloc[-1] if len(played) else None

    reg = games.loc[games["season"].eq(season) & games["game_type"].eq("REG")]
    unplayed = reg.loc[~reg["is_played"]]
    next_week = int(unplayed["week"].min()) if len(unplayed) else 19

    post = postseason(games, season)

    if phase == "regular_season":
        sim, model, latest = season_odds(inp, season, next_week, n_sims)
        prev = None
        if next_week > 1:
            prev = season_odds(inp, season, next_week - 1, max(n_sims // 4, 8192))[0].set_index("team")
        probs = Probs(inp, model, latest)
        field, kind = projected_field(sim), "projected"
        bracket = build_bracket(field, post.iloc[0:0], probs)
        teams = _season_rows(games, season, sim, prev)
        label = f"Projection before week {next_week}" if len(unplayed) else "Regular season complete; waiting for the playoff schedule"
    else:
        model = fit_model(inp, season, 99)
        probs = Probs(inp, model, latest_snapshot(inp, season, 99))
        field, kind = playoff_field(games, season), "actual"
        bracket = build_bracket(field, post, probs)
        odds = playoff_odds(field, bracket, probs, info["round_now"], n_draws, season * 100 + 99)
        teams = _playoff_rows(games, season, field, odds, bracket, info["round_now"])
        label = f"Playoffs: {ROUND_NAMES[info['round_now']]} round" if phase == "playoffs" else "Season complete"
    p_seed = {r["team"]: r["p_seed"] for r in teams}

    return {
        "schema": SCHEMA, "season": season, "playoffs_year": season + 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_through": None if last is None else {"game_id": last["game_id"], "date": str(last["gameday"])[:10], "week": int(last["week"])},
        "phase": phase, "round_now": info["round_now"], "label": label, "n_sims": n_sims,
        "model": {"game_model": "ratings_qb_logit_v1", "season_simulator": "season_sim_v2"},
        "field_kind": kind,
        "field": {c: [{"seed": i + 1, "team": t, "name": TEAM_NAMES[t], "p_seed": p_seed[t][i]} for i, t in enumerate(seeds)] for c, seeds in field.items()},
        "bracket": {"AFC": bracket["rounds"]["AFC"], "NFC": bracket["rounds"]["NFC"], "SB": bracket["SB"], "champion": bracket["champion"]},
        "teams": sorted(teams, key=lambda r: -r["p_win_sb"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--out", default="bracket/data/bracket.json", help="relative to the project root unless absolute")
    parser.add_argument("--sims", type=int, default=65536)
    args = parser.parse_args()

    season = args.season or live_season()
    payload = build_payload(load_inputs(), season, n_sims=args.sims)
    out = Path(args.out)
    if not out.is_absolute():
        out = project_path(*out.parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"{season}: {payload['label']} | data through {payload['data_through']['date']} | {out} ({out.stat().st_size / 1024:.0f} KB)")
    for c in CONFS:
        print(c, " > ".join(f"{x['seed']}:{x['team']}" for x in payload["field"][c]))
    print("favorite:", payload["bracket"]["champion"], "| top odds:", ", ".join(f"{t['team']} {t['p_win_sb']:.1%}" for t in payload["teams"][:5]))


if __name__ == "__main__":
    main()
