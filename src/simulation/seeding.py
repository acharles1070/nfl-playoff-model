"""Reconstruct playoff fields and seeds from the schedule.

nflverse schedules carry no seeds, so they are rebuilt from structure:

  * conference  = connected component of the WC/DIV/CON game graph
  * byes        = teams that appear in the Divisional round but not Wild Card
                  (seed 1; also seed 2 in the pre-2020 six-team format)
  * WC hosts    = the division winners seeded 2-4 (3-4 pre-2020), ordered by
                  regular-season record, ties by point differential; each
                  host's opponent is the mirror seed (2v7, 3v6, 4v5)

validate_seeds() replays the real results through the NFL's reseeding rules and
checks that every Divisional/Conference matchup and host came out right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _components(edges: list[tuple[str, str]]) -> list[set[str]]:
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)

    groups: dict[str, set[str]] = {}
    for team in parent:
        groups.setdefault(find(team), set()).add(team)

    return list(groups.values())


def regular_season_table(games: pd.DataFrame, season: int) -> pd.DataFrame:
    reg = games.loc[
        games["season"].eq(season) & ~games["is_postseason"] & games["is_played"]
    ]

    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        part = pd.DataFrame(
            {
                "team": reg[f"{side}_team"],
                "pf": reg[f"{side}_score"],
                "pa": reg[f"{other}_score"],
            }
        )
        part["win"] = (part["pf"] > part["pa"]).astype(float) + 0.5 * (part["pf"] == part["pa"])
        rows.append(part)

    table = pd.concat(rows).groupby("team").agg(
        wins=("win", "sum"), games=("win", "size"), pf=("pf", "sum"), pa=("pa", "sum")
    )
    table["win_pct"] = table["wins"] / table["games"]
    table["point_diff"] = table["pf"] - table["pa"]
    return table


def _candidate_orders(items: list[str]) -> list[list[str]]:
    """Record order first, then every other permutation (tiebreakers such as
    head-to-head cannot be rebuilt from the schedule, so I search)."""
    from itertools import permutations

    rest = [list(p) for p in permutations(items) if list(p) != items]
    return [items, *rest]


def reconstruct_field(games: pd.DataFrame, season: int) -> dict[str, list[str]]:
    """{'AFC-like conference label': [team seeded 1, 2, ...]} (labels 'A','B').

    Seed ORDER among byes and among Wild Card hosts starts from regular-season
    record and is then corrected, if needed, to the ordering whose reseeded
    Divisional and Conference matchups match what really happened.
    """
    post = games.loc[games["season"].eq(season) & games["is_postseason"]]

    wc = post.loc[post["game_type"].eq("WC")]
    div = post.loc[post["game_type"].eq("DIV")]
    pre_sb = post.loc[post["game_type"].isin(["WC", "DIV", "CON"])]

    groups = _components(list(zip(pre_sb["home_team"], pre_sb["away_team"])))

    if len(groups) != 2:
        raise ValueError(f"{season}: expected 2 conferences, found {len(groups)}")

    record = regular_season_table(games, season)
    rank = record.sort_values(["win_pct", "point_diff"], ascending=False).index.tolist()
    order = {t: i for i, t in enumerate(rank)}

    fields = {}

    for label, members in zip("AB", sorted(groups, key=lambda g: sorted(g)[0])):
        wc_games = wc.loc[wc["home_team"].isin(members)]
        wc_teams = set(wc_games["home_team"]) | set(wc_games["away_team"])
        byes = sorted(
            (set(div["home_team"]) | set(div["away_team"])) & members - wc_teams,
            key=lambda t: order[t],
        )

        hosts = sorted(wc_games["home_team"], key=lambda t: order[t])
        away_of = dict(zip(wc_games["home_team"], wc_games["away_team"]))

        # Among orderings that reproduce the real reseeded matchups, prefer the
        # one whose seeds best respect regular-season records in EVERY block
        # (byes, hosts, wild cards). The mirror pairing (2v7, 3v6, 4v5) ties the
        # host order to the wild-card order, so a tie between two hosts is
        # broken by the records of the wild cards they were paired with.
        def inversions(block, key):
            return sum(
                1 for a, b in zip(block, block[1:]) if record.loc[a, key] < record.loc[b, key]
            )

        best = None

        for byes_order in _candidate_orders(byes):
            for hosts_order in _candidate_orders(hosts):
                away_order = [away_of[h] for h in reversed(hosts_order)]
                seeds = list(byes_order) + hosts_order + away_order
                problems = validate_seeds(games, season, {label: seeds})

                cost = (
                    len(problems),
                    sum(inversions(b, "win_pct") for b in (byes_order, hosts_order, away_order)),
                    sum(inversions(b, "point_diff") for b in (byes_order, hosts_order, away_order)),
                )

                if best is None or cost < best[1]:
                    best = (seeds, cost)

        fields[label] = best[0]

    return fields


def reseed_pairs(alive: list[tuple[int, str]]) -> list[tuple[str, str]]:
    """NFL reseeding with 4 alive: 1 v 4, 2 v 3 (home = better seed)."""
    ordered = sorted(alive)
    return [(ordered[0][1], ordered[3][1]), (ordered[1][1], ordered[2][1])]


def validate_seeds(games: pd.DataFrame, season: int, fields: dict[str, list[str]]) -> list[str]:
    """Replay actual winners through reseeding; return a list of mismatches."""
    post = games.loc[games["season"].eq(season) & games["is_postseason"] & games["is_played"]].copy()
    post["winner"] = np.where(post["home_score"] > post["away_score"], post["home_team"], post["away_team"])

    problems = []

    for label, seeds in fields.items():
        seed_of = {t: i + 1 for i, t in enumerate(seeds)}
        members = set(seeds)

        wc = post.loc[post["game_type"].eq("WC") & post["home_team"].isin(members)]
        survivors = [t for t in seeds if t not in set(wc["home_team"]) | set(wc["away_team"])]
        survivors += list(wc["winner"])

        # WC matchups must be the mirror seeds with the better seed hosting:
        # 2v7, 3v6, 4v5 (7 teams) and 3v6, 4v5 (6 teams) all sum to 9.
        for g in wc.itertuples():
            if seed_of[g.home_team] + seed_of[g.away_team] != 9:
                problems.append(f"{season} {label} WC pairing {g.home_team}v{g.away_team}")
            if seed_of[g.home_team] > seed_of[g.away_team]:
                problems.append(f"{season} {label} WC host {g.home_team}")

        expected = reseed_pairs([(seed_of[t], t) for t in survivors])
        actual = post.loc[post["game_type"].eq("DIV") & post["home_team"].isin(members)]
        actual_pairs = {(g.home_team, g.away_team) for g in actual.itertuples()}

        for pair in expected:
            if pair not in actual_pairs:
                problems.append(f"{season} {label} DIV expected {pair}, actual {sorted(actual_pairs)}")

        div_winners = list(actual["winner"])
        con = post.loc[post["game_type"].eq("CON") & post["home_team"].isin(members)]
        hi, lo = sorted(div_winners, key=lambda t: seed_of[t])
        if not ((con["home_team"].iloc[0], con["away_team"].iloc[0]) == (hi, lo)):
            problems.append(f"{season} {label} CON expected host {hi}")

    return problems


if __name__ == "__main__":
    from src.config import configured_path

    games = pd.read_parquet(configured_path("processed") / "games.parquet")

    total = 0
    for season in range(2009, 2026):
        fields = reconstruct_field(games, season)
        problems = validate_seeds(games, season, fields)
        total += len(problems)
        print(season, {k: v[:3] + ['...'] for k, v in fields.items()}, "OK" if not problems else problems)

    print(f"\nTotal mismatches across 17 seasons: {total}")
