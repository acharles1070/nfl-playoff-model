"""Historical playoff futures odds (comparison target for the simulator).

Source: Covers "Sports Odds History" archive (BetMGM lines), playoff pages:

    /sportsoddshistory/nfl-post/?y=<season>&sa=nfl&a=<market>&p=pla

One page per season per market (sb = Super Bowl, afc, nfc). Each lists, for
every team, the American odds "prior to" each round (Wild Card, Divisional,
Conference Championship, Super Bowl) and flags the eventual winner.

These odds are used ONLY to score the simulator against the market, never as
model inputs. Raw HTML is cached under data/raw/odds (gitignored, personal
use); fetching is throttled and skips anything already cached.

Usage:
    python -m src.data.futures
"""

from __future__ import annotations

import html as htmllib
import re
import time
from collections.abc import Iterable

import numpy as np
import pandas as pd
import requests

from src.config import configured_path
from src.data.schedules import configured_seasons
from src.teams import validate_teams


BASE_URL = (
    "https://www.covers.com/sportsoddshistory/nfl-post/"
    "?y={season}&sa=nfl&a={market}&p=pla"
)
# Regular-season pages carry one price per team for every week ("prior to Wk N").
REG_URL = (
    "https://www.covers.com/sportsoddshistory/nfl-reg/"
    "?y={season}&sa=nfl&a={market}&p=reg"
)
USER_AGENT = "Mozilla/5.0 (personal research project)"
REQUEST_DELAY_SECONDS = 2.0

MARKETS = ("sb", "afc", "nfc")

ROUND_ORDER = ["WC", "DIV", "CON", "SB"]

# Covers column header (whitespace stripped) -> the project's round code
ROUND_HEADERS = {
    "WildCardRound": "WC",
    "DivisionalRound": "DIV",
    "ConferenceChampionships": "CON",
    "SuperBowl": "SB",
}

TEAM_NAMES = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
    "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Oakland Raiders": "LV",
    "Los Angeles Chargers": "LAC", "San Diego Chargers": "LAC",
    "Los Angeles Rams": "LAR", "St. Louis Rams": "LAR", "St Louis Rams": "LAR",
    "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO",
    "New York Giants": "NYG", "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN",
    "Washington Redskins": "WAS", "Washington Football Team": "WAS",
    "Washington Commanders": "WAS", "Washington": "WAS",
}


def _cache_path(season: int, market: str):
    return configured_path("raw") / "odds" / f"covers_post_{season}_{market}.html"


def _cache_path_reg(season: int, market: str):
    return configured_path("raw") / "odds" / f"covers_reg_{season}_{market}.html"


def fetch_page(season: int, market: str, *, refresh: bool = False, phase: str = "post") -> str:
    """Return the page HTML, downloading (throttled) only if not cached."""
    path = _cache_path(season, market) if phase == "post" else _cache_path_reg(season, market)

    if path.exists() and not refresh:
        return path.read_text(errors="ignore")

    time.sleep(REQUEST_DELAY_SECONDS)

    url = (BASE_URL if phase == "post" else REG_URL).format(season=season, market=market)

    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(response.text)

    return response.text


def american_to_implied(odds: float) -> float:
    """American odds -> raw implied probability (contains the vig)."""
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def _parse_odds(cell: str) -> float:
    cell = cell.replace("\xa0", " ").strip().upper()

    if cell in {"EVEN", "EV", "PK"}:
        return 100.0

    match = re.fullmatch(r"([+-])\s*(\d+)", cell)

    if not match:
        return np.nan

    value = float(match.group(2))

    return value if match.group(1) == "+" else -value


def _strip(cell: str) -> str:
    return re.sub(r"<[^>]+>", "", cell).strip()


def parse_page(page: str, season: int, market: str) -> pd.DataFrame:
    """Long table: one row per (team, round) with American odds."""
    page = htmllib.unescape(page)
    page = re.sub(r"<script.*?</script>|<style.*?</style>", "", page, flags=re.S)

    rows = [
        [_strip(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)]
        for r in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S)
    ]

    header_idx = next(
        (
            i
            for i, row in enumerate(rows)
            if row and any(c.replace("\n", "").replace(" ", "") in ROUND_HEADERS for c in row)
        ),
        None,
    )

    if header_idx is None:
        raise ValueError(f"No round header found for {season} {market}")

    rounds = [
        ROUND_HEADERS[c.replace("\n", "").replace(" ", "")]
        for c in rows[header_idx]
        if c.replace("\n", "").replace(" ", "") in ROUND_HEADERS
    ]

    out = []

    for row in rows[header_idx + 1:]:
        if not row or row[0] not in TEAM_NAMES:
            continue

        team = TEAM_NAMES[row[0]]
        cells = row[1:]

        # odds occupy the first len(rounds) cells; result is the final cell
        odds_cells = cells[: len(rounds)]
        result = cells[-1] if len(cells) > len(rounds) else ""
        won = "WINNER" in result.upper()

        for round_code, cell in zip(rounds, odds_cells):
            american = _parse_odds(cell)

            if np.isnan(american):
                continue

            out.append(
                {
                    "season": season,
                    "market": market,
                    "round_prior": round_code,
                    "team": team,
                    "american_odds": american,
                    "won_market": won,
                }
            )

    frame = pd.DataFrame(out)

    if frame.empty:
        raise ValueError(f"No odds parsed for {season} {market}")

    return frame


def parse_weekly_page(page: str, season: int, market: str) -> pd.DataFrame:
    """Long table of regular-season prices: one row per (team, week prior)."""
    page = htmllib.unescape(page)
    page = re.sub(r"<script.*?</script>|<style.*?</style>", "", page, flags=re.S)

    rows = [
        [_strip(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)]
        for r in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S)
    ]

    week_re = re.compile(r"^Wk\s*(\d+)$")
    header_idx = next(
        (i for i, row in enumerate(rows) if row and all(week_re.match(c) for c in row if c) and any(row)),
        None,
    )

    if header_idx is None:
        raise ValueError(f"No week header found for {season} {market}")

    weeks = [int(week_re.match(c).group(1)) for c in rows[header_idx] if c]

    out = []

    for row in rows[header_idx + 1:]:
        if not row or row[0] not in TEAM_NAMES:
            continue

        team = TEAM_NAMES[row[0]]
        cells = row[1:]
        odds_cells = cells[: len(weeks)]
        won = "WINNER" in (cells[-1] if len(cells) > len(weeks) else "").upper()

        for week, cell in zip(weeks, odds_cells):
            american = _parse_odds(cell)

            if np.isnan(american):
                continue

            out.append(
                {"season": season, "market": market, "week_prior": week,
                 "team": team, "american_odds": american, "won_market": won}
            )

    frame = pd.DataFrame(out)

    if frame.empty:
        raise ValueError(f"No weekly odds parsed for {season} {market}")

    return frame


def add_weekly_probabilities(odds: pd.DataFrame) -> pd.DataFrame:
    out = odds.copy()
    out["p_implied"] = out["american_odds"].map(american_to_implied)

    keys = ["season", "market", "week_prior"]
    out["overround"] = out.groupby(keys)["p_implied"].transform("sum")
    out["p_market"] = out["p_implied"] / out["overround"]
    return out


def collect_weekly_futures(
    seasons: Iterable[int] | None = None,
    markets: Iterable[str] = ("sb",),
    *,
    refresh_seasons: Iterable[int] = (),
) -> pd.DataFrame:
    """Weekly regular-season prices. The live season is always re-fetched."""
    from src.data.schedules import live_season

    seasons = sorted(set(seasons or configured_seasons()))
    always = {int(s) for s in refresh_seasons} | {live_season()}

    frames = []

    for season in seasons:
        if season < 2009:        # Covers weekly pages are used from 2009 on
            continue

        for market in markets:
            try:
                page = fetch_page(season, market, phase="reg", refresh=season in always)
                frames.append(parse_weekly_page(page, season, market))
            except Exception as exc:
                print(f"  ! {season} {market}: {type(exc).__name__}: {exc}")

    return add_weekly_probabilities(pd.concat(frames, ignore_index=True))


def add_probabilities(odds: pd.DataFrame) -> pd.DataFrame:
    """Raw implied and de-vigged (normalized within each market-round)."""
    out = odds.copy()

    out["p_implied"] = out["american_odds"].map(american_to_implied)

    keys = ["season", "market", "round_prior"]

    out["overround"] = out.groupby(keys)["p_implied"].transform("sum")
    out["p_market"] = out["p_implied"] / out["overround"]

    return out


def collect_futures(
    seasons: Iterable[int] | None = None,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    seasons = sorted(set(seasons or configured_seasons()))

    frames = []

    for season in seasons:
        for market in MARKETS:
            try:
                page = fetch_page(season, market, refresh=refresh)
                frames.append(parse_page(page, season, market))
            except Exception as exc:  # keep going; report gaps at the end
                print(f"  ! {season} {market}: {type(exc).__name__}: {exc}")

    if not frames:
        raise RuntimeError("No futures pages could be collected.")

    odds = pd.concat(frames, ignore_index=True)
    validate_teams(odds["team"], source="futures odds")

    return add_probabilities(odds)


def validate_against_results(odds: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Check each 'WINNER' flag against the actual champion in games.parquet."""
    post = games.loc[games["is_postseason"] & games["is_played"]].copy()

    post["winner"] = np.where(
        post["home_score"] > post["away_score"],
        post["home_team"],
        post["away_team"],
    )

    sb = post.loc[post["game_type"].eq("SB")].set_index("season")["winner"]

    rows = []

    for season, frame in odds.groupby("season"):
        flagged = frame.loc[frame["won_market"] & frame["market"].eq("sb"), "team"].unique()
        rows.append(
            {
                "season": season,
                "sb_winner_actual": sb.get(season),
                "sb_winner_flagged": ",".join(flagged) or None,
                "match": bool(len(flagged) == 1 and flagged[0] == sb.get(season)),
            }
        )

    return pd.DataFrame(rows)


def save_weekly() -> pd.DataFrame:
    weekly = collect_weekly_futures()
    path = configured_path("processed") / "futures_weekly_sb.parquet"
    weekly.to_parquet(path, index=False)
    print(f"Saved: {path} ({len(weekly)} price points)")
    return weekly


if __name__ == "__main__":
    odds = collect_futures()

    path = configured_path("processed") / "futures_odds.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    odds.to_parquet(path, index=False)
    print(f"Saved: {path} ({len(odds)} rows)")

    games = pd.read_parquet(configured_path("processed") / "games.parquet")
    check = validate_against_results(odds, games)

    print()
    print("Super Bowl winner flag vs actual result:")
    print(check.to_string(index=False))

    print()
    print("Coverage (teams with a price, per season/market/round):")
    print(
        odds.groupby(["season", "market", "round_prior"])
        .size()
        .unstack(["market", "round_prior"])
        .to_string()
    )

    print()
    print("Overround by round (sum of implied probabilities):")
    print(
        odds.drop_duplicates(["season", "market", "round_prior"])
        .groupby(["market", "round_prior"])["overround"]
        .describe()[["count", "min", "mean", "max"]]
        .round(3)
        .to_string()
    )

    print()
    weekly = save_weekly()
    print("\nWeekly regular-season Super Bowl prices: weeks priced per season, winner flags")
    print(
        weekly.groupby("season")
        .agg(weeks=("week_prior", "nunique"), teams=("team", "nunique"), winners=("won_market", lambda s: s.groupby(weekly.loc[s.index, "team"]).first().sum()))
        .T.to_string()
    )
