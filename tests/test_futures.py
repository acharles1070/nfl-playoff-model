"""Futures-odds parsing and de-vig math (no network)."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.futures import (
    _parse_odds,
    add_probabilities,
    american_to_implied,
    parse_page,
)


PAGE = """
<table><tr><th>Team</th><th>Playoffs,prior to...</th><th>Result</th></tr>
<tr><td>Wild Card Round</td><td>Divisional Round</td><td>Super Bowl</td></tr>
<tr><td>Philadelphia Eagles</td><td>+100</td><td>+100</td><td>-120</td><td>** WINNER **</td></tr>
<tr><td>Kansas City Chiefs</td><td>-110</td><td>-120</td><td>+100</td><td>&nbsp;</td></tr>
<tr><td>Cincinnati Bengals</td><td></td><td></td><td></td><td>&nbsp;</td></tr>
</table>
"""


def test_american_odds_conversion():
    assert american_to_implied(100) == pytest.approx(0.5)
    assert american_to_implied(-200) == pytest.approx(2 / 3)
    assert american_to_implied(300) == pytest.approx(0.25)


def test_parse_odds_variants():
    assert _parse_odds("+1600") == 1600
    assert _parse_odds("-120") == -120
    assert _parse_odds("EVEN") == 100
    assert np.isnan(_parse_odds(""))
    assert np.isnan(_parse_odds("\xa0"))


def test_parse_page_extracts_rounds_and_winner():
    # the parser follows the header: this fixture names three rounds
    odds = parse_page(PAGE, 2024, "sb")

    assert set(odds["team"]) == {"PHI", "KC"}          # all-blank team dropped
    assert set(odds["round_prior"]) == {"WC", "DIV", "SB"}
    assert odds.loc[odds["team"].eq("PHI"), "won_market"].all()
    assert not odds.loc[odds["team"].eq("KC"), "won_market"].any()


def test_devig_normalizes_each_round_to_one():
    odds = add_probabilities(parse_page(PAGE, 2024, "sb"))
    per_round = odds.groupby("round_prior")["p_market"].sum()

    assert np.allclose(per_round, 1.0)
    assert (odds["overround"] > 1.0).all()


def test_every_nfl_franchise_name_variant_maps():
    """A silently dropped team skews every normalized probability for that season."""
    from src.data.futures import TEAM_NAMES
    from src.teams import VALID_TEAMS

    assert set(TEAM_NAMES.values()) == set(VALID_TEAMS)
    for variant in ("St Louis Rams", "St. Louis Rams", "San Diego Chargers", "Oakland Raiders"):
        assert variant in TEAM_NAMES


def test_weekly_cached_pages_have_all_32_teams_in_week_one():
    """Week-1 prices exist for the whole league in every cached season."""
    import pandas as pd
    from src.config import configured_path

    path = configured_path("processed") / "futures_weekly_sb.parquet"
    if not path.exists():
        import pytest
        pytest.skip("run python -m src.data.futures first")

    w = pd.read_parquet(path)
    week1 = w.loc[w["week_prior"].eq(1)].groupby("season")["team"].nunique()
    assert (week1 == 32).all(), week1[week1 != 32].to_dict()
