"""Expected snap-share lost to injury, per team per week and position group.

    lost[team, week, group] = sum over reported players of
                              importance_i * P(out | status_i, group)

importance   the player's mean snap share over his last 4 games BEFORE this
             week (offense share for offensive groups, defense share for
             defensive groups); an as-of join keeps it strictly pregame.
P(out)       1 for Out, ~1 for Doubtful, and for Questionable the measured
             share of Questionable players who did not play, per position
             group. The meaning of "Questionable" drifted after 2016, so each
             season uses rates measured on the PRECEDING four seasons only.

Quarterbacks are excluded from the burden features: the QB layer already uses
the actual starter. QB availability is handled separately for projections.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


LOOKBACK_SEASONS = 4
DEFAULT_P_OUT_QUESTIONABLE = 0.45   # used before any history exists (2013)

GROUPS = {
    "QB": "QB",
    "T": "OL", "G": "OL", "C": "OL", "OL": "OL",
    "WR": "WR_TE", "TE": "WR_TE",
    "RB": "RB", "FB": "RB",
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL",
    "LB": "LB", "OLB": "LB", "ILB": "LB", "MLB": "LB",
    "CB": "DB", "S": "DB", "SS": "DB", "FS": "DB", "DB": "DB",
}

SIDE = {"QB": "off", "OL": "off", "WR_TE": "off", "RB": "off",
        "DL": "def", "LB": "def", "DB": "def"}

BURDEN_GROUPS = ["OL", "WR_TE", "RB", "DL", "LB", "DB"]
STATUS_OUT = {"Out": 1.0, "Doubtful": 0.99}


def prior_snap_share(snaps: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """Rolling mean snap share AFTER each appearance, keyed for as-of joins."""
    s = snaps.copy()
    s["time"] = s["season"] * 100 + s["week"]
    s = s.sort_values(["pfr_player_id", "time"])

    s["share"] = s[["offense_pct", "defense_pct"]].max(axis=1)
    s["share_after"] = (
        s.groupby("pfr_player_id")["share"]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )

    return s[["pfr_player_id", "time", "share_after"]]


def measure_availability(
    reports: pd.DataFrame,
    snaps: pd.DataFrame,
    id_map: pd.DataFrame,
    target_season: int,
) -> dict[str, float]:
    """P(out | Questionable, group) measured on the seasons BEFORE target_season."""
    lo = target_season - LOOKBACK_SEASONS
    x = reports.merge(id_map, on="gsis_id", how="left")
    x = x.loc[
        x["season"].between(lo, target_season - 1) & x["report_status"].eq("Questionable")
    ]
    x = x.assign(group=x["position"].map(GROUPS)).dropna(subset=["group", "pfr_id"])

    if x.empty:
        return {"_overall": DEFAULT_P_OUT_QUESTIONABLE}

    played_keys = set(zip(snaps["season"], snaps["week"], snaps["pfr_player_id"]))
    x["played"] = [(s, w, p) in played_keys for s, w, p in zip(x["season"], x["week"], x["pfr_id"])]

    table = x.groupby("group")["played"].agg(["sum", "size"])
    overall = 1.0 - x["played"].mean()

    # shrink small groups toward the overall rate (beta-binomial, prior strength 50)
    p_out = {
        g: float(((r["size"] - r["sum"]) + 50 * overall) / (r["size"] + 50))
        for g, r in table.iterrows()
    }
    p_out["_overall"] = float(overall)
    return p_out


def team_week_burden(
    reports: pd.DataFrame,
    snaps: pd.DataFrame,
    id_map: pd.DataFrame,
    p_out_by_season: dict[int, dict[str, float]],
) -> pd.DataFrame:
    """One row per (season, week, team) with lost_<group> columns."""
    x = reports.merge(id_map, on="gsis_id", how="left")
    x = x.assign(group=x["position"].map(GROUPS)).dropna(subset=["group", "pfr_id"])
    x = x.loc[x["report_status"].isin(["Out", "Doubtful", "Questionable"])].copy()

    x["p_out"] = x["report_status"].map(STATUS_OUT)
    is_q = x["report_status"].eq("Questionable")
    x.loc[is_q, "p_out"] = [
        p_out_by_season[int(season)].get(group, p_out_by_season[int(season)]["_overall"])
        for season, group in zip(x.loc[is_q, "season"], x.loc[is_q, "group"])
    ]

    x["time"] = x["season"] * 100 + x["week"]
    x = x.sort_values("time")

    share = prior_snap_share(snaps).rename(columns={"pfr_player_id": "pfr_id"}).sort_values("time")
    x = pd.merge_asof(
        x, share, on="time", by="pfr_id",
        allow_exact_matches=False,           # strictly BEFORE this week
    )
    x["share_after"] = x["share_after"].fillna(0.0)
    x["lost"] = x["share_after"] * x["p_out"]

    wide = (
        x.pivot_table(index=["season", "week", "team"], columns="group", values="lost", aggfunc="sum", fill_value=0.0)
        .reindex(columns=["QB", *BURDEN_GROUPS], fill_value=0.0)
        .add_prefix("lost_")
        .reset_index()
    )

    # A team-week that has report rows but nobody Out/Doubtful/Questionable is a
    # genuine zero; a team-week with NO report rows is unknown and stays absent
    # (attach_burden turns that into NaN, which the model imputes).
    reported = reports[["season", "week", "team"]].drop_duplicates()
    wide = reported.merge(wide, on=["season", "week", "team"], how="left")
    lost_cols = [c for c in wide.columns if c.startswith("lost_")]
    wide[lost_cols] = wide[lost_cols].fillna(0.0)
    wide["lost_off_total"] = wide[["lost_OL", "lost_WR_TE", "lost_RB"]].sum(axis=1)
    wide["lost_def_total"] = wide[["lost_DL", "lost_LB", "lost_DB"]].sum(axis=1)
    return wide


def attach_burden(matchups: pd.DataFrame, burden: pd.DataFrame) -> pd.DataFrame:
    """Home-minus-away burden differences (positive = HOME more hurt)."""
    cols = [c for c in burden.columns if c.startswith("lost_")]
    out = matchups

    for side in ("home", "away"):
        part = burden.rename(columns={"team": f"{side}_team", **{c: f"{side}_{c}" for c in cols}})
        out = out.merge(part, on=["season", "week", f"{side}_team"], how="left", validate="many_to_one")

    for c in cols:
        # NaN when either side has no injury report at all (unknown, not zero)
        out[f"diff_{c}"] = out[f"home_{c}"] - out[f"away_{c}"]

    if len(out) != len(matchups):
        raise ValueError("Row count changed while attaching injury burden.")
    return out
