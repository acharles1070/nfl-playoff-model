"""Who is actually unavailable: game-day inactives and injured reserve.

Importance of a player = his mean snap share over his last 4 games strictly
before the week (same as-of join as src.features.injuries).

lost_ir_<group>    starters on injured reserve the PREVIOUS week (status RES in
                   week w-1). Known any time before the game, so usable in a
                   Friday prediction. Captures long-term injuries that weekly
                   injury reports never list.
lost_ina_<group>   game-day inactives THIS week (status INA). Announced ~90 minutes
                   before kickoff: kickoff-time information, which is also what the
                   closing line knows. Not usable for earlier predictions.
"""

from __future__ import annotations

import pandas as pd

from src.features.injuries import BURDEN_GROUPS, GROUPS, prior_snap_share


def _grouped(x: pd.DataFrame) -> pd.DataFrame:
    x = x.assign(group=x["position"].map(GROUPS)).dropna(subset=["group"])
    return x.loc[x["group"].isin(BURDEN_GROUPS)]


def _as_of_share(x: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x["time"] = x["season"] * 100 + x["week"]
    x = x.sort_values("time")
    share = prior_snap_share(snaps).rename(columns={"pfr_player_id": "pfr_id"}).sort_values("time")
    x = pd.merge_asof(x, share, on="time", by="pfr_id", allow_exact_matches=False)
    x["share_after"] = x["share_after"].fillna(0.0)
    return x


def _wide(x: pd.DataFrame, prefix: str) -> pd.DataFrame:
    wide = (
        x.pivot_table(index=["season", "week", "team"], columns="group", values="share_after",
                      aggfunc="sum", fill_value=0.0)
        .reindex(columns=BURDEN_GROUPS, fill_value=0.0)
        .add_prefix(f"{prefix}_")
        .reset_index()
    )
    cols = [c for c in wide.columns if c.startswith(prefix)]
    wide[f"{prefix}_off_total"] = wide[[c for c in cols if c.split("_", 2)[-1] in ("OL", "WR_TE", "RB")]].sum(axis=1)
    wide[f"{prefix}_def_total"] = wide[[c for c in cols if c.split("_", 2)[-1] in ("DL", "LB", "DB")]].sum(axis=1)
    return wide


def roster_burden(rosters: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team) that has roster data."""
    x = _grouped(rosters)
    keys = rosters[["season", "week", "team"]].drop_duplicates()

    # --- game-day inactives, this week
    ina = _as_of_share(x.loc[x["status"].eq("INA")], snaps)
    ina_wide = _wide(ina, "lost_ina")

    # --- injured reserve as of LAST week, carried to this week
    ir = x.loc[x["status"].eq("RES")].copy()
    ir["week_for"] = ir["week"] + 1                      # last week's status -> this week
    ir = ir.drop(columns="week").rename(columns={"week_for": "week"})
    ir = _as_of_share(ir, snaps)
    ir_wide = _wide(ir, "lost_ir")

    out = keys.merge(ina_wide, on=["season", "week", "team"], how="left")
    out = out.merge(ir_wide, on=["season", "week", "team"], how="left")

    lost = [c for c in out.columns if c.startswith("lost_")]
    out[lost] = out[lost].fillna(0.0)

    # a team-week that has no previous-week roster cannot know its IR list
    first_week = out.groupby(["season", "team"])["week"].transform("min")
    out.loc[out["week"].eq(first_week), [c for c in lost if "_ir_" in c]] = 0.0
    return out


def attach_roster_burden(matchups: pd.DataFrame, burden: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in burden.columns if c.startswith("lost_")]
    out = matchups

    for side in ("home", "away"):
        part = burden.rename(columns={"team": f"{side}_team", **{c: f"{side}_{c}" for c in cols}})
        out = out.merge(part, on=["season", "week", f"{side}_team"], how="left", validate="many_to_one")

    for c in cols:
        out[f"diff_{c}"] = out[f"home_{c}"] - out[f"away_{c}"]

    if len(out) != len(matchups):
        raise ValueError("Row count changed while attaching roster burden.")
    return out
