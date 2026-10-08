"""Score the title-odds ledger once the season resolves.

    python -m src.simulation.title_score

Compares, for each logged week, -ln P(champion) from my simulation with the same
number from the market's de-vigged price for the same week (BetMGM via Covers),
and checks playoff-qualification calibration. Before the Super Bowl is played it
reports what has been logged so far.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import configured_path
from src.live import ledger
from src.simulation.title_odds import FIELDS, TITLE_LEDGER


def main() -> None:
    ok, message = ledger.verify(TITLE_LEDGER, FIELDS)
    print(message)
    if not ok:
        raise SystemExit("Title ledger integrity failure.")

    rows = pd.DataFrame(ledger.read_rows(TITLE_LEDGER))
    if rows.empty:
        print("Nothing logged yet.")
        return

    for c in ("p_win_sb", "p_playoffs", "p_division", "expected_wins"):
        rows[c] = pd.to_numeric(rows[c])
    rows["season"] = rows["season"].astype(int)
    rows["week_prior"] = rows["week_prior"].astype(int)

    games = pd.read_parquet(configured_path("processed") / "games.parquet")
    weekly = pd.read_parquet(configured_path("processed") / "futures_weekly_sb.parquet")

    print(f"{rows.groupby(['model_id', 'season'])['week_prior'].nunique().to_dict()} weeks logged per (model, season)")

    # score each model version separately: a bug-fix release (new model_id) must never be mixed
    # with the version it replaced, and a week logged by both would otherwise duplicate teams
    for (model_id, season), logged in rows.groupby(["model_id", "season"]):
        sb = games.loc[games["season"].eq(season) & games["game_type"].eq("SB") & games["is_played"]]
        if sb.empty:
            print(f"{model_id} {season}: Super Bowl not played yet; scoring later.")
            continue

        s = sb.iloc[0]
        champ = s["home_team"] if s["home_score"] > s["away_score"] else s["away_team"]
        out = []

        for week, g in logged.groupby("week_prior"):
            mk = weekly.loc[weekly["season"].eq(season) & weekly["week_prior"].eq(week)].set_index("team")["p_market"]
            common = mk.index.intersection(g["team"])
            pm = g.set_index("team").loc[common, "p_win_sb"]
            pm = pm / pm.sum()
            pk = mk.loc[common] / mk.loc[common].sum()
            if champ in common:
                out.append({"week": week, "model": -np.log(pm[champ]), "market": -np.log(pk[champ])})

        t = pd.DataFrame(out)
        print(f"\n{model_id}, {season} champion: {champ}")
        print(t.round(3).to_string(index=False))
        print(f"mean log score  model {t.model.mean():.3f}  market {t.market.mean():.3f}")


if __name__ == "__main__":
    main()
