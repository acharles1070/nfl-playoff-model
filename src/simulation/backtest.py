"""Score the bracket simulator against historical futures prices.

For each season S and each round R in {WC, DIV, CON, SB}:
  * fit the game model on seasons < S only (walk-forward),
  * read every alive team's ratings as of the start of round R,
  * simulate from that point with the ACTUAL survivors and seeds,
  * compare P(win conference / win Super Bowl) with the market's de-vigged
    price "prior to" round R (BetMGM via Covers), on the same teams.

Scores per (season, market, round): -log P(actual winner) and the multi-team
Brier score. The market is a comparison only; it is never a model input.

Usage:
    python -m src.simulation.backtest
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import configured_path
from src.data.schedules import last_complete_season
from src.features.ratings import RatingParams, attach_ratings_to_matchups, build_team_ratings
from src.models.backtest import build_logistic_model
from src.simulation.bracket import MatchupTable, game_prob, draw_shocks, probit_index, simulate_bracket
from src.simulation.seeding import reconstruct_field


FEATURES = ["diff_rating_off_total", "diff_rating_def"]
FIRST_MODEL_SEASON = 2010
PROBIT_PER_LOGIT = 0.6        # dm/dL around typical game probabilities


class GameModel:
    """The fitted logistic game model plus what the simulator needs from it."""

    def __init__(self, train: pd.DataFrame, features: list[str] = FEATURES):
        self.features = features
        self.pipeline = build_logistic_model(features)
        self.pipeline.fit(train, train["home_win"].astype(int).to_numpy())

        scaler = self.pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scaler"]
        lr = self.pipeline.named_steps["model"]

        self.intercept = float(lr.intercept_[0])
        self.gradient = lr.coef_[0] / scaler.scale_       # d logit / d raw feature

        # The model standardizes its inputs, so in RAW-feature space the constant (the home-field
        # term, since features are home-minus-away differences) is intercept - sum(coef * mean / scale),
        # not the bare intercept. Subtracting only the intercept left neutral-site logits slightly
        # asymmetric (about 0.015 logit, ~0.4 points of probability at 50%).
        self.home_logit_constant = self.intercept - float(np.sum(lr.coef_[0] * scaler.mean_ / scaler.scale_))

    def home_logit(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.decision_function(frame)


def build_table(snapshot: pd.DataFrame, model: GameModel, tau_scale: float = 1.0) -> MatchupTable:
    """snapshot: one row per alive team with pregame_* rating columns."""
    snap = snapshot.set_index("team")
    teams = list(snap.index)

    pairs = [(i, j) for i in teams for j in teams if i != j]

    frame = pd.DataFrame(
        {
            "diff_rating_off_total": [
                (snap.loc[i, "pregame_off"] + snap.loc[i, "pregame_qb"])
                - (snap.loc[j, "pregame_off"] + snap.loc[j, "pregame_qb"])
                for i, j in pairs
            ],
            "diff_rating_def": [
                snap.loc[j, "pregame_def"] - snap.loc[i, "pregame_def"] for i, j in pairs
            ],
        }
    )

    logit_home = model.home_logit(frame)
    logit_neutral = logit_home - model.home_logit_constant        # drop the home-field term

    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

    m_home = {p: probit_index(sigmoid(l)) for p, l in zip(pairs, logit_home)}
    m_neutral = {p: probit_index(sigmoid(l)) for p, l in zip(pairs, logit_neutral)}

    g_off, g_def = model.gradient
    tau2 = {}

    for t in teams:
        var_off = snap.loc[t, "pregame_off_var"] + np.nan_to_num(snap.loc[t, "pregame_qb_var"])
        var_def = snap.loc[t, "pregame_def_var"]
        tau2[t] = tau_scale * PROBIT_PER_LOGIT ** 2 * (g_off ** 2 * var_off + g_def ** 2 * var_def)

    # keep s_ij^2 = 1 - tau_i^2 - tau_j^2 positive (marginal preservation needs it)
    cap = 0.45
    tau2 = {t: min(v, cap) for t, v in tau2.items()}

    return MatchupTable(teams, m_home, m_neutral, tau2)


def round_snapshot(ratings: pd.DataFrame, post: pd.DataFrame, round_code: str, teams: list[str]) -> pd.DataFrame:
    """Each team's pregame ratings in its first game of the given round
    (bye teams: their Divisional game, i.e. the same state after the regular season)."""
    order = {"WC": 0, "DIV": 1, "CON": 2, "SB": 3}
    frames = []

    for team in teams:
        games = post.loc[
            (post["home_team"].eq(team) | post["away_team"].eq(team))
            & post["game_type"].map(order).ge(order[round_code])
        ].sort_values("week")
        gid = games["game_id"].iloc[0]
        frames.append(ratings.loc[(ratings["game_id"].eq(gid)) & ratings["team"].eq(team)])

    return pd.concat(frames, ignore_index=True)


def alive_by_round(post: pd.DataFrame, fields: dict[str, list[str]]) -> dict[str, dict[str, list[str]]]:
    out = {}
    for round_code in ("DIV", "CON"):
        teams = set(post.loc[post["game_type"].eq(round_code), "home_team"]) | set(
            post.loc[post["game_type"].eq(round_code), "away_team"]
        )
        out[round_code] = {label: [t for t in seeds if t in teams] for label, seeds in fields.items()}
    return out


def run_season(season, games, ratings, matchups, futures, tau_scale: float = 1.0) -> list[dict]:
    train = attach_ratings_to_matchups(
        matchups.loc[matchups["season"].between(FIRST_MODEL_SEASON, season - 1)], ratings
    )
    model = GameModel(train)

    post = games.loc[games["season"].eq(season) & games["is_postseason"] & games["is_played"]]
    fields = reconstruct_field(games, season)
    alive = alive_by_round(post, fields)

    all_teams = [t for s in fields.values() for t in s]
    conf_label = {t: label for label, seeds in fields.items() for t in seeds}

    sb_game = post.loc[post["game_type"].eq("SB")].iloc[0]
    finalists = [sb_game["home_team"], sb_game["away_team"]]

    rows = []

    for round_code in ("WC", "DIV", "CON", "SB"):
        if round_code == "SB":
            teams = finalists
        elif round_code == "WC":
            teams = all_teams
        else:
            teams = [t for v in alive[round_code].values() for t in v]

        table = build_table(round_snapshot(ratings, post, round_code, teams), model, tau_scale)

        if round_code == "SB":
            a, b = finalists
            p = float(game_prob(table, draw_shocks(teams, table.tau2, 4096, 0), a, b, neutral=True).mean())
            sim = pd.DataFrame(
                {"team": [a, b], "conference": [conf_label[a], conf_label[b]],
                 "p_win_sb": [p, 1 - p], "p_win_conf": [1.0, 1.0]}
            )
        else:
            sim = simulate_bracket(
                {l: [t for t in s if t in teams] for l, s in fields.items()} if round_code == "WC" else
                {l: [t for t in fields[l] if t in teams] for l in fields},
                table,
                n_draws=4096,
                seed=season,
                start=round_code if round_code != "WC" else "WC",
                alive=alive.get(round_code),
            )

        for market, column in (("sb", "p_win_sb"), ("afc", "p_win_conf"), ("nfc", "p_win_conf")):
            if round_code == "SB" and market != "sb":
                continue

            fut = futures.loc[
                futures["season"].eq(season) & futures["market"].eq(market) & futures["round_prior"].eq(round_code)
            ].set_index("team")

            if fut.empty:
                continue

            if market in ("afc", "nfc"):
                # the futures file labels conferences by name; match by members
                members = {t for t in fut.index}
                label = next(l for l, s in fields.items() if members <= set(s))
                model_p = sim.loc[sim["conference"].eq(label)].set_index("team")[column]
            else:
                model_p = sim.set_index("team")[column]

            common = fut.index.intersection(model_p.index)
            if len(common) < 2:
                continue

            p_model = model_p.loc[common] / model_p.loc[common].sum()
            p_market = fut.loc[common, "p_market"] / fut.loc[common, "p_market"].sum()
            won = fut.loc[common, "won_market"].astype(float)

            if won.sum() != 1:
                continue

            winner = won.idxmax()

            rows.append(
                {
                    "season": season, "market": market, "round": round_code, "teams": len(common),
                    "winner": winner,
                    "p_model_winner": float(p_model[winner]), "p_market_winner": float(p_market[winner]),
                    "brier_model": float(((p_model - won) ** 2).sum()),
                    "brier_market": float(((p_market - won) ** 2).sum()),
                }
            )

    return rows


def main() -> None:
    processed = configured_path("processed")
    outputs = configured_path("outputs")
    outputs.mkdir(parents=True, exist_ok=True)

    games = pd.read_parquet(processed / "games.parquet")
    team_game = pd.read_parquet(processed / "team_game_epa.parquet")
    qb_game = pd.read_parquet(processed / "qb_game.parquet")
    matchups = pd.read_parquet(processed / "historical_matchups_oa.parquet")
    futures = pd.read_parquet(processed / "futures_odds.parquet")

    played = games.loc[games["is_played"]]
    ratings = build_team_ratings(team_game, played, RatingParams(), qb_game)

    rows = []
    for season in range(2011, last_complete_season() + 1):
        rows += run_season(season, games, ratings, matchups, futures)

    result = pd.DataFrame(rows)
    result["logscore_model"] = -np.log(result["p_model_winner"])
    result["logscore_market"] = -np.log(result["p_market_winner"])
    result.to_csv(outputs / "bracket_backtest.csv", index=False)

    def summarize(frame, label):
        g = frame.groupby(["market", "round"]).agg(
            seasons=("season", "size"),
            model_logscore=("logscore_model", "mean"),
            market_logscore=("logscore_market", "mean"),
            model_brier=("brier_model", "mean"),
            market_brier=("brier_market", "mean"),
        )
        g["logscore_diff"] = g["model_logscore"] - g["market_logscore"]
        print(f"\n=== {label} (lower is better; diff<0 means model beats market) ===")
        print(g.round(4).to_string())

    summarize(result, "ALL SEASONS 2011-2025")
    summarize(result.loc[result["season"] >= 2017], "HOLDOUT 2017-2025")

    sb = result.loc[result["market"].eq("sb") & result["season"].ge(2017)]
    print("\nSuper Bowl-winner probability assigned, prior to Wild Card, holdout seasons:")
    wc = sb.loc[sb["round"].eq("WC"), ["season", "winner", "p_model_winner", "p_market_winner"]]
    print(wc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
