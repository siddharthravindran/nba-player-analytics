"""Who will the voters pick? A model of All-NBA voting, next to player_ranking's model of impact.

    from player_ranking import *
    from awards import *
    model = fit_rating_model(df_master)
    voters = fit_allnba_model(df_master, model)            # trains, prints how it did season by season
    predict_allnba(voters, df_master, model)               # this season: chance of a spot, projected team
    predict_allnba(voters, df_master, model, "2023-24")    # a past season, next to what really happened
    voters.weights                                         # what voters reward, largest first
    voters.check                                           # one row per season: hits out of 15

player_ranking answers "who has played best". This answers a different
question: given how voters have behaved since 2015-16, who gets a spot. The two
disagree where voters reward things impact metrics don't — scoring volume,
team record, games played and reputation.

How it works
  - One row per rotation player per season. Each stat is measured against that
    season's pool (z-scores), like the rating model.
  - A logistic regression on "made an All-NBA team", penalty chosen by leaving
    out one season at a time. The same leave-one-out runs give the accuracy
    check, so every season in voters.check was predicted without seeing it.
  - The 15 highest chances are the projected teams, five at a time. From
    2023-24 a player must be able to reach 65 games.

What it can't see: positions (teams were two guards, two forwards and a center
until 2022-23; the model ignores that), injuries still to come, appeals against
the 65-game rule, and narrative.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from nba_viz_utils import norm_name
from player_ranking import _regular_season, rank_players
from salaries import ALLNBA_WEIGHT, _allnba_selections

VOTER_STATS = ["PTS", "REB", "AST", "STL", "BLK", "TS_PCT", "USG_PCT", "MIN", "W_PCT", "PIE", "BBR_BPM",
               "GP_SHARE", "RATING", "WINS_PACE", "PRIOR"]
LABELS = {"PTS": "scoring", "REB": "rebounds", "AST": "assists", "STL": "steals", "BLK": "blocks",
          "TS_PCT": "efficiency", "USG_PCT": "usage", "MIN": "minutes", "W_PCT": "team record", "PIE": "all-round production",
          "BBR_BPM": "BPM", "GP_SHARE": "games played", "RATING": "impact rating", "WINS_PACE": "wins added",
          "PRIOR": "past All-NBA teams"}
POOL_MIN, POOL_GP_SHARE = 24.0, 0.45         # every All-NBA player since 2015-16 clears both
GAMES_RULE_FROM, GAMES_RULE, SEASON_GAMES = "2023-24", 65, 82
# The league's count is not always the stat sheet's: in 2025-26 three First Team players show 64 games here
# (two won "extraordinary circumstances" appeals). One game of slack keeps the filter from ruling out real picks.
GAMES_SLACK = 1
PENALTIES = (0.01, 0.03, 0.1, 0.3, 1.0)
TEAM_NAMES = {3: "1st", 2: "2nd", 1: "3rd"}


@dataclass
class AllNBAModel:
    weights: pd.Series               # per stat, in standard deviations; largest first
    intercept: float
    penalty: float
    seasons: list
    check: pd.DataFrame = field(default=None, repr=False)      # leave-one-season-out results
    missed: pd.DataFrame = field(default=None, repr=False)     # every wrong call in those runs


def _voter_table(df_master, rating_model):
    """One row per rotation player per season with the stats voters see, and what the voters did."""
    df = _regular_season(df_master)
    parts = []
    for season in sorted(df["SEASON"].unique()):
        rated = rank_players(rating_model, df_master, season, min_min=0, min_gp=0).set_index("PLAYER_ID")
        one = df[df["SEASON"] == season].set_index("PLAYER_ID")
        one = one.join(rated[["RATING", "WINS_82"]].rename(columns={"WINS_82": "WINS_PACE"}), how="inner")
        team_games = one.groupby("TEAM_ABBREVIATION")["GP"].transform("max").clip(lower=1)
        most = min(one["GP"].max(), SEASON_GAMES)             # a traded player can appear in 83 or 84
        # a finished season is judged against the league's schedule; one in progress against his own team's
        played = team_games if most < 0.9 * SEASON_GAMES and season == df["SEASON"].max() else most
        one["TEAM_GAMES"] = team_games if isinstance(played, pd.Series) else most
        one["GP_SHARE"] = (one["GP"] / played).clip(upper=1.0)
        parts.append(one.reset_index())
    table = pd.concat(parts, ignore_index=True)
    table["name_key"] = table["PLAYER_NAME"].map(norm_name)
    table["yr"] = table["SEASON"].str[:4].astype(int)

    picks = _allnba_selections()
    made = picks.set_index(["name_key", "yr"])["weight"]
    table["VOTES"] = [made.get(k, 0) for k in zip(table["name_key"], table["yr"])]
    table["VOTES"] = table["VOTES"].fillna(0).astype(int)
    table["DECIDED"] = table["yr"].isin(picks["yr"].unique())
    by_player = {k: g for k, g in picks.groupby("name_key")}
    prior = []
    for key, year in zip(table["name_key"], table["yr"]):
        g = by_player.get(key)
        prior.append(0 if g is None else g.loc[(g["yr"] < year) & (g["yr"] >= year - 3), "weight"].sum())
    table["PRIOR"] = prior

    pool = table[(table["MIN"] >= POOL_MIN) & (table["GP_SHARE"] >= POOL_GP_SHARE)].copy()
    left_out = table[(table["VOTES"] > 0) & ~table.index.isin(pool.index)]
    if len(left_out):
        print(f"⚠️ {len(left_out)} All-NBA players fall outside the pool: "
              f"{left_out[['PLAYER_NAME', 'SEASON']].values.tolist()[:5]}")
    X = pool[VOTER_STATS].astype(float)
    by_season = X.groupby(pool["SEASON"])
    Z = ((X - by_season.transform("mean")) / by_season.transform("std").replace(0, np.nan)).clip(-5, 5).fillna(0)
    return pool, Z


def _can_reach(pool):
    """From 2023-24: could he still finish with 65 games (less one of slack)? Earlier seasons had no minimum."""
    left = (SEASON_GAMES - pool["TEAM_GAMES"]).clip(lower=0)
    return (pool["SEASON"] < GAMES_RULE_FROM) | (pool["GP"] + left >= GAMES_RULE - GAMES_SLACK)


def _teams(chance, eligible):
    """The 15 highest chances among eligible players, five to a team."""
    order = chance[eligible].sort_values(ascending=False).index[:15]
    out = pd.Series("", index=chance.index)
    for i, idx in enumerate(order):
        out[idx] = TEAM_NAMES[3 - i // 5]
    return out


def fit_allnba_model(df_master, rating_model, penalty=None, verbose=True):
    """Train the voting model. Returns an AllNBAModel; .check is the season-by-season test.

    rating_model: from fit_rating_model — its rating and wins are two of the
    inputs, so the voting model can learn how much voters follow impact.
    penalty=None picks the regularisation by leaving out one season at a time.
    """
    pool, Z = _voter_table(df_master, rating_model)
    known = pool["DECIDED"].to_numpy()
    y = (pool["VOTES"] > 0).to_numpy()
    seasons = sorted(pool.loc[known, "SEASON"].unique())

    def held_out(C):
        chance = pd.Series(np.nan, index=pool.index)
        for s in seasons:
            test = known & (pool["SEASON"] == s).to_numpy()
            train = known & ~test
            fit = LogisticRegression(C=C, max_iter=5000).fit(Z[train], y[train])
            chance[test] = fit.predict_proba(Z[test])[:, 1]
        return chance

    if penalty is None:
        loss = {}
        for C in PENALTIES:
            p = held_out(C)[known].clip(1e-6, 1 - 1e-6)
            loss[C] = float(-(np.where(y[known], np.log(p), np.log(1 - p))).mean())
        penalty = min(loss, key=loss.get)

    chance = held_out(penalty)
    eligible = _can_reach(pool)
    rows, missed = [], []
    for s in seasons:
        here = (pool["SEASON"] == s).to_numpy()
        team = _teams(chance[here], eligible[here])
        real = pool.loc[here, "VOTES"].map(TEAM_NAMES).fillna("")
        hit = (team != "") & (real != "")
        by = lambda col: set(pool.loc[here].nlargest(15, col).index)            # simple yardsticks
        actual = set(real[real != ""].index)
        rows.append({"SEASON": s, "RIGHT_OF_15": int(hit.sum()), "FIRST_TEAM_OF_5": int(((team == "1st") & (real == "1st")).sum()),
                     "SAME_TEAM": int(((team == real) & hit).sum()),
                     "BY_WINS": len(by("WINS_PACE") & actual), "BY_BPM": len(by("BBR_BPM") & actual), "BY_POINTS": len(by("PTS") & actual)})
        wrong = (team != "") ^ (real != "")
        for idx in wrong[wrong].index:
            missed.append({"SEASON": s, "Player": pool.at[idx, "PLAYER_NAME"], "PROJECTED": team[idx], "ACTUAL": real[idx],
                           "CHANCE": chance[idx], "GP": int(pool.at[idx, "GP"])})
    check = pd.DataFrame(rows)

    fit = LogisticRegression(C=penalty, max_iter=5000).fit(Z[known], y[known])
    weights = pd.Series(fit.coef_[0], index=VOTER_STATS).sort_values(key=np.abs, ascending=False)
    if verbose:
        n = len(seasons) * 15
        print(f"\nAll-NBA voting model: {int(y[known].sum())} selections over {len(seasons)} seasons, "
              f"{int(known.sum())} player-seasons, penalty C={penalty:g}")
        print(f"Each season predicted without seeing it: {check['RIGHT_OF_15'].sum()} of {n} spots right "
              f"({check['RIGHT_OF_15'].mean():.1f} of 15 a season), first team {check['FIRST_TEAM_OF_5'].sum()} of {len(seasons) * 5}")
        print(f"  for comparison, the top 15 by wins added gets {check['BY_WINS'].sum()}, by BPM {check['BY_BPM'].sum()}, "
              f"by points per game {check['BY_POINTS'].sum()}")
        print("What voters reward most: " + ", ".join(f"{LABELS[k]} ({v:+.2f})" for k, v in weights.head(6).items()))
    return AllNBAModel(weights, float(fit.intercept_[0]), penalty, seasons, check, pd.DataFrame(missed))


def predict_allnba(voters, df_master, rating_model, season="2026-27", top=30):
    """Each player's chance of an All-NBA spot for a season, best first, with the projected teams.

    CHANCE      probability of making any of the three teams
    PROJECTED   1st / 2nd / 3rd if he is among the 15 most likely eligible players
    ACTUAL      what happened, for seasons already voted on
    CAN_REACH_65  False once he has missed too many games to qualify (2023-24 on)
    WHY / WHY_NOT  the stats pushing his chance up and down the most
    RATING_RANK   where the impact model has him in the same pool, for contrast

    During a season the inputs are per-game numbers and his team's record so
    far, so this is "if the vote were today on today's pace". It can't know
    about games he is going to miss.
    """
    pool, Z = _voter_table(df_master, rating_model)
    here = (pool["SEASON"] == season).to_numpy()
    if not here.any():
        raise ValueError(f"no rotation players for {season}")
    pool, Z = pool[here], Z[here]
    w = voters.weights.reindex(VOTER_STATS)
    pull = Z * w
    chance = pd.Series(1 / (1 + np.exp(-(pull.sum(axis=1) + voters.intercept))), index=pool.index)
    eligible = _can_reach(pool)

    def reasons(row, sign):
        picked = (row * sign).nlargest(2)
        return ", ".join(LABELS[k] for k, v in picked.items() if v > 0.15)

    out = pd.DataFrame({
        "Player": pool["PLAYER_NAME"], "TEAM": pool["TEAM_ABBREVIATION"], "GP": pool["GP"].astype(int),
        "TEAM_GAMES": pool["TEAM_GAMES"].astype(int), "PTS": pool["PTS"], "W_PCT": pool["W_PCT"],
        "CHANCE": chance, "PROJECTED": _teams(chance, eligible),
        "CAN_REACH_65": eligible, "RATING_RANK": pool["RATING"].rank(ascending=False).astype(int),
        "WHY": [reasons(r, 1) for _, r in pull.iterrows()], "WHY_NOT": [reasons(r, -1) for _, r in pull.iterrows()],
    })
    if pool["DECIDED"].any():
        out.insert(8, "ACTUAL", pool["VOTES"].map(TEAM_NAMES).fillna(""))
    out = out.sort_values("CHANCE", ascending=False).reset_index(drop=True)
    out.insert(0, "RANK", np.arange(1, len(out) + 1))
    out.attrs.update(season=season, n=len(out))
    return out.head(top) if top else out
