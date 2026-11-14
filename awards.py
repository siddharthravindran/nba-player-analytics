"""Who will the voters pick? A model of All-NBA voting, next to player_ranking's model of impact.

    from player_ranking import *
    from awards import *
    model = fit_rating_model(df_master)
    voters = fit_allnba_model(df_master, model)            # trains, prints how it did season by season
    predict_allnba(voters, df_master, model)               # this season: chance of a spot, projected team
    predict_allnba(voters, df_master, model, "2023-24")    # a past season, next to what really happened
    voters.weights                                         # what voters reward, largest first
    voters.check                                           # one row per season: hits out of 15
    predict_allstars(voters, df_master, model)             # 12 a conference plus the next five, same voting model
    pick_all_defense(model, df_master)                     # All-Defensive teams, from a second model trained on that vote
    fit_alldefense_model(df_master, model).weights         # what All-Defensive voters reward
    mip = fit_mip_model(df_master, model)                  # Most Improved: trained on who finished top five in that vote
    predict_mip(mip, df_master, model)                     # this season's candidates, with what changed

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

from bbref_stats import DATA_DIR, _get
from nba_viz_utils import SEASONS, norm_name
from player_ranking import _regular_season, rank_players
from salaries import ALLNBA_WEIGHT, _allnba_selections

VOTER_STATS = ["PTS", "REB", "AST", "STL", "BLK", "TS_PCT", "USG_PCT", "MIN", "W_PCT", "PIE", "BBR_BPM",
               "GP_SHARE", "RATING", "WINS_PACE", "PRIOR"]
LABELS = {"PTS": "scoring", "REB": "rebounds", "AST": "assists", "STL": "steals", "BLK": "blocks",
          "TS_PCT": "efficiency", "USG_PCT": "usage", "MIN": "minutes", "W_PCT": "team record", "PIE": "all-round production",
          "BBR_BPM": "BPM", "GP_SHARE": "games played", "RATING": "impact rating", "WINS_PACE": "wins added",
          "PRIOR": "past selections"}
DEFENSE_STATS = ["STL", "BLK", "DEFLECTIONS", "CONTESTED_SHOTS", "CHARGES_DRAWN", "DREB_PCT", "PCT_PLUSMINUS", "D_FG_PCT",
                 "DEF_RIM_FGA", "DEF_RIM_FG_PCT", "DEF_RATING", "ONOFF_DEF_RATING", "DEF_WS", "MIN", "GP_SHARE", "W_PCT",
                 "BBR_DBPM", "D_RATING", "PRIOR"]
LABELS.update({"DEFLECTIONS": "deflections", "CONTESTED_SHOTS": "shots contested", "CHARGES_DRAWN": "charges drawn",
               "DREB_PCT": "defensive rebounding", "PCT_PLUSMINUS": "opponents shooting below their norm",
               "D_FG_PCT": "opponent FG%", "DEF_RIM_FGA": "rim shots defended", "DEF_RIM_FG_PCT": "opponent FG% at the rim",
               "DEF_RATING": "team defense with him on", "ONOFF_DEF_RATING": "defense on vs off court",
               "DEF_WS": "defensive win shares", "BBR_DBPM": "defensive BPM", "D_RATING": "defensive impact rating"})
# Most Improved is about change, so its inputs are this season minus last (UP_...), plus who voters pass over:
# established stars (PRIOR_NBA), older players, and second-year players, whose jump is expected.
MIP_STATS = ["UP_PTS", "UP_MIN", "UP_AST", "UP_REB", "UP_TS_PCT", "UP_USG_PCT", "UP_PIE", "UP_RATING", "UP_WINS_PACE", "UP_W_PCT",
             "PTS", "MIN", "W_PCT", "GP_SHARE", "BBR_AGE", "SOPHOMORE", "PRIOR_NBA", "PRIOR"]
_CHANGES = ["PTS", "MIN", "AST", "REB", "TS_PCT", "USG_PCT", "PIE", "RATING", "WINS_PACE", "W_PCT"]
LABELS.update({"UP_PTS": "scoring jump", "UP_MIN": "more minutes", "UP_AST": "more assists", "UP_REB": "more rebounds",
               "UP_TS_PCT": "better efficiency", "UP_USG_PCT": "bigger role", "UP_PIE": "more all-round production",
               "UP_RATING": "better impact rating", "UP_WINS_PACE": "more wins added", "UP_W_PCT": "team improved",
               "BBR_AGE": "age", "SOPHOMORE": "second-year player", "PRIOR_NBA": "already an All-NBA player"})
MIP_POOL_MIN, MIP_PREV_GP = 20.0, 20         # needs a real previous season to improve on
POOL_MIN, POOL_GP_SHARE = 24.0, 0.45         # every All-NBA player since 2015-16 clears both
DEFENSE_POOL_MIN = 20.0                      # All-Defensive specialists play less
GAMES_RULE_FROM, GAMES_RULE, SEASON_GAMES = "2023-24", 65, 82
# The league's count is not always the stat sheet's: in 2025-26 three First Team players show 64 games here
# (two won "extraordinary circumstances" appeals). One game of slack keeps the filter from ruling out real picks.
GAMES_SLACK = 1
# A season in progress: games played are projected to the end of the schedule for every award (_project_season).
# For All-Defensive the stats are also weighed against this many games of last season. 25 is a judgment, not fitted.
EARLY_PRIOR_GAMES = 25
_PULL_TO_LAST_SEASON = {"defense"}           # the awards whose stats are pulled toward last season; games are projected for all
EARLY_PRIOR_MIN_GP = 20                      # a "last season" shorter than this is skipped for the one before
_TEAM_STATS = ["W_PCT", "DEF_RATING", "ONOFF_DEF_RATING"]         # say little about him once he has changed teams
_NOT_PROJECTED = ["PRIOR", "PRIOR_NBA", "SOPHOMORE", "BBR_AGE"]   # facts about him, not this season's play
PENALTIES = (0.01, 0.03, 0.1, 0.3, 1.0)
TEAM_NAMES = {3: "1st", 2: "2nd", 1: "3rd"}
DEFENSE_TEAM_NAMES = {2: "1st", 1: "2nd"}
EAST = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"}

__all__ = ["AllNBAModel", "fit_allnba_model", "predict_allnba", "predict_allstars", "fit_alldefense_model", "pick_all_defense",
           "fit_mip_model", "predict_mip"]


@dataclass
class AllNBAModel:
    weights: pd.Series               # per stat, in standard deviations; largest first
    intercept: float
    penalty: float
    seasons: list
    check: pd.DataFrame = field(default=None, repr=False)      # leave-one-season-out results
    missed: pd.DataFrame = field(default=None, repr=False)     # every wrong call in those runs
    award: str = "nba"                                         # "nba" or "defense"


def _alldefense_selections(refresh=False):
    """All-Defensive teams from Basketball-Reference, one row per selection: name_key, yr, weight (1st = 2, 2nd = 1)."""
    cache = DATA_DIR / "alldefense_raw.parquet"
    if cache.exists() and not refresh:
        raw = pd.read_parquet(cache)
    else:
        from io import StringIO
        html = _get("https://www.basketball-reference.com/awards/all_defense.html")
        start = html.find('<table', html.find('id="div_awards_all_defense"'))
        raw = pd.read_html(StringIO(html[start:html.find("</table>", start) + 8]))[0]
        raw.columns = [str(c) for c in raw.columns]
        raw = raw.astype(object).where(raw.notna(), None).astype("string")
        raw.to_parquet(cache)
    raw = raw[raw["Lg"] == "NBA"]
    long = raw.melt(id_vars=["Season", "Tm"], value_vars=[c for c in raw.columns if c.startswith("Unnamed")],
                    value_name="player").dropna(subset=["player"])
    long["name_key"] = long["player"].str.replace(r"\s*\(T\)$", "", regex=True).map(norm_name)
    long = long[long["Season"].isin(SEASONS)]
    long["yr"] = long["Season"].str[:4].astype(int)
    long["weight"] = long["Tm"].map({"1st": 2, "2nd": 1})
    return long[["name_key", "yr", "weight"]]


def _mip_selections(refresh=False):
    """Top five in each season's Most Improved Player vote (Basketball-Reference): name_key, yr, weight, finish."""
    cache = DATA_DIR / "mip_voting.parquet"
    if cache.exists() and not refresh:
        raw = pd.read_parquet(cache)
    else:
        from io import StringIO
        frames = []
        for year in range(2017, int(max(SEASONS)[:4]) + 1):
            html = _get(f"https://www.basketball-reference.com/awards/awards_{year}.html")
            at = html.find('id="mip"')
            if at < 0:
                continue                                                    # not voted on yet
            t = pd.read_html(StringIO(html[html.rfind("<table", 0, at):html.find("</table>", at) + 8]))[0]
            t.columns = [c[-1] if isinstance(c, tuple) else c for c in t.columns]
            t = t[["Rank", "Player", "Age", "Tm", "First", "Pts Won", "Pts Max", "Share"]].copy()
            t.insert(0, "SEASON", f"{year - 1}-{str(year)[2:]}")
            frames.append(t)
        raw = pd.concat(frames, ignore_index=True)
        raw = raw.astype(object).where(raw.notna(), None).astype("string")
        raw.to_parquet(cache)
    raw = raw.assign(finish=raw.groupby("SEASON").cumcount() + 1)           # the page lists most votes first
    top = raw[raw["finish"] <= 5]
    return pd.DataFrame({"name_key": top["Player"].map(norm_name), "yr": top["SEASON"].str[:4].astype(int),
                         "weight": 1, "finish": top["finish"]})


_AWARDS = {   # award -> (selections, stats, minimum minutes, team labels)
    "nba": (_allnba_selections, VOTER_STATS, POOL_MIN, TEAM_NAMES),
    "defense": (_alldefense_selections, DEFENSE_STATS, DEFENSE_POOL_MIN, DEFENSE_TEAM_NAMES),
    "mip": (_mip_selections, MIP_STATS, MIP_POOL_MIN, {1: "top 5"}),
}


def _voter_table(df_master, rating_model, award="nba"):
    """One row per rotation player per season with the stats voters see, and what the voters did."""
    selections, stats, pool_min, _ = _AWARDS[award]
    df = _regular_season(df_master)
    parts, in_progress = [], None
    for season in sorted(df["SEASON"].unique()):
        rated = rank_players(rating_model, df_master, season, min_min=0, min_gp=0).set_index("PLAYER_ID")
        one = df[df["SEASON"] == season].set_index("PLAYER_ID")
        one = one.join(rated[["RATING", "D_RATING", "WINS_82"]].rename(columns={"WINS_82": "WINS_PACE"}), how="inner")
        team_games = one.groupby("TEAM_ABBREVIATION")["GP"].transform("max").clip(lower=1)
        most = min(one["GP"].max(), SEASON_GAMES)             # a traded player can appear in 83 or 84
        # a finished season is judged against the league's schedule; one in progress against his own team's
        played = team_games if most < 0.9 * SEASON_GAMES and season == df["SEASON"].max() else most
        in_progress = season if isinstance(played, pd.Series) else in_progress
        one["TEAM_GAMES"] = team_games if isinstance(played, pd.Series) else most
        one["GP_SHARE"] = (one["GP"] / played).clip(upper=1.0)
        parts.append(one.reset_index())
    table = pd.concat(parts, ignore_index=True)
    table = table.sort_values("GP").drop_duplicates(["PLAYER_ID", "SEASON"], keep="last").sort_index().reset_index(drop=True)
    table["name_key"] = table["PLAYER_NAME"].map(norm_name)
    table["yr"] = table["SEASON"].str[:4].astype(int)

    # this season against last: UP_PTS and so on, and whether last season was his first
    last = table[["PLAYER_ID", "yr", "GP"] + _CHANGES].apply(lambda c: pd.to_numeric(c, errors="coerce") if c.name != "PLAYER_ID" else c)
    last = last.assign(yr=last["yr"] + 1).rename(columns={c: f"PREV_{c}" for c in ["GP"] + _CHANGES})
    table = table.merge(last, on=["PLAYER_ID", "yr"], how="left")
    for c in _CHANGES:
        table[f"UP_{c}"] = pd.to_numeric(table[c], errors="coerce") - table[f"PREV_{c}"]
    debut = table.groupby("PLAYER_ID")["yr"].transform("min")
    first_data_year = table["yr"].min()                    # everyone "debuts" in the first season of the data
    table["SOPHOMORE"] = ((debut == table["yr"] - 1) & (debut > first_data_year)).astype(float)

    picks = selections()
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
    if award == "mip":
        stars = {k: g for k, g in _allnba_selections().groupby("name_key")}
        table["PRIOR_NBA"] = [0 if stars.get(k) is None else stars[k].loc[stars[k]["yr"] < y, "weight"].sum()
                              for k, y in zip(table["name_key"], table["yr"])]
        if "finish" in picks.columns:
            table["FINISH"] = [picks.set_index(["name_key", "yr"])["finish"].get(k, np.nan) for k in zip(table["name_key"], table["yr"])]
        table = table[table["PREV_GP"] >= MIP_PREV_GP]

    pool = table[(table["MIN"] >= pool_min) & (table["GP_SHARE"] >= POOL_GP_SHARE)].copy()
    left_out = table[(table["VOTES"] > 0) & ~table.index.isin(pool.index)]
    if len(left_out):
        print(f"⚠️ {len(left_out)} selected players fall outside the pool: "
              f"{left_out[['PLAYER_NAME', 'SEASON']].values.tolist()[:5]}")
    pool["STOCKS"] = pd.to_numeric(pool["STL"], errors="coerce") + pd.to_numeric(pool["BLK"], errors="coerce")
    X = pool[stats].apply(pd.to_numeric, errors="coerce").astype(float)
    by_season = X.groupby(pool["SEASON"])
    Z = ((X - by_season.transform("mean")) / by_season.transform("std").replace(0, np.nan)).clip(-5, 5).fillna(0)
    if in_progress and (pool["SEASON"] == in_progress).any():
        # Games are projected for every award. Stats are pulled toward last season only for All-Defensive:
        # defensive numbers are the noisiest after a dozen games, and that vote leans on reputation. Scoring and
        # role settle fast and a real leap (a second-year player) should count, so All-NBA, All-Star and
        # Most Improved are scored on this season as it stands.
        Z = _project_season(table, pool, X, Z, in_progress, stats_too=award in _PULL_TO_LAST_SEASON)
    return pool, Z


def _project_season(table, pool, X, Z, season, stats_too=True):
    """Score a season in progress on where each player is headed, not on the games so far.

    Twelve games are noisy, and the vote is on 82. Each stat is pulled toward
    the player's own last season (the one before, if he barely played), by how
    much of the season is left: about 40% this season's numbers after 12 games,
    80% at the halfway mark, all of it by the end. A player with no last season
    is pulled toward the average rotation player. Team stats are pulled toward
    average if he has changed teams. The UP_ changes are pulled toward "no change".

    Games played is projected too: the games he has, plus the rest of the
    schedule at the rate players with his last season have gone on to play
    (_games_rate). Otherwise four missed games in November count like missing
    a third of a season.

    Only the scoring changes. The stats shown in the tables stay the real ones.
    stats_too=False projects the games and leaves the stats as they are.
    """
    here = (pool["SEASON"] == season).to_numpy()
    cur, stats, year = pool[here], list(Z.columns), int(season[:4])
    by_season = X.groupby(pool["SEASON"])
    mean, spread = by_season.mean(), by_season.std().replace(0, np.nan)
    ups = [c for c in stats if c.startswith("UP_")]
    levels = [c for c in stats if c in table.columns and c not in ups + _NOT_PROJECTED + ["GP_SHARE"]]

    prior = None                                           # each player's last real season, in that season's z-scores
    for back in (1, 2):
        then = table[(table["yr"] == year - back) & (table["GP"] >= EARLY_PRIOR_MIN_GP)].set_index("PLAYER_ID")
        if then.empty or then["SEASON"].iloc[0] not in mean.index:
            continue
        s = then["SEASON"].iloc[0]
        z = ((then[levels].apply(pd.to_numeric, errors="coerce") - mean.loc[s, levels]) / spread.loc[s, levels]).clip(-5, 5)
        z = z.assign(_TEAM=then["TEAM_ABBREVIATION"])
        prior = z if prior is None else prior.combine_first(z)
    if prior is None:
        return Z
    prior = prior.reindex(cur["PLAYER_ID"].to_numpy())
    prior.index = cur.index

    target = pd.DataFrame(0.0, index=cur.index, columns=stats)
    target[levels] = prior[levels].astype(float).fillna(0.0)
    moved = (prior["_TEAM"] != cur["TEAM_ABBREVIATION"]).to_numpy()
    target.loc[moved, [c for c in _TEAM_STATS if c in levels]] = 0.0
    for c in ups:
        target[c] = -mean.loc[season, c] / spread.loc[season, c]             # "no change from last season"

    games = cur["GP"].astype(float)
    left = (SEASON_GAMES - cur["TEAM_GAMES"]).clip(lower=0).astype(float)
    rest = games / (games + EARLY_PRIOR_GAMES)             # how far to trust his numbers for the games still to come
    now = ((games + left * rest) / (games + left)).to_numpy()[:, None] if stats_too else 1.0
    projected = Z[here] * now + target.fillna(0.0) * (1 - now)
    fixed = [c for c in _NOT_PROJECTED if c in stats]
    projected[fixed] = Z.loc[cur.index, fixed]

    if "GP_SHARE" in stats:
        past = pool.loc[~here].groupby("SEASON")["GP_SHARE"]
        usual, usual_spread = past.mean().mean(), past.std().mean()
        before = table[table["yr"] < year].groupby("PLAYER_ID")["yr"].min()          # rookies have no entry
        was = table[table["yr"] == year - 1].set_index("PLAYER_ID")["GP_SHARE"]
        was = pd.Series(cur["PLAYER_ID"].map(was).to_numpy(), index=cur.index)
        was = was.where(was.notna() | ~cur["PLAYER_ID"].isin(before.index), 0.0)     # a veteran who sat out all of last season
        base, slope = _games_rate(table, season)
        rate = (base + slope * was).fillna(usual).clip(0.0, 1.0)
        share = ((games + left * rate) / (cur["TEAM_GAMES"] + left)).clip(upper=1.0)
        projected["GP_SHARE"] = ((share - usual) / usual_spread).clip(-5, 5)

    Z = Z.copy()
    Z.loc[cur.index] = projected[stats]
    return Z


def _games_rate(table, season, min_min=20.0):
    """Share of games a rotation player goes on to play, from the share he played the season before.

    A straight-line fit on every finished season. One injured season is only
    partly repeated: the slope is well under 1.
    """
    done = table[table["SEASON"] < season]
    was = done[["PLAYER_ID", "yr", "GP_SHARE", "MIN"]].assign(yr=done["yr"] + 1).rename(columns={"GP_SHARE": "WAS", "MIN": "MIN_WAS"})
    both = done.merge(was, on=["PLAYER_ID", "yr"])
    both = both[both["MIN_WAS"] >= min_min]
    slope, base = np.polyfit(both["WAS"].astype(float), both["GP_SHARE"].astype(float), 1)
    return base, slope


def _can_reach(pool):
    """From 2023-24: could he still finish with 65 games (less one of slack)? Earlier seasons had no minimum."""
    left = (SEASON_GAMES - pool["TEAM_GAMES"]).clip(lower=0)
    return (pool["SEASON"] < GAMES_RULE_FROM) | (pool["GP"] + left >= GAMES_RULE - GAMES_SLACK)


def _teams(chance, eligible, names=TEAM_NAMES):
    """The highest chances among eligible players, five to a team."""
    order = chance[eligible].sort_values(ascending=False).index[:5 * len(names)]
    out = pd.Series("", index=chance.index)
    for i, idx in enumerate(order):
        out[idx] = names[len(names) - i // 5]
    return out


def fit_allnba_model(df_master, rating_model, penalty=None, verbose=True):
    """Train the voting model. Returns an AllNBAModel; .check is the season-by-season test.

    rating_model: from fit_rating_model — its rating and wins are two of the
    inputs, so the voting model can learn how much voters follow impact.
    penalty=None picks the regularisation by leaving out one season at a time.
    """
    return _fit_votes(df_master, rating_model, "nba", penalty, verbose, "All-NBA",
                      {"BY_WINS": "WINS_PACE", "BY_BPM": "BBR_BPM", "BY_POINTS": "PTS"},
                      "the top 15 by wins added gets {BY_WINS}, by BPM {BY_BPM}, by points per game {BY_POINTS}")


def fit_alldefense_model(df_master, rating_model, penalty=None, verbose=True):
    """Train the All-Defensive voting model, the same way as fit_allnba_model but on that vote and defensive stats.

    Ten spots a season instead of fifteen, so there is less to learn from and
    it is the less accurate of the two; .check has the season-by-season test.
    PRIOR here is All-Defensive selections in the previous three seasons.
    """
    return _fit_votes(df_master, rating_model, "defense", penalty, verbose, "All-Defensive",
                      {"BY_D_RATING": "D_RATING", "BY_DBPM": "BBR_DBPM", "BY_STOCKS": "STOCKS"},
                      "the top 10 by defensive impact rating gets {BY_D_RATING}, by defensive BPM {BY_DBPM}, "
                      "by steals plus blocks {BY_STOCKS}")


def fit_mip_model(df_master, rating_model, penalty=None, verbose=True):
    """Train the Most Improved Player model on who finished in the top five of that vote, 2016-17 on.

    Its inputs are changes from the previous season (scoring, minutes, role,
    efficiency, impact) plus the things that make voters look elsewhere: age,
    being in a second season, and having already made an All-NBA team. Players
    without 20 games the season before are left out, which removes rookies.
    Five spots a season is very little to learn from — read .check.
    """
    return _fit_votes(df_master, rating_model, "mip", penalty, verbose, "Most Improved Player",
                      {"BY_SCORING_JUMP": "UP_PTS", "BY_RATING_JUMP": "UP_RATING", "BY_MINUTES_JUMP": "UP_MIN"},
                      "the five biggest scoring jumps get {BY_SCORING_JUMP}, impact-rating jumps {BY_RATING_JUMP}, "
                      "minutes jumps {BY_MINUTES_JUMP}")


def _fit_votes(df_master, rating_model, award, penalty, verbose, title, yardsticks, yardstick_text):
    stats, names = _AWARDS[award][1], _AWARDS[award][3]
    spots = 5 * len(names)
    pool, Z = _voter_table(df_master, rating_model, award)
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
        team = _teams(chance[here], eligible[here], names)
        real = pool.loc[here, "VOTES"].map(names).fillna("")
        hit = (team != "") & (real != "")
        by = lambda col: set(pool.loc[here].nlargest(spots, col).index)         # simple yardsticks
        actual = set(real[real != ""].index)
        rows.append({"SEASON": s, f"RIGHT_OF_{spots}": int(hit.sum()), "FIRST_TEAM_OF_5": int(((team == "1st") & (real == "1st")).sum()),
                     "SAME_TEAM": int(((team == real) & hit).sum()),
                     **{name: len(by(col) & actual) for name, col in yardsticks.items()}})
        wrong = (team != "") ^ (real != "")
        for idx in wrong[wrong].index:
            missed.append({"SEASON": s, "Player": pool.at[idx, "PLAYER_NAME"], "PROJECTED": team[idx], "ACTUAL": real[idx],
                           "CHANCE": chance[idx], "GP": int(pool.at[idx, "GP"])})
    check = pd.DataFrame(rows)

    fit = LogisticRegression(C=penalty, max_iter=5000).fit(Z[known], y[known])
    weights = pd.Series(fit.coef_[0], index=stats).sort_values(key=np.abs, ascending=False)
    if verbose:
        n, right = len(seasons) * spots, check[f"RIGHT_OF_{spots}"]
        print(f"\n{title} voting model: {int(y[known].sum())} selections over {len(seasons)} seasons, "
              f"{int(known.sum())} player-seasons, penalty C={penalty:g}")
        first = f", first team {check['FIRST_TEAM_OF_5'].sum()} of {len(seasons) * 5}" if "1st" in names.values() else ""
        print(f"Each season predicted without seeing it: {right.sum()} of {n} spots right "
              f"({right.mean():.1f} of {spots} a season){first}")
        print("  for comparison, " + yardstick_text.format(**{k: check[k].sum() for k in yardsticks}))
        print("What voters reward most: " + ", ".join(f"{LABELS[k]} ({v:+.2f})" for k, v in weights.head(6).items()))
    return AllNBAModel(weights, float(fit.intercept_[0]), penalty, seasons, check, pd.DataFrame(missed), award)


def _voter_scores(voters, df_master, rating_model, season):
    """One season's rotation players, each stat's pull on the vote, and the chance of an All-NBA spot."""
    pool, Z = _voter_table(df_master, rating_model, voters.award)
    here = (pool["SEASON"] == season).to_numpy()
    if not here.any():
        raise ValueError(f"no rotation players for {season}")
    pool, Z = pool[here], Z[here]
    pull = Z * voters.weights.reindex(Z.columns)
    chance = pd.Series(1 / (1 + np.exp(-(pull.sum(axis=1) + voters.intercept))), index=pool.index)
    return pool, pull, chance


def _reasons(row, sign):
    picked = (row * sign).nlargest(2)
    return ", ".join(LABELS[k] for k, v in picked.items() if v > 0.15)


def predict_allnba(voters, df_master, rating_model, season="2026-27", top=30):
    """Each player's chance of an All-NBA spot for a season, best first, with the projected teams.

    CHANCE      probability of making any of the three teams
    PROJECTED   1st / 2nd / 3rd if he is among the 15 most likely eligible players
    ACTUAL      what happened, for seasons already voted on
    CAN_REACH_65  False once he has missed too many games to qualify (2023-24 on)
    WHY / WHY_NOT  the stats pushing his chance up and down the most
    RATING_RANK   where the impact model has him in the same pool, for contrast

    During a season the inputs are this season's per-game numbers and his
    team's record so far: "if the vote were today on today's pace". Only his
    games are projected to the end of the schedule (_project_season). It can't
    know about injuries still to come.
    """
    pool, pull, chance = _voter_scores(voters, df_master, rating_model, season)
    eligible = _can_reach(pool)
    reasons = _reasons

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


def predict_allstars(voters, df_master, rating_model, season="2026-27", spots=12, next_up=5):
    """All-Stars by conference: the `spots` players voters would rate highest, then the next few in line.

    SPOT     "starter" for a conference's top five, "reserve" for the rest of the 12, "next" for the replacements
    SCORE    the voting model's score (higher is better); it keeps separating players after CHANCE has hit zero
    CHANCE   his chance of an All-NBA spot, for scale
    RATING_RANK  where the impact model has him in his conference, for contrast

    This borrows the All-NBA voting model — there is no All-Star voting history
    in the data to train on. The two honours reward the same things (scoring,
    winning, playing), but All-Star starters are half fan vote, and popularity
    is not something the data sees. Positions are ignored, and there is no
    65-game rule: the vote happens in January.
    """
    pool, pull, chance = _voter_scores(voters, df_master, rating_model, season)
    out = pd.DataFrame({
        "CONF": np.where(pool["TEAM_ABBREVIATION"].isin(EAST), "East", "West"),
        "Player": pool["PLAYER_NAME"], "TEAM": pool["TEAM_ABBREVIATION"], "GP": pool["GP"].astype(int),
        "PTS": pool["PTS"], "W_PCT": pool["W_PCT"], "SCORE": pull.sum(axis=1) + voters.intercept, "CHANCE": chance,
        "RATING": pool["RATING"], "WHY": [_reasons(r, 1) for _, r in pull.iterrows()],
    })
    out["RATING_RANK"] = out.groupby("CONF")["RATING"].rank(ascending=False).astype(int)
    out = out.sort_values(["CONF", "SCORE"], ascending=[True, False])
    out["RANK"] = out.groupby("CONF").cumcount() + 1
    out = out[out["RANK"] <= spots + next_up].copy()
    out["SPOT"] = np.select([out["RANK"] <= 5, out["RANK"] <= spots], ["starter", "reserve"], default="next")
    cols = ["CONF", "SPOT", "RANK", "Player", "TEAM", "GP", "PTS", "W_PCT", "SCORE", "CHANCE", "RATING_RANK", "WHY"]
    out = out[cols].reset_index(drop=True)
    out.attrs.update(season=season)
    return out


_defense_cache = {}


def pick_all_defense(rating_model, df_master, season="2026-27", top=20, voters=None):
    """All-Defensive teams: each player's chance of a spot, best first, with the two projected teams.

    CHANCE     probability of making either team, from the All-Defensive voting model
    PROJECTED  1st / 2nd for the ten most likely eligible players
    ACTUAL     what happened, for seasons already voted on
    WHY        what is pushing his chance up the most
    The rest are context: this model's D_RATING, defensive BPM, steals, blocks
    and deflections per game, how far below their usual percentage opponents
    shoot against him (OPP_FG_DIFF, negative is good) and what they shoot at
    the rim.

    voters: a model from fit_alldefense_model; fitted here (once per master)
    when not given. Same 65-game rule as All-NBA. Reputation counts for a lot
    in this vote, so a first-time candidate is rated lower than his numbers.
    """
    if voters is None:
        key = (id(df_master), id(rating_model))
        if key not in _defense_cache:
            _defense_cache.clear()
            _defense_cache[key] = fit_alldefense_model(df_master, rating_model, verbose=False)
        voters = _defense_cache[key]
    pool, pull, chance = _voter_scores(voters, df_master, rating_model, season)
    eligible = _can_reach(pool)
    get = lambda c: pd.to_numeric(pool[c], errors="coerce") if c in pool.columns else np.nan
    out = pd.DataFrame({
        "Player": pool["PLAYER_NAME"], "TEAM": pool["TEAM_ABBREVIATION"], "GP": pool["GP"].astype(int), "MIN": pool["MIN"],
        "CHANCE": chance, "PROJECTED": _teams(chance, eligible, DEFENSE_TEAM_NAMES), "CAN_REACH_65": eligible,
        "D_RATING": pool["D_RATING"], "DBPM": get("BBR_DBPM"), "STL": get("STL"), "BLK": get("BLK"),
        "DEFLECTIONS": get("DEFLECTIONS"), "OPP_FG_DIFF": get("PCT_PLUSMINUS"), "RIM_FG_PCT": get("DEF_RIM_FG_PCT"),
        "WHY": [_reasons(r, 1) for _, r in pull.iterrows()],
    })
    if pool["DECIDED"].any():
        out.insert(6, "ACTUAL", pool["VOTES"].map(DEFENSE_TEAM_NAMES).fillna(""))
    out = out.sort_values("CHANCE", ascending=False).reset_index(drop=True)
    out.insert(0, "RANK", np.arange(1, len(out) + 1))
    out.attrs.update(season=season, n=len(out))
    return out.head(top) if top else out


def predict_mip(voters, df_master, rating_model, season="2026-27", top=10, strict=True):
    """Most Improved Player candidates for a season, most likely first.

    strict=True leaves out the players voters almost never choose however big
    the jump: second-year players (none of the ten winners since 2016-17, though
    9 of the 50 top-five finishers) and anyone who has already made an All-NBA
    team (1 of 50). The model already marks both down; strict removes them.
    strict=False shows everyone, with the reason in WHY_NOT.

    CHANCE     probability of finishing in the top five of the vote
    PTS_WAS / MIN_WAS / RATING_WAS   last season's numbers next to this season's
    SOPHOMORE  True for second-year players, who rarely win
    FINISH     where he actually finished, for seasons already voted on
    WHY / WHY_NOT  what is pushing his chance up and down the most

    Only players with 20+ games last season are candidates, so no rookies, and
    the 65-game rule applies. Early in a season the changes are this season's
    first weeks against all of last season, so hot starts are overrated.
    """
    pool, pull, chance = _voter_scores(voters, df_master, rating_model, season)
    out = pd.DataFrame({
        "Player": pool["PLAYER_NAME"], "TEAM": pool["TEAM_ABBREVIATION"], "AGE": pool["BBR_AGE"], "GP": pool["GP"].astype(int),
        "CHANCE": chance, "PTS": pool["PTS"], "PTS_WAS": pool["PREV_PTS"], "MIN": pool["MIN"], "MIN_WAS": pool["PREV_MIN"],
        "RATING": pool["RATING"], "RATING_WAS": pool["PREV_RATING"], "SOPHOMORE": pool["SOPHOMORE"] == 1,
        "ALL_NBA_BEFORE": pool["PRIOR_NBA"] > 0, "CAN_REACH_65": _can_reach(pool),
        "WHY": [_reasons(r, 1) for _, r in pull.iterrows()], "WHY_NOT": [_reasons(r, -1) for _, r in pull.iterrows()],
    })
    if pool["DECIDED"].any() and "FINISH" in pool.columns:
        out.insert(5, "FINISH", pool["FINISH"])
    out = out[out["CAN_REACH_65"]]
    if strict:
        out = out[~out["SOPHOMORE"] & ~out["ALL_NBA_BEFORE"]]
    out = out.sort_values("CHANCE", ascending=False).reset_index(drop=True)
    out.insert(0, "RANK", np.arange(1, len(out) + 1))
    out.attrs.update(season=season, n=len(out), strict=strict)
    return out.head(top) if top else out
