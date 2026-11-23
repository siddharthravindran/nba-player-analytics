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
    mvp = fit_mvp_model(df_master, model)                  # MVP: trained on every player's share of the vote
    predict_mvp(mvp, df_master, model)                     # chance of winning and expected share of the vote

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
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from bbref_stats import DATA_DIR, _get, fetch_injuries, fetch_mvp_tracker
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
             "OVER_BEST_PTS", "PTS", "MIN", "W_PCT", "GP_SHARE", "BBR_AGE", "SOPHOMORE", "PRIOR_NBA", "PRIOR"]
_CHANGES = ["PTS", "MIN", "AST", "REB", "TS_PCT", "USG_PCT", "PIE", "RATING", "WINS_PACE", "W_PCT"]
LABELS.update({"UP_PTS": "scoring jump", "UP_MIN": "more minutes", "UP_AST": "more assists", "UP_REB": "more rebounds",
               "UP_TS_PCT": "better efficiency", "UP_USG_PCT": "bigger role", "UP_PIE": "more all-round production",
               "UP_RATING": "better impact rating", "UP_WINS_PACE": "more wins added", "UP_W_PCT": "team improved",
               "OVER_BEST_PTS": "scoring above his previous best", "BBR_AGE": "age", "SOPHOMORE": "second-year player", "PRIOR_NBA": "already an All-NBA player"})
MIP_POOL_MIN, MIP_PREV_GP = 20.0, 15         # a "real" previous season: 15+ games. Held-out is the same at 10, 15
                                             # or 20; 15 is a judgment (Ty Jerome's 15 games in 2025-26 count as his leap)
POOL_MIN, POOL_GP_SHARE = 24.0, 0.45         # every All-NBA player since 2015-16 clears both
DEFENSE_POOL_MIN = 20.0                      # All-Defensive specialists play less
GAMES_RULE_FROM, GAMES_RULE, SEASON_GAMES = "2023-24", 65, 82
# The league's count is not always the stat sheet's: in 2025-26 three First Team players show 64 games here
# (two won "extraordinary circumstances" appeals). One game of slack keeps the filter from ruling out real picks.
GAMES_SLACK = 1
# A season in progress: games played are projected to the end of the schedule for every award (_project_season).
# For All-Defensive the stats are also weighed against this many games of last season. 25 is a judgment, not fitted.
EARLY_PRIOR_GAMES = 25
_PULL_TO_LAST_SEASON = {"defense", "defense_norep"}
# Games a player is known to be missing from today, for the season in progress: {"Stephen Curry": 12}.
# Set it in the notebook (KNOWN_ABSENCES["Stephen Curry"] = 12) and every award counts those games as missed, in
# the games projection and the 65-game rule. Weeks to games: about 3.5 games a week. Clear it when he is back.
KNOWN_ABSENCES = {}
GAMES_PER_WEEK = 3.5
INDEFINITE_GAMES = 10                        # "out" with no timeline: about three weeks. A judgment, not fitted
# All-Star: coaches and media pass over players who have missed a lot by the vote. Starters are picked about
# 40 games in, reserves about 50; more than about 10 games missed by then is roughly the cut-off (an estimate).
ALLSTAR_VOTE_AT = {"starter": 40, "reserve": 50}
ALLSTAR_MISSED_LIMIT = 10           # the awards whose stats are pulled toward last season; games are projected for all
EARLY_PRIOR_MIN_GP = 20                      # a "last season" shorter than this is skipped for the one before
_TEAM_STATS = ["W_PCT", "DEF_RATING", "ONOFF_DEF_RATING"]         # say little about him once he has changed teams
_NOT_PROJECTED = ["PRIOR", "PRIOR_NBA", "SOPHOMORE", "BBR_AGE"]   # facts about him, not this season's play
PENALTIES = (0.01, 0.03, 0.1, 0.3, 1.0)
TEAM_NAMES = {3: "1st", 2: "2nd", 1: "3rd"}
DEFENSE_TEAM_NAMES = {2: "1st", 1: "2nd"}
EAST = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"}

__all__ = ["KNOWN_ABSENCES", "use_injury_report", "add_absences", "fit_allstar_model", "save_award_snapshot", "award_movers", "AllNBAModel", "fit_roy_model", "predict_roy", "fit_mip_winner_model", "fit_smoy_model", "predict_smoy", "fit_allnba_model", "predict_allnba", "predict_allstars", "fit_alldefense_model", "pick_all_defense",
           "fit_mip_model", "predict_mip", "fit_mvp_model", "predict_mvp"]


@dataclass
class AllNBAModel:
    weights: pd.Series               # per stat, in standard deviations; largest first
    intercept: float
    penalty: float
    seasons: list
    check: pd.DataFrame = field(default=None, repr=False)      # leave-one-season-out results
    missed: pd.DataFrame = field(default=None, repr=False)     # every wrong call in those runs
    award: str = "nba"                                         # "nba", "defense", "mip" or "mvp"
    temperature: float = 1.0                                   # MVP only: sharpens vote shares into a chance of winning


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


def _mvp_votes(refresh=False):
    """Every player who got an MVP vote (Basketball-Reference): name_key, yr, weight, finish, share.

    share is the page's Share column: points won over points possible, so the
    winner is near 1 and a season's shares add up to about 2.6. weight is the
    share in thousandths, an integer like the other awards' weights; PRIOR adds
    up the last three seasons of it ("past MVP votes").
    """
    cache = DATA_DIR / "mvp_voting.parquet"
    if cache.exists() and not refresh:
        raw = pd.read_parquet(cache)
    else:
        from io import StringIO
        frames = []
        for year in range(int(min(SEASONS)[:4]) + 1, int(max(SEASONS)[:4]) + 1):
            html = _get(f"https://www.basketball-reference.com/awards/awards_{year}.html")
            at = html.find('id="mvp"')
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
    share = pd.to_numeric(raw["Share"], errors="coerce").fillna(0.0)
    return pd.DataFrame({"name_key": raw["Player"].map(norm_name), "yr": raw["SEASON"].str[:4].astype(int),
                         "weight": (share * 1000).round().astype(int), "finish": raw.groupby("SEASON").cumcount() + 1,
                         "share": share})


RISING_STARS_TEAMS = {"Team Candace"}       # 2025's tournament included a rookie/sophomore team; not All-Stars


def _allstar_marked(refresh=False):
    """Players marked All-Star ("AS") in each season's Basketball-Reference per-game table: name_key, yr.

    Catches selections the game pages miss (players who sat out for protocols
    or anything else); it includes replacements, which _allstar_selections
    takes back out using the game notes.
    """
    cache = DATA_DIR / "allstar_awards.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    from io import StringIO
    rows = []
    for year in range(int(min(SEASONS)[:4]) + 1, int(max(SEASONS)[:4]) + 1):
        html = _get(f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html")
        at = html.find('id="per_game_stats"')
        if at < 0:
            continue
        t = pd.read_html(StringIO(html[html.rfind("<table", 0, at):html.find("</table>", at) + 8]))[0]
        if "Awards" not in t.columns:
            continue
        names = t.loc[t["Awards"].astype(str).str.contains(r"\bAS\b", na=False), "Player"].unique()
        rows += [{"name_key": norm_name(n), "yr": year - 1} for n in names]
    out = pd.DataFrame(rows).drop_duplicates()
    out.to_parquet(cache)
    return out


def _allstar_selections(refresh=False):
    """Every All-Star selection since 2015-16 (Basketball-Reference box scores): name_key, yr, weight.

    The box scores list who played; the notes add players selected who didn't
    play (injury, health and safety protocols, coach's decision) and name their replacements, who are taken
    out: a replacement is the commissioner's pick, not the vote's. From 2025
    the game is a tournament over several pages, and the Rising Stars team is
    left out (RISING_STARS_TEAMS).
    """
    cache = DATA_DIR / "allstar_rosters.parquet"
    if cache.exists() and not refresh:
        raw = pd.read_parquet(cache)
    else:
        from io import StringIO
        rows = []
        for year in range(int(min(SEASONS)[:4]) + 1, int(max(SEASONS)[:4]) + 1):
            main = _get(f"https://www.basketball-reference.com/allstar/NBA_{year}.html")
            games = sorted(set(re.findall(rf'href="/allstar/({year}02\d+NBA)\.html"', main)))
            pages = [main] + [_get(f"https://www.basketball-reference.com/allstar/{g}.html") for g in games]
            for html in pages:
                if "line_score" not in html:
                    continue
                for tid in re.findall(r'<table[^>]*id="([^"]+)"', html):
                    if tid == "line_score":
                        continue
                    at = html.find(f'id="{tid}"')
                    t = pd.read_html(StringIO(html[html.rfind("<table", 0, at):html.find("</table>", at) + 8]))[0]
                    for p in t.iloc[:, 0]:
                        if isinstance(p, str) and p not in ("Reserves", "Starters", "Team Totals"):
                            rows.append({"yr": year - 1, "player": p, "team": tid, "kind": "played"})
                text = re.sub(r"<[^>]+>", "", html)
                # any "did not play" note: hurt, health and safety protocols, coach's decision; all were selected
                for who, team, rest in re.findall(r"([A-Z][\w.'\- ]{2,40}?) \(([^()]+)\): Did not play[^;\n]*;?([^\n]{0,120})", text):
                    rows.append({"yr": year - 1, "player": who.strip(), "team": team, "kind": "injured"})
                    for sub in re.findall(r"Replaced by ([A-Z][\w.'\- ]+?)(?:$| and|\s{2})", rest) + \
                               re.findall(r"([A-Z][\w.'\- ]+?) named as replacement", rest):
                        rows.append({"yr": year - 1, "player": sub.strip(), "team": team, "kind": "replacement"})
        raw = pd.DataFrame(rows).drop_duplicates()
        raw.to_parquet(cache)
    raw = raw[~raw["team"].isin(RISING_STARS_TEAMS)].assign(name_key=lambda r: r["player"].map(norm_name))
    marked = _allstar_marked(refresh)                       # every All-Star by Basketball-Reference's awards column
    out = pd.concat([raw.loc[raw["kind"].isin(["played", "injured"]), ["name_key", "yr"]], marked]).drop_duplicates()
    subs = raw.loc[raw["kind"] == "replacement", ["name_key", "yr"]].drop_duplicates()
    out = out.merge(subs, how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
    return out.assign(weight=1)


_AWARDS = {   # award -> (selections, stats, minimum minutes, team labels)
    "nba": (_allnba_selections, VOTER_STATS, POOL_MIN, TEAM_NAMES),
    "defense": (_alldefense_selections, DEFENSE_STATS, DEFENSE_POOL_MIN, DEFENSE_TEAM_NAMES),
    "defense_norep": (_alldefense_selections, [s for s in DEFENSE_STATS if s != "PRIOR"], DEFENSE_POOL_MIN, DEFENSE_TEAM_NAMES),
    "mip": (_mip_selections, MIP_STATS, MIP_POOL_MIN, {1: "top 5"}),
    "mvp": (_mvp_votes, VOTER_STATS, POOL_MIN, {1: "MVP"}),
    "star": (_allstar_selections, VOTER_STATS, POOL_MIN, {1: "All-Star"}),
}
MVP_PENALTIES = (0.003, 0.01, 0.03, 0.1, 0.3)


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
    out = table["PLAYER_NAME"].map(KNOWN_ABSENCES) if KNOWN_ABSENCES else pd.Series(np.nan, index=table.index)
    table["OUT"] = out.where(table["SEASON"] == in_progress).fillna(0).astype(float)
    unknown = set(KNOWN_ABSENCES) - set(df.loc[df["SEASON"] == in_progress, "PLAYER_NAME"])
    if unknown:
        print(f"⚠️ KNOWN_ABSENCES names not found this season: {sorted(unknown)}")
    table = table.sort_values("GP").drop_duplicates(["PLAYER_ID", "SEASON"], keep="last").sort_index().reset_index(drop=True)
    table["name_key"] = table["PLAYER_NAME"].map(norm_name)
    table["yr"] = table["SEASON"].str[:4].astype(int)

    # this season against his last real season (20+ games): last season, or the one before if he barely played
    # (Ty Jerome 2026-27: 15 games in 2025-26, so 2024-25 is the baseline). UP_PTS and so on.
    # every season in the master, not only the rated ones: a short season can be missing from rank_players
    every = df.sort_values("GP").drop_duplicates(["PLAYER_ID", "SEASON"], keep="last")
    every = every.assign(yr=every["SEASON"].str[:4].astype(int))[["PLAYER_ID", "yr", "GP"] + [c for c in _CHANGES if c in df.columns]]
    every = every.merge(table[["PLAYER_ID", "yr"] + [c for c in _CHANGES if c not in df.columns]], on=["PLAYER_ID", "yr"], how="left")
    seasons = every[["PLAYER_ID", "yr", "GP"] + _CHANGES].apply(lambda c: pd.to_numeric(c, errors="coerce") if c.name != "PLAYER_ID" else c)
    seasons = seasons[seasons["GP"] >= MIP_PREV_GP].rename(columns={c: f"PREV_{c}" for c in ["GP"] + _CHANGES})
    last = None
    for back in (1, 2):
        b = seasons.assign(yr=seasons["yr"] + back).set_index(["PLAYER_ID", "yr"])
        last = b if last is None else last.combine_first(b)
    table = table.merge(last.reset_index(), on=["PLAYER_ID", "yr"], how="left")
    for c in _CHANGES:
        table[f"UP_{c}"] = pd.to_numeric(table[c], errors="coerce") - table[f"PREV_{c}"]
    # scoring against his best earlier season, not just last season: voters reward a new level, not a bounce-back
    # (49 of 50 top-five finishers were above their previous best; none of 86 three-point bounce-backs made it)
    real = every.loc[every["GP"] >= MIP_PREV_GP, ["PLAYER_ID", "yr", "PTS"]].assign(PTS=lambda t: pd.to_numeric(t["PTS"], errors="coerce"))
    then = table[["PLAYER_ID", "yr"]].reset_index().merge(real, on="PLAYER_ID", suffixes=("", "_then"))
    table["BEST_PTS"] = then[then["yr_then"] < then["yr"]].groupby("index")["PTS"].max().reindex(table.index)
    table["OVER_BEST_PTS"] = (pd.to_numeric(table["PTS"], errors="coerce") - table["BEST_PTS"]).fillna(table["UP_PTS"])
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
    pool["REACH_65"] = _reach_chance(pool, in_progress)
    for spot, at in ALLSTAR_VOTE_AT.items():
        pool[f"ALLSTAR_OK_{spot.upper()}"] = _allstar_chance(pool, in_progress, at)
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
    schedule at the usual rate for a rotation player (about 82%), the same for
    everyone. So the games he has already missed count for what they are, and
    four missed games in November no longer look like missing a third of a season.

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
        # the rest of the schedule at the usual rate for everyone: last season's injuries are not held against him
        out = cur["OUT"].clip(upper=left) if "OUT" in cur.columns else 0.0            # KNOWN_ABSENCES
        share = ((games + (left - out) * usual) / (cur["TEAM_GAMES"] + left)).clip(upper=1.0)
        projected["GP_SHARE"] = ((share - usual) / usual_spread).clip(-5, 5)

    Z = Z.copy()
    Z.loc[cur.index] = projected[stats]
    return Z


def _availability(pool, in_progress):
    """Beta parameters for the share of games a rotation player plays, from finished seasons."""
    done = pool.loc[(pool["SEASON"] < in_progress), "GP_SHARE"].clip(0.005, 0.995)
    m, v = done.mean(), done.var()
    k = m * (1 - m) / v - 1
    return m * k, (1 - m) * k


def _chance_to_play(need, can_play, a, b):
    """Chance of playing at least `need` of `can_play` games: beta-binomial, so missed games come in clumps."""
    from scipy.stats import betabinom
    return np.array([1.0 if g <= 0 else 0.0 if g > n else float(betabinom.sf(g - 1, n, a, b))
                     for g, n in zip(np.asarray(need, int), np.asarray(can_play, int))])


def _allstar_chance(pool, in_progress, at):
    """Chance he has missed no more than ALLSTAR_MISSED_LIMIT games by the time the All-Star vote is `at` games in."""
    out = pd.Series(1.0, index=pool.index)
    now = pool["SEASON"] == in_progress
    if in_progress is None or not now.any():
        return out
    a, b = _availability(pool, in_progress)
    cur = pool[now]
    window = (at - cur["TEAM_GAMES"]).clip(lower=0)
    out_then = cur.get("OUT", 0)
    out_then = np.minimum(out_then, window) if np.ndim(out_then) else 0
    missed = cur["TEAM_GAMES"] - cur["GP"] + out_then
    can_play = (window - out_then).clip(lower=0)
    need = can_play - (ALLSTAR_MISSED_LIMIT - missed)              # games he must play of the ones he can
    p = _chance_to_play(need, can_play, a, b)
    p[(ALLSTAR_MISSED_LIMIT - missed).to_numpy() < 0] = 0.0
    out.loc[cur.index] = p
    return out


def _reach_chance(pool, in_progress):
    """Chance he finishes with 65 games (less one of slack), 1.0 before the rule.

    Finished seasons: 1 or 0. The season in progress: the games he still needs
    against a beta-binomial for how many of the rest he plays, fitted to the
    share of games rotation players played in finished seasons (missed games
    come in clumps, so it is wider than a plain coin-flip per game). Games in
    KNOWN_ABSENCES are taken off the games he can still play.
    """
    out = _can_reach(pool).astype(float)
    now = (pool["SEASON"] == in_progress) & (pool["SEASON"] >= GAMES_RULE_FROM)
    if in_progress is None or not now.any():
        return out
    a, b = _availability(pool, in_progress)
    cur = pool[now]
    can_play = ((SEASON_GAMES - cur["TEAM_GAMES"]).clip(lower=0) - cur.get("OUT", 0)).clip(lower=0)
    out.loc[cur.index] = _chance_to_play(GAMES_RULE - GAMES_SLACK - cur["GP"], can_play, a, b)
    return out


def _can_reach(pool):
    """From 2023-24: could he still finish with 65 games (less one of slack)? Earlier seasons had no minimum."""
    left = (SEASON_GAMES - pool["TEAM_GAMES"]).clip(lower=0) - pool.get("OUT", 0)
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


def fit_alldefense_model(df_master, rating_model, penalty=None, verbose=True, reputation=True):
    """Train the All-Defensive voting model, the same way as fit_allnba_model but on that vote and defensive stats.

    Ten spots a season instead of fifteen, so there is less to learn from and
    it is the less accurate of the two; .check has the season-by-season test.
    PRIOR here is All-Defensive selections in the previous three seasons.
    reputation=False drops PRIOR: the vote as if judged on this season's defense alone.
    """
    return _fit_votes(df_master, rating_model, "defense" if reputation else "defense_norep", penalty, verbose,
                      "All-Defensive" if reputation else "All-Defensive (no reputation)",
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


def _voter_scores(voters, df_master, rating_model, season, games_rule=True):
    """One season's rotation players, each stat's pull on the vote, and the chance of an All-NBA spot.

    With games_rule the chance is multiplied by REACH_65, the chance he gets to
    65 games; All-Star has no such rule and passes False.
    """
    pool, Z = _voter_table(df_master, rating_model, voters.award)
    here = (pool["SEASON"] == season).to_numpy()
    if not here.any():
        raise ValueError(f"no rotation players for {season}")
    pool, Z = pool[here], Z[here]
    pull = Z * voters.weights.reindex(Z.columns)
    chance = pd.Series(1 / (1 + np.exp(-(pull.sum(axis=1) + voters.intercept))), index=pool.index)
    if games_rule:
        chance = chance * pool["REACH_65"]
    return pool, pull, chance


def _reasons(row, sign):
    picked = (row * sign).nlargest(2)
    return ", ".join(LABELS[k] for k, v in picked.items() if v > 0.15)


_BOX = ["PTS", "REB", "AST", "STL", "BLK", "FG3_PCT", "EFG_PCT"]          # shown next to the chances; not all are model inputs


def _box(pool):
    return {c: pd.to_numeric(pool[c], errors="coerce") if c in pool.columns else np.nan for c in _BOX}


def predict_allnba(voters, df_master, rating_model, season="2026-27", top=30):
    """Each player's chance of an All-NBA spot for a season, best first, with the projected teams.

    CHANCE      probability of making any of the three teams
    PROJECTED   1st / 2nd / 3rd if he is among the 15 most likely eligible players
    ACTUAL      what happened, for seasons already voted on
    CAN_REACH_65  False once he has missed too many games to qualify (2023-24 on)
    REACH_65    chance he gets there; CHANCE is already multiplied by it
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
        "TEAM_GAMES": pool["TEAM_GAMES"].astype(int), **_box(pool), "W_PCT": pool["W_PCT"],
        "CHANCE": chance, "PROJECTED": _teams(chance, eligible),
        "CAN_REACH_65": eligible, "REACH_65": pool["REACH_65"], "RATING_RANK": pool["RATING"].rank(ascending=False).astype(int),
        "WHY": [reasons(r, 1) for _, r in pull.iterrows()], "WHY_NOT": [reasons(r, -1) for _, r in pull.iterrows()],
    })
    if pool["DECIDED"].any():
        out.insert(out.columns.get_loc("CAN_REACH_65"), "ACTUAL", pool["VOTES"].map(TEAM_NAMES).fillna(""))
    out = out.sort_values("CHANCE", ascending=False).reset_index(drop=True)
    out.insert(0, "RANK", np.arange(1, len(out) + 1))
    out.attrs.update(season=season, n=len(out))
    return out.head(top) if top else out


def predict_allstars(voters, df_master, rating_model, season="2026-27", spots=12, next_up=12):
    """All-Stars by conference: the `spots` players voters would rate highest, then the next few in line.

    SPOT     "starter" for a conference's top five, "reserve" for the rest of the 12, "next" for the next 12 in line (24 per conference)
    SCORE    the voting model's score (higher is better); it keeps separating players after CHANCE has hit zero
    AVAILABLE  chance he has missed no more than ALLSTAR_MISSED_LIMIT games by the vote
    CHANCE   his chance of being picked, from the All-Star voting model
    RATING_RANK  where the impact model has him in his conference, for contrast

    Uses the All-Star voting model (fit_allstar_model), trained on real
    All-Star selections; pass any voters, it fits that model once if needed.
    All-Star starters are half fan vote, and popularity is only seen through
    past selections. Positions are ignored, and there is no
    65-game rule: the vote happens in January. Coaches and media do pass over
    players who have missed a lot, though: starters are ranked on their chance
    of having missed no more than ALLSTAR_MISSED_LIMIT games by game 40, the
    rest on the same by game 50 (AVAILABLE). KNOWN_ABSENCES count as missed.
    """
    if voters is None or voters.award != "star":
        key = (id(df_master), id(rating_model))
        if key not in _allstar_cache:
            _allstar_cache.clear()
            _allstar_cache[key] = fit_allstar_model(df_master, rating_model, verbose=False)
        voters = _allstar_cache[key]
    pool, pull, chance = _voter_scores(voters, df_master, rating_model, season, games_rule=False)
    score = pull.sum(axis=1) + voters.intercept
    with np.errstate(divide="ignore"):                       # odds times the chance he's available; 0 sinks him
        as_starter = score + np.log(pool["ALLSTAR_OK_STARTER"])
        as_reserve = score + np.log(pool["ALLSTAR_OK_RESERVE"])
    out = pd.DataFrame({
        "CONF": np.where(pool["TEAM_ABBREVIATION"].isin(EAST), "East", "West"),
        "Player": pool["PLAYER_NAME"], "TEAM": pool["TEAM_ABBREVIATION"], "GP": pool["GP"].astype(int),
        **_box(pool), "W_PCT": pool["W_PCT"], "SCORE": score, "CHANCE": chance,
        "RATING": pool["RATING"], "WHY": [_reasons(r, 1) for _, r in pull.iterrows()],
        "_STARTER": as_starter, "_RESERVE": as_reserve,
        "OK_STARTER": pool["ALLSTAR_OK_STARTER"], "OK_RESERVE": pool["ALLSTAR_OK_RESERVE"],
    })
    out["RATING_RANK"] = out.groupby("CONF")["RATING"].rank(ascending=False).astype(int)
    parts = []
    for conf, g in out.groupby("CONF"):
        starters = g.nlargest(5, "_STARTER")
        rest = g.drop(starters.index).sort_values("_RESERVE", ascending=False)
        g = pd.concat([starters, rest.head(spots + next_up - 5)])
        g["RANK"] = np.arange(1, len(g) + 1)
        g["AVAILABLE"] = np.where(g["RANK"] <= 5, g["OK_STARTER"], g["OK_RESERVE"])
        parts.append(g)
    out = pd.concat(parts)
    out["SPOT"] = np.select([out["RANK"] <= 5, out["RANK"] <= spots], ["starter", "reserve"], default="next")
    cols = ["CONF", "SPOT", "RANK", "Player", "TEAM", "GP", *_BOX, "W_PCT", "SCORE", "CHANCE", "AVAILABLE", "RATING_RANK", "WHY"]
    out = out[cols].reset_index(drop=True)
    out.attrs.update(season=season)
    return out


_defense_cache = {}


def pick_all_defense(rating_model, df_master, season="2026-27", top=20, voters=None, reputation=True):
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
    in this vote, so a first-time candidate is rated lower than his numbers;
    reputation=False uses a model fitted without past selections (PRIOR).
    """
    if voters is None:
        key = (id(df_master), id(rating_model), reputation)
        if key not in _defense_cache:
            for old in [k for k in _defense_cache if k[:2] != key[:2]]:   # a new master or rating model: drop the old fits
                del _defense_cache[old]
            _defense_cache[key] = fit_alldefense_model(df_master, rating_model, verbose=False, reputation=reputation)
        voters = _defense_cache[key]
    pool, pull, chance = _voter_scores(voters, df_master, rating_model, season)
    eligible = _can_reach(pool)
    get = lambda c: pd.to_numeric(pool[c], errors="coerce") if c in pool.columns else np.nan
    out = pd.DataFrame({
        "Player": pool["PLAYER_NAME"], "TEAM": pool["TEAM_ABBREVIATION"], "GP": pool["GP"].astype(int), "MIN": pool["MIN"],
        "CHANCE": chance, "PROJECTED": _teams(chance, eligible, DEFENSE_TEAM_NAMES), "CAN_REACH_65": eligible,
        "REACH_65": pool["REACH_65"],
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


def predict_mip(voters, df_master, rating_model, season="2026-27", top=10, strict=True, winner=None):
    """Most Improved Player candidates for a season, most likely first.

    strict=True leaves out the players voters almost never choose however big
    the jump: second-year players (none of the ten winners since 2016-17, though
    9 of the 50 top-five finishers) and anyone who has already made an All-NBA
    team (1 of 50). The model already marks both down; strict removes them.
    strict=False shows everyone, with the reason in WHY_NOT.

    WIN_PCT    chance of winning, in percent, adding up to 100 over the candidates
               shown (only with winner= from fit_mip_winner_model; sorted by it then)
    CHANCE     probability of finishing in the top five of the vote (TOP5_PCT: the same in percent;
               these add up to about 500 over everyone, since five players finish top five)
    PTS_WAS / MIN_WAS / RATING_WAS   last season's numbers next to this season's
    PTS_BEST   his best scoring season before this one; a return to an old level rarely gets votes
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
        "CHANCE": chance, "PTS": pool["PTS"], "PTS_WAS": pool["PREV_PTS"], "PTS_BEST": pool["BEST_PTS"], "MIN": pool["MIN"], "MIN_WAS": pool["PREV_MIN"],
        "RATING": pool["RATING"], "RATING_WAS": pool["PREV_RATING"], "SOPHOMORE": pool["SOPHOMORE"] == 1,
        "ALL_NBA_BEFORE": pool["PRIOR_NBA"] > 0, "CAN_REACH_65": _can_reach(pool), "REACH_65": pool["REACH_65"],
        "WHY": [_reasons(r, 1) for _, r in pull.iterrows()], "WHY_NOT": [_reasons(r, -1) for _, r in pull.iterrows()],
    })
    if pool["DECIDED"].any() and "FINISH" in pool.columns:
        out.insert(5, "FINISH", pool["FINISH"])
    out = out[out["CAN_REACH_65"]]
    if strict:
        out = out[~out["SOPHOMORE"] & ~out["ALL_NBA_BEFORE"]]
    out.insert(out.columns.get_loc("CHANCE") + 1, "TOP5_PCT", (out["CHANCE"] * 100).round(1))
    order = "CHANCE"
    if winner is not None:                    # chance of winning, over the candidates shown; adds up to 100
        _, wz = _voter_table(df_master, rating_model, "mip")
        score = wz.loc[out.index] @ winner.weights.reindex(wz.columns).fillna(0)
        out.insert(out.columns.get_loc("CHANCE"), "WIN_PCT",
                   (_weighted_softmax(winner.temperature * score.to_numpy(), out["REACH_65"].to_numpy(float)) * 100).round(1))
        order = "WIN_PCT"
    out = out.sort_values(order, ascending=False).reset_index(drop=True)
    out.insert(0, "RANK", np.arange(1, len(out) + 1))
    out.attrs.update(season=season, n=len(out), strict=strict)
    return out.head(top) if top else out


def fit_mip_winner_model(df_master, rating_model, penalty=None, verbose=True):
    """Most Improved winner model, built like MVP: a softmax over each season's candidates fitted to vote shares.

    Same inputs and pool as fit_mip_model (changes from last season, age, second
    year, past All-NBA). Pass it to predict_mip(..., winner=) for WIN_PCT, each
    candidate's chance of winning, adding up to 100.
    """
    from scipy.optimize import minimize_scalar
    _mip_selections()                                                     # makes sure the vote cache exists
    raw = pd.read_parquet(DATA_DIR / "mip_voting.parquet")
    share = pd.Series(pd.to_numeric(raw["Share"], errors="coerce").fillna(0).to_numpy(),
                      index=pd.MultiIndex.from_arrays([raw["Player"].map(norm_name), raw["SEASON"].str[:4].astype(int)]))
    share = share.groupby(level=[0, 1]).max()
    pool, Z = _voter_table(df_master, rating_model, "mip")
    pool = pool.assign(SHARE=[share.get(k, 0.0) for k in zip(pool["name_key"], pool["yr"])])
    X = Z.to_numpy()
    seasons = sorted(s for s in pool.loc[pool["DECIDED"], "SEASON"].unique() if pool.loc[pool["SEASON"] == s, "SHARE"].sum() > 0)
    rows = {s: np.flatnonzero((pool["SEASON"] == s).to_numpy()) for s in seasons}
    sh = pool["SHARE"].to_numpy()
    target = {s: sh[rows[s]] / sh[rows[s]].sum() for s in seasons}

    def held_out(lam):
        out = {}
        for s in seasons:
            train = [x for x in seasons if x != s]
            out[s] = X[rows[s]] @ _fit_shares(X, [rows[x] for x in train], [target[x] for x in train], lam)
        return out

    if penalty is None:
        penalty = min(MVP_PENALTIES, key=lambda l: -np.mean([target[s] @ np.log(_softmax(sc) + 1e-12) for s, sc in held_out(l).items()]))
    scores = held_out(penalty)
    win = {s: int(np.argmax(sh[rows[s]])) for s in seasons}
    temperature = float(minimize_scalar(lambda t: -sum(np.log(_softmax(t * scores[s])[win[s]] + 1e-12) for s in seasons),
                                        bounds=(0.5, 4.0), method="bounded").x)
    rank = {s: int(np.flatnonzero(np.argsort(-scores[s]) == win[s])[0]) + 1 for s in seasons}
    jump = pd.to_numeric(pool["UP_PTS"], errors="coerce").to_numpy() if "UP_PTS" in pool.columns else None
    by_jump = {s: int(np.flatnonzero(np.argsort(-np.nan_to_num(jump[rows[s]], nan=-99)) == win[s])[0]) + 1 for s in seasons} if jump is not None else {}
    check = pd.DataFrame({"SEASON": seasons, "WINNER_RANK": [rank[s] for s in seasons],
                          "WINNER_CHANCE": [float(_softmax(temperature * scores[s])[win[s]]) for s in seasons],
                          **({"BIGGEST_SCORING_JUMP_RANK": [by_jump[s] for s in seasons]} if by_jump else {})})
    w = _fit_shares(X, list(rows.values()), list(target.values()), penalty)
    weights = pd.Series(w, index=Z.columns).sort_values(key=np.abs, ascending=False)
    if verbose:
        n = len(seasons)
        print(f"\nMost Improved winner model: {n} seasons of vote shares, penalty {penalty:g}, temperature {temperature:.2f}")
        print(f"Each season predicted without seeing it: winner first {sum(r == 1 for r in rank.values())} of {n}, "
              f"top three {sum(r <= 3 for r in rank.values())} of {n}"
              + (f"; the biggest scoring jump was the winner {sum(r == 1 for r in by_jump.values())} of {n}" if by_jump else ""))
        print("What voters reward: " + ", ".join(f"{LABELS.get(k, k)} ({v:+.2f})" for k, v in weights.head(6).items()))
    return AllNBAModel(weights, 0.0, penalty, seasons, check, None, "mip", temperature)


# ---------------------------------------------------------------- MVP
# One winner a season is too few to learn from (11 since 2015-16), so the MVP model learns from every player's
# share of the vote: a softmax over each season's pool, fitted to the shares. Its scores, sharpened by one
# temperature fitted on held-out seasons, give each player's chance of winning; the chances add up to 1.

def _softmax(s):
    e = np.exp(s - s.max())
    return e / e.sum()


def _weighted_softmax(s, weight):
    """Softmax where each player's term is scaled by the chance he is eligible at all."""
    e = np.exp(s - s.max()) * weight
    return e / e.sum()


def _fit_shares(X, groups, targets, lam):
    """Weights that make each season's softmax of X @ w match its vote shares, with an L2 penalty."""
    from scipy.optimize import minimize

    def loss(w):
        s, total, grad = X @ w, lam * w @ w, 2 * lam * w
        for ix, q in zip(groups, targets):
            p = _softmax(s[ix])
            total -= q @ np.log(p + 1e-12) / len(groups)
            grad -= X[ix].T @ (q - p) / len(groups)
        return total, grad
    return minimize(loss, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def _mvp_frame(df_master, rating_model):
    pool, Z = _voter_table(df_master, rating_model, "mvp")
    votes = _mvp_votes().set_index(["name_key", "yr"])
    pool["SHARE"] = [votes["share"].get(k, 0.0) for k in zip(pool["name_key"], pool["yr"])]
    pool["FINISH"] = [votes["finish"].get(k, np.nan) for k in zip(pool["name_key"], pool["yr"])]
    pool["ELIGIBLE"] = _can_reach(pool)
    team_record = pool.sort_values("GP").groupby(["SEASON", "TEAM_ABBREVIATION"])["W_PCT"].last()
    pool["TEAM_RANK"] = [team_record[s].rank(ascending=False, method="min")[t]
                         for s, t in zip(pool["SEASON"], pool["TEAM_ABBREVIATION"])]
    return pool, Z


def fit_mvp_model(df_master, rating_model, penalty=None, verbose=True):
    """MVP voting model, trained on every vote-getter's share since 2015-16, tested one season at a time.

    Accuracy is the winner picked, the winner's rank, and how many of the real
    top three are in the predicted top three, every season predicted without
    seeing it. Yardsticks: the best impact rating on a top-three team, the best
    impact rating anywhere, and the top scorer.
    """
    pool, Z = _mvp_frame(df_master, rating_model)
    stats = list(Z.columns)
    known = pool["DECIDED"].to_numpy() & pool["ELIGIBLE"].to_numpy()
    seasons = sorted(pool.loc[pool["DECIDED"], "SEASON"].unique())
    rows_of = {s: np.flatnonzero(known & (pool["SEASON"] == s).to_numpy()) for s in seasons}
    X, share = Z.to_numpy(float), pool["SHARE"].to_numpy(float)
    target = {s: share[ix] / share[ix].sum() for s, ix in rows_of.items()}

    def held_out(lam):
        scores = {}
        for s in seasons:
            train = [x for x in seasons if x != s]
            w = _fit_shares(X, [rows_of[x] for x in train], [target[x] for x in train], lam)
            scores[s] = X[rows_of[s]] @ w
        return scores

    if penalty is None:
        loss = {lam: -np.mean([target[s] @ np.log(_softmax(sc) + 1e-12) for s, sc in held_out(lam).items()])
                for lam in MVP_PENALTIES}
        penalty = min(loss, key=loss.get)
    scores = held_out(penalty)

    def winner_loglik(t):
        return -sum(np.log(_softmax(t * scores[s])[np.argmax(share[rows_of[s]])] + 1e-12) for s in seasons)
    from scipy.optimize import minimize_scalar
    temperature = float(minimize_scalar(winner_loglik, bounds=(0.5, 4.0), method="bounded").x)

    rows = []
    for s in seasons:
        ix = rows_of[s]
        here, sc = pool.iloc[ix], scores[s]
        order = np.argsort(-sc)
        real = np.argsort(-share[ix])
        win = real[0]
        yard = lambda m: here["PLAYER_NAME"].iloc[np.argmax(np.where(m, here["RATING"].to_numpy(float), -np.inf))]
        rows.append({"SEASON": s, "WINNER": here["PLAYER_NAME"].iloc[win], "PICKED": here["PLAYER_NAME"].iloc[order[0]],
                     "WINNER_RANK": int(np.flatnonzero(order == win)[0]) + 1,
                     "WIN_CHANCE": float(_softmax(temperature * sc)[win]),
                     "TOP3_OF_3": len(set(order[:3]) & set(real[:3])),
                     "TOP_TEAM_RATING": yard(here["TEAM_RANK"].to_numpy() <= 3),
                     "BEST_RATING": yard(np.ones(len(ix), bool)),
                     "TOP_SCORER": here["PLAYER_NAME"].iloc[np.argmax(here["PTS"].to_numpy(float))]})
    check = pd.DataFrame(rows)

    w = _fit_shares(X, list(rows_of.values()), list(target.values()), penalty)
    weights = pd.Series(w, index=stats).sort_values(key=np.abs, ascending=False)
    if verbose:
        n = len(seasons)
        hits = lambda c: int((check[c] == check["WINNER"]).sum())
        print(f"\nMVP voting model: {int((share[known] > 0).sum())} vote-getters over {n} seasons, "
              f"{int(known.sum())} player-seasons, penalty {penalty:g}, temperature {temperature:.2f}")
        print(f"Each season predicted without seeing it: winner picked {hits('PICKED')} of {n}, "
              f"winner in the top three {int((check['WINNER_RANK'] <= 3).sum())} of {n}, "
              f"real top three found {check['TOP3_OF_3'].sum()} of {3 * n}")
        print(f"  for comparison, the best impact rating on a top-three team picks the winner {hits('TOP_TEAM_RATING')}, "
              f"the best impact rating anywhere {hits('BEST_RATING')}, the top scorer {hits('TOP_SCORER')}")
        print("What voters reward most: " + ", ".join(f"{_mvp_label(k)} ({v:+.2f})" for k, v in weights.head(6).items()))
    return AllNBAModel(weights, 0.0, penalty, seasons, check, None, "mvp", temperature)


def _mvp_label(k):
    return "past MVP votes" if k == "PRIOR" else LABELS[k]


def _tracker(season):
    """Basketball-Reference's MVP tracker by name_key, or None (past season, or the fetch failed)."""
    if season != max(SEASONS):
        return None
    try:
        t = fetch_mvp_tracker()
    except Exception as e:
        print(f"⚠️ Basketball-Reference MVP tracker not available ({type(e).__name__}: {e})")
        return None
    return t.assign(name_key=t["Player"].map(norm_name)).set_index("name_key")["TRACKER_PROB"]


def predict_mvp(voters, df_master, rating_model, season="2026-27", top=15, tracker=True):
    """Each eligible player's chance of winning MVP for a season, and his expected share of the vote.

    WIN_CHANCE   chance he wins; adds up to 1, and counts his chance of reaching 65 games (REACH_65)
    SHARE_OF_POINTS  his expected share of all the voting points cast; adds up to 1 over the players
    ACTUAL_SHARE / FINISH   what happened, for seasons already voted on
    TEAM_RANK    his team's record among all teams
    WHY / WHY_NOT  the stats pushing him up and down the most
    RATING_RANK  where the impact model has him, for contrast
    TRACKER_PROB Basketball-Reference's MVP tracker (their own model), this season
                 only: a second opinion to compare against, not an input

    Players who can no longer reach 65 games are left out (from 2023-24).
    Early in a season this is "if the vote were today", on per-game numbers
    and records so far, with games projected to the end of the schedule. The
    chances don't know how much a season can still change: in November they
    are far too sure of themselves.
    """
    pool, Z = _mvp_frame(df_master, rating_model)
    here = (pool["SEASON"] == season).to_numpy()
    if not here.any():
        raise ValueError(f"no rotation players for {season}")
    rank_all = pool.loc[here, "RATING"].rank(ascending=False).astype(int)
    keep = here & pool["ELIGIBLE"].to_numpy()
    pool, Z = pool[keep], Z[keep]
    pull = Z * voters.weights.reindex(Z.columns)
    score = pull.sum(axis=1).to_numpy()
    reach = pool["REACH_65"].to_numpy(float)
    reasons = lambda row, sign: ", ".join(_mvp_label(k) for k, v in (row * sign).nlargest(2).items() if v > 0.15)
    out = pd.DataFrame({
        "Player": pool["PLAYER_NAME"], "TEAM": pool["TEAM_ABBREVIATION"], "GP": pool["GP"].astype(int),
        "PTS": pool["PTS"], "REB": pool["REB"], "AST": pool["AST"], "STL": pool["STL"], "BLK": pool["BLK"], "W_PCT": pool["W_PCT"],
        "TEAM_RANK": pool["TEAM_RANK"].astype(int),
        "WIN_CHANCE": _weighted_softmax(voters.temperature * score, reach), "SHARE_OF_POINTS": _softmax(score),
        "REACH_65": reach,
        "RATING_RANK": rank_all.reindex(pool.index),
        "WHY": [reasons(r, 1) for _, r in pull.iterrows()], "WHY_NOT": [reasons(r, -1) for _, r in pull.iterrows()],
    }, index=pool.index)
    if pool["DECIDED"].any():
        out.insert(out.columns.get_loc("REACH_65"), "ACTUAL_SHARE", pool["SHARE"])
        out.insert(out.columns.get_loc("REACH_65"), "FINISH", pool["FINISH"])
    tr = _tracker(season) if tracker else None
    if tr is not None:
        out.insert(out.columns.get_loc("SHARE_OF_POINTS") + 1, "TRACKER_PROB",
                   out["Player"].map(norm_name).map(tr).fillna(0.0).to_numpy())
    out = out.sort_values("WIN_CHANCE", ascending=False).reset_index(drop=True)
    out.insert(0, "RANK", np.arange(1, len(out) + 1))
    out.insert(out.columns.get_loc("WIN_CHANCE") + 1, "WIN_PCT", (out["WIN_CHANCE"] * 100).round(1))   # the same chance as a percent
    out.attrs.update(season=season, n=len(out))
    return out.head(top) if top else out


# ---------------------------------------------------------------- injuries
_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a few": 2, "a couple of": 2, "several": 3}
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30}
_DURATION = re.compile(r"(\d+|" + "|".join(sorted(_NUMBERS, key=len, reverse=True)) + r")"
                       r"(?:\s*(?:-|to|or)\s*(?:\d+|" + "|".join(_NUMBERS) + r"))?\s+(?:more\s+|additional\s+)?(day|week|month)s?", re.I)


def _games_out(row, today, left):
    """Games still to miss from one injury-report line: (games, how it was read)."""
    text = str(row["Description"])
    if text.lower().startswith("out for season") or re.search(
            r"(?:end|rest|remainder) of the (?:regular )?season|season-ending|miss the (?:20\d\d-\d\d )?season", text, re.I):
        return left, "out for season"
    if not text.lower().startswith("out"):
        return 0.0, "day to day"
    m = _DURATION.search(text)
    if m:
        n = m.group(1).lower()
        days = (int(n) if n.isdigit() else _NUMBERS[n]) * _UNIT_DAYS[m.group(2).lower()]
        since = (today - row["Update"]).days if pd.notna(row["Update"]) else 0
        return round(max(days - since, 0) / 7 * GAMES_PER_WEEK), f"{m.group(0)} from {row['Update']:%b %d}"
    if re.search(r"(?:out|OUT|Out) (?:for|on) \w+'?s? \(", text) or re.search(r"did not play (?:on|in) \w+", text, re.I):
        return 1.0, "out for one game"
    return float(INDEFINITE_GAMES), "out indefinitely"


def use_injury_report(df_master, season="2026-27", today=None, refresh=False):
    """Fill KNOWN_ABSENCES from Basketball-Reference's injury report and return what it read.

    Timelines are read from the text ("re-evaluated in approximately six weeks",
    "at least four to six weeks", "3-4 months"), always the shorter end and
    counted from the update date, at GAMES_PER_WEEK. "Re-evaluated in" is a
    checkup, not a return, so these lean short. "Out" for a named game counts
    as one game; "out" with no timeline as INDEFINITE_GAMES; day-to-day as none. Set your own numbers after calling this:
    KNOWN_ABSENCES["Stephen Curry"] = 12 overrides the report.
    """
    rep = fetch_injuries(force=refresh)
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    cur = _regular_season(df_master)
    cur = cur[cur["SEASON"] == season]
    names = dict(zip(cur["PLAYER_NAME"].map(norm_name), cur["PLAYER_NAME"]))
    left = SEASON_GAMES - cur.groupby("TEAM_ABBREVIATION")["GP"].max().max()
    rows = []
    for _, r in rep.iterrows():
        games, why = _games_out(r, today, left)
        rows.append({"Player": names.get(norm_name(r["Player"]), r["Player"]), "IN_DATA": norm_name(r["Player"]) in names,
                     "TEAM": r["Team"], "GAMES": games, "READ_AS": why, "Description": r["Description"]})
    out = pd.DataFrame(rows).sort_values("GAMES", ascending=False).reset_index(drop=True)
    KNOWN_ABSENCES.clear()
    KNOWN_ABSENCES.update({p: g for p, g, ok in zip(out["Player"], out["GAMES"], out["IN_DATA"]) if ok and g > 0})
    return out



# ---------------------------------------------------------------- All-Star
def fit_allstar_model(df_master, rating_model, penalty=None, verbose=True):
    """All-Star voting model, trained on every All-Star selection since 2015-16.

    A logistic regression like All-NBA, but on All-Star selections (injured
    selections in, their replacements out), so it learns what All-Star voters
    and coaches reward rather than borrowing the All-NBA vote. PRIOR is past
    All-Star selections. Checked one season at a time: each conference's 12
    most likely against who was really picked.

    Caveat: the stats are full-season numbers, but the vote happens about 50
    games in, so the training sees a little of what came after the vote.
    """
    pool, Z = _voter_table(df_master, rating_model, "star")
    known = pool["DECIDED"].to_numpy()
    y = (pool["VOTES"] > 0).to_numpy()
    conf = np.where(pool["TEAM_ABBREVIATION"].isin(EAST), "East", "West")
    seasons = sorted(pool.loc[known, "SEASON"].unique())

    def held_out(C):
        chance = pd.Series(np.nan, index=pool.index)
        for s in seasons:
            test = known & (pool["SEASON"] == s).to_numpy()
            fit = LogisticRegression(C=C, max_iter=5000).fit(Z[known & ~test], y[known & ~test])
            chance[test] = fit.predict_proba(Z[test])[:, 1]
        return chance

    if penalty is None:
        loss = {}
        for C in PENALTIES:
            p = held_out(C)[known].clip(1e-6, 1 - 1e-6)
            loss[C] = float(-(np.where(y[known], np.log(p), np.log(1 - p))).mean())
        penalty = min(loss, key=loss.get)
    chance = held_out(penalty)

    rows = []
    for s in seasons:
        for c in ("East", "West"):
            here = (pool["SEASON"] == s).to_numpy() & (conf == c)
            g = pool[here]
            real = set(g.index[y[here]])
            top = lambda col: set(g[col].nlargest(12).index) if isinstance(col, str) else set(col[here].nlargest(12).index)
            rows.append({"SEASON": s, "CONF": c, "PICKED": len(real), "RIGHT_OF_12": len(top(chance) & real),
                         "TOP_SCORERS": len(top("PTS") & real), "TOP_RATING": len(top("RATING") & real)})
    check = pd.DataFrame(rows)
    fit = LogisticRegression(C=penalty, max_iter=5000).fit(Z[known], y[known])
    weights = pd.Series(fit.coef_[0], index=Z.columns).sort_values(key=np.abs, ascending=False)
    if verbose:
        n = 24 * len(seasons)
        print(f"\nAll-Star voting model: {int(y[known].sum())} selections over {len(seasons)} seasons, "
              f"{int(known.sum())} player-seasons, penalty C={penalty:g}")
        print(f"Each season predicted without seeing it: {check['RIGHT_OF_12'].sum()} of {n} spots right "
              f"({check['RIGHT_OF_12'].sum() / len(seasons):.1f} of 24 a season)")
        print(f"  for comparison, each conference's 12 top scorers get {check['TOP_SCORERS'].sum()}, "
              f"its 12 best impact ratings {check['TOP_RATING'].sum()}")
        lab = lambda k: "past All-Star selections" if k == "PRIOR" else LABELS[k]
        print("What voters reward most: " + ", ".join(f"{lab(k)} ({v:+.2f})" for k, v in weights.head(6).items()))
    return AllNBAModel(weights, float(fit.intercept_[0]), penalty, seasons, check, None, "star")


_allstar_cache = {}


# ---------------------------------------------------------------- history
AWARD_HISTORY = DATA_DIR / "award_history.parquet"
AWARD_ORDER = ["MVP", "MVP (B-Ref tracker)", "All-NBA", "All-Star", "All-Defensive", "Most Improved",
               "Rookie of the Year", "Sixth Man of the Year"]


def _for_history(df_master, season):
    """The master as the history measures it: this season's Basketball-Reference stats left out (they were only
    joined from Nov 13, 2026, so earlier days can't have them), ages kept. Every day is then measured the same way."""
    df = df_master.copy()
    now = df.index.get_level_values("SEASON") == season
    drop = [c for c in df.columns if c.startswith("BBR_") and c != "BBR_AGE"]
    df.loc[now, drop] = np.nan
    return df


def _group_snapshot(dm, rating_model, season, pen):
    """Rookie of the Year and Sixth Man chances in the history's long format, and the penalties used."""
    out, used = [], {}
    for award, title in (("roy", "Rookie of the Year"), ("smoy", "Sixth Man of the Year")):
        voters = _fit_group_model(award, dm, rating_model, pen.get(title), False, title)
        t = _predict_group(voters, dm, rating_model, season, None, title)
        out.append(pd.DataFrame({"AWARD": title, "Player": t["Player"], "TEAM": t["TEAM"], "GP": t["GP"],
                                 "CHANCE": t["WIN_PCT"].astype(float) / 100, "RANK": t["RANK"]}))
        used[title] = voters.penalty
    return pd.concat(out, ignore_index=True), used


def award_snapshot(df_master, rating_model, season="2026-27", penalties=None):
    """Every player's chance in every award race today, long format: AWARD, Player, TEAM, GP, CHANCE, RANK.

    Measured the way the history is (_for_history), with no injury absences:
    there are no old injury reports, so past days couldn't have them either.
    penalties: {award: penalty} to skip the penalty search (faster; the history
    backfill passes today's).
    """
    pen = penalties or {}
    saved = dict(KNOWN_ABSENCES)
    KNOWN_ABSENCES.clear()
    try:
        dm = _for_history(df_master, season)
        nba = fit_allnba_model(dm, rating_model, penalty=pen.get("All-NBA"), verbose=False)
        star = fit_allstar_model(dm, rating_model, penalty=pen.get("All-Star"), verbose=False)
        dfn = fit_alldefense_model(dm, rating_model, penalty=pen.get("All-Defensive"), verbose=False)
        mip = fit_mip_model(dm, rating_model, penalty=pen.get("Most Improved"), verbose=False)
        mvp = fit_mvp_model(dm, rating_model, penalty=pen.get("MVP"), verbose=False)
        parts = {
            "MVP": predict_mvp(mvp, dm, rating_model, season, top=None).rename(columns={"WIN_CHANCE": "CHANCE"}),
            "All-NBA": predict_allnba(nba, dm, rating_model, season, top=None),
            "All-Star": predict_allstars(star, dm, rating_model, season, next_up=40),
            "All-Defensive": pick_all_defense(rating_model, dm, season, top=None, voters=dfn),
            "Most Improved": predict_mip(mip, dm, rating_model, season, top=None),
        }
        groups, group_pens = _group_snapshot(dm, rating_model, season, pen)
    finally:
        KNOWN_ABSENCES.clear()
        KNOWN_ABSENCES.update(saved)
    out = []
    for award, t in parts.items():
        t = t.sort_values("CHANCE", ascending=False).reset_index(drop=True)
        out.append(pd.DataFrame({"AWARD": award, "Player": t["Player"], "TEAM": t["TEAM"],
                                 "GP": t["GP"] if "GP" in t.columns else np.nan, "CHANCE": t["CHANCE"].astype(float),
                                 "RANK": np.arange(1, len(t) + 1)}))
    tr = _tracker(season)                                 # saved alongside, for the season-end comparison
    if tr is not None:
        m = parts["MVP"][["Player", "TEAM", "GP"]].assign(CHANCE=lambda t: t["Player"].map(norm_name).map(tr).fillna(0.0))
        m = m.sort_values("CHANCE", ascending=False).reset_index(drop=True)
        out.append(m.assign(AWARD="MVP (B-Ref tracker)", RANK=np.arange(1, len(m) + 1))[["AWARD", "Player", "TEAM", "GP", "CHANCE", "RANK"]])
    out.append(groups)
    out = pd.concat(out, ignore_index=True)
    out.attrs["penalties"] = {**group_pens, "All-NBA": nba.penalty, "All-Star": star.penalty, "All-Defensive": dfn.penalty,
                              "Most Improved": mip.penalty, "MVP": mvp.penalty}
    return out


def save_award_snapshot(df_master, rating_model, date=None, season="2026-27", penalties=None, path=AWARD_HISTORY):
    """Add today's (or `date`'s) award chances to data/award_history.parquet, replacing that date if it's there."""
    date = pd.Timestamp(date or pd.Timestamp.today()).normalize()
    snap = award_snapshot(df_master, rating_model, season, penalties).assign(DATE=date)
    pens = snap.attrs.get("penalties")
    full = snap
    if path.exists():
        old = pd.read_parquet(path)
        full = pd.concat([old[old["DATE"] != date], snap], ignore_index=True)
    full.sort_values(["DATE", "AWARD", "RANK"]).to_parquet(path, index=False)
    snap.attrs["penalties"] = pens
    return snap


def award_movers(since=None, until=None, award=None, top=8, min_change=0.005, path=AWARD_HISTORY):
    """Who gained and lost the most in each award race between two saved days.

    since / until: dates (the nearest saved day on or after `since`, on or
    before `until`); default the first and last saved days. Returns a table per
    award with CHANCE_THEN, CHANCE_NOW, CHANGE and RANK_THEN -> RANK_NOW, the
    biggest gainers then the biggest losers. A player missing on one day (not
    yet in the pool, or dropped out of it: too few games, or hurt) counts as 0
    there. Changes under `min_change` are left out.
    """
    h = pd.read_parquet(path)
    days = sorted(h["DATE"].unique())
    d0 = min(d for d in days if since is None or d >= pd.Timestamp(since))
    d1 = max(d for d in days if until is None or d <= pd.Timestamp(until))
    out = []
    for a in ([award] if award else [x for x in AWARD_ORDER if x in set(h["AWARD"])]):
        then = h[(h["DATE"] == d0) & (h["AWARD"] == a)].set_index("Player")
        now = h[(h["DATE"] == d1) & (h["AWARD"] == a)].set_index("Player")
        j = now[["TEAM", "CHANCE", "RANK"]].join(then[["CHANCE", "RANK"]], rsuffix="_THEN", how="outer")
        j["TEAM"] = j["TEAM"].fillna(then["TEAM"].reindex(j.index))
        j = j.rename(columns={"CHANCE": "CHANCE_NOW", "RANK": "RANK_NOW"}).fillna({"CHANCE_NOW": 0.0, "CHANCE_THEN": 0.0})
        j["CHANGE"] = j["CHANCE_NOW"] - j["CHANCE_THEN"]
        up = j[j["CHANGE"] >= min_change].nlargest(top, "CHANGE").assign(DIRECTION="gained")
        down = j[j["CHANGE"] <= -min_change].nsmallest(top, "CHANGE").assign(DIRECTION="lost")
        part = pd.concat([up, down]).reset_index().assign(AWARD=a)
        out.append(part[["AWARD", "DIRECTION", "Player", "TEAM", "CHANCE_THEN", "CHANCE_NOW", "CHANGE", "RANK_THEN", "RANK_NOW"]])
    res = pd.concat(out, ignore_index=True)
    res.attrs.update(since=str(pd.Timestamp(d0).date()), until=str(pd.Timestamp(d1).date()))
    return res


def add_absences(df_master, overrides, season="2026-27", today=None):
    """Your own absence estimates, counted down as the team plays, so they expire on their own.

        add_absences(df_master, {"Duncan Robinson": (4, "2026-11-17")})   # 4 games from Nov 17

    Each entry is (games he'll miss, the date you set it). Games still to miss =
    games minus his team's games played on or after that date; at 0 the entry
    has expired and is dropped. Remaining ones go into KNOWN_ABSENCES (after
    use_injury_report, so they win over it). Returns a table of every entry.
    """
    from team_projection import load_games
    from player_ranking import _regular_season
    games = load_games(season)
    cur = _regular_season(df_master)
    cur = cur[cur["SEASON"] == season].sort_values("GP").drop_duplicates("PLAYER_ID", keep="last")
    team = dict(zip(cur["PLAYER_NAME"], cur["TEAM_ABBREVIATION"]))
    rows = []
    for name, (n, since) in overrides.items():
        t = team.get(name)
        if t is None:
            rows.append({"Player": name, "TEAM": None, "SET": since, "GAMES": n, "PLAYED_SINCE": None, "LEFT": None,
                         "STATUS": "name not found this season"})
            continue
        day = pd.Timestamp(since)
        played = int((((games["HOME"] == t) | (games["AWAY"] == t)) & (games["DATE"] >= day)).sum())
        left = max(n - played, 0)
        rows.append({"Player": name, "TEAM": t, "SET": since, "GAMES": n, "PLAYED_SINCE": played, "LEFT": left,
                     "STATUS": "out" if left else "expired: delete this line"})
        if left:
            KNOWN_ABSENCES[name] = left
        else:
            KNOWN_ABSENCES.pop(name, None)       # the report may still list him; your read says he's back
    out = pd.DataFrame(rows)
    done = out.loc[out["STATUS"] != "out", "Player"].tolist()
    if done:
        print(f"⚠️ expired or not found, delete from the cell: {done}")
    return out


# ---------------------------------------------------------------- Rookie of the Year and Sixth Man of the Year
# Same listwise approach as MVP: a softmax over each season's pool, weights fitted to vote shares, a temperature
# fitted on held-out seasons so the chances read as percentages. Pools: Rookie of the Year = first-year players;
# Sixth Man = bench players (not starters, by Basketball-Reference's games-started column).

def _vote_shares(name, years):
    """Basketball-Reference award vote shares for a season-end year range: name_key, SEASON, share."""
    from io import StringIO
    from nba_viz_utils import cached_fetch as _cf
    def fetch():
        frames = []
        for year in years:
            try:
                html = _get(f"https://www.basketball-reference.com/awards/awards_{year}.html")
            except Exception as e:                       # a season's page isn't up yet (in progress)
                print(f"   no {name} page for {year} yet ({type(e).__name__})")
                continue
            at = html.find(f'id="{name}"')
            if at < 0:
                continue
            t = pd.read_html(StringIO(html[html.rfind("<table", 0, at):html.find("</table>", at) + 8]))[0]
            t.columns = [c[-1] if isinstance(c, tuple) else c for c in t.columns]
            frames.append(t[["Player", "Share"]].assign(SEASON=f"{year - 1}-{str(year)[2:]}"))
        return pd.concat(frames, ignore_index=True).astype({"Player": str, "Share": str})
    raw = _cf(f"{name}_voting", fetch)
    raw["name_key"] = raw.Player.map(norm_name)
    raw["share"] = pd.to_numeric(raw.Share, errors="coerce").fillna(0.0)
    return raw[["name_key", "SEASON", "share"]]


def _starts_table():
    """Games and starts per player-season (G, GS) and a starter flag (started at least half his games), Basketball-Reference.

    Past seasons are cached for good; the season in progress is refetched when its cache is a day old.
    """
    from io import StringIO
    import time as _time
    from nba_viz_utils import CACHE_DIR, cached_fetch as _cf
    now_year = int(max(SEASONS)[:4]) + 1
    def fetch(years):
        frames = []
        for year in years:
            try:
                html = _get(f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html")
            except Exception as e:
                print(f"   no per-game page for {year} yet ({type(e).__name__})")
                continue
            at = html.find('id="per_game_stats"')
            if at < 0:
                continue
            t = pd.read_html(StringIO(html[html.rfind("<table", 0, at):html.find("</table>", at) + 8]))[0]
            t = t[t["Player"].astype(str) != "Player"]
            frames.append(t[["Player", "Team", "G", "GS"]].assign(SEASON=f"{year - 1}-{str(year)[2:]}"))
        return pd.concat(frames, ignore_index=True)
    past = _cf("bbref_starts", lambda: fetch(range(2016, now_year)))
    past = past[past["SEASON"] != max(SEASONS)]                        # an old cache may hold a stale copy of this season
    path = CACHE_DIR / "bbref_starts_current.parquet"
    stale = not path.exists() or _time.time() - path.stat().st_mtime > 86400
    gs = pd.concat([past, _cf("bbref_starts_current", lambda: fetch([now_year]), force_refresh=stale)], ignore_index=True)
    gs["GS"] = pd.to_numeric(gs.GS, errors="coerce"); gs["G"] = pd.to_numeric(gs.G, errors="coerce")
    tot = gs[gs.Team == "TOT"].set_index(["Player", "SEASON"])
    one = gs[gs.Team != "TOT"].drop_duplicates(["Player", "SEASON"], keep="last").set_index(["Player", "SEASON"])
    s = pd.concat([tot, one[~one.index.isin(tot.index)]])[["G", "GS"]]
    s.index = pd.MultiIndex.from_tuples([(norm_name(p), se) for p, se in s.index])
    s = s[~s.index.duplicated(keep="first")]
    return s.assign(STARTER=s.GS >= s.G / 2)


def _bench_split(season):
    """Per-game stats in games he came off the bench (NBA stats' starter/bench split), by PLAYER_ID; cached a day."""
    import time as _time
    from nba_api.stats.endpoints import LeagueDashPlayerStats
    from nba_viz_utils import CACHE_DIR, cached_fetch as _cf
    name = f"bench_split_{season}"
    path = CACHE_DIR / f"{name}.parquet"
    stale = season == max(SEASONS) and (not path.exists() or _time.time() - path.stat().st_mtime > 86400)
    b = _cf(name, lambda: LeagueDashPlayerStats(season=season, starter_bench_nullable="Bench", per_mode_detailed="PerGame",
                                                timeout=60).get_data_frames()[0], force_refresh=stale)
    return b.set_index(b["PLAYER_ID"].astype(str))[["GP", "MIN", "PTS", "REB", "AST"]]


def _award_pool(df_master, rating_model, award):
    """One row per candidate per season, with the vote share, the season's inputs and a 'pool' flag."""
    from player_ranking import rank_players as _rank
    import contextlib, io
    df = _regular_season(df_master)
    with contextlib.redirect_stdout(io.StringIO()):
        rated = {s: _rank(rating_model, df_master, s, min_min=0, min_gp=0).assign(PID=lambda t: t.PLAYER_ID.astype(str)).groupby("PID").RATING.mean()
                 for s in SEASONS}
    d = df.sort_values("GP").drop_duplicates(["PLAYER_ID", "SEASON"], keep="last").copy()
    d["PID"] = d["PLAYER_ID"].astype(str)
    d["key"] = d["PLAYER_NAME"].map(norm_name)
    d["RATING"] = [rated[s].get(p, np.nan) for s, p in zip(d["SEASON"], d["PID"])]
    d["MINUTES_PG"] = pd.to_numeric(d["MIN"], errors="coerce")
    d["GAMES"] = pd.to_numeric(d["GP"], errors="coerce")
    d["TEAM_RANK"] = d.groupby(["SEASON", "TEAM_ABBREVIATION"])["MINUTES_PG"].rank(ascending=False, method="first")
    # games needed, as a share of the season's games so far (league max GP): 82-game seasons give 30% / 24%
    season_games = d["SEASON"].map(d.groupby("SEASON")["GAMES"].max())
    if award == "roy":
        first = d.groupby("PID")["SEASON"].min()
        d = d[d["PID"].map(first) == d["SEASON"]]
        d = d[d["SEASON"] >= "2016-17"]
        d = d[(d["MINUTES_PG"] >= 12) & (d["GAMES"] >= 0.3 * season_games.loc[d.index])]
        votes = _vote_shares("roy", range(2016, int(max(SEASONS)[:4]) + 2)).set_index(["name_key", "SEASON"]).share
    else:
        starts = _starts_table()
        found = starts.reindex(pd.MultiIndex.from_arrays([d["key"], d["SEASON"]]))
        d["STARTS"] = found["GS"].to_numpy()
        d["STARTER"] = found["STARTER"].fillna(False).astype(bool).to_numpy()
        d = d[(~d["STARTER"]) & (d["MINUTES_PG"] >= 15) & (d["GAMES"] >= 0.244 * season_games.loc[d.index])]
        d = d[d["SEASON"] >= "2016-17"]
        votes = _vote_shares("smoy", range(2016, int(max(SEASONS)[:4]) + 2)).set_index(["name_key", "SEASON"]).share
    d["SHARE"] = [votes.get((k, s), 0.0) for k, s in zip(d["key"], d["SEASON"])]
    return d.reset_index(drop=True)


_VOTE_STATS = {
    "roy": ["PTS", "MINUTES_PG", "TS_PCT", "USG_PCT", "RATING", "AST", "REB", "W_PCT", "GAMES"],
    "smoy": ["PTS", "MINUTES_PG", "TS_PCT", "USG_PCT", "RATING", "AST", "REB", "W_PCT", "GAMES"],
}


def _fit_group_model(award, df_master, rating_model, penalty=None, verbose=True, title=""):
    d = _award_pool(df_master, rating_model, award)
    stats = _VOTE_STATS[award]
    X = d[stats].apply(pd.to_numeric, errors="coerce")
    Z = ((X - X.groupby(d["SEASON"]).transform("mean")) / X.groupby(d["SEASON"]).transform("std")).clip(-5, 5).fillna(0).to_numpy()
    seasons = sorted(s for s in d["SEASON"].unique() if s < max(SEASONS))
    idx = {s: np.flatnonzero((d["SEASON"] == s).to_numpy()) for s in d["SEASON"].unique()}
    sh = d["SHARE"].to_numpy()
    tgt = {s: sh[idx[s]] / max(sh[idx[s]].sum(), 1e-9) for s in seasons}
    def held(lam):
        out = {}
        for s in seasons:
            tr = [x for x in seasons if x != s]
            w = _fit_shares(Z, [idx[x] for x in tr], [tgt[x] for x in tr], lam)
            out[s] = Z[idx[s]] @ w
        return out
    if penalty is None:
        penalty = min((0.003, 0.01, 0.03, 0.1, 0.3), key=lambda l: -np.mean([tgt[s] @ np.log(_softmax(sc) + 1e-12) for s, sc in held(l).items()]))
    scores = held(penalty)
    from scipy.optimize import minimize_scalar
    def winner_loglik(t):
        return -sum(np.log(_softmax(t * scores[s])[np.argmax(sh[idx[s]])] + 1e-12) for s in seasons)
    temperature = float(minimize_scalar(winner_loglik, bounds=(0.5, 4.0), method="bounded").x)
    ranks = {s: int(np.flatnonzero(np.argsort(-scores[s]) == np.argmax(sh[idx[s]]))[0]) + 1 for s in seasons}
    top_scorer = {s: int(np.flatnonzero(np.argsort(-d["PTS"].to_numpy(float)[idx[s]]) == np.argmax(sh[idx[s]]))[0]) + 1 for s in seasons}
    w = _fit_shares(Z, [idx[s] for s in seasons], [tgt[s] for s in seasons], penalty)
    weights = pd.Series(w, index=stats).sort_values(key=np.abs, ascending=False)
    check = pd.DataFrame({"SEASON": seasons, "WINNER_RANK": [ranks[s] for s in seasons], "TOP_SCORER_RANK": [top_scorer[s] for s in seasons]})
    if verbose:
        n = len(seasons)
        print(f"\n{title} voting model: {int((sh > 0).sum())} vote-getters over {n} seasons, penalty {penalty:g}, temperature {temperature:.2f}")
        print(f"Each season predicted without seeing it: winner first {sum(r == 1 for r in ranks.values())} of {n}; "
              f"top scorer first {sum(r == 1 for r in top_scorer.values())} of {n}")
        print("What voters reward: " + ", ".join(f"{k} ({v:+.2f})" for k, v in weights.head(6).items()))
    model = AllNBAModel(weights, 0.0, penalty, seasons, check, None, award, temperature)
    model.pool = d
    return model


def _predict_group(voters, df_master, rating_model, season, top, title):
    d = _award_pool(df_master, rating_model, voters.award)
    cur = d[d["SEASON"] == season].copy()
    if cur.empty:
        raise ValueError(f"no candidates for {season}")
    # standardize within each season's pool, the same way the fitted model did, then keep this season's rows
    X = d[_VOTE_STATS[voters.award]].apply(pd.to_numeric, errors="coerce")
    Z = ((X - X.groupby(d["SEASON"]).transform("mean")) / X.groupby(d["SEASON"]).transform("std")).clip(-5, 5).fillna(0).to_numpy()
    rows = np.flatnonzero((d["SEASON"] == season).to_numpy())
    score = Z[rows] @ voters.weights.reindex(_VOTE_STATS[voters.award]).to_numpy()
    chance = _softmax(voters.temperature * score)
    out = pd.DataFrame({"Player": cur["PLAYER_NAME"].to_numpy(), "TEAM": cur["TEAM_ABBREVIATION"].to_numpy(),
                        "GP": cur["GAMES"].astype(int).to_numpy(), "MIN": cur["MINUTES_PG"].round(1).to_numpy(),
                        "PTS": cur["PTS"].round(1).to_numpy(), "REB": cur["REB"].round(1).to_numpy(),
                        "AST": cur["AST"].round(1).to_numpy(), "STL": cur["STL"].round(1).to_numpy(),
                        "BLK": cur["BLK"].round(1).to_numpy(), "TS_PCT": cur["TS_PCT"].round(3).to_numpy(),
                        "RATING": cur["RATING"].round(2).to_numpy(), "TEAM_W_PCT": cur["W_PCT"].round(3).to_numpy(),
                        "WIN_PCT": (chance * 100).round(1)})
    if voters.award == "smoy":                    # how much of his season came off the bench
        out.insert(out.columns.get_loc("GP") + 1, "STARTS", cur["STARTS"].fillna(0).astype(int).to_numpy())
        try:
            bench = _bench_split(season).reindex(cur["PID"].to_numpy())
            out["BENCH_GP"] = bench["GP"].fillna(0).astype(int).to_numpy()
            out["BENCH_MIN"] = bench["MIN"].round(1).to_numpy()
            out["BENCH_PTS"] = bench["PTS"].round(1).to_numpy()
        except Exception as e:
            print(f"⚠️ bench split not available ({type(e).__name__})")
    if cur["SHARE"].max() > 0:
        out["ACTUAL_SHARE"] = cur["SHARE"].to_numpy()
    out = out.sort_values("WIN_PCT", ascending=False).reset_index(drop=True)
    out.insert(0, "RANK", np.arange(1, len(out) + 1))
    out.attrs.update(season=season, title=title)
    return out.head(top) if top else out


def fit_roy_model(df_master, rating_model, penalty=None, verbose=True):
    """Rookie of the Year voting model: the first-year players who got real minutes, trained on each season's vote shares."""
    return _fit_group_model("roy", df_master, rating_model, penalty, verbose, "Rookie of the Year")


def predict_roy(voters, df_master, rating_model, season="2026-27", top=10):
    """Each rookie's percentage chance of winning Rookie of the Year, for a season (WIN_PCT adds up to 100 over the pool)."""
    return _predict_group(voters, df_master, rating_model, season, top, "Rookie of the Year")


def fit_smoy_model(df_master, rating_model, penalty=None, verbose=True):
    """Sixth Man of the Year voting model: bench players (not starters), trained on each season's vote shares."""
    return _fit_group_model("smoy", df_master, rating_model, penalty, verbose, "Sixth Man of the Year")


def predict_smoy(voters, df_master, rating_model, season="2026-27", top=10):
    """Each bench player's percentage chance of winning Sixth Man of the Year, for a season (WIN_PCT adds up to 100)."""
    return _predict_group(voters, df_master, rating_model, season, top, "Sixth Man of the Year")
