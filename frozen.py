"""Dated copies of the predictions that are never changed, to grade once the season is over.

    from frozen import *
    freeze_predictions(df_master, model, proj)   # run after the absences cell; does nothing if the last copy is under a week old
    frozen_dates()
    load_frozen("win_totals")                    # every copy stacked, with a FROZEN date column
    grade_win_totals()                           # each copy against the final records
    grade_player_wins(df_master, model)          # each copy's rest-of-season player wins against what happened

The live models carry on as before; a copy only reads them. Each one is a folder, data/frozen/<date>/ (kept in git, so the
commit and push dates back up when it was made):
  awards.csv          that day's rows of the award history (measured its way: no injury absences, no Basketball-Reference stats)
  win_totals.csv      project_season with that day's absences
  trade_values.csv    trade_value, plus MARKET_VALUE_M
  player_seasons.csv  the year-by-year projection behind trade_value. WINS is a full-season rate; REST_WINS is the part still
                      to play (this season's games left; all of a later season). Wins don't use KNOWN_ABSENCES, like trade_value
  meta.json           games played, absences, the code's commit (and whether the code had uncommitted changes), settings

An existing date is never overwritten. The first copy is Nov 23, 2026, so these are early-season forecasts, not preseason ones.
"""
import json
import subprocess

import numpy as np
import pandas as pd

import trade_value as T
from awards import AWARD_HISTORY, KNOWN_ABSENCES, save_award_snapshot
from nba_viz_utils import DATA_DIR
from player_ranking import _regular_season, rank_players
from salary_market import contracts_vs_market, score_market
from team_projection import load_games, project_season
from trade_machine import add_market_value

__all__ = ["freeze_predictions", "frozen_dates", "load_frozen", "grade_win_totals", "grade_player_wins"]

FROZEN_DIR = DATA_DIR / "frozen"
SEASON = "2026-27"
SEASON_GAMES = 82


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=DATA_DIR.parent, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def frozen_dates(root=FROZEN_DIR):
    return sorted(p.parent.name for p in root.glob("*/meta.json"))


def freeze_predictions(df_master, model, proj, every=7, absences=None, root=FROZEN_DIR):
    """Save today's copy, unless the last one is under `every` days old (every=0: any day without a copy). Returns its folder, or None."""
    today = pd.Timestamp.today().normalize()
    done = frozen_dates(root)
    if done:
        age = (today - pd.Timestamp(done[-1])).days
        if age < max(every, 1):
            print(f"Last copy is {done[-1]} ({age} days old); next one due {(pd.Timestamp(done[-1]) + pd.Timedelta(days=max(every, 1))).date()}")
            return None
    absences = dict(KNOWN_ABSENCES if absences is None else absences)
    if not absences:
        raise ValueError("no absences: run the injury cell first (use_injury_report, add_absences)")
    games = load_games(SEASON)
    played = pd.concat([games["HOME"], games["AWAY"]]).value_counts()
    cur = _regular_season(df_master)
    cur = cur[cur["SEASON"] == SEASON]
    gap = (played - cur.groupby("TEAM_ABBREVIATION")["GP"].max()).median()
    if gap >= 1:
        raise ValueError(f"df_master is about {gap:.0f} game(s) behind the results: refresh it first")

    history = pd.read_parquet(AWARD_HISTORY)
    if history["DATE"].max() != today:
        save_award_snapshot(df_master, model)
        history = pd.read_parquet(AWARD_HISTORY)
    awards = history[history["DATE"] == today].drop(columns="DATE").reset_index(drop=True)
    wins = project_season(proj, df_master, absences=absences)
    contracts = T.fetch_contract_years()
    curve, price = T.fit_aging_curve(model, df_master), T.price_of_a_win(model, df_master)
    odds = T.fit_title_odds()
    tv = T.trade_value(model, df_master, contracts=contracts, curve=curve, price=price, title_odds=odds, verbose=False)
    tv = add_market_value(tv, contracts_vs_market(score_market(df_master)))
    y, _ = T._years(model, df_master, 0.07, contracts, price, curve, title=(*odds, T.TITLE_VALUE), market=T.market_shares(df_master))
    y = y[y["PLAYER_ID"].notna() & ~y["RESIGN"]]
    y["REST_WINS"] = y["WINS"] * np.where(y["AHEAD"] == 0, ((SEASON_GAMES - y["TEAM_GAMES"]) / SEASON_GAMES).clip(0, 1), 1.0)
    cols = ["PLAYER_ID", "Player", "TEAM", "SEASON", "AHEAD", "AGE_THEN", "TEAM_GAMES", "AVAILABILITY", "RATING", "MPG_THEN", "WINS",
            "REST_WINS", "WORTH_M", "SALARY_M", "SURPLUS_M", "OPTION", "COUNTED"]
    # what he has already won, so REST_WINS can be graded as final minus this
    so_far = rank_players(model, df_master, SEASON, min_min=0, min_gp=0).drop_duplicates("PLAYER_ID").set_index("PLAYER_ID")["WINS"]
    y = y[cols].assign(WINS_SO_FAR=np.where(y["AHEAD"] == 0, y["PLAYER_ID"].map(so_far).fillna(0.0), 0.0))

    code = _git("status", "--porcelain", "--", "*.py")
    meta = {"frozen_at": pd.Timestamp.now(tz="America/Los_Angeles").isoformat(timespec="seconds"), "season": SEASON,
            "games_through": str(games["DATE"].max().date()), "games_played": int(len(games)), "commit": _git("rev-parse", "HEAD"),
            "uncommitted_code": None if code is None else bool(code), "price_of_a_win": float(price),
            "settings": {"cap_growth": 0.07, "TITLE_VALUE": T.TITLE_VALUE, "RESIGN_YEARS": T.RESIGN_YEARS, "REAL_TERMS": T.REAL_TERMS,
                         "ROOKIE_GROWTH": T.ROOKIE_GROWTH, "MINUTES_CHANGE": T.MINUTES_CHANGE},
            "absences": {k: int(v) for k, v in sorted(absences.items())}}
    out = root / str(today.date())
    tmp = root / f".{today.date()}.partial"
    tmp.mkdir(parents=True, exist_ok=True)
    for name, t in {"awards": awards, "win_totals": wins, "trade_values": tv, "player_seasons": y}.items():
        t.round(6).to_csv(tmp / f"{name}.csv", index=False)
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    tmp.rename(out)                                     # all or nothing, and fails if the date exists
    print(f"🧊 Frozen: {out.name} ({len(awards):,} award rows, {len(wins)} teams, {len(tv)} trade values; games through {meta['games_through']})"
          + (" ⚠️ code had uncommitted changes" if code else ""))
    return out


def load_frozen(name, date=None, root=FROZEN_DIR):
    """One table ("awards", "win_totals", "trade_values", "player_seasons") from every copy, or one date's."""
    dates = [date] if date else frozen_dates(root)
    return pd.concat([pd.read_csv(root / d / f"{name}.csv").assign(FROZEN=d) for d in dates], ignore_index=True)


def _final_games():
    games = load_games(SEASON)
    if len(games) < SEASON_GAMES * 30 / 2:
        raise ValueError(f"the season isn't over ({len(games):,} of 1,230 games played)")
    return games


def grade_win_totals(root=FROZEN_DIR, games=None):
    """Per copy: average miss on final wins, against the pace that day, and how many teams landed in the 80% range."""
    games = _final_games() if games is None else games
    won = pd.concat([games["HOME"][games["MARGIN"] > 0], games["AWAY"][games["MARGIN"] < 0]]).value_counts()
    t = load_frozen("win_totals", root=root)
    t["FINAL"] = t["TEAM"].map(won).fillna(0)
    t["MISS"] = (t["PROJ_W"] - t["FINAL"]).abs()
    t["PACE_MISS"] = (SEASON_GAMES * t["W"] / (t["W"] + t["L"]) - t["FINAL"]).abs()
    t["IN_80"] = t["FINAL"].between(t["LOW_80"], t["HIGH_80"])
    return t.groupby("FROZEN").agg(MISS=("MISS", "mean"), PACE_MISS=("PACE_MISS", "mean"), IN_80=("IN_80", "mean")).round(2)


def grade_player_wins(df_master, model, root=FROZEN_DIR, check=True):
    """Per copy: REST_WINS against this season's final wins minus the wins already won that day."""
    if check:
        _final_games()
    final = rank_players(model, df_master, SEASON, min_min=0, min_gp=0).drop_duplicates("PLAYER_ID").set_index("PLAYER_ID")["WINS"]
    final.index = final.index.astype(str)
    p = load_frozen("player_seasons", root=root).query("AHEAD == 0").copy()
    p["ACTUAL"] = p["PLAYER_ID"].astype(str).map(final).fillna(0.0) - p["WINS_SO_FAR"]
    p["ERR"] = p["REST_WINS"] - p["ACTUAL"]
    g = p.groupby("FROZEN")
    return pd.DataFrame({"PLAYERS": g.size(), "RMSE": g["ERR"].apply(lambda e: float(np.sqrt((e ** 2).mean()))), "BIAS": g["ERR"].mean(),
                         "CORR": g.apply(lambda x: x["REST_WINS"].corr(x["ACTUAL"]), include_groups=False)}).round(2)
