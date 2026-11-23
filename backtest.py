"""Did trade_value's contracts pay off? Past seasons rebuilt with only what was known then.

    from backtest import *
    bt = run_backtest(df_master)                 # ~10 minutes; one row per contract-season per setting
    backtest_summary(bt)                         # prints the accuracy tables
    bt = load_backtest()                         # the saved run

    python3 backtest.py                          # runs it, saves data/backtest.parquet and prints the summary

For each season-end in STARTS the rating model, aging curve, price of a win, minutes, rookie growth and availability are
refitted on that season and earlier. Every contract is then projected up to three seasons ahead and compared with the
wins the player really produced (rank_players' WINS; 0 if he didn't play), priced the same way.

Read with care
  - There are no old contracts, only salaries. A salary that continues at normal raises (0.90 to 1.095 of the year
    before, or a rookie-scale step) is taken as the same deal, which favors players who stayed useful.
  - Outcomes come from the same family of rating model, so a blind spot of the rating is invisible here.
  - It is a season-end test: no 16-game samples. Title value, re-signing rights and options are left out.
  - The first two starts have too little history to price a win, so they borrow the next one's price.
"""
import contextlib
import io

import numpy as np
import pandas as pd

import trade_value as T
from bbref_stats import DATA_DIR, name_key
from nba_viz_utils import SALARY_CAP, SEASONS
from player_ranking import fit_rating_model, rank_players
from salaries import load_salaries

__all__ = ["run_backtest", "backtest_summary", "load_backtest", "STARTS"]

STARTS = ("2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25")
SETTINGS = {"full": dict(MINUTES_CHANGE="all", ROOKIE_GROWTH=True), "no minutes": dict(MINUTES_CHANGE=None, ROOKIE_GROWTH=True),
            "no growth": dict(MINUTES_CHANGE="all", ROOKIE_GROWTH=False), "neither": dict(MINUTES_CHANGE=None, ROOKIE_GROWTH=False)}
PATH = DATA_DIR / "backtest.parquet"
CAP_NOW_M = SALARY_CAP[max(SEASONS)] / 1e6


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def _contracts(pay, salary, start, ahead, years_in):
    """Contract rows for everyone paid the season after `start`: that season, and later ones while it looks like the same deal."""
    rows = []
    for n in pay.loc[pay["SEASON"] == ahead[0], "name_key"].unique():
        rows.append((start, n, float(salary.get((n, start), 0.0))))
        for k, s in enumerate(ahead):
            v = salary.get((n, s))
            if v is None:
                break
            if k:
                r = v / salary[(n, ahead[k - 1])]
                rookie_scale = years_in.get(n, 99) + k + 1 <= 3 and r <= 1.95
                if not (0.9 <= r <= 1.095 or rookie_scale):
                    break
            rows.append((s, n, float(v)))
    return pd.DataFrame(rows, columns=["SEASON", "name_key", "Salary"]).assign(BBR_ID="none", Player="", Tm="", OPTION="")


def run_backtest(df_master, starts=STARTS, settings=None, save=True, verbose=True):
    """One row per (setting, start, contract season): predicted wins PRED_W, actual wins ACTUAL_W, SALARY as a share of the cap, PRICE of a win."""
    settings = SETTINGS if settings is None else settings
    saved = {k: getattr(T, k) for k in ("MINUTES_CHANGE", "ROOKIE_GROWTH", "REAL_TERMS")}
    current = max(SEASONS)
    done = [s for s in SEASONS if s < current]
    rows = []
    try:
        T.REAL_TERMS = False
        with _quiet():
            full = fit_rating_model(df_master)
            pay = load_salaries(save=False).drop_duplicates(["name_key", "SEASON"])
            actual = {s: rank_players(full, df_master, s, min_min=0, min_gp=0).drop_duplicates("PLAYER_ID").set_index("PLAYER_ID") for s in done}
        salary = pay.set_index(["name_key", "SEASON"])["Salary"].to_dict()
        for start in starts:
            i = done.index(start)
            ahead = done[i + 1:i + 4]
            known = df_master[df_master.index.get_level_values("SEASON") <= start]
            with _quiet():
                model = fit_rating_model(known, test_season=start, verbose=False)
                price = T.price_of_a_win(model, known)
                curve = T.fit_aging_curve(model, known)
                availability = T.fit_availability(known)
                level = T.current_level(model, known, availability=availability)
                first = T._first_seasons(T._regular_season(known))
            first.index = first.index.astype(str)
            years_in = {name_key(n): int(start[:4]) - first.get(str(p), 2000) for p, n in zip(level.index, level["Player"])}
            contracts = _contracts(pay, salary, start, ahead, years_in)
            for name, setting in settings.items():
                for k, v in setting.items():
                    setattr(T, k, v)
                with _quiet():
                    y, _ = T._years(model, known, 0.0, contracts, price, curve, availability=availability, title=None)
                y = y[y["AHEAD"] > 0]
                for r in y.itertuples():
                    a = actual[r.SEASON]
                    played = r.PLAYER_ID in a.index
                    rows.append({"SETTING": name, "START": start, "SEASON": r.SEASON, "AHEAD": r.AHEAD, "PLAYER_ID": str(r.PLAYER_ID),
                                 "Player": level.at[r.PLAYER_ID, "Player"], "AGE": r.AGE, "YEARS_IN": years_in.get(r.name_key, 99),
                                 "LEVEL": r.LEVEL, "MPG": r.MPG, "PRED_W": r.WINS * min(float(a["GP"].max()), 82) / 82,
                                 "ACTUAL_W": max(float(a["WINS"].get(r.PLAYER_ID, 0.0)), 0.0) if played else 0.0,
                                 "SALARY": r.Salary / SALARY_CAP[r.SEASON], "PRICE": price})
            if verbose:
                print(f"{start}: {len(contracts):,} contract rows, price of a win {price:.2%}", flush=True)
    finally:
        for k, v in saved.items():
            setattr(T, k, v)
    out = pd.DataFrame(rows)
    out["PRICE_BORROWED"] = out["PRICE"].isna()
    out["PRICE"] = out["PRICE"].bfill()                      # the earliest starts use the next start's price
    if save:
        out.to_parquet(PATH, index=False)
    return out


def load_backtest():
    return pd.read_parquet(PATH)


def _contract_totals(bt):
    d = bt.assign(PRED_M=(bt["PRED_W"] * bt["PRICE"] - bt["SALARY"]) * CAP_NOW_M, ACTUAL_M=(bt["ACTUAL_W"] * bt["PRICE"] - bt["SALARY"]) * CAP_NOW_M)
    return d.groupby(["SETTING", "START", "PLAYER_ID"]).agg(
        Player=("Player", "first"), YEARS_IN=("YEARS_IN", "first"), AGE=("AGE", "first"), SALARY=("SALARY", "sum"),
        PRED_M=("PRED_M", "sum"), ACTUAL_M=("ACTUAL_M", "sum"), SEASONS=("AHEAD", "size")).reset_index()


def backtest_summary(bt):
    """Prints wins error by horizon, whole-contract surplus by setting, calibration by tenth, and each start's best and worst 25."""
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    f = bt[bt["SETTING"] == "full"]
    print("Wins a season: rmse (ours vs a straight line on salary), correlation")
    for h, g in f.groupby("AHEAD"):
        line = np.polyval(np.polyfit(g["SALARY"], g["ACTUAL_W"], 1), g["SALARY"])
        print(f"  {h} ahead  {len(g):,} contract-seasons  {rmse(g['PRED_W'], g['ACTUAL_W']):.2f} vs {rmse(line, g['ACTUAL_W']):.2f}   "
              f"{g['PRED_W'].corr(g['ACTUAL_W']):.2f}")
    t = _contract_totals(bt)
    print(f"\nWhole-contract surplus, $M at today's cap ({(t['SETTING'] == 'full').sum():,} contracts)")
    for s, g in t.groupby("SETTING", sort=False):
        print(f"  {s:11s} correlation {g['PRED_M'].corr(g['ACTUAL_M']):.3f}, rmse {rmse(g['PRED_M'], g['ACTUAL_M']):.1f}, "
              f"average {g['PRED_M'].mean():+.1f} predicted, {g['ACTUAL_M'].mean():+.1f} actual")
    g = t[t["SETTING"] == "full"]
    print(f"  salary alone: correlation {(-g['SALARY']).corr(g['ACTUAL_M']):.3f}")
    print("\nBy tenth of predicted surplus")
    tenth = pd.qcut(g["PRED_M"].rank(method="first"), 10, labels=False) + 1
    print(g.groupby(tenth).agg(PREDICTED=("PRED_M", "mean"), ACTUAL=("ACTUAL_M", "mean"),
                               POSITIVE=("ACTUAL_M", lambda x: (x > 0).mean())).rename_axis("TENTH").round(2).to_string())
    print("\nBest and worst 25 at each start")
    for s, x in g.groupby("START"):
        top, low = x.nlargest(25, "PRED_M"), x.nsmallest(25, "PRED_M")
        print(f"  {s}  best {top['PRED_M'].mean():.0f} -> {top['ACTUAL_M'].mean():.0f} ({(top['ACTUAL_M'] > 0).mean():.0%} paid off)   "
              f"worst {low['PRED_M'].mean():.0f} -> {low['ACTUAL_M'].mean():.0f} ({(low['ACTUAL_M'] < 0).mean():.0%} lost)")
    print("\nRookies (average surplus, and the five best-rated at each start)")
    for s, x in t[t["YEARS_IN"] == 0].groupby("SETTING", sort=False):
        best = x.sort_values("PRED_M", ascending=False).groupby("START").head(5)
        print(f"  {s:11s} all {x['PRED_M'].mean():.1f} predicted, {x['ACTUAL_M'].mean():.1f} actual;  top five {best['PRED_M'].mean():.1f}, {best['ACTUAL_M'].mean():.1f}")


if __name__ == "__main__":
    import warnings
    from nba_viz_utils import INDEX_COLS
    warnings.filterwarnings("ignore")
    backtest_summary(run_backtest(pd.read_parquet(DATA_DIR / "df_master_current.parquet").set_index(INDEX_COLS).sort_index()))
