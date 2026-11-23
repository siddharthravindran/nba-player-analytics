"""Did trade_value's contracts pay off? Past seasons rebuilt with only what was known then.

    from backtest import *
    bt = run_backtest(df_master)                 # ~10 minutes; one row per contract-season per setting
    backtest_summary(bt)                         # prints the accuracy tables
    bt = load_backtest()                         # the saved run
    carried_forward(bt, df_master)               # against a tougher baseline: last season's wins, unchanged (~1 minute)
    calibration_chart(bt, "docs/backtest_calibration.png")

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

__all__ = ["run_backtest", "backtest_summary", "load_backtest", "carried_forward", "calibration_chart", "STARTS"]

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


def carried_forward(bt, df_master, boot=1000):
    """Prints our error against assuming every season repeats the last one (WINS from a rating model fitted as of then, per 82 games).

    The ranges resample players, not contracts: the same player is in several starts.
    """
    f = bt[bt["SETTING"] == "full"].copy()
    last = {}
    with _quiet():
        for start in f["START"].unique():
            known = df_master[df_master.index.get_level_values("SEASON") <= start]
            r = rank_players(fit_rating_model(known, test_season=start, verbose=False), known, start, min_min=0, min_gp=0)
            r = r.drop_duplicates("PLAYER_ID")
            last[start] = (r["WINS"].clip(lower=0) * 82 / min(r["GP"].max(), 82)).set_axis(r["PLAYER_ID"].astype(str)).to_dict()
    games = T._regular_season(df_master).groupby("SEASON")["GP"].max().clip(upper=82)
    f["CARRIED"] = [last[s].get(p, 0.0) * games[t] / 82 for s, p, t in zip(f["START"], f["PLAYER_ID"], f["SEASON"])]
    rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))
    for h, g in f.groupby("AHEAD"):
        print(f"  {h} ahead  wins rmse {rmse(g['PRED_W'] - g['ACTUAL_W']):.2f} vs {rmse(g['CARRIED'] - g['ACTUAL_W']):.2f} carried forward")
    ours, old = _contract_totals(f), _contract_totals(f.assign(PRED_W=f["CARRIED"]))
    a, p, n, sal = ours["ACTUAL_M"].to_numpy(), ours["PRED_M"].to_numpy(), old["PRED_M"].to_numpy(), -ours["SALARY"].to_numpy()
    corr = lambda x, i: np.corrcoef(x[i], a[i])[0, 1]
    rows = {k: np.flatnonzero(ours["PLAYER_ID"] == k) for k in ours["PLAYER_ID"].unique()}
    rng, keys = np.random.default_rng(0), list(rows)
    draws = []
    for _ in range(boot):
        i = np.concatenate([rows[keys[k]] for k in rng.integers(len(keys), size=len(keys))])
        draws.append((corr(p, i), corr(n, i), corr(sal, i), corr(p, i) - corr(n, i)))
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    i = np.arange(len(a))
    for k, (name, v) in enumerate((("ours", corr(p, i)), ("carried forward", corr(n, i)), ("salary alone", corr(sal, i)),
                                   ("ours minus carried", corr(p, i) - corr(n, i)))):
        print(f"  surplus correlation, {name:18s} {v:.3f}  (95% range {lo[k]:.2f} to {hi[k]:.2f})")
    print(f"  surplus rmse {rmse(p - a):.1f} vs {rmse(n - a):.1f} carried forward ({len(a):,} contracts, {len(keys)} players)")


def calibration_chart(bt, path=None):
    """Predicted and actual surplus by tenth of predicted surplus."""
    import matplotlib.pyplot as plt
    g = _contract_totals(bt[bt["SETTING"] == "full"])
    tenth = pd.qcut(g["PRED_M"].rank(method="first"), 10, labels=False) + 1
    d = g.groupby(tenth)[["PRED_M", "ACTUAL_M"]].mean()
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=160)
    x, w = np.arange(len(d)), 0.36
    for off, col, color, label in ((-w / 2 - 0.02, "PRED_M", "#a3a19b", "Predicted"), (w / 2 + 0.02, "ACTUAL_M", "#2a78d6", "What happened")):
        ax.bar(x + off, d[col], w, color=color, label=label)
        for k in (0, len(d) - 1):
            v = d[col].iloc[k]
            ax.annotate(f"{'-' if v < 0 else ''}${abs(v):.0f}M", (k + off, v), xytext=(0, 3 if v > 0 else -3), textcoords="offset points",
                        ha="center", va="bottom" if v > 0 else "top", fontsize=8, color="#333")
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xticks(x, ["Worst\ntenth", *map(str, range(2, 10)), "Best\ntenth"])
    ax.set_ylabel("Average surplus ($M, today's cap)", color="#333")
    ax.yaxis.grid(True, color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(length=0, colors="#333")
    ax.margins(y=0.1)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(f"Contracts ranked by predicted surplus, {min(g['START'])} to {max(g['START'])} ({len(g):,} contracts)", loc="left", fontsize=11, color="#222")
    fig.tight_layout()
    if path:
        fig.savefig(path, facecolor="white")
    return ax


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
