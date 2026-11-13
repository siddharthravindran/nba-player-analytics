"""Month-by-month splits for players and teams, kept apart from df_master.

df_master holds one row per player-season. A month column would multiply every
row by seven and break every chart that assumes one row per player, so the
splits live in their own frame: one row per player (or team) per month.

    from monthly_splits import *
    monthly = build_monthly(["2026-27"])                 # players, Base + Advanced
    team_monthly = build_monthly(["2026-27"], who="team")

    month_over_month(monthly, "PTS", season="2026-27")   # biggest risers and fallers
    plot_monthly_trend(monthly, ["Stephen Curry", "Nikola Jokic"], "TS_PCT", "2026-27")
    plot_month_change(monthly, "PTS", "2026-27")

Each month is its own request (measure x month x season), so the default is
the two measures most charts need. A finished month never changes and is read
from cache for good; only the month in progress is fetched again.

Column names match df_master, so the existing helpers work on a month's slice:
    plot_interactive_barh(monthly[monthly["MONTH"] == "2026-11"], x_col="PTS", ...)
"""
import calendar
import time
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
from nba_api.stats.endpoints import LeagueDashPlayerStats, LeagueDashTeamStats
from nba_api.stats.static import teams as static_teams
from unidecode import unidecode

from nba_viz_utils import DATA_DIR, cached_fetch, drop_redundant_cols, safe_cache_name

TEAM_ABBR = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}

# Calendar months with regular-season games. Two seasons ran off the usual calendar.
SEASON_MONTHS = {
    "default": [10, 11, 12, 1, 2, 3, 4],
    "2019-20": [10, 11, 12, 1, 2, 3, 7, 8],        # suspended in March, resumed in the bubble
    "2020-21": [12, 1, 2, 3, 4, 5],                # started December 22
}

ENDPOINTS = {"player": (LeagueDashPlayerStats, "PLAYER_ID"), "team": (LeagueDashTeamStats, "TEAM_ID")}
BLUE, LIGHT_BLUE, RED, GRID, MUTED, INK = "#2a78d6", "#9ec5f4", "#e34948", "#e4e3de", "#8a897f", "#0b0b0b"


# =============================================================================
# FETCH
# =============================================================================

def month_windows(season, today=None):
    """The season's months as (label, name, date_from, date_to, finished).

    Months that haven't started are left out. label is 'YYYY-MM' so it sorts
    in season order; the dates are the MM/DD/YYYY strings the API takes.
    """
    today = today or date.today()
    start_year = int(season[:4])
    windows = []
    for month in SEASON_MONTHS.get(season, SEASON_MONTHS["default"]):
        year = start_year if month >= 9 else start_year + 1
        first = date(year, month, 1)
        last = date(year, month, calendar.monthrange(year, month)[1])
        if first > today:
            continue
        windows.append((f"{year}-{month:02d}", calendar.month_abbr[month],
                        first.strftime("%m/%d/%Y"), last.strftime("%m/%d/%Y"), last < today))
    return windows


def _fetch_month(who, measure, season, season_type, window, per_mode, timeout, force_refresh):
    label, _, date_from, date_to, finished = window
    endpoint, _ = ENDPOINTS[who]

    def fetch_raw():
        return endpoint(
            measure_type_detailed_defense=measure, per_mode_detailed=per_mode,
            season=season, season_type_all_star=season_type,
            date_from_nullable=date_from, date_to_nullable=date_to, timeout=timeout,
        ).get_data_frames()[0]

    name = safe_cache_name("monthly", who, measure, season, season_type, per_mode, label)
    try:
        return cached_fetch(name, fetch_raw, force_refresh=force_refresh or not finished)
    except Exception as e:
        print(f"   ⚠️ {name}: {e}")
        return None
    finally:
        time.sleep(0.8)                             # on the failures too


def build_monthly(seasons, who="player", measures=("Base", "Advanced"),
                  season_type="Regular Season", per_mode="PerGame",
                  timeout=30, force_refresh=False, save=True, today=None):
    """One row per player (or team) per month.

    Adds MONTH ('2026-11'), MONTH_NAME ('Nov') and MONTH_COMPLETE; GP and MIN
    are that month's. Saved to data/df_monthly_{who}s.parquet, replacing only
    the seasons passed in, so old seasons survive a refresh of the current one.
    """
    _, key = ENDPOINTS[who]
    frames = []
    for season in seasons:
        windows = month_windows(season, today)
        print(f"🗓️  {season}: {len(windows)} months x {len(measures)} measures")
        for window in windows:
            label, name, _, _, finished = window
            wide = None
            for measure in measures:
                df = _fetch_month(who, measure, season, season_type, window, per_mode, timeout, force_refresh)
                if df is None or df.empty:
                    continue
                df = drop_redundant_cols(df.drop(
                    columns=[c for c in df.columns if c.startswith("CF") or c == "NICKNAME"]))
                if wide is None:
                    wide = df
                else:
                    # the measures repeat GP, MIN and each other's headline columns
                    new = [c for c in df.columns if c not in wide.columns]
                    wide = wide.merge(df[[key] + new], on=key, how="outer")
            if wide is None:
                continue
            wide["SEASON"], wide["SEASON_TYPE"] = season, season_type
            wide["MONTH"], wide["MONTH_NAME"], wide["MONTH_COMPLETE"] = label, name, finished
            frames.append(wide)

    if not frames:
        print("   ⚠️ nothing came back")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    if who == "player":
        out["PLAYER_NAME"] = out["PLAYER_NAME"].map(lambda x: unidecode(str(x)).strip())   # as in the master
    else:
        out.insert(0, "TEAM", out["TEAM_ID"].map(TEAM_ABBR))

    lead = [c for c in ("SEASON", "SEASON_TYPE", "MONTH", "MONTH_NAME", "MONTH_COMPLETE") if c in out.columns]
    out = out[lead + [c for c in out.columns if c not in lead]]

    if save:
        path = DATA_DIR / f"df_monthly_{who}s.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            fetched = set(zip(out["SEASON"], out["SEASON_TYPE"]))
            keep = [pair not in fetched for pair in zip(old["SEASON"], old["SEASON_TYPE"])]
            out = pd.concat([old[keep], out], ignore_index=True)
        out = out.sort_values(["SEASON", "MONTH"]).reset_index(drop=True)
        out.to_parquet(path, index=False)
        print(f"💾 {path.name}: {len(out)} rows, {out['SEASON'].nunique()} seasons")
    return out


def load_monthly(who="player"):
    return pd.read_parquet(DATA_DIR / f"df_monthly_{who}s.parquet")


# =============================================================================
# MONTH TO MONTH
# =============================================================================

def _entity(df):
    return "PLAYER_NAME" if "PLAYER_NAME" in df.columns else "TEAM"


def monthly_pivot(monthly, stat, season, season_type="Regular Season", min_gp=5, min_min=None):
    """Wide table: one row per player/team, one column per month, in season order.

    A month only counts if the player played min_gp games in it (and averaged
    min_min minutes, if given) — three games in October isn't a month.
    Team frames carry total minutes per game, so min_min is for players.
    """
    df = monthly[(monthly["SEASON"] == season) & (monthly["SEASON_TYPE"] == season_type)]
    df = df[df["GP"] >= min_gp]
    if min_min is not None:
        df = df[df["MIN"] >= min_min]
    wide = df.pivot_table(index=_entity(df), columns="MONTH", values=stat, aggfunc="first")
    return wide[sorted(wide.columns)]


def month_over_month(monthly, stat, season, month=None, season_type="Regular Season",
                     min_gp=5, min_min=None):
    """Change in `stat` from the month before to `month` (default: the latest).

    Returns PREV, CURRENT, CHANGE sorted biggest riser first. Only players or
    teams that qualify in both months are compared.
    """
    wide = monthly_pivot(monthly, stat, season, season_type, min_gp, min_min)
    months = list(wide.columns)
    month = month or months[-1]
    if month not in months or months.index(month) == 0:
        raise ValueError(f"need a month with one before it; have {months}")
    prev = months[months.index(month) - 1]
    out = wide[[prev, month]].dropna().rename(columns={prev: "PREV", month: "CURRENT"})
    out["CHANGE"] = out["CURRENT"] - out["PREV"]
    out.attrs.update(prev=prev, month=month, stat=stat)
    return out.sort_values("CHANGE", ascending=False)


def plot_monthly_trend(monthly, names, stat, season, season_type="Regular Season",
                       min_gp=1, figsize=(10, 5)):
    """One line per player/team across the season's months."""
    if isinstance(names, str):
        names = [names]
    wide = monthly_pivot(monthly, stat, season, season_type, min_gp=min_gp)
    month_names = dict(zip(monthly["MONTH"], monthly["MONTH_NAME"]))

    fig, ax = plt.subplots(figsize=figsize)
    for name in names:
        if name not in wide.index:
            print(f"⚠️ No rows for {name} in {season}.")
            continue
        row = wide.loc[name].dropna()
        ax.plot(row.index, row.values, marker="o", lw=2, label=name)
    ax.set_xticks(range(len(wide.columns)))
    ax.set_xticklabels([month_names[m] for m in wide.columns])
    ax.set_title(f"{stat.replace('_', ' ')} by month — {season}", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel(stat.replace("_", " "))
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def plot_month_change(monthly, stat, season, month=None, season_type="Regular Season",
                      top_n=10, direction="both", style="bars", min_gp=5, min_min=None, figsize=None):
    """Biggest month-to-month moves, ordered by the size of the change.

    style="bars"      the bar IS the change, largest increase on top; the label
                      gives the change and the two values behind it.
    style="dumbbell"  both months' values as dots on the stat's own scale. Rows
                      are still ordered by change, but the eye follows the dots'
                      positions, so it reads as levels more than as a ranking.
    direction: "both" (top_n risers and top_n fallers), "up" or "down".
    """
    changes = month_over_month(monthly, stat, season, month, season_type, min_gp, min_min)
    prev, month = changes.attrs["prev"], changes.attrs["month"]
    month_names = dict(zip(monthly["MONTH"], monthly["MONTH_NAME"]))
    if direction == "up":
        shown = changes.head(top_n)
    elif direction == "down":
        shown = changes.tail(top_n)
    else:
        n = min(top_n, len(changes) // 2)
        shown = pd.concat([changes.head(n), changes.tail(n)])
    shown = shown[::-1]                                  # matplotlib draws the first row at the bottom
    y = range(len(shown))

    def signed(v):
        return f"{v:+.3f}" if abs(v) < 1 else f"{v:+.1f}"

    def plain(v):
        return f"{v:.3f}" if abs(v) < 1 else f"{v:.1f}"

    fig, ax = plt.subplots(figsize=figsize or (9, max(3.5, 0.36 * len(shown) + 1.6)))
    name = stat.replace("_", " ")
    if style == "dumbbell":
        ax.hlines(y, shown["PREV"], shown["CURRENT"], color=LIGHT_BLUE, lw=2, zorder=1)
        ax.scatter(shown["PREV"], y, color=LIGHT_BLUE, s=55, zorder=2, label=month_names[prev])
        ax.scatter(shown["CURRENT"], y, color=BLUE, s=55, zorder=3, label=month_names[month])
        for i, (_, row) in zip(y, shown.iterrows()):
            ax.annotate(signed(row["CHANGE"]), (max(row["PREV"], row["CURRENT"]), i), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=8, color=MUTED)
        ax.set_xlabel(f"{name}  (min {min_gp} games in each month)")
        ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1), ncol=2)
        ax.margins(x=0.12)
    else:
        ax.barh(list(y), shown["CHANGE"], height=0.68,
                color=[BLUE if v >= 0 else RED for v in shown["CHANGE"]], zorder=2)
        ax.axvline(0, color=MUTED, lw=1, zorder=3)
        for i, (_, row) in zip(y, shown.iterrows()):
            up = row["CHANGE"] >= 0
            ax.annotate(f"{signed(row['CHANGE'])}   ({plain(row['PREV'])} → {plain(row['CURRENT'])})",
                        (row["CHANGE"], i), xytext=(6 if up else -6, 0), textcoords="offset points",
                        va="center", ha="left" if up else "right", fontsize=8, color=INK)
        ax.set_xlabel(f"Change in {name}, {month_names[prev]} to {month_names[month]}  "
                      f"(min {min_gp} games in each month)")
        ax.margins(x=0.32)

    ax.set_yticks(list(y))
    ax.set_yticklabels(shown.index)
    ax.set_title(f"{name}: biggest moves, {month_names[prev]} to {month_names[month]} {season}",
                 loc="left", fontsize=13, fontweight="bold")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    plt.tight_layout()
    plt.show()
