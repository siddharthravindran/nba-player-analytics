"""Basketball-Reference advanced stats and current contracts.

df_master's "advanced" columns are NBA.com's (TS%, usage, assist rate, PIE).
Basketball-Reference's are a different set — PER, win shares, box plus-minus,
VORP — and are not in the master. This pulls them, one page per season.

    from bbref_stats import *
    adv = load_advanced()                                   # 2015-16 .. this season
    season_change(adv, "BBR_BPM", "2025-26", "2026-27")     # biggest risers and fallers
    plot_season_change(adv, "BBR_BPM", "2025-26", "2026-27")
    df_master = add_bbref_advanced(df_master, adv)          # optional: BBR_ columns in the master

    contracts = fetch_contracts()                           # this season's salaries, one page

Regular season only — that is what the league-wide advanced page covers. A
traded player gets his combined line (the "2TM" row), matching how the master
keeps one row per player-season.
"""
import re
import time
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from unidecode import unidecode

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR, DATA_DIR = PROJECT_DIR / "cache", PROJECT_DIR / "data"

BASE = "https://www.basketball-reference.com"
HDR = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
PAUSE = 4.0                                             # the site allows 20 requests a minute

# page column -> our name. BBR_ keeps them apart from NBA.com's TS_PCT, USG_PCT, AST_PCT.
ADVANCED_COLS = {
    "per": "BBR_PER", "ts_pct": "BBR_TS_PCT", "fg3a_per_fga_pct": "BBR_FG3A_RATE",
    "fta_per_fga_pct": "BBR_FT_RATE", "orb_pct": "BBR_ORB_PCT", "drb_pct": "BBR_DRB_PCT",
    "trb_pct": "BBR_TRB_PCT", "ast_pct": "BBR_AST_PCT", "stl_pct": "BBR_STL_PCT",
    "blk_pct": "BBR_BLK_PCT", "tov_pct": "BBR_TOV_PCT", "usg_pct": "BBR_USG_PCT",
    "ows": "BBR_OWS", "dws": "BBR_DWS", "ws": "BBR_WS", "ws_per_48": "BBR_WS_48",
    "obpm": "BBR_OBPM", "dbpm": "BBR_DBPM", "bpm": "BBR_BPM", "vorp": "BBR_VORP",
}
BBR_LABELS = {
    "BBR_PER": "PER", "BBR_BPM": "Box plus-minus", "BBR_OBPM": "Offensive box plus-minus",
    "BBR_DBPM": "Defensive box plus-minus", "BBR_VORP": "VORP", "BBR_WS": "Win shares",
    "BBR_OWS": "Offensive win shares", "BBR_DWS": "Defensive win shares",
    "BBR_WS_48": "Win shares per 48", "BBR_TS_PCT": "True shooting %", "BBR_USG_PCT": "Usage %",
    "BBR_TOV_PCT": "Turnover %", "BBR_STL_PCT": "Steal %", "BBR_BLK_PCT": "Block %",
}
# Counting totals: they grow all season, so a partial season can't be compared with a full one.
CUMULATIVE = {"BBR_OWS", "BBR_DWS", "BBR_WS", "BBR_VORP"}

BLUE, RED, GRID, MUTED, INK = "#2a78d6", "#e34948", "#e4e3de", "#8a897f", "#0b0b0b"

_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_CELL = re.compile(r'<t[dh][^>]*data-stat="([^"]+)"[^>]*>(.*?)</t[dh]>', re.S)
_SLUG = re.compile(r'data-append-csv="([^"]+)"')
_TAG = re.compile(r"<[^>]+>")


def season_label(end_year):
    return f"{end_year - 1}-{str(end_year)[2:]}"


def current_end_year(today=None):
    today = today or date.today()
    return today.year + 1 if today.month >= 10 else today.year


def name_key(name):
    """Same normalisation as nba_viz_utils.norm_name, so the two sources join."""
    x = unidecode(str(name)).lower().strip().replace(".", "")
    x = re.sub(r"\s+(jr|sr|iv|iii|ii|v)$", "", x)
    return re.sub(r"\s+", " ", x).strip()


def _get(url):
    try:
        r = requests.get(url, headers=HDR, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} for {url}")
        # the site is UTF-8 but doesn't say so in its headers; r.text would guess
        # Latin-1 and turn "Jokić" into "JokiÄ\x87"
        return r.content.decode("utf-8", errors="replace")
    finally:
        time.sleep(PAUSE)                               # on failures too


def _table_rows(html, table_id):
    start = html.find(f'id="{table_id}"')
    if start < 0:
        raise ValueError(f"table '{table_id}' not found")
    end = html.find("</table>", start)
    for row in _ROW.findall(html[start:end]):
        slug = _SLUG.search(row)
        if slug:                                        # header and league-average rows have none
            cells = {k: _TAG.sub("", v).replace("&nbsp;", " ").strip() for k, v in _CELL.findall(row)}
            yield slug.group(1), cells


# =============================================================================
# ADVANCED
# =============================================================================

def parse_advanced(html, end_year):
    rows = []
    for slug, c in _table_rows(html, "advanced"):
        row = {"BBR_ID": slug, "PLAYER": c.get("name_display", ""), "BBR_TEAM": c.get("team_name_abbr", ""),
               "BBR_POS": c.get("pos", ""), "BBR_AGE": c.get("age", ""), "BBR_G": c.get("games", ""),
               "BBR_MP": c.get("mp", "")}
        row.update({ours: c.get(theirs, "") for theirs, ours in ADVANCED_COLS.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    numeric = ["BBR_AGE", "BBR_G", "BBR_MP"] + list(ADVANCED_COLS.values())
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    # a traded player's combined row ("2TM") is listed before his per-team rows
    df["BBR_TEAMS"] = df.groupby("BBR_ID")["BBR_ID"].transform("size").sub(1).clip(lower=1)
    df = df.drop_duplicates("BBR_ID", keep="first").reset_index(drop=True)
    df.insert(0, "SEASON", season_label(end_year))
    df.insert(2, "name_key", df["PLAYER"].map(name_key))
    return df


def fetch_advanced(end_year, force=False):
    """One season. A finished season is read from cache; the current one is fetched again."""
    cache = CACHE_DIR / f"bbref_advanced_{end_year}.parquet"
    if cache.exists() and not (force or end_year >= current_end_year()):
        return pd.read_parquet(cache)
    df = parse_advanced(_get(f"{BASE}/leagues/NBA_{end_year}_advanced.html"), end_year)
    CACHE_DIR.mkdir(exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"   🌐 {season_label(end_year)} advanced: {len(df)} players")
    return df


def load_advanced(end_years=None, force=False, save=True):
    """Every season stacked. Default: 2015-16 through the season in progress."""
    end_years = end_years or range(2016, current_end_year() + 1)
    frames = []
    for year in end_years:
        try:
            frames.append(fetch_advanced(year, force=force))
        except Exception as e:
            print(f"   ⚠️ {season_label(year)}: {e}")
    out = pd.concat(frames, ignore_index=True)
    if save:
        out.to_parquet(DATA_DIR / "bbref_advanced.parquet", index=False)
    return out


def add_bbref_advanced(df_master, advanced=None):
    """Write the BBR_ columns into the master's regular-season rows.

    Joined on season + normalised name, since the two sites share no player id.
    Two players with the same name in one season are left unmatched rather than
    guessed at. Fills existing rows; creates none.
    """
    advanced = load_advanced(save=False) if advanced is None else advanced
    indexed = isinstance(df_master.index, pd.MultiIndex)
    index_cols = list(df_master.index.names) if indexed else None
    m = df_master.reset_index() if indexed else df_master.copy()

    cols = [c for c in advanced.columns if c.startswith("BBR_")]
    adv = advanced[~advanced.duplicated(["SEASON", "name_key"], keep=False)]
    lookup = adv.set_index(["SEASON", "name_key"])[cols]
    m = m.drop(columns=[c for c in cols if c in m.columns])

    regular = m["SEASON_TYPE"] == "Regular Season"
    keys = pd.MultiIndex.from_arrays([m.loc[regular, "SEASON"], m.loc[regular, "PLAYER_NAME"].map(name_key)])
    # the master may carry the same name twice in a season; only fill the unambiguous ones
    filled = lookup.reindex(keys).to_numpy(dtype=object)
    filled[keys.duplicated(keep=False)] = np.nan
    block = pd.DataFrame(np.nan, index=m.index, columns=cols).astype(object)
    block.loc[regular] = filled
    text_cols = ("BBR_ID", "BBR_TEAM", "BBR_POS")
    m = pd.concat([m, pd.DataFrame({c: block[c] if c in text_cols else pd.to_numeric(block[c], errors="coerce")
                                    for c in cols})], axis=1)

    by_season = m[regular].groupby("SEASON")["BBR_ID"].apply(lambda s: s.notna().mean())
    print(f"🔗 bbref advanced matched {m.loc[regular, 'BBR_ID'].notna().sum()}/{int(regular.sum())} "
          f"regular-season rows (worst season {by_season.idxmin()}: {by_season.min():.0%})")
    return m.set_index(index_cols).sort_index() if indexed else m


# =============================================================================
# SEASON TO SEASON
# =============================================================================

def season_change(df, stat, season_from, season_to, min_mp_from=1000, min_mp_to=None, name_col="PLAYER"):
    """Change in `stat` between two seasons, biggest riser first.

    Works on the bbref frame as is, or on any frame with SEASON and a name
    column. min_mp filters on total minutes (BBR_MP); min_mp_to defaults to a
    quarter of the minutes the busiest player has logged in the later season,
    which keeps a season in progress from being dominated by 40-minute samples.
    """
    if stat in CUMULATIVE:
        print(f"⚠️ {stat} is a running total — a season in progress will always look lower. "
              f"Use a rate (BBR_BPM, BBR_WS_48, BBR_PER) to compare across seasons.")
    a, b = df[df["SEASON"] == season_from], df[df["SEASON"] == season_to]
    if "SEASON_TYPE" in df.columns:
        a, b = a[a["SEASON_TYPE"] == "Regular Season"], b[b["SEASON_TYPE"] == "Regular Season"]
    if "BBR_MP" in df.columns:
        if min_mp_to is None:
            min_mp_to = 0.25 * b["BBR_MP"].max()
        a, b = a[a["BBR_MP"] >= min_mp_from], b[b["BBR_MP"] >= min_mp_to]
    key = "BBR_ID" if "BBR_ID" in df.columns else name_col
    a = a.dropna(subset=[key, stat]).drop_duplicates(key)
    b = b.dropna(subset=[key, stat]).drop_duplicates(key)
    out = b[[key, name_col, stat]].merge(a[[key, stat]], on=key, suffixes=("_cur", "_prev"))
    out = out.rename(columns={f"{stat}_prev": "PREV", f"{stat}_cur": "CURRENT"}).set_index(name_col)
    out["CHANGE"] = out["CURRENT"] - out["PREV"]
    out.attrs.update(stat=stat, season_from=season_from, season_to=season_to,
                     min_mp_from=min_mp_from, min_mp_to=min_mp_to)
    return out[["PREV", "CURRENT", "CHANGE"]].sort_values("CHANGE", ascending=False)


def plot_season_change(df, stat, season_from, season_to, top_n=10, direction="both", figsize=None, **kw):
    """Bars of the season-to-season change, largest increase on top."""
    changes = season_change(df, stat, season_from, season_to, **kw)
    if direction == "up":
        shown = changes.head(top_n)
    elif direction == "down":
        shown = changes.tail(top_n)
    else:
        n = min(top_n, len(changes) // 2)
        shown = pd.concat([changes.head(n), changes.tail(n)])
    shown = shown[::-1]
    y = range(len(shown))

    def fmt(v, sign=False):
        digits = 3 if abs(shown[["PREV", "CURRENT"]].to_numpy()).max() < 1 else 1
        return f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"

    fig, ax = plt.subplots(figsize=figsize or (9, max(3.5, 0.36 * len(shown) + 1.6)))
    ax.barh(list(y), shown["CHANGE"], height=0.68,
            color=[BLUE if v >= 0 else RED for v in shown["CHANGE"]], zorder=2)
    ax.axvline(0, color=MUTED, lw=1, zorder=3)
    for i, (_, row) in zip(y, shown.iterrows()):
        up = row["CHANGE"] >= 0
        ax.annotate(f"{fmt(row['CHANGE'], True)}   ({fmt(row['PREV'])} → {fmt(row['CURRENT'])})",
                    (row["CHANGE"], i), xytext=(6 if up else -6, 0), textcoords="offset points",
                    va="center", ha="left" if up else "right", fontsize=8, color=INK)
    name = BBR_LABELS.get(stat, stat.replace("_", " "))
    ax.set_yticks(list(y))
    ax.set_yticklabels(shown.index)
    ax.set_title(f"{name}: biggest moves, {season_from} to {season_to}", loc="left", fontsize=13, fontweight="bold")
    note = ""
    if changes.attrs.get("min_mp_to") is not None:
        note = (f"  (min {changes.attrs['min_mp_from']:.0f} min in {season_from}, "
                f"{changes.attrs['min_mp_to']:.0f} in {season_to})")
    ax.set_xlabel(f"Change in {name}{note}", fontsize=9)
    ax.margins(x=0.32)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    plt.tight_layout()
    plt.show()


# =============================================================================
# CONTRACTS
# =============================================================================

def parse_contracts(html):
    """The league-wide contracts page -> one row per player for the current season."""
    start = html.find('id="player-contracts"')
    season = re.search(r'data-stat="y1"[^>]*>\s*(\d{4}-\d{2})', html[start:])
    rows = []
    for slug, c in _table_rows(html, "player-contracts"):
        salary = re.sub(r"[^0-9]", "", c.get("y1", ""))
        if salary:
            rows.append({"BBR_ID": slug, "Player": c.get("player", ""), "Tm": c.get("team_id", ""),
                         "Salary": int(salary)})
    df = pd.DataFrame(rows)
    # a player whose pay is split between teams (waived and re-signed) is listed once
    # per team, each row already showing his full amount — keep one, don't add them
    df = (df.groupby("BBR_ID", as_index=False)
            .agg(Player=("Player", "first"), Tm=("Tm", "last"), Salary=("Salary", "max")))
    df.insert(0, "SEASON", season.group(1) if season else season_label(current_end_year()))
    df["name_key"] = df["Player"].map(name_key)
    return df.sort_values("Salary", ascending=False).reset_index(drop=True)


_YEAR_HEAD = re.compile(r'<th[^>]*data-stat="(y\d)"[^>]*>\s*(\d{4}-\d{2})')
_YEAR_CELL = re.compile(r'<td class="([^"]*)" data-stat="(y\d)"[^>]*csk="(\d+)"')
_OPTION = {"salary-pl": "player", "salary-tm": "team", "salary-et": "early termination"}


def parse_contract_years(html):
    """The same page, every season it lists: one row per player per contract year.

    OPTION is "player", "team" or "early termination" where that year is not yet
    certain, and "" for a guaranteed year.
    """
    start = html.find('id="player-contracts"')
    end = html.find("</table>", start)
    seasons = dict(_YEAR_HEAD.findall(html[start:end]))
    rows = []
    for row in _ROW.findall(html[start:end]):
        slug = _SLUG.search(row)
        if not slug:
            continue
        cells = {k: _TAG.sub("", v).replace("&nbsp;", " ").strip() for k, v in _CELL.findall(row)}
        for css, col, amount in _YEAR_CELL.findall(row):
            option = next((label for key, label in _OPTION.items() if key in css), "")
            rows.append({"BBR_ID": slug.group(1), "Player": cells.get("player", ""), "Tm": cells.get("team_id", ""),
                         "SEASON": seasons.get(col, col), "Salary": int(amount), "OPTION": option})
    df = pd.DataFrame(rows)
    # split between teams after a buyout: each row already shows the full amount (see parse_contracts)
    df = (df.sort_values("Salary").groupby(["BBR_ID", "SEASON"], as_index=False)
            .agg(Player=("Player", "first"), Tm=("Tm", "last"), Salary=("Salary", "max"), OPTION=("OPTION", "last")))
    df["name_key"] = df["Player"].map(name_key)
    return df.sort_values(["SEASON", "Salary"], ascending=[True, False]).reset_index(drop=True)


def fetch_contract_years(force=False, max_age_days=7):
    """Every remaining season of every contract, with option years marked. Cached for a week."""
    cache = DATA_DIR / "bbref_contract_years.parquet"
    if cache.exists() and not force and (time.time() - cache.stat().st_mtime) < max_age_days * 86400:
        return pd.read_parquet(cache)
    df = parse_contract_years(_get(f"{BASE}/contracts/players.html"))
    df.to_parquet(cache, index=False)
    print(f"   🌐 contract years: {df['BBR_ID'].nunique()} players, {df['SEASON'].min()} to {df['SEASON'].max()}")
    return df


def fetch_contracts(force=False, max_age_days=7):
    """This season's salaries. Cached for a week — contracts change with trades and signings."""
    cache = DATA_DIR / "bbref_contracts_current.parquet"
    if cache.exists() and not force and (time.time() - cache.stat().st_mtime) < max_age_days * 86400:
        return pd.read_parquet(cache)
    df = parse_contracts(_get(f"{BASE}/contracts/players.html"))
    df.to_parquet(cache, index=False)
    print(f"   🌐 contracts: {len(df)} players, {df['SEASON'].iloc[0]}")
    return df
