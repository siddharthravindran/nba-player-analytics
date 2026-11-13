"""When do the NBA standings settle?

    python3 standings_convergence.py            # seasons ending 2016..2026, 82-game ones only

Source: Basketball-Reference's "standings by date" pages, one per conference
per season. Each row is a date and each cell a team with its record that day,
which is enough to rebuild every team's record after each game it played.

Two different questions hide in "when do the standings normalize", and they
have different answers:

  vs final   How close is the table after g games to the final table? This
             reaches 1.0 by construction — the first g games are part of the
             final record — so it describes how much of the season is already
             banked, not how much the early games tell you.
  vs rest    How well does the record after g games rank teams over the games
             still to come? Nothing is counted twice here. This is the one
             that says when early records become informative.

Seeds here are ranked on win% alone (ties share a rank); the league's
head-to-head tiebreakers are not applied.
"""
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR, DATA_DIR, FIG_DIR = PROJECT_DIR / "cache", PROJECT_DIR / "data", PROJECT_DIR / "figures"

BASE = "https://www.basketball-reference.com"
HDR = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
CONFERENCES = ("eastern", "western")

# 2019-20 (bubble) and 2020-21 (72 games) are left out: a shortened season
# reaches its final table sooner, which would bend the average curve.
DEFAULT_END_YEARS = [y for y in range(2016, 2027) if y not in (2020, 2021)]

BLUE, ORANGE, AQUA, GRID, MUTED, INK = "#2a78d6", "#eb6834", "#1baf7a", "#e4e3de", "#8a897f", "#0b0b0b"

_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_DATE = re.compile(r'data-stat="date"[^>]*>\s*<a[^>]*>([^<]+)</a>')
_CELL = re.compile(r"&nbsp;([A-Z]{3})&nbsp;.*?\((\d+)-(\d+)\)", re.S)


# =============================================================================
# FETCH
# =============================================================================

def season_label(end_year):
    return f"{end_year - 1}-{str(end_year)[2:]}"


def parse_standings_page(html, end_year, conference):
    """One standings-by-date page -> long frame: one row per team per date."""
    start = html.find('id="standings_by_date"')
    if start < 0:
        raise ValueError("standings_by_date table not found")
    rows = []
    for row in _ROW.findall(html[start:]):
        date = _DATE.search(row)
        if not date:
            continue                                    # month headers, column headers
        for cell in row.split("<td")[1:]:
            m = _CELL.search(cell)
            if m:
                rows.append((date.group(1), m.group(1), int(m.group(2)), int(m.group(3))))
    df = pd.DataFrame(rows, columns=["DATE", "TEAM", "W", "L"])
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["SEASON"], df["CONF"] = season_label(end_year), conference
    return df


def fetch_conference(end_year, conference, force=False, sleep=4.0):
    """Cached per page. The pause keeps this under the site's 20 requests a minute."""
    cache = CACHE_DIR / f"bbref_standings_by_date_{end_year}_{conference}.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    url = f"{BASE}/leagues/NBA_{end_year}_standings_by_date_{conference}_conference.html"
    try:
        r = requests.get(url, headers=HDR, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        df = parse_standings_page(r.text, end_year, conference)
    finally:
        time.sleep(sleep)                               # on failures too
    CACHE_DIR.mkdir(exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"   🌐 {season_label(end_year)} {conference}: {df['TEAM'].nunique()} teams, {df['DATE'].nunique()} dates")
    return df


def load_standings(end_years=DEFAULT_END_YEARS, force=False):
    frames = []
    for year in end_years:
        for conference in CONFERENCES:
            try:
                frames.append(fetch_conference(year, conference, force=force))
            except Exception as e:
                print(f"   ⚠️ {season_label(year)} {conference}: {e}")
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# ANALYSIS
# =============================================================================

def by_game(standings):
    """Record after each game a team has played: one row per team per game number."""
    df = standings.assign(G=standings["W"] + standings["L"])
    df = df.sort_values("DATE").drop_duplicates(["SEASON", "TEAM", "G"], keep="first")
    df["PCT"] = df["W"] / df["G"]
    return df[df["G"] > 0]


def _spearman(a, b):
    """Rank correlation without scipy. Ties share a rank."""
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _top(pct, wins, k):
    """The k best records; wins break ties so the cut is deterministic."""
    return set(pd.DataFrame({"p": pct, "w": wins}).sort_values(["p", "w"], ascending=False).index[:k])


def season_curve(games):
    """Agreement with the final table after every game number, for one season."""
    wins = games.pivot(index="G", columns="TEAM", values="W")
    conf = games.drop_duplicates("TEAM").set_index("TEAM")["CONF"]
    n_games = int(wins.notna().all(axis=1)[::-1].idxmax())     # last game number every team reached
    wins = wins.loc[1:n_games]
    final_w = wins.loc[n_games]
    final_pct = final_w / n_games

    final_seed = final_pct.groupby(conf).rank(ascending=False)
    final_top = {(c, k): _top(final_pct[conf == c], final_w[conf == c], k)
                 for c in conf.unique() for k in (6, 10)}

    rows = []
    for g in range(1, n_games + 1):
        w = wins.loc[g]
        pct = w / g
        seed = pct.groupby(conf).rank(ascending=False)
        row = {
            "G": g,
            "VS_FINAL": _spearman(pct, final_pct),
            "SEED_ERROR": float((seed - final_seed).abs().mean()),
        }
        if g <= n_games - 10:                           # too few games left to rank on past this
            row["VS_REST"] = _spearman(pct, (final_w - w) / (n_games - g))
        for k in (6, 10):
            hits = sum(len(_top(pct[conf == c], w[conf == c], k) & final_top[(c, k)]) for c in conf.unique())
            row[f"TOP{k}_HELD"] = hits / (k * conf.nunique())
        rows.append(row)

    out = pd.DataFrame(rows)
    out["SEASON"], out["N_GAMES"] = games["SEASON"].iloc[0], n_games
    return out


def convergence(standings):
    games = by_game(standings)
    return pd.concat([season_curve(g) for _, g in games.groupby("SEASON")], ignore_index=True)


def typical_date(standings, game_numbers):
    """Median days from opening night to a team's g-th game, across seasons."""
    games = by_game(standings)
    opener = games.groupby("SEASON")["DATE"].transform("min")
    days = (games["DATE"] - opener).dt.days
    return days.groupby(games["G"]).median().reindex(game_numbers)


def first_stays(series, threshold, above=True):
    """First game number from which the series stays on the right side of threshold."""
    ok = (series >= threshold) if above else (series <= threshold)
    bad = ok[~ok]
    if bad.empty:
        return int(ok.index[0])
    nxt = ok.index[ok.index > bad.index[-1]]
    return int(nxt[0]) if len(nxt) else None


def season_length_table(standings, lengths=(10, 20, 30, 41, 50, 58, 66, 72, 82, 100)):
    """How much of the standings is talent, and how much luck, at each season length.

    Teams' final win% varies more than coin-flipping alone would produce. The
    extra spread is real differences in strength; the rest is luck, and luck
    shrinks as 1/N. So for an N-game season

        talent share = true variance / (true variance + p(1-p)/N)

    which is also the correlation you'd expect between two independent N-game
    seasons played by the same teams. `check` compares that prediction with the
    data: the correlation between each team's record in odd- and even-numbered
    games (two interleaved 41-game seasons).

    It treats a team's strength as fixed for the season. Trades, injuries and
    tanking move it, so read the numbers as an upper bound on how well any
    schedule length can sort teams.
    """
    games = by_game(standings)
    obs, luck, halves = [], [], []
    for _, g in games.groupby("SEASON"):
        final = g.sort_values("G").drop_duplicates("TEAM", keep="last")
        if final["G"].min() < 81:
            continue
        obs.append(final["PCT"].var(ddof=1))
        luck.append((final["PCT"] * (1 - final["PCT"])).mean())
        wins = g.pivot(index="G", columns="TEAM", values="W").loc[1:82]
        won = wins.diff().fillna(wins.iloc[0])                    # 1 if that game was a win
        halves.append(np.corrcoef(won.iloc[0::2].mean(), won.iloc[1::2].mean())[0, 1])
    pq = float(np.mean(luck))
    true_var = float(np.mean(obs)) - pq / 82
    table = pd.DataFrame({"GAMES": lengths})
    table["TALENT_SHARE"] = true_var / (true_var + pq / table["GAMES"])
    table["CORR_WITH_TRUE_STRENGTH"] = np.sqrt(table["TALENT_SHARE"])
    table["LUCK_SD_WINS_PER_82"] = np.sqrt(pq / table["GAMES"]) * 82
    table.attrs.update(half_luck_games=pq / true_var, sd_talent=np.sqrt(true_var),
                       check={"observed_41": float(np.mean(halves)),
                              "predicted_41": true_var / (true_var + pq / 41)})
    return table


# =============================================================================
# REPORT
# =============================================================================

def report(curves, standings):
    full = curves[curves["N_GAMES"] >= 81]
    mean = full.groupby("G").mean(numeric_only=True)
    days = typical_date(standings[standings["SEASON"].isin(full["SEASON"].unique())], mean.index)
    n = full["SEASON"].nunique()

    def when(g):
        if g is None:
            return "never"
        d = pd.Timestamp("2026-10-20") + pd.Timedelta(days=float(days.loc[g]))
        return f"game {g:>2}  (~day {days.loc[g]:.0f}, about {d:%b %-d} on an Oct 20 start)"

    print(f"\n{n} full seasons: {', '.join(sorted(full['SEASON'].unique()))}\n")
    print("Order of the whole league vs the FINAL standings (rank correlation, mean across seasons)")
    for t in (0.80, 0.90, 0.95):
        print(f"   stays above {t:.2f} from {when(first_stays(mean['VS_FINAL'], t))}")
    print("\nConference seed, average miss vs the final seed")
    for t in (2.0, 1.5, 1.0):
        print(f"   within {t:.1f} seeds from {when(first_stays(mean['SEED_ERROR'], t, above=False))}")
    # picking 6 of 15 at random already gets 40% of the top 6, and 10 of 15 gets
    # 67% of the top 10, so the top-10 number starts high and means little early
    print("\nShare of the final top 6 in each conference already in the top 6  (40% by chance)")
    for t in (0.70, 0.80, 0.90):
        print(f"   {t:.0%} from {when(first_stays(mean['TOP6_HELD'], t))}")
    rest = mean["VS_REST"].dropna()
    print("\nRecord so far vs the REST of the season (no double counting)")
    for g in (5, 10, 15, 20, 25, 30, 41):
        print(f"   after {g:>2} games: {rest.loc[g]:.2f}")
    reached = rest[rest >= 0.9 * rest.max()].index[0]
    print(f"   peak {rest.max():.2f} at game {int(rest.idxmax())}; first within 10% of that at game {int(reached)}")
    return mean, days


def plot(curves, mean, days, path):
    full = curves[curves["N_GAMES"] >= 81]
    n = full["SEASON"].nunique()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))

    def each_season(ax, col):
        for _, s in full.groupby("SEASON"):
            ax.plot(s["G"], s[col], color=GRID, lw=1, zorder=1)

    ax = axes[0]
    each_season(ax, "VS_FINAL")
    ax.plot(mean.index, mean["VS_FINAL"], color=BLUE, lw=2, zorder=3)
    rest = mean["VS_REST"].dropna()
    ax.plot(rest.index, rest, color=ORANGE, lw=2, zorder=3)
    ax.text(46, float(mean["VS_FINAL"].loc[46]) - 0.1, "vs final standings", color=INK, fontsize=9, ha="left")
    ax.text(46, float(rest.loc[46]) - 0.075, "vs rest of season", color=INK, fontsize=9, ha="left")
    ax.plot([], [], color=BLUE, lw=2, label="Record so far vs final standings")
    ax.plot([], [], color=ORANGE, lw=2, label="Record so far vs rest of season")
    ax.plot([], [], color=GRID, lw=1, label="Single seasons (vs final)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Rank correlation, all 30 teams")
    ax.set_title("How well the current order matches", loc="left", fontsize=11, fontweight="bold")

    ax = axes[1]
    each_season(ax, "SEED_ERROR")
    ax.plot(mean.index, mean["SEED_ERROR"], color=BLUE, lw=2, zorder=3)
    ax.set_ylim(0, None)
    ax.set_ylabel("Seeds off, average team")
    ax.set_title("Average miss on the final conference seed", loc="left", fontsize=11, fontweight="bold")

    ax = axes[2]
    ax.plot(mean.index, mean["TOP6_HELD"] * 100, color=BLUE, lw=2)
    ax.plot(mean.index, mean["TOP10_HELD"] * 100, color=AQUA, lw=2)
    ax.text(47, float(mean["TOP6_HELD"].loc[47]) * 100 - 7, "top 6", color=INK, fontsize=9)
    ax.text(47, float(mean["TOP10_HELD"].loc[47]) * 100 - 6, "top 10", color=INK, fontsize=9)
    for chance, name in ((40, "top 6"), (66.7, "top 10")):
        ax.axhline(chance, color=MUTED, lw=1, ls=(0, (2, 3)))
        ax.text(81, chance + 1.2, f"{name} by chance", color=MUTED, fontsize=8, ha="right")
    ax.plot([], [], color=BLUE, lw=2, label="Final top 6 (playoff spots)")
    ax.plot([], [], color=AQUA, lw=2, label="Final top 10 (incl. play-in)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_ylim(35, 101)
    ax.set_ylabel("% already in that group")
    ax.set_title("Final top 6 / top 10 already in place", loc="left", fontsize=11, fontweight="bold")

    ticks = [1, 10, 20, 30, 41, 50, 60, 70, 82]
    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{g}\n{(pd.Timestamp('2026-10-20') + pd.Timedelta(days=float(days.loc[g]))):%b %-d}"
                            for g in ticks], fontsize=8)
        ax.set_xlim(1, 82)
        ax.set_xlabel("Games played (typical date on an Oct 20 start)", fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED)

    fig.suptitle(f"When do the NBA standings settle?  Mean of {n} full seasons, "
                 f"{full['SEASON'].min()} to {full['SEASON'].max()}", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, -0.02, "Source: Basketball-Reference standings by date. Seeds ranked on win% only; "
             "2019-20 and 2020-21 excluded (shortened).", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n📈 {path.relative_to(PROJECT_DIR)}")


def where_are_we(end_year, mean):
    """Put the season in progress on the historical curve."""
    try:
        now = pd.concat([fetch_conference(end_year, c, force=True) for c in CONFERENCES])
    except Exception as e:
        print(f"\n(could not read {season_label(end_year)}: {e})")
        return
    latest = now.sort_values("DATE").drop_duplicates("TEAM", keep="last")
    g = int((latest["W"] + latest["L"]).median())
    if g not in mean.index:
        return
    m = mean.loc[g]
    print(f"\n{season_label(end_year)} as of {latest['DATE'].max():%b %-d}: teams have played about {g} games. "
          f"In past seasons, at that point:")
    print(f"   order vs final standings {m['VS_FINAL']:.2f} · order vs rest of season {m['VS_REST']:.2f}")
    print(f"   average team {m['SEED_ERROR']:.1f} seeds from where it finished")
    print(f"   {m['TOP6_HELD']:.0%} of the final top 6 and {m['TOP10_HELD']:.0%} of the final top 10 already there")


def main(end_years=DEFAULT_END_YEARS, current_end_year=2027):
    print("🏀 Standings by date")
    standings = load_standings(end_years)
    curves = convergence(standings)
    mean, days = report(curves, standings)
    table = season_length_table(standings)
    a = table.attrs
    print(f"\nHow long does the season need to be?  A record is half talent, half luck after "
          f"{a['half_luck_games']:.0f} games.")
    print(table.round(2).to_string(index=False))
    print(f"   check: odd vs even games correlate {a['check']['observed_41']:.2f}; "
          f"the model predicts {a['check']['predicted_41']:.2f}")
    DATA_DIR.mkdir(exist_ok=True)
    curves.to_parquet(DATA_DIR / "standings_convergence.parquet", index=False)
    plot(curves, mean, days, FIG_DIR / "standings_convergence.png")
    if current_end_year:
        where_are_we(current_end_year, mean)
    return curves


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or DEFAULT_END_YEARS
    main(years)
