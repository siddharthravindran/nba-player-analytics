"""Refresh the in-progress season in the master — the notebook's "Refresh the master" cells as a script.

    python3 update_master.py              # refresh, check, save + timestamped backup
    python3 update_master.py --dry-run    # refresh and check, write nothing
    python3 update_master.py --extras     # also team tables and monthly splits
    python3 update_master.py --keep-backups 30

Meant to run unattended every morning (see launchd/com.nbadatavis.update-master.plist),
so it is careful about what it writes:

  - nothing is written unless the refreshed master passes the checks below;
  - the canonical file is replaced in one step, so a crash can't leave half a file;
  - if no stat changed (no games last night), nothing is written at all.

The Basketball-Reference columns are refreshed in the same run. That site posts
last night's games a few hours after the NBA does, so a run before it has
updated leaves those columns one day behind until the next run.

The season it refreshes is ACTIVE_SEASON / ACTIVE_SEASON_TYPE in nba_viz_utils.py.
"""
import argparse
import os
import sys
import time

import pandas as pd

from bbref_stats import add_bbref_advanced, load_advanced
from nba_viz_utils import (ACTIVE_SEASON, ACTIVE_SEASON_TYPE, DATA_DIR, INDEX_COLS, ON_OFF_COLS, add_on_off,
                           backfill_master, get_active_slice, refresh_active_season_data)

MASTER = DATA_DIR / "df_master_current.parquet"


def refresh(df_master):
    """The same two paths as the notebook: patch the season in place, or build it if it's new."""
    if ACTIVE_SEASON in set(df_master.index.get_level_values("SEASON")):
        df_master, _ = refresh_active_season_data(
            df_master, active_season=ACTIVE_SEASON, active_season_type=ACTIVE_SEASON_TYPE,
            force_refresh=True, timeout=180)
    else:
        print(f"{ACTIVE_SEASON} not in master — building its rows from scratch.")
        df_master = backfill_master(df_master, [ACTIVE_SEASON], season_types=(ACTIVE_SEASON_TYPE,))
    # on/off is not part of the standard refresh; keep it current once it has been added
    if all(c in df_master.columns for c in ON_OFF_COLS):
        df_master = add_on_off(df_master, [ACTIVE_SEASON], season_types=(ACTIVE_SEASON_TYPE,), force_refresh=True)
    return refresh_bbref(df_master)


def refresh_bbref(df_master, advanced=None):
    """Re-join the Basketball-Reference columns (BPM, age, ...), which the NBA refresh above doesn't touch.

    Like on/off, only once they have been added. If Basketball-Reference can't
    be reached or returns nothing for this season, the columns are left as they
    were rather than blanked — a day-old BPM is better than none.
    """
    if not any(c.startswith("BBR_") for c in df_master.columns):
        return df_master
    try:
        advanced = load_advanced() if advanced is None else advanced
        if ACTIVE_SEASON not in set(advanced["SEASON"]):
            raise ValueError(f"no {ACTIVE_SEASON} rows came back")
        return add_bbref_advanced(df_master, advanced)[df_master.columns]          # keep the column order
    except Exception as e:
        print(f"⚠️ Basketball-Reference columns not refreshed ({type(e).__name__}: {e}) — keeping the old ones")
        return df_master


def drop_stale_team_rows(df_master):
    """Remove the row a player leaves behind when he changes teams mid-season.

    The master is indexed by team, and a season row carries the player's totals
    under his latest team. After a trade the refresh adds a row for the new team
    but the old team's row stays, frozen at the day he left. Of a player's rows
    for the active season, keep the one with the most games (then the fullest).
    Returns the cleaned master and the number of rows removed.
    """
    flat = df_master.reset_index()
    active = (flat["SEASON"] == ACTIVE_SEASON) & (flat["SEASON_TYPE"] == ACTIVE_SEASON_TYPE)
    rows = flat[active].assign(_gp=pd.to_numeric(flat.loc[active, "GP"], errors="coerce").fillna(-1),
                               _filled=flat[active].notna().sum(axis=1))
    keep = rows.sort_values(["_gp", "_filled"]).drop_duplicates("PLAYER_ID", keep="last").index
    stale = rows.index.difference(keep)
    if len(stale):
        gone = flat.loc[stale]
        print(f"🧹 removed {len(stale)} stale rows left by team changes: "
              + ", ".join(f"{n} ({t})" for n, t in zip(gone["PLAYER_NAME"], gone["TEAM_ABBREVIATION"])))
        flat = flat.drop(index=stale)
    return flat.set_index(list(df_master.index.names)).sort_index(), len(stale)


def problems(before, after, removed=0):
    """Reasons not to save. An empty list means the refreshed master is safe to write.

    removed: stale rows dropped on purpose by drop_stale_team_rows, which don't count as lost.
    """
    found = []
    if len(after) + removed < len(before):
        found.append(f"rows went down: {len(before):,} -> {len(after):,}")
    lost = sorted(set(before.columns) - set(after.columns))
    if lost:
        found.append(f"{len(lost)} columns disappeared, e.g. {lost[:5]}")
    more_dupes = int(after.index.duplicated().sum()) - int(before.index.duplicated().sum())
    if more_dupes > 0:
        found.append(f"{more_dupes} new duplicate index rows")

    seasons = after.index.get_level_values("SEASON")
    old_seasons = before.index.get_level_values("SEASON")
    if (seasons != ACTIVE_SEASON).sum() != (old_seasons != ACTIVE_SEASON).sum():
        found.append("the number of rows in finished seasons changed")

    now, was = after[seasons == ACTIVE_SEASON], before[old_seasons == ACTIVE_SEASON]
    if now.empty:
        found.append(f"no {ACTIVE_SEASON} rows after the refresh")
    elif "GP" in now.columns and not was.empty:
        gp_now, gp_was = pd.to_numeric(now["GP"], errors="coerce").max(), pd.to_numeric(was["GP"], errors="coerce").max()
        if gp_now < gp_was:
            found.append(f"most games played went down: {gp_was:g} -> {gp_now:g}")
        # a season slice that comes back mostly empty means an endpoint returned nothing
        if now.notna().mean().mean() < 0.8 * was.notna().mean().mean():
            found.append("the season's rows are much emptier than before")
    return found


def save(df_master, keep_backups=None):
    flat = df_master.reset_index()
    tmp = MASTER.with_suffix(".parquet.tmp")
    flat.to_parquet(tmp, index=False)
    os.replace(tmp, MASTER)                                    # one step: never a half-written master
    stamp = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y%m%d_%H%M")
    backup = DATA_DIR / f"df_master_current_backup_{stamp}.parquet"
    flat.to_parquet(backup, index=False)
    print(f"💾 saved {MASTER.name} and {backup.name}")
    if keep_backups:
        old = sorted(DATA_DIR.glob("df_master_current_backup_*.parquet"))[:-keep_backups]
        for path in old:
            path.unlink()
        if old:
            print(f"🧹 removed {len(old)} backups older than the newest {keep_backups}")


def extras():
    """Tables that live beside the master. A failure here never affects the master."""
    from monthly_splits import build_monthly
    from team_stats import build_team_data
    jobs = {
        "team tables": lambda: build_team_data(ACTIVE_SEASON, ACTIVE_SEASON_TYPE, force_refresh=True),
        "monthly players": lambda: build_monthly([ACTIVE_SEASON], season_type=ACTIVE_SEASON_TYPE, force_refresh=True),
        "monthly teams": lambda: build_monthly([ACTIVE_SEASON], who="team", season_type=ACTIVE_SEASON_TYPE,
                                               force_refresh=True),
    }
    for name, job in jobs.items():
        try:
            job()
        except Exception as e:
            print(f"⚠️ {name} failed: {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="refresh and check, write nothing")
    parser.add_argument("--extras", action="store_true", help="also refresh team tables and monthly splits")
    parser.add_argument("--keep-backups", type=int, default=None, metavar="N",
                        help="delete all but the newest N backups (default: keep every one)")
    args = parser.parse_args()

    started = time.time()
    print(f"\n===== {pd.Timestamp.now(tz='America/Los_Angeles'):%Y-%m-%d %H:%M %Z} — "
          f"refreshing {ACTIVE_SEASON} {ACTIVE_SEASON_TYPE} =====")
    before = pd.read_parquet(MASTER).set_index(INDEX_COLS).sort_index()
    after, removed = drop_stale_team_rows(refresh(before.copy()))

    found = problems(before, after, removed)
    if found:
        print("❌ NOT SAVED — the refreshed master failed its checks:")
        for line in found:
            print(f"   - {line}")
        return 1

    was = get_active_slice(before, ACTIVE_SEASON, ACTIVE_SEASON_TYPE)
    now = get_active_slice(after, ACTIVE_SEASON, ACTIVE_SEASON_TYPE)
    changed = not was.sort_index().equals(now.sort_index().reindex(columns=was.columns))
    gp = pd.to_numeric(now["GP"], errors="coerce").max() if "GP" in now.columns and len(now) else float("nan")
    print(f"✅ checks passed: {len(after):,} rows, {after.shape[1]} columns, "
          f"{len(now)} {ACTIVE_SEASON} players, most games played {gp:g}")

    if not changed:
        print("no stat changed since the last run — nothing written")
    elif args.dry_run:
        print("dry run — nothing written")
    else:
        save(after, args.keep_backups)

    if args.extras and not args.dry_run:
        extras()
    print(f"done in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
