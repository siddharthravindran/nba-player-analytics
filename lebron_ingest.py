"""
LEBRON impact-metric ingestion for the NBA salary model (v2).

Loads per-season LEBRON exports (one file per season, single-season grain),
standardizes the season format to match the salary master ('YYYY-YY'),
concatenates them, and merges onto the model dataframe on PLAYER_ID + SEASON.

Join contract:
    LEBRON file            model master
    ------------           ------------
    nba_id           <->   PLAYER_ID
    Seasons ('2020') <->   SEASON ('2019-20')   [ending-year -> hyphenated]
"""
from __future__ import annotations
from pathlib import Path
import glob
import re
import pandas as pd

# Columns we keep from LEBRON. Drop Player/Team/Pos/MPG — they're either
# redundant with the master or would collide. Keep the impact metrics + roles.
LEBRON_VALUE_COLS = ["LEBRON", "O-LEBRON", "D-LEBRON", "WAR", "OffRole", "DefRole"]

# single-season files look like 'lebron-data-2019.csv'; the multi-season
# aggregate file (wrong grain) has multiple years and must be excluded.
_SINGLE_SEASON_RE = re.compile(r"lebron-data-(\d{4})\.csv$")


def _ending_year_to_season(year: int) -> str:
    """2020 -> '2019-20'  (NBA season string, ending-year convention)."""
    start = year - 1
    return f"{start}-{str(year)[-2:]}"


def load_lebron_seasons(folder: str) -> pd.DataFrame:
    """
    Discover and load every single-season LEBRON csv in `folder`,
    standardize the season column, and stack into one long table.

    Returns a frame indexed by (PLAYER_ID, SEASON) with LEBRON_VALUE_COLS,
    prefixed 'LBR_' so nothing collides with the master's own columns.
    """
    frames = []
    found_years = []

    for path in sorted(glob.glob(str(Path(folder) / "lebron-data-*.csv"))):
        m = _SINGLE_SEASON_RE.search(Path(path).name)
        if not m:
            # skips the multi-season aggregate file (wrong grain) automatically
            continue

        year = int(m.group(1))
        found_years.append(year)

        df = pd.read_csv(path)  # csv module / pandas handles the quoted role fields fine

        # standardize keys
        df = df.rename(columns={"nba_id": "PLAYER_ID"})
        df["SEASON"] = _ending_year_to_season(year)

        keep = ["PLAYER_ID", "SEASON"] + [c for c in LEBRON_VALUE_COLS if c in df.columns]
        df = df[keep]

        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No single-season LEBRON files found in {folder}. "
            "Expected files named like 'lebron-data-2019.csv'."
        )

    out = pd.concat(frames, ignore_index=True)

    # prefix value cols so they never clash with master columns
    rename = {c: f"LBR_{c.replace('-', '_')}" for c in LEBRON_VALUE_COLS if c in out.columns}
    out = out.rename(columns=rename)

    # integrity: one row per player-season
    dupes = out.duplicated(subset=["PLAYER_ID", "SEASON"]).sum()
    if dupes:
        print(f"⚠️  {dupes} duplicate PLAYER_ID+SEASON rows in LEBRON data — keeping first.")
        out = out.drop_duplicates(subset=["PLAYER_ID", "SEASON"], keep="first")

    print(f"✅ Loaded LEBRON for seasons: {sorted(_ending_year_to_season(y) for y in found_years)}")
    print(f"   {len(out)} player-season rows, cols: {[c for c in out.columns if c.startswith('LBR_')]}")
    return out


def merge_lebron(model_df: pd.DataFrame, lebron_df: pd.DataFrame,
                 how: str = "left") -> pd.DataFrame:
    """
    Merge LEBRON onto the model dataframe on PLAYER_ID + SEASON.

    `model_df` may be MultiIndex'd (SEASON, SEASON_TYPE, PLAYER_NAME, PLAYER_ID, ...);
    we reset, merge on the two shared keys, and restore the original index.
    """
    original_index = model_df.index.names if model_df.index.names[0] is not None else None
    m = model_df.reset_index() if original_index else model_df.copy()

    for key in ("PLAYER_ID", "SEASON"):
        if key not in m.columns:
            raise KeyError(f"model_df is missing join key '{key}' after reset_index().")

    # normalize PLAYER_ID dtype on both sides to avoid int/str mismatch silently nulling
    m["PLAYER_ID"] = m["PLAYER_ID"].astype(str).str.strip()
    lebron_df = lebron_df.copy()
    lebron_df["PLAYER_ID"] = lebron_df["PLAYER_ID"].astype(str).str.strip()

    before = len(m)
    merged = m.merge(lebron_df, on=["PLAYER_ID", "SEASON"], how=how)
    assert len(merged) == before, "row count changed on merge — check for dup keys in LEBRON data"

    # coverage report — how many model rows actually got a LEBRON match
    lbr_cols = [c for c in lebron_df.columns if c.startswith("LBR_")]
    if lbr_cols:
        matched = merged[lbr_cols[0]].notna().sum()
        print(f"🔗 LEBRON matched {matched}/{before} model rows "
              f"({matched/before:.0%}). Unmatched are pre-2016 or sub-threshold players.")

    if original_index:
        merged = merged.set_index(list(original_index))
    return merged


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    lbr = load_lebron_seasons(folder)
    print(lbr.head(10).to_string())
