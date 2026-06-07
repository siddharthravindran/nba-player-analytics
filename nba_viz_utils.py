from __future__ import annotations
from typing import List, Union, Optional, Tuple, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holoviews as hv
import hvplot.pandas  # noqa: F401
import re
import time
import unicodedata
from unidecode import unidecode
from pathlib import Path

hv.extension("bokeh")

from nba_api.stats.endpoints import (
    LeagueDashPlayerStats,
    LeagueDashPtStats,
    SynergyPlayTypes,
    CommonAllPlayers,
    LeagueDashPlayerPtShot,
    LeagueDashPlayerBioStats,
    LeagueHustleStatsPlayer
)

# =============================================================================
# SECTION 1: CONSTANTS & CONFIG
# =============================================================================

PROJECT_DIR = Path("/content/drive/MyDrive/nba_data_vis")

CACHE_DIR = PROJECT_DIR / "cache"
DATA_DIR  = PROJECT_DIR / "data"
FIG_DIR   = PROJECT_DIR / "figures"

for d in [CACHE_DIR, DATA_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

minimal_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/"
}

PT_STAT_TYPES = {
    'SpeedDistance', 'Rebounding', 'Possessions', 'CatchShoot', 'PullUpShot',
    'Drives', 'Passing', 'ElbowTouch', 'PostTouch', 'PaintTouch', 'Efficiency', 'Defense'
}

PLAYER_STAT_TYPES = {
    'Base', 'Advanced', 'Misc', 'Four Factors', 'Scoring',
    'Opponent', 'Usage', 'Defense'
}

SYNERGY_PLAY_TYPES = {
    'Cut', 'Handoff', 'Isolation', 'Misc', 'OffScreen', 'Postup',
    'PRBallHandler', 'PRRollman', 'OffRebound', 'Spotup', 'Transition'
}

PREFIXES = {
    'PRBallHandler': 'PNRBH_', 'PRRollman': 'RM_', 'Isolation': 'ISO_',
    'Transition': 'TRANS_', 'Postup': 'POST_', 'Spotup': 'SPOT_',
    'Drives': 'DRIVE_', 'Handoff': 'DHO_', 'Cut': 'CUT_',
    'ElbowTouch': 'ELBOW_', 'PostTouch': 'POSTT_', 'PaintTouch': 'PAINT_',
    'General': 'SHOT_', 'ShotClock': 'SC_', 'Dribble': 'DRIBB_',
    'TouchTime': 'TOUCH_', 'ClosestDefender': 'CLOSESTDEF_', 'OffScreen': 'OFFSCREEN_',
    'OffRebound':'PUTBACK_'
}

EXCLUDE = {
    'SEASON', 'TEAM_ABBREVIATION', 'PLAYER_NAME', 'PLAYER_ID', 'TEAM_ID',
    'GP', 'G', 'MIN', 'W', 'L', 'W_PCT','DRIVES', 'AGE', 'NICKNAME',
    'PLAYER_LAST_TEAM_ID', 'PLAYER_LAST_TEAM_ABBREVIATION',
    'TEAM_NAME', 'SEASON_ID', 'PLAY_TYPE', 'TYPE_GROUPING', 'PAINT_TOUCHES',
    'index'
}

BUCKET_CONFIG = {
    'Dribble': {
        'param': 'dribble_range_nullable',
        'prefix': 'DRIBB_',
        'buckets': ['0 Dribbles', '1 Dribble', '2 Dribbles', '3-6 Dribbles', '7+ Dribbles']
    },
    'ShotClock': {
        'param': 'shot_clock_range_nullable',
        'prefix': 'SC_',
        'buckets': ['24-22', '22-18 Very Early', '18-15 Early', '15-7 Average', '7-4 Late', '4-0 Very Late']
    },
    'ClosestDefender': {
        'param': 'close_def_dist_range_nullable',
        'prefix': 'CLOSESTDEF_',
        'buckets': ['0-2 Feet - Very Tight', '2-4 Feet - Tight', '4-6 Feet - Open', '6+ Feet - Wide Open']
    },
    'TouchTime': {
    'param': 'touch_time_range_nullable',
    'prefix': 'TOUCH_', 
    # By starting with the code, .split(' ')[0] grabs 'LT2', '2_6', etc.
    'buckets': ['LT2 < 2 Seconds', '2_6 2-6 Seconds', '6P 6+ Seconds']
}
}


# --- add near BUCKET_CONFIG (Section 1) ---
DEF_BREAKDOWN_CONFIG = {
    # LeagueDashPtDefend defense_category : column prefix
    '2 Pointers':        'DEF2_',
    '3 Pointers':        'DEF3_',
    'Less Than 6Ft':     'DEFLT6_',
    'Less Than 10Ft':    'DEFLT10_',
    'Greater Than 15Ft': 'DEFGT15_',
}

INDEX_COLS = ['SEASON', 'SEASON_TYPE','PLAYER_NAME', 'PLAYER_ID', 'TEAM_ABBREVIATION', 'TEAM_ID']
_DEF_KEEP_KEYS = {'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'TEAM_ID', 'SEASON', 'SEASON_TYPE'}
_DEF_DROP = (set(EXCLUDE) - set(INDEX_COLS)) | {'PLAYER_POSITION'} # box-score dupes; these come from Base


misc_cols = [
    'PTS_OFF_TOV',
    'PTS_2ND_CHANCE',
    'PTS_FB',
    'OPP_PTS_OFF_TOV',
    'OPP_PTS_2ND_CHANCE',
    'OPP_PTS_FB',
    'OPP_PTS_PAINT'
]

shoot_cols = [
    'Restricted_Area_FGM', 'Restricted_Area_FGA', 'Restricted_Area_FG_PCT',
    'In_The_Paint_Non_RA_FGM', 'In_The_Paint_Non_RA_FGA', 'In_The_Paint_Non_RA_FG_PCT',
    'Mid_Range_FGM', 'Mid_Range_FGA', 'Mid_Range_FG_PCT',
    'Corner_3_FGM', 'Corner_3_FGA', 'Corner_3_FG_PCT',
    'Above_the_Break_3_FGM', 'Above_the_Break_3_FGA', 'Above_the_Break_3_FG_PCT'
]

reb_cols = [
    'OREB_CONTEST', 'OREB_UNCONTEST',
       'OREB_CONTEST_PCT', 'OREB_CHANCES', 'OREB_CHANCE_PCT',
       'OREB_CHANCE_DEFER', 'OREB_CHANCE_PCT_ADJ', 'AVG_OREB_DIST',
       'DREB_CONTEST', 'DREB_UNCONTEST', 'DREB_CONTEST_PCT', 'DREB_CHANCES',
       'DREB_CHANCE_PCT', 'DREB_CHANCE_DEFER', 'DREB_CHANCE_PCT_ADJ',
       'AVG_DREB_DIST', 'REB_CONTEST', 'REB_UNCONTEST',
       'REB_CONTEST_PCT', 'REB_CHANCES', 'REB_CHANCE_PCT', 'REB_CHANCE_DEFER',
       'REB_CHANCE_PCT_ADJ', 'AVG_REB_DIST'
]

general_shots_cols = [
    'SHOT_FG2A_FREQUENCY',
    'SHOT_FG2M',
    'SHOT_FG2A',
    'SHOT_FG2_PCT',
    'SHOT_FG3A_FREQUENCY',
]

def_cols = [
    'D_FGM',
    'D_FGA',
    'D_FG_PCT',
    'PCT_PLUSMINUS',
    'PLAYER_POSITION'
]

player_defense_cols = ["DEF_WS"]


REFRESH_NBA_STATS_CONFIG = {
    "Misc": {
        "suffix": "misc",
        "keep_cols": misc_cols,
    },
    "Rebounding": {
        "suffix": None,
        "keep_cols": reb_cols,
    },
    "General": {
        "suffix": "gen_shots",
        "keep_cols": general_shots_cols,
    },
    "Player Defense": {
        "stat_type": "Defense",
        "source_override": "player",
        "suffix": "def_player",
        "keep_cols": player_defense_cols,
    }
}


FULL_NBA_STATS = [
    "Base",
    "Advanced",
    "Usage",
    "Scoring",

    "Drives",
    "Possessions",
    "PullUpShot",
    "CatchShoot",
    "Passing",
    "Defense",
    "PaintTouch",
    "ElbowTouch",
    "SpeedDistance",
    "PostTouch",

    "PRBallHandler",
    "PRRollman",
    "Isolation",
    "Handoff",
    "Postup",
    "OffScreen",
    "Transition",
    "Spotup",
    "Cut",
    "OffRebound",
]

CORE_BOX_SCORE_COLS = {
    "GP", "G", "MIN",
    "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB",
    "AST", "TOV", "STL", "BLK",
    "BLKA", "PF", "PFD",
    "PTS", "PLUS_MINUS",
}


SHOT_LOCATION_FGA_COLS = [
    "Restricted_Area_FGA",
    "In_The_Paint_Non_RA_FGA",
    "Mid_Range_FGA",
    "Corner_3_FGA",
    "Above_the_Break_3_FGA",
]

PLAY_TYPE_FGA_COLS = [
    "PNRBH_FGA",
    "RM_FGA",
    "ISO_FGA",
    "DHO_FGA",
    "POST_FGA",
    "OFFSCREEN_FGA",
    "TRANS_FGA",
    "CUT_FGA",
    "SPOTUP_FGA",
    "PUTBACK_FGA",
]


# =============================================================================
# SECTION 2: ROUTING
# =============================================================================

def _route_stat_type(stat_type):
    pt_shot_dashboard = ['General', 'ShotClock', 'Dribble', 'ClosestDefender', 'TouchTime']
    synergy_types     = ['PRBallHandler', 'PRRollman', 'Isolation', 'Transition', 'Postup',
                         'Spotup', 'Handoff', 'Cut', 'OffScreen', 'OffRebound']
    pt_stats          = ['Drives', 'Defense', 'CatchShoot', 'Passing', 'PullUpShot',
                         'Rebounding', 'Possessions', 'ElbowTouch', 'PostTouch',
                         'PaintTouch', 'Efficiency', 'SpeedDistance']

    if stat_type in pt_shot_dashboard: return 'pt_shot'
    elif stat_type in synergy_types:   return 'synergy'
    elif stat_type in pt_stats:        return 'pt'
    else:                              return 'player'


# =============================================================================
# SECTION 3: API FETCH HELPERS
# =============================================================================

def _fetch_pt_shot(season, per_mode, season_type, timeout=120,**kwargs):
    return LeagueDashPlayerPtShot(
        season=season, per_mode_simple=per_mode,
        season_type_all_star=season_type, timeout=timeout,**kwargs
    ).get_data_frames()[0]


def _fetch_pt_stats(stat_type, season, player_or_team, per_mode, season_type, timeout=120):
    return LeagueDashPtStats(
        pt_measure_type=stat_type, player_or_team=player_or_team,
        per_mode_simple=per_mode, season=season,
        season_type_all_star=season_type,
        timeout=timeout
    ).get_data_frames()[0]


def _fetch_player_stats(stat_type, season, per_mode, season_type, timeout=120):
    return LeagueDashPlayerStats(
        season=season,
        per_mode_detailed=per_mode,
        measure_type_detailed_defense=stat_type,
        season_type_all_star=season_type,
        timeout=timeout
    ).get_data_frames()[0]


def _fetch_synergy(stat_type, season, player_or_team, per_mode, season_type, synergy_grouping, timeout=120):
    abbrev = 'P' if player_or_team == 'Player' else 'T'
    return SynergyPlayTypes(
        play_type_nullable=stat_type,
        player_or_team_abbreviation=abbrev,
        per_mode_simple=per_mode, season=season,
        season_type_all_star=season_type,
        type_grouping_nullable=synergy_grouping.capitalize(),
        timeout=timeout
    ).get_data_frames()[0]


# =============================================================================
# SECTION 4: CACHE & REFRESH HELPERS
# =============================================================================

def safe_cache_name(*parts):
    return "_".join(
        str(p).replace(" ", "_").replace("/", "_").replace("+", "P")
        for p in parts
    ).lower()


def cached_fetch(cache_name, fetch_fn, force_refresh=False):
    path = CACHE_DIR / f"{cache_name}.parquet"

    if path.exists() and not force_refresh:
        print(f"📂 Loading cache: {path.name}")
        return pd.read_parquet(path)

    print(f"🌐 Fetching fresh: {path.name}")
    df = fetch_fn()

    if df is not None and not df.empty:
        df.reset_index().to_parquet(path, index=False)
        print(f"💾 Saved cache: {path.name}")

    return df


def should_refresh(season, season_type):
    return (
        season == "2025-26"
        and season_type == "Playoffs"
    )
    

# =============================================================================
# SECTION 5: DATA PROCESSING HELPERS
# =============================================================================

def _normalize_names(df):
    if 'PLAYER_NAME' in df.columns:
        df['PLAYER_NAME'] = (df['PLAYER_NAME']
                             .str.normalize('NFKD')
                             .str.encode('ascii', errors='ignore')
                             .str.decode('utf-8'))
    return df



def _apply_prefix(df, stat_type, prefixes=PREFIXES, exclude=EXCLUDE):
    if stat_type not in prefixes:
        return df

    # Drop generic total touches, but keep PAINT_TOUCHES / ELBOW_TOUCHES / POST_TOUCHES
    if stat_type in {"PaintTouch", "ElbowTouch", "PostTouch"}:
        df = df.drop(columns=["TOUCHES"], errors="ignore")

    prefix = prefixes[stat_type]

    semantic_starts = {
        "PaintTouch": "PAINT_TOUCH",
        "ElbowTouch": "ELBOW_TOUCH",
        "PostTouch": "POST_TOUCH",
    }

    semantic_start = semantic_starts.get(stat_type)

    rename_map = {}

    for c in df.columns:
        c_str = str(c)

        if c in exclude:
            continue

        if c_str.startswith(prefix):
            continue

        # Prevent PAINT_PAINT_TOUCHES and PAINT_PAINT_TOUCH_FGM, etc.
        if semantic_start and c_str.startswith(semantic_start):
            continue

        rename_map[c] = f"{prefix}{c_str}"

    return df.rename(columns=rename_map)



def clean_bucket(b):
    parts = b.split(' ')
    meaningful = [p for p in parts if p[0].isdigit() or p[0] in '<>+']
    if len(meaningful) >= 2 and meaningful[0] in ('<', '>'):
        # Handle "< 2" or "> 6" — combine symbol + number
        num = parts[parts.index(meaningful[0]) + 1]
        return meaningful[0].replace('<', 'LT_').replace('>', 'GT_') + num
    elif meaningful:
        return meaningful[0].replace('-', '_').replace('+', 'P').replace('<', 'LT_').replace('>', 'GT_')
    return parts[0].replace('-', '_').replace('+', 'P')


def apply_bucket_suffix(df, bucket, prefix, exclude_cols):
    b = clean_bucket(bucket)
    new_cols = {}
    for c in df.columns:
        if c in exclude_cols:
            continue
        if c.startswith(prefix):
            stat = c.replace(prefix, "", 1)
            new_cols[c] = f"{prefix}{b}_{stat}"
    return df.rename(columns=new_cols)


def _consolidate(dfs, stat_type, source):
    result = pd.concat(dfs, ignore_index=True)

    if 'TEAM_ID' not in result.columns and 'PLAYER_LAST_TEAM_ID' in result.columns:
        result = result.rename(columns={'PLAYER_LAST_TEAM_ID': 'TEAM_ID'})
    if 'TEAM_ABBREVIATION' not in result.columns and 'PLAYER_LAST_TEAM_ABBREVIATION' in result.columns:
        result = result.rename(columns={'PLAYER_LAST_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION'})

    subset_cols = [c for c in ['PLAYER_ID', 'SEASON', 'TEAM_ID', 'TEAM_ABBREVIATION'] if c in result.columns]
    result = result.drop_duplicates(subset=subset_cols)

    game_cols = [c for c in ['GP', 'G'] if c in result.columns]
    if not game_cols:
        non_index = [c for c in result.columns if c not in INDEX_COLS]
        result = result.dropna(subset=non_index, how='all')
    else:
        result = result.dropna(subset=game_cols, how='all')

    valid_index = [c for c in INDEX_COLS if c in result.columns]
    result.set_index(valid_index, inplace=True)
    return result


def consolidate_player_stats(df):
    # 1. Clean the index/columns immediately
    df = df.reset_index()
    if 'index' in df.columns: df = df.drop(columns=['index'])
    if 'level_0' in df.columns: df = df.drop(columns=['level_0'])

    # 2. Standardize Team Column Names
    df = df.rename(columns={
        'PLAYER_LAST_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION',
        'PLAYER_LAST_TEAM_ID': 'TEAM_ID'
    })
    
    # 3. Aggressive Name & ID Cleaning
    # We do this BEFORE setting the index to ensure every level is clean
    if 'PLAYER_NAME' in df.columns:
        df['PLAYER_NAME'] = df['PLAYER_NAME'].apply(
            lambda x: unidecode(str(x)).strip() if pd.notnull(x) else x
        )
        # Remove double spaces
        df['PLAYER_NAME'] = df['PLAYER_NAME'].str.replace(r'\s+', ' ', regex=True)

    # 4. THE NUCLEAR OPTION: Force every INDEX_COL to a stripped string
    # This kills the "Regular Season " vs "Regular Season" bug forever
    for col in INDEX_COLS:
        if col in df.columns:
            # For IDs, we convert to float -> int -> str to handle '203999.0'
            if 'ID' in col:
                df[col] = df[col].astype(float).fillna(0).astype(int).astype(str).str.strip()
            else:
                df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = "N/A"

    # 5. Handle Traded Players (TOT Logic)
    if 'TEAM_ABBREVIATION' in df.columns:
        tot_mask = df['TEAM_ABBREVIATION'] == 'TOT'
        if tot_mask.any():
            traded_ids = df[tot_mask]['PLAYER_ID'].unique()
            df = pd.concat([df[tot_mask], df[~df['PLAYER_ID'].isin(traded_ids)]])

    # 6. Final Deduplication and Index Lock
    df = df.drop_duplicates(subset=INDEX_COLS)
    
    # Return a sorted, string-indexed dataframe
    return df.set_index(INDEX_COLS).sort_index()
    
    
def drop_redundant_cols(df, keep_pace_estimate=True):
    """Drop NBA-API redundancy: rank twins, sp_work_ duplicates,
    estimated (E_) variants except E_PACE, and fantasy points.
    Safe on both indexed and reset frames — index levels aren't in .columns."""
    drop = [c for c in df.columns
            if c.endswith('_RANK')
            or c.startswith('sp_work_')
            or (c.startswith('E_') and not (keep_pace_estimate and c == 'E_PACE'))
            or 'FANTASY' in c]
    return df.drop(columns=drop, errors='ignore')
    


# =============================================================================
# SECTION 6: MAIN FETCH FUNCTIONS
# =============================================================================

def fetch_nba_stats(
    stat_type,
    seasons,
    player_or_team="Player",
    per_mode="PerGame",
    season_type=None,
    season_types=("Regular Season", "Playoffs"),
    synergy_grouping="offensive",
    source_override=None,
    timeout=120,
    **kwargs
):
    if season_type is not None:
        season_types = (season_type,)

    force_refresh = kwargs.pop("force_refresh", False)

    source = source_override or _route_stat_type(stat_type)
    print(f"DEBUG: Routing {stat_type} to source: {source}")

    dfs = []

    for season in seasons:
        for stype in season_types:
            
            refresh_this = force_refresh or should_refresh(season, stype)
            
            try:
                cache_name = safe_cache_name(
                    "nba_stats",
                    source,
                    stat_type,
                    season,
                    stype,
                    player_or_team,
                    per_mode,
                    synergy_grouping,
                    *[f"{k}_{v}" for k, v in sorted(kwargs.items())]
                )
                
                def fetch_raw():
                    if source == 'pt_shot':
                        return _fetch_pt_shot(season, per_mode, stype, timeout=timeout, **kwargs)
                    elif source == 'pt':
                        return _fetch_pt_stats(stat_type, season, player_or_team, per_mode, stype, timeout=timeout)
                    elif source == 'player':
                        return _fetch_player_stats(stat_type, season, per_mode, stype, timeout=timeout)
                    elif source == 'synergy':
                        return _fetch_synergy(stat_type, season, player_or_team, per_mode, stype, synergy_grouping, timeout=timeout)
                
                df = cached_fetch(cache_name, fetch_raw, force_refresh=refresh_this)
            except Exception as e:
                print(f"DEBUG: API Error for {stat_type} in {season} ({stype}): {e}")
                continue

            if df is None or df.empty:
                print(f"DEBUG: No data for {season} ({stype}). Skipping...")
                continue

            print(f"DEBUG: Fetched {len(df)} rows for {season} ({stype})")

            df = _normalize_names(df)
            df = _apply_prefix(df, stat_type)
            df = drop_redundant_cols(df)          # governs all new + rebuilt data

            drop_cols = ['NICKNAME', 'AGE']
            if source != 'player' or stat_type != 'Base':
                drop_cols.append('GP')
            df = df.drop(columns=drop_cols, errors='ignore')

            df['SEASON'] = season
            df['SEASON_TYPE'] = stype

            dfs.append(df)
            time.sleep(0.8)

    if not dfs:
        print(f"DEBUG: No data collected for {stat_type}.")
        return pd.DataFrame()

    return _consolidate(dfs, stat_type, source)


def build_bucketed_features(
    stat_type,
    seasons,
    buckets,
    prefix,
    param_name,
    fetch_fn,
    season_types=("Regular Season", "Playoffs"),
    exclude_cols=None,
    sleep_time=5
):
    if exclude_cols is None:
        exclude_cols = [
            'SEASON', 'SEASON_TYPE', 'PLAYER_NAME',
            'PLAYER_ID', 'TEAM_ID', 'TEAM_ABBREVIATION'
        ]

    season_type_dfs = []

    for season_type in season_types:
        season_master = None

        for b in buckets:
            print(f"🚀 {stat_type}: {b} ({season_type})")

            try:
                df = fetch_fn(
                    stat_type=stat_type,
                    seasons=seasons,
                    season_type=season_type,
                    **{param_name: b}
                )
            except Exception as e:
                print(f"❌ Error fetching {b} ({season_type}): {e}")
                continue

            if df is None or df.empty:
                print(f"⚠️ No data for {b} ({season_type})")
                continue

            df = apply_bucket_suffix(df, b, prefix, exclude_cols)
            df = df.drop(columns=['G'], errors='ignore')
            df = consolidate_player_stats(df)

            if season_master is None:
                season_master = df
            else:
                season_master = season_master.join(df, how='outer')

            print(f"✅ Added {b} ({season_type})")
            time.sleep(sleep_time)

        if season_master is not None:
            season_type_dfs.append(season_master)

    if not season_type_dfs:
        return pd.DataFrame()

    master_df = pd.concat(season_type_dfs).sort_index()

    vol_cols = [
        c for c in master_df.columns
        if any(x in c for x in ['_FGA', '_FGM', '_G'])
    ]
    master_df[vol_cols] = master_df[vol_cols].fillna(0)

    return master_df
    
    
def fetch_nba_defensive_dashboard(
    seasons: list,
    per_mode: str = 'PerGame',
    season_type: str = 'Regular Season',
    timeout: int = 120,
    force_refresh: bool = False
):
    from nba_api.stats.endpoints import leaguedashptdefend

    dfs = []

    for season in seasons:
        print(f"🛡️ Fetching Defensive Dashboard for {season} ({season_type})...")

        try:
            cache_name = safe_cache_name(
                "defensive_dashboard",
                season,
                season_type,
                per_mode
            )
            
            refresh_this = force_refresh or should_refresh(season, season_type)

            def fetch_raw():
                return leaguedashptdefend.LeagueDashPtDefend(
                    defense_category='Overall',
                    per_mode_simple=per_mode,
                    season=season,
                    season_type_all_star=season_type,
                    timeout=timeout
                ).get_data_frames()[0]

            df = cached_fetch(
                cache_name,
                fetch_raw,
                force_refresh=refresh_this
            )

            df['SEASON'] = season
            df['SEASON_TYPE'] = season_type

            df.rename(columns={
                'CLOSE_DEF_PERSON_ID': 'PLAYER_ID',
                'PLAYER_LAST_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION',
                'PLAYER_LAST_TEAM_ID': 'TEAM_ID'
            }, inplace=True)

            dfs.append(df)

        except Exception as e:
            print(f"Error on {season}: {e}")

    if not dfs:
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)

    full_df.drop_duplicates(
        subset=['PLAYER_ID', 'SEASON', 'SEASON_TYPE'],
        inplace=True
    )

    full_df.dropna(subset=['PLAYER_ID', 'PLAYER_NAME'], inplace=True)

    if 'PCT_PLUSMINUS' in full_df.columns:
        full_df['PCT_PLUSMINUS'] = full_df['PCT_PLUSMINUS'] * 100

    valid_index = [c for c in INDEX_COLS if c in full_df.columns]
    return full_df.set_index(valid_index).sort_index()
    
    
    
def fetch_nba_defensive_breakdowns(
    seasons, per_mode='PerGame', season_type='Regular Season',
    categories=None, timeout=120, force_refresh=False,
):
    """Loop LeagueDashPtDefend over shot-type/distance categories and merge
    into one player-indexed frame. Each category's stat columns are namespaced
    by prefix (DEF2_, DEF3_, ...) so nothing collides. 'Overall' is excluded —
    you already have it as D_* / PCT_PLUSMINUS."""
    from nba_api.stats.endpoints import leaguedashptdefend
    config = categories or DEF_BREAKDOWN_CONFIG
    season_frames = []

    for season in seasons:
        cat_frames = []
        for category, prefix in config.items():
            print(f"🛡️ Defense [{category}] — {season} ({season_type})")
            cache_name = safe_cache_name("def_breakdown", category, season, season_type, per_mode)

            def fetch_raw(category=category, season=season):
                return leaguedashptdefend.LeagueDashPtDefend(
                    defense_category=category, per_mode_simple=per_mode,
                    season=season, season_type_all_star=season_type, timeout=timeout,
                ).get_data_frames()[0]

            try:
                df = cached_fetch(cache_name, fetch_raw,
                                  force_refresh=force_refresh or should_refresh(season, season_type))
            except Exception as e:
                print(f"❌ {category} ({season}) failed: {e}")
                continue
            if df is None or df.empty:
                print(f"⚠️ No data for {category} ({season})")
                continue

            df = df.rename(columns={
                'CLOSE_DEF_PERSON_ID': 'PLAYER_ID',
                'PLAYER_LAST_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION',
                'PLAYER_LAST_TEAM_ID': 'TEAM_ID',
            })
            df['SEASON'] = season
            df['SEASON_TYPE'] = season_type
            df = df.drop(columns=_DEF_DROP, errors='ignore')

            for c in df.columns:                      # match your *100 convention
                if 'PCT_PLUSMINUS' in c:
                    df[c] = df[c] * 100

            df = df.rename(columns={c: f"{prefix}{c}"
                                    for c in df.columns if c not in _DEF_KEEP_KEYS})
            cat_frames.append(consolidate_player_stats(df))

        if not cat_frames:
            continue
        season_df = cat_frames[0]
        for extra in cat_frames[1:]:
            season_df = season_df.join(extra, how='outer')
        season_frames.append(season_df)

    return pd.concat(season_frames).sort_index() if season_frames else pd.DataFrame()
    
    
    
def fetch_nba_shot_locations(
    seasons: List[str],
    per_mode: str = 'PerGame',
    season_type: str = 'Regular Season',
    timeout: int = 120,
    force_refresh: bool = False
) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguedashplayershotlocations
    import pandas as pd

    dfs = []

    for season in seasons:
        print(f"Fetching Shot Locations for {season} ({season_type})...")

        try:
            cache_name = safe_cache_name(
                "shot_locations",
                season,
                season_type,
                per_mode
            )

            refresh_this = force_refresh or should_refresh(season, season_type)

            def fetch_raw():
                raw_data = leaguedashplayershotlocations.LeagueDashPlayerShotLocations(
                    distance_range='By Zone',
                    season=season,
                    per_mode_detailed=per_mode,
                    season_type_all_star=season_type,
                    timeout=timeout
                )
                return raw_data.get_data_frames()[0]

            df = cached_fetch(
                cache_name,
                fetch_raw,
                force_refresh=refresh_this
            )

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    f"{col[0]}_{col[1]}".strip('_') if col[0] != '' else col[1]
                    for col in df.columns.values
                ]

            df['SEASON'] = season
            df['SEASON_TYPE'] = season_type

            dfs.append(df)

        except Exception as e:
            print(f"Error fetching {season}: {e}")

    if not dfs:
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)

    clean_cols = []
    for col in full_df.columns:
        c = str(col).replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        while '__' in c:
            c = c.replace('__', '_')
        clean_cols.append(c.strip('_'))

    full_df.columns = clean_cols

    return consolidate_player_stats(full_df)
    
    

def fetch_nba_hustle_stats(
    seasons,
    season_type="Regular Season",
    per_mode="PerGame",
    timeout=120,
    force_refresh=False,
):
    chunks = []

    for season in seasons:
        cache_name = safe_cache_name(
            "hustle_stats",
            season,
            season_type,
            per_mode
        )

        def fetch_raw():
            return LeagueHustleStatsPlayer(
                season=season,
                season_type_all_star=season_type,
                per_mode_time=per_mode,
                timeout=timeout
            ).get_data_frames()[0]

        df = cached_fetch(
            cache_name,
            fetch_raw,
            force_refresh=force_refresh or should_refresh(season, season_type)
        )

        if df is None or df.empty:
            print(f"⚠️ No hustle data for {season} {season_type}")
            continue

        df = _normalize_names(df)

        df["SEASON"] = season
        df["SEASON_TYPE"] = season_type
        df = df.drop(columns=["AGE", "G", "MIN"], errors="ignore")

        chunks.append(df)

    if not chunks:
        return pd.DataFrame()

    return _consolidate(chunks, "Hustle", source="hustle")
    

def get_player_registry(season):
    print(f"🚀 Fetching {season}...")
    df = LeagueDashPlayerBioStats(
        season=season, headers=minimal_headers, timeout=60
    ).get_data_frames()[0]
    df['SEASON'] = season
    df = df.rename(columns={
        'PLAYER_HEIGHT_INCHES': 'HT_INCHES',
        'DRAFT_YEAR': 'DRAFT_YR',
        'DRAFT_NUMBER': 'DRAFT_POSITION'
    })
    return df


def get_multi_season_registry(seasons_list):
    all_registries = []
    for season in seasons_list:
        try:
            all_registries.append(get_player_registry(season))
            print(f"✅ {season} Done.")
            time.sleep(5)
        except Exception as e:
            print(f"❌ {season} Failed: {e}")

    if not all_registries:
        return pd.DataFrame()

    master_df = pd.concat(all_registries, ignore_index=True)
    return master_df.set_index(INDEX_COLS).sort_index()


# =============================================================================
# SECTION 7: MASTER DF MANAGEMENT
# =============================================================================

def build_master_df(seasons, stat_types, season_types=('Regular Season', 'Playoffs')):
    masters = []
    
    for season_type in season_types:
        print(f"\n📅 Building {season_type} master...")
        
        base_df = fetch_nba_stats(stat_types[0], seasons, season_type=season_type)
        
        if base_df is None or base_df.empty:
            print(f"⚠️ No base data for {season_type}. Skipping.")
            continue
        
        for stat in stat_types[1:]:
            new_df = fetch_nba_stats(stat, seasons, season_type=season_type)
            base_df = update_master_df(base_df, new_df, suffix=stat.lower())
        
        masters.append(base_df)
    
    if not masters:
        return pd.DataFrame()
    
    return pd.concat(masters).sort_index()
    
    

def update_master_df(master_df: pd.DataFrame, new_df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    if new_df is None or new_df.empty: return master_df

    # 1. THE NUCLEAR ALIGNMENT (The piece you should add)
    # This forces both DFs to have clean, string-based, unidecoded indices
    # so that 'Nikola Jokić' (int ID) and 'Nikola Jokic' (string ID) become the same row.
    for df_to_clean in [master_df, new_df]:
        # Reset and clean the name specifically
        temp = df_to_clean.reset_index()
        if 'PLAYER_NAME' in temp.columns:
            temp['PLAYER_NAME'] = temp['PLAYER_NAME'].apply(
                lambda x: unidecode(str(x)).strip() if pd.notnull(x) else x
            )
        
        # Force all Big 5 Index columns to be stripped strings
        for col in INDEX_COLS:
            if col in temp.columns:
                temp[col] = temp[col].astype(str).str.strip()
        
        # Re-assign the cleaned version back to the variable
        if df_to_clean is master_df:
            m_df = temp
        else:
            n_df = temp

    # 2. Define join keys
    join_keys = [c for c in INDEX_COLS if c in m_df.columns and c in n_df.columns]

    # 3. Filter columns (Keep the logic you already had)
    metadata = {'PLAYER_NAME', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'NICKNAME',
                'AGE', 'GP', 'G', 'MIN', 'W', 'L', 'W_PCT'}
    keep_cols = join_keys + [c for c in n_df.columns
                            if c not in join_keys
                            and c not in metadata          # 🔒 never re-merge metadata
                            and c not in m_df.columns      # 🔥 prevent duplicate stat columns
                        ]

    # 4. Perform the merge
    updated = m_df.merge(n_df[keep_cols], on=join_keys, how='left', suffixes=('', f'_{suffix}'))

    # 5. --- DEDUPLICATION LOGIC ---
    # Prioritize 'TOT' (Total) rows for traded players so they only have 1 row per season
    if 'TEAM_ABBREVIATION' in updated.columns:
        # Create a helper column to sort: 'TOT' rows come first
        updated['_is_tot'] = updated['TEAM_ABBREVIATION'] == 'TOT'
        
        # Sort by Player/Season and put 'TOT' rows at the top
        updated = updated.sort_values(
                        ['PLAYER_ID', 'SEASON', 'SEASON_TYPE', '_is_tot'],
                        ascending=[True, True, True, False]
                    )
        
        # Drop duplicates, keeping the first one (which will be 'TOT' if it exists)
        updated = updated.drop_duplicates(
            subset=['PLAYER_ID', 'SEASON', 'SEASON_TYPE'],
            keep='first'
        )
        
        # Remove helper column
        updated = updated.drop(columns=['_is_tot'])

    # 6. Re-apply the Big 5 Index
    return updated.set_index(INDEX_COLS).sort_index()


def check_master_integrity(df):
    if df is None or df.empty:
        print("❌ CRITICAL: Master DF is None or Empty!")
        return

    print(f"📊 Integrity Check for df_master:")
    print(f"  - Shape: {df.shape}")
    print(f"  - Total Players: {len(df.index.get_level_values('PLAYER_ID').unique())}")
    print(f"  - Seasons Present: {df.index.get_level_values('SEASON').unique().tolist()}")

    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        print(f"  - ⚠️ WARNING: Found {duplicates} duplicate index entries!")
    else:
        print("  - ✅ No duplicate indices found.")

    all_nan_cols = df.columns[df.isna().all()].tolist()
    if all_nan_cols:
        print(f"  - ⚠️ WARNING: {len(all_nan_cols)} columns are completely empty (all NaN).")
    else:
        print("  - ✅ All columns contain at least some data.")


def polish_master_df(df: pd.DataFrame) -> pd.DataFrame:
    df_polished = df.copy()
    if 'GP' in df_polished.columns:
        df_polished['GP'] = df_polished['GP'].fillna(0).astype(int)
    if 'PLAYER_POSITION' in df_polished.columns:
        pos_col = df_polished.pop('PLAYER_POSITION')
        df_polished.insert(1, 'PLAYER_POSITION', pos_col)
    return df_polished
    
    
    
def ingest_stat(
    df_master,
    fetch_fn,
    seasons=['2025-26'],
    *,
    suffix=None,
    keep_cols=None,
    season_types=("Regular Season", "Playoffs"),
    **kwargs
):
    chunks = []

    for season_type in season_types:
        df = fetch_fn(
            seasons=seasons,
            season_type=season_type,
            **kwargs
        )

        if df is None or df.empty:
            continue

        if not isinstance(df.index, pd.MultiIndex):
            df = consolidate_player_stats(df)

        chunks.append(df)

    if not chunks:
        print("⚠️ No data returned. Skipping.")
        return df_master

    combined = pd.concat(chunks)

    if keep_cols is not None:
        available = [c for c in keep_cols if c in combined.columns]
        combined = combined[available]

    combined = (
        combined
        .reset_index()
        .drop_duplicates(subset=INDEX_COLS, keep='first')
        .set_index(INDEX_COLS)
        .sort_index()
    )

    suffix = suffix or fetch_fn.__name__.replace("fetch_nba_", "").replace("_dashboard", "")

    return update_master_df(df_master, combined, suffix=suffix)
    
    
def ingest_nba_stat(df_master, stat_type, seasons, suffix=None, season_types=("Regular Season", "Playoffs"), keep_cols=None, **kwargs):
    return ingest_stat(
        df_master,
        lambda seasons, season_type, **kw: fetch_nba_stats(
            stat_type,
            seasons,
            season_type=season_type,
            **kw
        ),
        seasons=seasons,
        suffix=suffix or stat_type.lower(),
        season_types=season_types,
        keep_cols=keep_cols,
        **kwargs
    )



def sync_and_patch_stat(df_master, new_data_df, index_cols=None, add_new_cols=True):
    if index_cols is None:
        index_cols = INDEX_COLS

    incoming_stat_cols = set(new_data_df.columns)

    def normalize_df(df):
        df = df.reset_index()

        if "PLAYER_NAME" in df.columns:
            df["PLAYER_NAME"] = df["PLAYER_NAME"].apply(lambda x: unidecode(str(x)).strip())

        for col in index_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        return (
            df
            .drop_duplicates(subset=index_cols, keep="first")
            .set_index(index_cols)
            .sort_index()
        )

    master = normalize_df(df_master)
    new = normalize_df(new_data_df)

    if add_new_cols:
        for col in incoming_stat_cols:
            if col in new.columns and col not in master.columns:
                master[col] = pd.NA

    overlap_cols = [
        c for c in incoming_stat_cols
        if c in new.columns and c in master.columns
    ]

    master.update(new[overlap_cols])

    return master.sort_index()
    
    

REFRESH_CUSTOM_CONFIG = [
    {
        "name": "Shot Locations",
        "fetch_fn": fetch_nba_shot_locations,
        "suffix": "shoot",
        "keep_cols": shoot_cols,
    },
    {
        "name": "Defensive Dashboard",
        "fetch_fn": fetch_nba_defensive_dashboard,
        "suffix": "def",
        "keep_cols": def_cols,
    },
    {
        "name": "Defensive Breakdowns",
        "fetch_fn": fetch_nba_defensive_breakdowns,
        "suffix": None,
        "keep_cols": None,
    },
    {
    "name": "Hustle Stats",
    "fetch_fn": fetch_nba_hustle_stats,
    "suffix": "hustle",
    "keep_cols": None,
    }   
]


def get_active_slice(df, season, season_type):
    """
    Return a copy of one season/season_type slice from df_master.
    """
    return df.xs(
        (season, season_type),
        level=("SEASON", "SEASON_TYPE")
    ).copy()
    
    

def refresh_and_patch_nba_stat(
    df_master,
    stat_type,
    active_season,
    active_season_type,
    keep_cols=None,
    source_override=None,
    force_refresh=True,
    timeout=180,
    per_mode="PerGame",
):
    fresh_df = fetch_nba_stats(
        stat_type,
        [active_season],
        season_type=active_season_type,
        per_mode=per_mode,
        source_override=source_override,
        force_refresh=force_refresh,
        timeout=timeout
    )

    if fresh_df is None or fresh_df.empty:
        print(f"⚠️ No fresh data returned for {stat_type}.")
        return df_master

    if keep_cols is not None:
        available = [c for c in keep_cols if c in fresh_df.columns]
        missing = [c for c in keep_cols if c not in fresh_df.columns]

        if missing:
            print(f"⚠️ Missing requested columns for {stat_type}: {missing}")

        fresh_df = fresh_df[available]

    danger_cols = ["FGM", "FGA", "GP", "PTS", "REB", "AST"]
    raw_danger = [c for c in danger_cols if c in fresh_df.columns]

    if stat_type != "Base" and raw_danger:
        print(f"🚨 {stat_type} has raw danger cols before protection: {raw_danger}")

    if stat_type != "Base":
        protected = [c for c in CORE_BOX_SCORE_COLS if c in fresh_df.columns]

        if protected:
            print(f"🛡️ Dropping protected core columns from {stat_type}: {protected}")
            fresh_df = fresh_df.drop(columns=protected, errors="ignore")

    return sync_and_patch_stat(df_master, fresh_df)
    
    
    
def check_active_refresh_change(before_df, after_df, check_cols=None, verbose=True):
    """
    Compare aggregate totals before/after refreshing an active season slice.

    Useful for confirming that new data was actually added or updated.
    """
    if check_cols is None:
        check_cols = ["GP", "MIN", "PTS", "FGA", "FGM", "FG3A", "FTA", "AST", "REB", "TOV"]

    check_cols = [
        c for c in check_cols
        if c in before_df.columns and c in after_df.columns
    ]

    if not check_cols:
        raise ValueError("None of the requested check_cols exist in both dataframes.")

    before_summary = before_df[check_cols].sum(numeric_only=True)
    after_summary = after_df[check_cols].sum(numeric_only=True)

    summary = pd.DataFrame({
        "before": before_summary,
        "after": after_summary,
    })

    summary["diff"] = summary["after"] - summary["before"]
    summary = summary.sort_values("diff", ascending=False)

    changed = summary["diff"].abs().sum() != 0

    if verbose:
        print("📊 Active slice refresh diff:")
        print(summary)

        if changed:
            print("✅ Refresh changed active slice data.")
        else:
            print("⚠️ No detectable change in checked columns.")


    return summary
    

def refresh_and_patch_custom_stat(
    df_master,
    fetch_fn,
    active_season,
    active_season_type,
    keep_cols=None,
    force_refresh=True,
    per_mode="PerGame",
    timeout=180,
):
    fresh_df = fetch_fn(
        seasons=[active_season],
        season_type=active_season_type,
        per_mode=per_mode,
        force_refresh=force_refresh,
        timeout=timeout
    )

    if fresh_df is None or fresh_df.empty:
        print("⚠️ No fresh custom data returned.")
        return df_master

    if keep_cols is not None:
        available = [c for c in keep_cols if c in fresh_df.columns]
        missing = [c for c in keep_cols if c not in fresh_df.columns]

        if missing:
            print(f"⚠️ Missing requested custom columns: {missing}")

        fresh_df = fresh_df[available]

    danger_cols = ["FGM", "FGA", "GP", "PTS", "REB", "AST"]
    raw_danger = [c for c in danger_cols if c in fresh_df.columns]

    if raw_danger:
        print(f"🚨 Custom fetcher has raw danger cols before protection: {raw_danger}")

    protected = [c for c in CORE_BOX_SCORE_COLS if c in fresh_df.columns]

    if protected:
        print(f"🛡️ Dropping protected core columns from custom fetcher: {protected}")
        fresh_df = fresh_df.drop(columns=protected, errors="ignore")

    return sync_and_patch_stat(df_master, fresh_df)
    
    
def refresh_active_season_data(
    df_master,
    active_season="2025-26",
    active_season_type="Playoffs",
    full_stats=None,
    curated_config=None,
    custom_config=None,
    force_refresh=True,
    per_mode="PerGame",
    timeout=180,
):
    full_stats = full_stats or FULL_NBA_STATS
    curated_config = curated_config or REFRESH_NBA_STATS_CONFIG
    custom_config = custom_config or REFRESH_CUSTOM_CONFIG

    before_cols = set(df_master.columns)

    for stat_type in full_stats:
        print(f"\n🔄 Refreshing {stat_type} — {active_season} {active_season_type}")

        df_master = refresh_and_patch_nba_stat(
            df_master,
            stat_type,
            active_season,
            active_season_type,
            force_refresh=force_refresh,
            timeout=timeout,
            per_mode=per_mode,
        )

    for stat_type, cfg in curated_config.items():
        actual_stat_type = cfg.get("stat_type", stat_type)
        source_override = cfg.get("source_override")
    
        print(f"\n🔄 Refreshing {stat_type} — {active_season} {active_season_type}")
    
        df_master = refresh_and_patch_nba_stat(
            df_master,
            actual_stat_type,
            active_season,
            active_season_type,
            keep_cols=cfg.get("keep_cols"),
            source_override=source_override,
            force_refresh=force_refresh,
            timeout=timeout,
            per_mode=per_mode,
        )

    for cfg in custom_config:
        print(f"\n🔄 Refreshing {cfg['name']} — {active_season} {active_season_type}")

        df_master = refresh_and_patch_custom_stat(
            df_master,
            cfg["fetch_fn"],
            active_season,
            active_season_type,
            keep_cols=cfg.get("keep_cols"),
            force_refresh=force_refresh,
            timeout=timeout,
            per_mode=per_mode,
        )

    new_cols = sorted(set(df_master.columns) - before_cols)

    print(f"\n🧱 New columns added: {len(new_cols)}")
    print(new_cols)

    return df_master, new_cols
    
    
    
def compare_active_slice(before_df, after_df, cols=None, verbose=True):
    """
    Compare row-level changes between before/after active season slices.

    Returns:
    - before_changed: before values for changed rows
    - after_changed: after values for changed rows
    - change_summary: dict with changed/new/dropped row counts
    """
    if cols is None:
        cols = ["GP", "MIN", "PTS", "FGA", "FGM", "AST", "REB", "USG_PCT"]

    cols = [
        c for c in cols
        if c in before_df.columns and c in after_df.columns
    ]

    if not cols:
        raise ValueError("None of the requested cols exist in both dataframes.")

    aligned_before, aligned_after = before_df[cols].align(
        after_df[cols],
        join="outer",
        axis=0
    )

    changed = (
        (aligned_after != aligned_before)
        & ~(aligned_after.isna() & aligned_before.isna())
    )

    changed_rows = changed.any(axis=1)

    new_rows = aligned_after.index.difference(aligned_before.index)
    dropped_rows = aligned_before.index.difference(aligned_after.index)

    change_summary = {
        "changed_rows": int(changed_rows.sum()),
        "new_rows": int(len(new_rows)),
        "dropped_rows": int(len(dropped_rows)),
    }

    if verbose:
        print(f"Changed rows: {change_summary['changed_rows']}")
        print(f"New rows: {change_summary['new_rows']}")
        print(f"Dropped rows: {change_summary['dropped_rows']}")

    return aligned_before.loc[changed_rows], aligned_after.loc[changed_rows], change_summary


# =============================================================================
# SECTION 8: VISUALIZATION
# =============================================================================

def scatter_nba_stats(
    df, x_stat, y_stat, season, season_type='Regular Season',
    color_stat=None, size_stat=None, custom_query=None,
    min_pts=None, min_min=None, min_gp=None,
    cmap='plasma', figsize=(15, 10), title=None
):
    color_stat = color_stat or y_stat
    plot_df = df.reset_index().copy()

    # 1. Temporal Filters: Year AND Type
    filters = [f"SEASON == '{season}'"]
    
    # Catch both standard naming and potential case sensitivity
    if 'SEASON_TYPE' in plot_df.columns:
        filters.append(f"SEASON_TYPE == '{season_type}'")
    elif 'season_type' in plot_df.columns:
        filters.append(f"season_type == '{season_type}'")

    # 2. Volume Filters
    if min_pts: filters.append(f"PTS >= {min_pts}")
    if min_min: filters.append(f"MIN >= {min_min}")
    if min_gp:  filters.append(f"GP >= {min_gp}")
    
    # Apply filters
    plot_df = plot_df.query(" & ".join(filters)).copy()

    if custom_query:
        try:
            plot_df = plot_df.query(custom_query).copy()
        except Exception as e:
            print(f"Query Error: {e}")

    # 3. Updated Deduplication: Include SEASON_TYPE in the subset
    # This prevents accidentally dropping a playoff row if a reg season row exists
    plot_df = plot_df.drop_duplicates(subset=['PLAYER_NAME', 'SEASON', 'SEASON_TYPE', 'PLAYER_ID'])
    
    if plot_df.empty:
        print(f"Warning: Query returned an empty DataFrame for {season} {season_type}.")
        return

    # ... (Keep the rest of your scatter logic for 's', 'fig', 'ax' the same)
    s = 100
    if size_stat and size_stat in plot_df.columns:
        size_values = plot_df[size_stat].fillna(0)
        s = ((size_values - size_values.min()) / (size_values.max() - size_values.min() + 1e-6)) * 400 + 50

    fig, ax = plt.subplots(figsize=figsize, facecolor='#f0f0f0')
    scatter = ax.scatter(
        plot_df[x_stat], plot_df[y_stat],
        c=plot_df[color_stat], s=s, cmap=cmap,
        edgecolors='white', alpha=0.7, zorder=3
    )

    for i, name in enumerate(plot_df['PLAYER_NAME']):
        ax.annotate(name, (plot_df[x_stat].iloc[i], plot_df[y_stat].iloc[i]),
                    xytext=(0, 7), textcoords='offset points',
                    fontsize=8, ha='center', alpha=0.8, clip_on=True)

    # 4. Dynamic Title
    full_title = title or f'{x_stat} vs {y_stat} ({season} {season_type})'
    ax.set_title(full_title, fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlabel(x_stat.replace('_', ' '), fontsize=12, fontweight='bold')
    ax.set_ylabel(y_stat.replace('_', ' '), fontsize=12, fontweight='bold')
    plt.colorbar(scatter, label=color_stat)
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.show()


def plot_top_players(
    df: pd.DataFrame, sort_column: str, query_condition: str,
    *, top_n: int = 60, title: str = "Top Players",
    palette: str = "flare", show: bool = True
) -> plt.Figure:
    required_cols = {"PLAYER_NAME", sort_column}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    filtered = (df.query(query_condition)
                  .sort_values(by=sort_column, ascending=False)
                  .head(top_n).reset_index(drop=True))

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(24, 12), dpi=80)
    sns.barplot(data=filtered, x="PLAYER_NAME", y=sort_column, palette=palette, ax=ax)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Player", fontsize=12)
    ax.set_ylabel(sort_column, fontsize=12)
    ax.tick_params(axis="x", rotation=35)

    for i, v in enumerate(filtered[sort_column].to_list()):
        try:
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", rotation=45)
        except Exception:
            pass

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_interactive_barh(
    df: pd.DataFrame, *, x_col: str, y_col: str,
    season: Optional[str] = None, season_type = 'Regular Season',query_condition: Optional[str] = None,
    sort_by: Optional[str] = None, ascending: bool = True,
    title: str = "Custom Bar Plot", ylim: Optional[Tuple[float, float]] = None,
    cmap: str = "coolwarm", height: int = 500, width: int = 1000,
) -> Any:
    filtered = df.reset_index().copy()

    if season is not None:
        filtered = filtered[filtered["SEASON"] == season]
        
    # 2. THE MISSING FILTER: Season Type (Reg vs Playoffs)
    if "SEASON_TYPE" in filtered.columns:
        filtered = filtered[filtered["SEASON_TYPE"] == season_type]
    elif "season_type" in filtered.columns: # Catch case sensitivity
        filtered = filtered[filtered["season_type"] == season_type]
        
    if query_condition:
        filtered = filtered.query(query_condition)

    filtered = filtered.sort_values(by=sort_by or x_col, ascending=ascending)
    filtered[x_col] = pd.to_numeric(filtered[x_col], errors='coerce').astype(float)
    filtered[y_col] = filtered[y_col].astype(str)
    filtered = filtered.dropna(subset=[x_col])

    plot = filtered.hvplot.barh(
        x=y_col, y=x_col, height=height, width=width,
        color=x_col, cmap=cmap, colorbar=True, title=title,
        xlabel=y_col, ylabel=x_col,
    )
    if ylim:
        plot = plot.opts(ylim=ylim)
    return plot

    
def analyze_player_changes(
    df: pd.DataFrame,
    stat_column: str,
    query_conditions=None,
    *,
    compare_col: str = "SEASON_TYPE",
    base: str = "Regular Season",
    comp: str = "Playoffs",
    season_filter: str = "2025-26",
    n: int = 25,
    palette: str = "coolwarm",
    show: bool = True,
):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from unidecode import unidecode

    # 1. THE "BRIDGE" BUILDER (Identity Resolution)
    # This part was missing and causing the "Vanishing Column" error
    subset = df.reset_index().copy()
    
    # Force IDs and Labels to clean strings to ensure the 'bridge' exists
    sync_cols = ['PLAYER_ID', 'SEASON', 'SEASON_TYPE', 'TEAM_ID']
    for col in sync_cols:
        if col in subset.columns:
            subset[col] = subset[col].astype(str).str.strip()
    
    # Normalize names to prevent "Jokic" vs "Jokić" issues
    if 'PLAYER_NAME' in subset.columns:
        subset['PLAYER_NAME'] = subset['PLAYER_NAME'].apply(lambda x: unidecode(str(x)).strip())

    # 2. APPLY SEASON FILTER
    if season_filter:
        subset = subset[subset['SEASON'] == str(season_filter)]

    # 3. QUALIFIED PLAYERS FILTER
    if query_conditions:
        # Standardize query format
        if isinstance(query_conditions, list):
            q_str = " and ".join([f"({c})" for c in query_conditions])
        else:
            q_str = query_conditions
            
        # Get IDs of players who qualify in the 'comp' group
        comp_rows = subset[subset[compare_col] == str(comp)]
        if comp_rows.empty:
            print(f"❌ Error: No '{comp}' rows found for {season_filter}")
            return pd.DataFrame(), None
            
        qualifier_ids = comp_rows.query(q_str)['PLAYER_ID'].unique()
        subset = subset[subset['PLAYER_ID'].isin(qualifier_ids)]

    # 4. THE PIVOT
    pivoted = subset.pivot_table(index='PLAYER_NAME', columns=compare_col, values=stat_column, aggfunc='mean')

    # 5. SYMMETRY CHECK
    if base not in pivoted.columns or comp not in pivoted.columns:
        available = pivoted.columns.tolist()
        print(f"❌ Error: '{base}' or '{comp}' missing from pivot. Available: {available}")
        return pd.DataFrame(), None

    # 6. CALCULATION
    pivoted = pivoted.dropna(subset=[base, comp])
    pivoted['DIFFERENCE'] = pivoted[comp] - pivoted[base]
    diff_df = pivoted.sort_values('DIFFERENCE', ascending=False).reset_index()

    # 7. PLOTTING
    if diff_df.empty:
        print("No players found in both groups.")
        return pd.DataFrame(), None

    plot_df = pd.concat([diff_df.head(n), diff_df.tail(n)]).drop_duplicates()
    plot_df = plot_df.sort_values('DIFFERENCE', ascending=False)

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.4)))
    
    # Dynamic Colors
    if palette in ["coolwarm", "coolwarm_r"]:
        pos, neg = ('#4C72B0', '#C44E52')
        if palette == "coolwarm_r": pos, neg = ('#C44E52', '#4C72B0')
        colors = [pos if x > 0 else neg for x in plot_df["DIFFERENCE"]]
    else:
        colors = sns.color_palette(palette, n_colors=len(plot_df))

    sns.barplot(data=plot_df, x="DIFFERENCE", y="PLAYER_NAME", palette=colors, ax=ax, hue="PLAYER_NAME", legend=False)
    ax.set_title(f"Change in {stat_column}: {base} vs {comp} ({season_filter})")
    plt.tight_layout()
    
    if show: plt.show()
    return diff_df, fig
    
    

def plot_percentile_violin(
    df,
    stat,
    season,
    season_type="Regular Season",
    highlight_players=None,
    min_label_gap=0.04,
    label_x_offset=0.08,
):
    """
    Violin plot for a stat distribution with optional highlighted players.
    
    min_label_gap:
        Minimum vertical gap between label positions as a fraction of stat range.
        Increase if labels still overlap.
    """

    data = (
        df.xs((season, season_type), level=("SEASON", "SEASON_TYPE"))
        .reset_index()
    )

    data = data.dropna(subset=[stat]).copy()

    if data.empty:
        print(f"⚠️ No data available for {stat} — {season} {season_type}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.violinplot(
        y=data[stat],
        inner="quartile",
        color="skyblue",
        ax=ax
    )

    # ------------------------------------------------------------
    # Highlight players with non-overlapping labels
    # ------------------------------------------------------------
    if highlight_players:
        stat_min = data[stat].min()
        stat_max = data[stat].max()
        stat_range = stat_max - stat_min

        if stat_range == 0:
            stat_range = 1

        min_gap = stat_range * min_label_gap

        highlight_rows = []

        for p in highlight_players:
            player_row = data.loc[data["PLAYER_NAME"] == p]

            if player_row.empty:
                print(f"⚠️ {p} not found in {season} {season_type}")
                continue

            val = player_row[stat].iloc[0]
            pct = data[stat].rank(pct=True).loc[player_row.index[0]] * 100

            highlight_rows.append({
                "PLAYER_NAME": p,
                "value": val,
                "percentile": pct,
            })

        highlight_rows = sorted(highlight_rows, key=lambda x: x["value"])

        label_positions = []

        for row in highlight_rows:
            y = row["value"]

            if label_positions and y - label_positions[-1] < min_gap:
                y = label_positions[-1] + min_gap

            label_positions.append(y)

            ax.plot(
                0,
                row["value"],
                "ro",
                zorder=5
            )

            ax.annotate(
                f"{row['PLAYER_NAME']} ({row['percentile']:.0f}th pct)",
                xy=(0, row["value"]),
                xytext=(label_x_offset, y),
                textcoords="data",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
                arrowprops=dict(
                    arrowstyle="-",
                    lw=0.8,
                    color="gray"
                )
            )

    ax.set_title(
        f"{stat} Distribution — {season} {season_type} | Per Game",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_ylabel(stat.replace("_", " "))
    ax.set_xlabel("")
    ax.set_xticks([])

    # Give labels room on right side
    ax.set_xlim(-0.5, 1.0)

    plt.tight_layout()
    plt.show()
    
    
    
def plot_shot_mix(
    df,
    player_name,
    season,
    season_type="Regular Season",
    team=None,
    attempt_cols=None,
    min_pct_label=3,
    other_threshold=2,
    figsize=(12, 2.5),
    return_data=False
):
    """
    Horizontal stacked bar showing a player's selected shot attempt mix.

    Use attempt_cols for either:
    - shot location FGA columns
    - play-type FGA columns

    other_threshold:
        Categories below this percent are grouped into Other.
    min_pct_label:
        Only show text labels inside segments above this percent.
    """

    data = (
        df.xs((season, season_type), level=("SEASON", "SEASON_TYPE"))
        .reset_index()
        .copy()
    )

    player_data = data[data["PLAYER_NAME"] == player_name]

    if team is not None:
        player_data = player_data[player_data["TEAM_ABBREVIATION"] == team]

    if player_data.empty:
        print(f"⚠️ No data found for {player_name} in {season} {season_type}")
        return None

    if attempt_cols is None:
        attempt_cols = [
            c for c in df.columns
            if c.endswith("_FGA")
            and any(c.startswith(p) for p in PREFIXES.values())
        ]

    attempt_cols = [c for c in attempt_cols if c in player_data.columns]

    if not attempt_cols:
        print("⚠️ No attempt columns found.")
        return None

    mix = (
        player_data[attempt_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .sum()
        .sort_values(ascending=False)
    )

    mix = mix[mix > 0]

    if mix.empty:
        print("⚠️ No positive attempt mix found.")
        return None

    mix_pct = mix / mix.sum() * 100

    small = mix_pct[mix_pct < other_threshold]
    large = mix_pct[mix_pct >= other_threshold]

    if not small.empty:
        mix_pct = pd.concat([
            large,
            pd.Series({"Other": small.sum()})
        ])

    clean_labels = []
    for c in mix_pct.index:
        if c == "Other":
            clean_labels.append("Other")
        else:
            clean_labels.append(
                c.replace("_FGA", "")
                 .replace("_", " ")
                 .title()
            )

    mix_pct.index = clean_labels

    fig, ax = plt.subplots(figsize=figsize)

    left = 0
    colors = sns.color_palette("rocket", len(mix_pct))

    for (label, val), color in zip(mix_pct.items(), colors):
        ax.barh(
            0,
            val,
            left=left,
            color=color,
            label=f"{label} ({val:.1f}%)"
        )

        if val >= min_pct_label:
            ax.text(
                left + val / 2,
                0,
                f"{val:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold"
            )

        left += val

    ax.set_xlim(0, 100)
    ax.set_xlabel("% of selected attempts")
    ax.set_yticks([])
    ax.set_title(
        f"Shot Mix — {player_name} ({season} {season_type})",
        fontsize=14,
        fontweight="bold"
    )

    ax.legend(
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=8,
        frameon=False
    )

    plt.tight_layout()
    plt.show()

    if return_data:
        return mix_pct
