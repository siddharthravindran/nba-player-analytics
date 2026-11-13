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
import requests
from io import StringIO
from bs4 import BeautifulSoup, Comment
from unidecode import unidecode
import glob

from nba_api.stats.library.http import NBAStatsHTTP
from nba_api.stats.endpoints import (
    LeagueDashPlayerStats,
    LeagueDashPtStats,
    SynergyPlayTypes,
    CommonAllPlayers,
    LeagueDashPlayerPtShot,
    LeagueDashPlayerBioStats,
    LeagueHustleStatsPlayer,
    TeamPlayerOnOffSummary,
)
from nba_api.stats.static import teams as _static_teams

# =============================================================================
# SECTION 1: NBA.com CONSTANTS & CONFIG
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CACHE_DIR = PROJECT_DIR / "cache"
DATA_DIR  = PROJECT_DIR / "data"
FIG_DIR   = PROJECT_DIR / "figures"

for d in [CACHE_DIR, DATA_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# stats.nba.com drops API requests that don't look like they came from the site's
# own front end — not with a 429 or an error page, it just never answers, so the
# connection hangs until the read timeout fires. nba_api 1.10.2 ships most of the
# browser headers in STATS_HEADERS but not these three, so add them to its session
# once here, at import. NBAStatsHTTP.headers is a class attribute and the calls in
# this module never pass headers= of their own, so every endpoint picks these up.
NBA_STATS_HEADERS = {
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}
NBAStatsHTTP.headers.update(NBA_STATS_HEADERS)

SALARY_CAP = {'2015-16':70_000_000,'2016-17':94_143_000,'2017-18':99_093_000,
              '2018-19':101_869_000,'2019-20':109_140_000,'2020-21':109_140_000,
              '2021-22':112_414_000,'2022-23':123_655_000,'2023-24':136_021_000,
              '2024-25':140_588_000,'2025-26':154_647_000,
              '2026-27':164_961_000}

SEASONS = tuple(SALARY_CAP.keys())   # one source of truth — add a cap year, the season list follows

# The season currently in progress. should_refresh() reads these, so the cache is
# bypassed for the live season and reused for everything already final. Bump both
# when a new season tips off, or the pipeline will keep serving last year's cache.
ACTIVE_SEASON = "2026-27"
ACTIVE_SEASON_TYPE = "Regular Season"

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
    'buckets': ["Touch < 2 Seconds", "Touch 2-6 Seconds", "Touch 6+ Seconds"]
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


# ── transpose: RS + PO side-by-side, one row per player-season ──────────────
ID_KEYS     = ['PLAYER_ID', 'SEASON']
DESCRIPTIVE = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'TEAM_ID', 'PLAYER_POSITION']


# =============================================================================
# SECTION 2: BBREFERENCE CONSTANTS & CONFIG
# =============================================================================


HDR = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.basketball-reference.com/",
}

CACHE = CACHE_DIR / "bbref_salary_cache"
for d in [CACHE]:
    d.mkdir(parents=True, exist_ok=True)
BASE = "https://www.basketball-reference.com"


# =============================================================================
# SECTION 3: ROUTING
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
# SECTION 4: API FETCH HELPERS
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
# SECTION 5: CACHE & REFRESH HELPERS
# =============================================================================

def safe_cache_name(*parts):
    return "_".join(
        str(p).replace(" ", "_").replace("/", "_").replace("+", "P")
        for p in parts
    ).lower()


def cached_fetch(cache_name, fetch_fn, force_refresh=False, retries=3, backoff=2.0):
    path = CACHE_DIR / f"{cache_name}.parquet"
    if path.exists() and not force_refresh:
        print(f"📂 Loading cache: {path.name}")
        return pd.read_parquet(path)

    print(f"🌐 Fetching fresh: {path.name}")
    df = None
    for attempt in range(retries):
        try:
            df = fetch_fn()
            break
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"   ↻ retry {attempt + 1}/{retries}: {e}")
            time.sleep(backoff * (attempt + 1))

    if df is not None and not df.empty:
        df.to_parquet(path, index=False)
        print(f"💾 Saved cache: {path.name}")
    return df
    


def should_refresh(season, season_type):
    return (
        season == ACTIVE_SEASON
        and season_type == ACTIVE_SEASON_TYPE
    )
    

# =============================================================================
# SECTION 6: DATA PROCESSING HELPERS
# =============================================================================

def _normalize_names(df):
    if 'PLAYER_NAME' in df.columns:
        df['PLAYER_NAME'] = (df['PLAYER_NAME']
                             .str.normalize('NFKD')
                             .str.encode('ascii', errors='ignore')
                             .str.decode('utf-8'))
    return df
    
    
def norm_name(x):
    x = unidecode(str(x)).lower().strip().replace('.', '')   # P.J. -> pj
    x = re.sub(r'\s+(jr|sr|iv|iii|ii|v)$', '', x)            # drop Jr/Sr/III suffixes
    return re.sub(r'\s+', ' ', x).strip()



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
    """Combine a list of fetched frames (one per season / season-type) into a
    single player-season-indexed table. Light-touch: dedupes by
    (player, season, team), so traded players keep a row per stint AND their
    TOT row — it does NOT collapse them. (For one-row-per-player-season with
    traded players merged to TOT, use consolidate_player_stats instead.)

    Note: stat_type and source are passed in by fetch_nba_stats but unused here.
    """
    # Stack every season/season-type chunk into one frame.
    result = pd.concat(dfs, ignore_index=True)

    # Some endpoints return PLAYER_LAST_TEAM_* instead of TEAM_* — normalize.
    if 'TEAM_ID' not in result.columns and 'PLAYER_LAST_TEAM_ID' in result.columns:
        result = result.rename(columns={'PLAYER_LAST_TEAM_ID': 'TEAM_ID'})
    if 'TEAM_ABBREVIATION' not in result.columns and 'PLAYER_LAST_TEAM_ABBREVIATION' in result.columns:
        result = result.rename(columns={'PLAYER_LAST_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION'})

    # One row per (player, season, team); traded players keep each stint + TOT.
    subset_cols = [c for c in ['PLAYER_ID', 'SEASON', 'TEAM_ID', 'TEAM_ABBREVIATION'] if c in result.columns]
    result = result.drop_duplicates(subset=subset_cols)

    # Drop empty rows: by games-played if that column exists, else rows where
    # every non-index column is NaN.
    game_cols = [c for c in ['GP', 'G'] if c in result.columns]
    if not game_cols:
        non_index = [c for c in result.columns if c not in INDEX_COLS]
        result = result.dropna(subset=non_index, how='all')
    else:
        result = result.dropna(subset=game_cols, how='all')

    # Lock in the player-season index (only the index cols actually present).
    valid_index = [c for c in INDEX_COLS if c in result.columns]
    result.set_index(valid_index, inplace=True)
    return result


def consolidate_player_stats(df):
    """Canonicalize ONE frame's identity columns so it merges/indexes reliably,
    then set the player-season index. Heavier than _consolidate: aggressively
    cleans names/IDs/whitespace and collapses traded players to their TOT row.
    Use on a single custom-fetched frame before joining it into the master.
    """
    # 1. Flatten any existing index and drop stray reset artifacts.
    df = df.reset_index()
    if 'index' in df.columns: df = df.drop(columns=['index'])
    if 'level_0' in df.columns: df = df.drop(columns=['level_0'])

    # 2. Normalize team column names (some endpoints use PLAYER_LAST_TEAM_*).
    df = df.rename(columns={
        'PLAYER_LAST_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION',
        'PLAYER_LAST_TEAM_ID': 'TEAM_ID'
    })

    # 3. Clean PLAYER_NAME before it becomes an index level: strip accents
    #    (Jokić -> Jokic), trim, collapse repeated spaces. This is what lets the
    #    same player match across endpoints/sources later.
    if 'PLAYER_NAME' in df.columns:
        df['PLAYER_NAME'] = df['PLAYER_NAME'].apply(
            lambda x: unidecode(str(x)).strip() if pd.notnull(x) else x
        )
        df['PLAYER_NAME'] = df['PLAYER_NAME'].str.replace(r'\s+', ' ', regex=True)

    # 4. Force every index column to a clean stripped string so merges can't
    #    fail on type/whitespace ("Regular Season " vs "Regular Season"). IDs go
    #    float -> int -> str to kill "203999.0"-style floats; missing index
    #    columns are filled "N/A" so the index is always complete.
    for col in INDEX_COLS:
        if col in df.columns:
            if 'ID' in col:
                df[col] = df[col].astype(float).fillna(0).astype(int).astype(str).str.strip()
            else:
                df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = "N/A"

    # 5. Traded players: if a TOT (season total) row exists, keep ONLY the TOT
    #    row for those players and drop their per-team rows.
    if 'TEAM_ABBREVIATION' in df.columns:
        tot_mask = df['TEAM_ABBREVIATION'] == 'TOT'
        if tot_mask.any():
            traded_ids = df[tot_mask]['PLAYER_ID'].unique()
            df = pd.concat([df[tot_mask], df[~df['PLAYER_ID'].isin(traded_ids)]])

    # 6. One row per player-season, then lock and sort the index.
    df = df.drop_duplicates(subset=INDEX_COLS)
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
    
    
def normalize_master(df):
    df = df.reset_index()
    df['PLAYER_NAME'] = df['PLAYER_NAME'].apply(lambda x: unidecode(str(x)).strip())
    for col in INDEX_COLS:
        if col not in df.columns:
            continue
        if col.endswith('ID'):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64').astype(str)
        else:
            df[col] = df[col].astype(str).str.strip()
    # when near-dups collapse, keep the MOST-populated row, not a random one
    df = df.assign(_n=df.notna().sum(axis=1)).sort_values('_n', ascending=False)
    df = df.drop_duplicates(subset=INDEX_COLS, keep='first').drop(columns='_n')
    return df.set_index(INDEX_COLS).sort_index()
    


# =============================================================================
# SECTION 7: MAIN FETCH FUNCTIONS
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
                df = None
            finally:
                # Pause after every request attempt, including the ones that failed. A
                # `continue` here used to skip the pause, so the moment stats.nba.com
                # started refusing us the loop stopped backing off and hammered it flat
                # out — which is what keeps a throttle alive.
                time.sleep(0.8)

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
    sleep_time=5,
    force_refresh=False,
    timeout=120
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
                    force_refresh=force_refresh,
                    timeout=timeout,
                    **{param_name: b}
                )
            except Exception as e:
                print(f"❌ Error fetching {b} ({season_type}): {e}")
                df = None
            finally:
                # Same reason as in fetch_nba_stats(): back off after every bucket,
                # not just the ones that came back with data.
                time.sleep(sleep_time)

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
    import ast

    dfs = []
    for season in seasons:
        print(f"Fetching Shot Locations for {season} ({season_type})...")
        try:
            cache_name = safe_cache_name("shot_locations", season, season_type, per_mode)
            refresh_this = force_refresh or should_refresh(season, season_type)

            def fetch_raw():
                raw = leaguedashplayershotlocations.LeagueDashPlayerShotLocations(
                    distance_range='By Zone', season=season,
                    per_mode_detailed=per_mode, season_type_all_star=season_type,
                    timeout=timeout
                ).get_data_frames()[0]
                # Flatten BEFORE caching so the parquet never stores a MultiIndex
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = ["_".join(str(x) for x in col if str(x) != "")
                                   for col in raw.columns]
                return raw

            df = cached_fetch(cache_name, fetch_raw, force_refresh=refresh_this)

            # Belt-and-suspenders: old caches reload as tuple-strings — re-flatten them
            def _flat(col):
                s = str(col)
                if s.startswith("(") and "," in s:
                    try:
                        return "_".join(str(x) for x in ast.literal_eval(s) if str(x) != "")
                    except Exception:
                        return s
                return s
            df.columns = [_flat(c) for c in df.columns]

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
    
    
ON_OFF_COLS = [
    'ONCOURT_NET_RATING', 'OFFCOURT_NET_RATING', 'ONOFF_NET_RATING',
    'ONCOURT_OFF_RATING', 'OFFCOURT_OFF_RATING', 'ONOFF_OFF_RATING',
    'ONCOURT_DEF_RATING', 'OFFCOURT_DEF_RATING', 'ONOFF_DEF_RATING',
    'ONCOURT_MIN', 'OFFCOURT_MIN',
]


def fetch_nba_on_off(
    seasons,
    season_type="Regular Season",
    per_mode="PerGame",
    timeout=120,
    force_refresh=False,
    team_ids=None,
):
    """Team net rating with each player on the court vs off it (nba.com On/Off).

    The endpoint is per team, so this is one request per team per season and
    season type (30 for a regular season). `team_ids` limits it to those teams.
    One row per player per team.

    Columns (all per 100 possessions except the minutes):
      ONCOURT_*  team rating while he plays      OFFCOURT_*  while he sits
      ONOFF_NET_RATING = on minus off — the swing. Positive is good.
      ONOFF_DEF_RATING = on minus off — NEGATIVE is good (fewer points allowed).
      ONCOURT_MIN / OFFCOURT_MIN are season TOTALS whatever per_mode says: the
      swing is only as trustworthy as the smaller of the two samples.

    The prefixes are ONCOURT/OFFCOURT rather than ON/OFF because OFF_RATING
    already means offensive rating.
    """
    wanted = None if team_ids is None else {str(t) for t in team_ids}
    chunks = []

    for season in seasons:
        refresh_this = force_refresh or should_refresh(season, season_type)
        for team in _static_teams.get_teams():
            if wanted is not None and str(team["id"]) not in wanted:
                continue
            cache_name = safe_cache_name("on_off", team["abbreviation"], season, season_type)

            def fetch_raw(team_id=team["id"]):
                frames = TeamPlayerOnOffSummary(
                    team_id=team_id, season=season, season_type_all_star=season_type,
                    per_mode_detailed="Totals", timeout=timeout,
                ).get_data_frames()
                # result sets: team overall, players OFF court, players ON court.
                # COURT_STATUS says which is which, so stacking them loses nothing.
                return pd.concat([f for f in frames if "VS_PLAYER_ID" in f.columns], ignore_index=True)

            try:
                raw = cached_fetch(cache_name, fetch_raw, force_refresh=refresh_this)
            except Exception as e:
                print(f"⚠️ On/off failed for {team['abbreviation']} {season} {season_type}: {e}")
                raw = None
            finally:
                time.sleep(0.8)

            if raw is None or raw.empty:
                continue

            keys = ["VS_PLAYER_ID", "VS_PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"]
            stats = ["NET_RATING", "OFF_RATING", "DEF_RATING", "MIN"]
            sides = {}
            for status, prefix in (("On", "ONCOURT_"), ("Off", "OFFCOURT_")):
                part = raw[raw["COURT_STATUS"].astype(str).str.strip().str.lower() == status.lower()]
                sides[status] = part[keys + stats].rename(columns={c: prefix + c for c in stats})
            df = sides["On"].merge(sides["Off"].drop(columns=["VS_PLAYER_NAME", "TEAM_ABBREVIATION"]),
                                   on=["VS_PLAYER_ID", "TEAM_ID"], how="outer")
            for stat in ("NET_RATING", "OFF_RATING", "DEF_RATING"):
                df[f"ONOFF_{stat}"] = df[f"ONCOURT_{stat}"] - df[f"OFFCOURT_{stat}"]

            df = df.rename(columns={"VS_PLAYER_ID": "PLAYER_ID", "VS_PLAYER_NAME": "PLAYER_NAME"})
            # this endpoint may write names "Last, First"; the master has "First Last"
            df["PLAYER_NAME"] = df["PLAYER_NAME"].map(
                lambda n: " ".join(reversed([p.strip() for p in str(n).split(",", 1)])) if "," in str(n) else n)
            df = _normalize_names(df)
            df["TEAM_ABBREVIATION"] = team["abbreviation"]
            df["SEASON"] = season
            df["SEASON_TYPE"] = season_type
            chunks.append(df[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
                              "SEASON", "SEASON_TYPE"] + ON_OFF_COLS])

    if not chunks:
        return pd.DataFrame()

    return _consolidate(chunks, "OnOff", source="on_off")


def add_on_off(df_master, seasons, season_types=("Regular Season", "Playoffs"),
               timeout=120, force_refresh=False):
    """Write the on/off columns into the master for these seasons.

        df_master = add_on_off(df_master, SEASONS)                        # backfill
        df_master = add_on_off(df_master, [ACTIVE_SEASON],                # keep current
                               season_types=(ACTIVE_SEASON_TYPE,))

    Joins on player id + team id, never on the name: this endpoint spells names
    its own way, and a name mismatch would silently drop the player. It fills
    rows that already exist and creates none. Only teams the master has for
    that season are requested, so playoffs cost 16 requests, not 30.

    The master carries one row per player-season under his latest team, so a
    traded player gets his on/off with that team only.
    """
    indexed = isinstance(df_master.index, pd.MultiIndex)
    m = df_master.reset_index() if indexed else df_master.copy()

    def as_key(col):
        return pd.to_numeric(col, errors="coerce").astype("Int64").astype(str)

    m["_pid"], m["_tid"] = as_key(m["PLAYER_ID"]), as_key(m["TEAM_ID"])
    for col in ON_OFF_COLS:
        if col not in m.columns:
            m[col] = np.nan

    for season in seasons:
        for season_type in season_types:
            mask = (m["SEASON"] == season) & (m["SEASON_TYPE"] == season_type)
            if not mask.any():
                continue
            on_off = fetch_nba_on_off([season], season_type, timeout=timeout,
                                      force_refresh=force_refresh,
                                      team_ids=sorted(m.loc[mask, "_tid"].unique()))
            if on_off.empty:
                print(f"⚠️ No on/off data for {season} {season_type}")
                continue
            on_off = on_off.reset_index()
            on_off["_pid"], on_off["_tid"] = as_key(on_off["PLAYER_ID"]), as_key(on_off["TEAM_ID"])
            lookup = on_off.drop_duplicates(["_pid", "_tid"]).set_index(["_pid", "_tid"])[ON_OFF_COLS]
            values = lookup.reindex(pd.MultiIndex.from_frame(m.loc[mask, ["_pid", "_tid"]]))
            m.loc[mask, ON_OFF_COLS] = values.to_numpy()
            matched = int(values["ONCOURT_NET_RATING"].notna().sum())
            print(f"🔁 On/off {season} {season_type}: {matched}/{int(mask.sum())} master rows filled")

    m = m.drop(columns=["_pid", "_tid"])
    return m.set_index(INDEX_COLS).sort_index() if indexed else m


def fetch_nba_clutch_stats(
    seasons,
    season_type="Regular Season",
    per_mode="PerGame",
    measure_types=("Base", "Advanced"),
    timeout=120,
    force_refresh=False,
):
    """Clutch = last 5 min, margin <= 5 (the endpoint's defaults).
    Pulls each measure type, prefixes everything CLUTCH_, joins into one frame.
    Base + Advanced is the meaningful signal; clutch is small-sample, so skip
    Misc/Scoring/Usage unless you really want the noise."""
    from nba_api.stats.endpoints import leaguedashplayerclutch

    keep_keys = set(INDEX_COLS) | {"PLAYER_ID", "PLAYER_NAME",
                                   "TEAM_ABBREVIATION", "TEAM_ID",
                                   "SEASON", "SEASON_TYPE"}
    drop_meta = ["NICKNAME", "AGE", "GP", "G", "W", "L", "W_PCT", "MIN", "GROUP_SET", "TEAM_COUNT",
    "DD2", "TD3", "index", "level_0"]

    season_frames = []
    for season in seasons:
        measure_frames = []
        for mt in measure_types:
            print(f"🕐 Clutch [{mt}] — {season} ({season_type})")
            cache_name = safe_cache_name("clutch", mt, season, season_type, per_mode)

            def fetch_raw(mt=mt, season=season):
                return leaguedashplayerclutch.LeagueDashPlayerClutch(
                    measure_type_detailed_defense=mt,
                    per_mode_detailed=per_mode,
                    season=season,
                    season_type_all_star=season_type,
                    timeout=timeout,
                ).get_data_frames()[0]

            try:
                df = cached_fetch(cache_name, fetch_raw,
                                  force_refresh=force_refresh or should_refresh(season, season_type))
            except Exception as e:
                print(f"❌ Clutch [{mt}] {season} failed: {e}")
                continue
            if df is None or df.empty:
                print(f"⚠️ No clutch data for {mt} {season}")
                continue

            df = _normalize_names(df)
            df["SEASON"] = season
            df["SEASON_TYPE"] = season_type
            df = df.drop(columns=drop_meta, errors="ignore")
            df = drop_redundant_cols(df)  
            df = df.rename(columns={c: f"CLUTCH_{c}"
                                    for c in df.columns if c not in keep_keys})
            measure_frames.append(consolidate_player_stats(df))

        if not measure_frames:
            continue
        season_df = measure_frames[0]
        for extra in measure_frames[1:]:
            new_cols = [c for c in extra.columns if c not in season_df.columns]
            season_df = season_df.join(extra[new_cols], how="outer")
        season_frames.append(season_df)

    return pd.concat(season_frames).sort_index() if season_frames else pd.DataFrame()
    

def get_player_registry(season, retries=4, timeout=30):
    import time
    print(f"🚀 Fetching {season}...")
    for attempt in range(1, retries + 1):
        try:
            df = LeagueDashPlayerBioStats(season=season, timeout=timeout).get_data_frames()[0]
            break                                   # ↑ no headers= on purpose, so NBA_STATS_HEADERS applies
        except Exception as e:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"   ⏳ {type(e).__name__} on {season}, retry {attempt}/{retries-1} in {wait}s")
            time.sleep(wait)
    df['SEASON'] = season
    df = df.rename(columns={
        'PLAYER_HEIGHT_INCHES': 'HT_INCHES',
        'DRAFT_YEAR': 'DRAFT_YR',
        'DRAFT_NUMBER': 'DRAFT_POSITION'
    })
    return df


def get_draft_table(seasons, cache_path="data/bio_registry.parquet"):
    """Per-player draft info for merging on PLAYER_ID. Collapses across seasons —
    bio has no SEASON_TYPE, so we never set_index(INDEX_COLS)."""
    from pathlib import Path
    cache = Path(cache_path)
    if cache.exists():
        reg = pd.read_parquet(cache)
    else:
        reg = pd.concat([get_player_registry(s) for s in seasons], ignore_index=True)
        reg.to_parquet(cache)
    bio = reg.groupby('PLAYER_ID')[['DRAFT_POSITION','DRAFT_YR']].first().reset_index()
    bio['DRAFT_YR']       = pd.to_numeric(bio['DRAFT_YR'], errors='coerce')
    bio['DRAFT_POSITION'] = pd.to_numeric(bio['DRAFT_POSITION'], errors='coerce')
    bio['PLAYER_ID']      = bio['PLAYER_ID'].astype(str)
    return bio
    
    

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
    },
    {"name": "Clutch Stats",
    "fetch_fn": fetch_nba_clutch_stats,
    "suffix": "clutch",
    "keep_cols": None
    }
]


# =============================================================================
# SECTION 8: MASTER DF MANAGEMENT
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



def _normalize_index(df, index_cols=None):
    """Put `df` on INDEX_COLS using one shared definition of row identity.

    Anything that decides whether two rows are the same player-season has to
    normalize keys identically, or a row gets patched in one place and duplicated
    in another. sync_and_patch_stat() and ensure_active_season_rows() both come
    through here so they can't drift apart.
    """
    if index_cols is None:
        index_cols = INDEX_COLS

    df = df.reset_index()
    df = df.drop(columns=[c for c in ("index", "level_0") if c in df.columns])

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


def sync_and_patch_stat(df_master, new_data_df, index_cols=None, add_new_cols=True):
    if index_cols is None:
        index_cols = INDEX_COLS

    incoming_stat_cols = set(new_data_df.columns) - {'TEAM_NAME', 'NICKNAME', 'AGE'}

    master = _normalize_index(df_master, index_cols)
    new = _normalize_index(new_data_df, index_cols)

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
    
    
    
def ensure_active_season_rows(
    df_master,
    active_season,
    active_season_type,
    per_mode="PerGame",
    timeout=120,
    force_refresh=True,
):
    """Give the active season a row for every player who has played in it.

    The refresh path patches with DataFrame.update(), which writes only into rows
    that already exist and never creates them. Anyone who wasn't in the league's
    stat line the last time the season was built therefore stays invisible however
    often you refresh — a late signing, someone back from injury, a call-up. The
    same goes for a traded player: TEAM_ABBREVIATION and TEAM_ID are part of
    INDEX_COLS, so a trade changes his key, his fresh row stops matching, and the
    pre-trade row keeps stale numbers nothing will ever correct.

    Base is the authoritative roster for a season — one row per player with at
    least one game played — so its index is the row set the master ought to have.
    Adding the missing rows here, before any patching, leaves every later update()
    with somewhere to write; they arrive holding Base's columns and the remaining
    stat families fill in the rest on their normal pass. Costs one extra Base
    request per refresh, because should_refresh() forces the active season past
    the cache.

    Returns (df_master, added_player_names).
    """
    base = fetch_nba_stats(
        "Base",
        [active_season],
        season_type=active_season_type,
        per_mode=per_mode,
        timeout=timeout,
        force_refresh=force_refresh,
    )

    if base is None or base.empty:
        print("⚠️ No Base data returned — skipping the missing-player check.")
        return df_master, []

    master = _normalize_index(df_master)
    fresh = _normalize_index(base)

    missing = fresh.index.difference(master.index)

    if not len(missing):
        print(f"✅ {active_season}: master already has every player.")
        return master, []

    added = fresh.loc[missing].reindex(columns=master.columns)
    combined = pd.concat([master, added]).sort_index()

    names = sorted(set(missing.get_level_values("PLAYER_NAME")))
    print(f"➕ {active_season}: added {len(missing)} player rows the master was missing.")
    print(f"   {', '.join(names[:8])}{' …' if len(names) > 8 else ''}")

    return combined, names


def refresh_active_season_data(
    df_master,
    active_season="2025-26",
    active_season_type="Playoffs",
    full_stats=None,
    curated_config=None,
    custom_config=None,
    bucket_config=None,
    force_refresh=True,
    per_mode="PerGame",
    timeout=180,
):
    full_stats = full_stats or FULL_NBA_STATS
    curated_config = curated_config or REFRESH_NBA_STATS_CONFIG
    custom_config = custom_config or REFRESH_CUSTOM_CONFIG
    bucket_config = bucket_config or BUCKET_CONFIG

    before_cols = set(df_master.columns)

    # update() writes only into rows that already exist, so the season needs a row
    # for every player who has played in it *before* any patching starts.
    df_master, _added_players = ensure_active_season_rows(
        df_master,
        active_season,
        active_season_type,
        per_mode=per_mode,
        timeout=timeout,
        force_refresh=force_refresh,
    )

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
        
    for stat_type, cfg in bucket_config.items():
        print(f"\n🔄 Refreshing {stat_type} buckets — {active_season} {active_season_type}")
        bucket_df = build_bucketed_features(
            stat_type=stat_type, seasons=[active_season],
            buckets=cfg["buckets"], prefix=cfg["prefix"], param_name=cfg["param"],
            fetch_fn=fetch_nba_stats,
            season_types=(active_season_type,),
            force_refresh=force_refresh,
            timeout=timeout,
        )
        if bucket_df is None or bucket_df.empty:
            print(f"⚠️ No bucket data for {stat_type}")
            continue
        # the shot dashboard carries its own MIN — don't let it overwrite the real box score
        protect = [c for c in CORE_BOX_SCORE_COLS if c in bucket_df.columns]
        protect += [c for c in ("index", "level_0") if c in bucket_df.columns]
        bucket_df = bucket_df.drop(columns=protect, errors="ignore")
        df_master = sync_and_patch_stat(df_master, bucket_df)

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
    
    
    
def backfill_master(existing_master, seasons,
                    season_types=('Regular Season', 'Playoffs')):
    """Sibling of refresh_active_season_data, for historical seasons.
    Uses the ADD path (no force_refresh — historical data is final), then
    stacks the new season rows beneath the existing master."""

    # 1. Base + all FULL_NBA_STATS families → creates the historical ROWS
    hist = build_master_df(seasons, FULL_NBA_STATS, season_types=season_types)
    if hist.empty:
        print("⚠️ No base data for these seasons.")
        return existing_master

    # 2. Curated nba_stats families (misc / rebounding / general / player-defense)
    for stat_type, cfg in REFRESH_NBA_STATS_CONFIG.items():
        hist = ingest_nba_stat(
            hist, cfg.get('stat_type', stat_type), seasons,
            suffix=cfg.get('suffix') or stat_type.lower(),
            keep_cols=cfg.get('keep_cols'),
            source_override=cfg.get('source_override'),
            season_types=season_types,
        )

    # 3. Custom fetchers (shot locations, defense, breakdowns, hustle, clutch)
    for cfg in REFRESH_CUSTOM_CONFIG:
        hist = ingest_stat(
            hist, cfg['fetch_fn'], seasons=seasons,
            suffix=cfg.get('suffix'), keep_cols=cfg.get('keep_cols'),
            season_types=season_types,
        )
        
    # 3b. Bucketed shot-dashboard families (closest defender, dribbles, shot
    #     clock, touch time) — built via build_bucketed_features, not the
    #     configs, so the driver has to add them explicitly.
    for stat_type, cfg in BUCKET_CONFIG.items():
        bucket_df = build_bucketed_features(
            stat_type=stat_type, seasons=seasons,
            buckets=cfg['buckets'], prefix=cfg['prefix'],
            param_name=cfg['param'], fetch_fn=fetch_nba_stats,
            season_types=season_types,
        )
        hist = update_master_df(hist, bucket_df, suffix=stat_type.lower())

    # 4. Stack historical rows beneath the existing master
    combined = pd.concat([existing_master, hist]).sort_index()
    return combined[~combined.index.duplicated(keep='last')]


# =============================================================================
# SECTION 9: VISUALIZATION
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


# Stats where a smaller number is the better one. Charts and percentiles read
# this so that "100th percentile" always means best, whichever way the stat runs.
LOWER_IS_BETTER = {
    'TOV', 'TM_TOV_PCT', 'PF', 'BLKA', 'DEF_RATING',
    'D_FG_PCT', 'PCT_PLUSMINUS', 'DEF_RIM_FG_PCT',
    'DEF2_FG2_PCT', 'DEF3_FG3_PCT', 'DEFLT6_LT_06_PCT', 'DEFLT10_LT_10_PCT', 'DEFGT15_GT_15_PCT',
    'DEF2_PLUSMINUS', 'DEF3_PLUSMINUS', 'DEFLT6_PLUSMINUS', 'DEFLT10_PLUSMINUS', 'DEFGT15_PLUSMINUS',
    'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE', 'OPP_PTS_FB', 'OPP_PTS_PAINT',
    'DRIVE_TOV', 'DRIVE_TOV_PCT', 'ONCOURT_DEF_RATING', 'ONOFF_DEF_RATING',
    'OPP_EFG_PCT', 'OPP_FTA_RATE', 'OPP_OREB_PCT', 'OPP_PTS',
}


def is_lower_better(col: str) -> bool:
    """True for a stat where smaller is better (opponent FG% allowed, turnovers, ...)."""
    col = str(col)
    base = col[len('CLUTCH_'):] if col.startswith('CLUTCH_') else col
    return (base in LOWER_IS_BETTER
            or base.endswith('_TOV_POSS_PCT')           # play-type turnover rates
            or base.endswith('_TOV_PER36') or base.endswith('_PF_PER36'))


def plot_interactive_barh(
    df: pd.DataFrame, *, x_col: str, y_col: str,
    season: Optional[str] = None, season_type = 'Regular Season',query_condition: Optional[str] = None,
    sort_by: Optional[str] = None, ascending: Optional[bool] = None,
    title: str = "Custom Bar Plot", ylim: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = None, height: int = 500, width: int = 1000,
    lower_is_better: Optional[bool] = None, percentile_among: str = "shown",
    top_n: Optional[int] = None, keep_top: Optional[dict] = None,
    hover: Optional[list] = None,
) -> Any:
    """Horizontal bar chart of one stat, best on top, with a percentile on hover.

    hover: extra columns to show in the tooltip, e.g. ["TEAM_ABBREVIATION", "GP"].
    keep_top: a volume filter by rank instead of by a fixed number,
        {column: share kept}. keep_top={"PULL_UP_FG3A": 0.75} drops the quarter
        of the pool that takes the fewest pull-up threes. It is applied after
        season, season type and query_condition, so "top 75%" means of the
        players who passed those. The cutoff it worked out is printed.

    lower_is_better: None looks the stat up in LOWER_IS_BETTER (DEF_RIM_FG_PCT,
        TM_TOV_PCT, ...). When true the smallest value is the top bar, the colour
        map is reversed so the best value keeps the warm colour, and the
        percentile is flipped so the best is still the 100th. Passing ascending
        or cmap yourself overrides those two.
    percentile_among: "shown" ranks each player among the rows that pass
        query_condition — the qualified pool you defined. "league" ranks among
        every player that season, which for a rate stat includes players with a
        handful of attempts.
    top_n: keep only the top_n bars. The percentile is still taken over the
        whole pool, so a top-15 chart doesn't rank its players 1 to 15.
    """
    if lower_is_better is None:
        lower_is_better = is_lower_better(x_col)
    if ascending is None:
        ascending = not lower_is_better                 # bokeh draws the last row on top
    if cmap is None:
        cmap = "coolwarm_r" if lower_is_better else "coolwarm"

    filtered = df.reset_index().copy()

    if season is not None:
        filtered = filtered[filtered["SEASON"] == season]
        
    # 2. THE MISSING FILTER: Season Type (Reg vs Playoffs)
    if "SEASON_TYPE" in filtered.columns:
        filtered = filtered[filtered["SEASON_TYPE"] == season_type]
    elif "season_type" in filtered.columns: # Catch case sensitivity
        filtered = filtered[filtered["season_type"] == season_type]
        
    filtered[x_col] = pd.to_numeric(filtered[x_col], errors='coerce').astype(float)

    def percentile(values):
        # best value -> 100, whichever direction the stat runs
        return (values.rank(pct=True, ascending=not lower_is_better) * 100).round(0)

    if percentile_among == "league":
        filtered["PERCENTILE"] = percentile(filtered[x_col])
    if query_condition:
        filtered = filtered.query(query_condition)
    for col, share in (keep_top or {}).items():
        if not 0 < share <= 1:
            raise ValueError(f"keep_top[{col!r}] must be a share between 0 and 1, got {share}")
        volume = pd.to_numeric(filtered[col], errors="coerce")
        cutoff = volume.quantile(1 - share)
        kept = volume >= cutoff
        print(f"keep_top: {col} >= {cutoff:.2f} keeps {int(kept.sum())} of {len(filtered)} players")
        filtered = filtered[kept]
    if percentile_among != "league":
        filtered["PERCENTILE"] = percentile(filtered[x_col])

    filtered = filtered.sort_values(by=sort_by or x_col, ascending=ascending)
    filtered[y_col] = filtered[y_col].astype(str)
    filtered = filtered.dropna(subset=[x_col])
    if top_n:
        filtered = filtered.tail(top_n)                 # the last rows are the top bars

    plot = filtered.hvplot.barh(
        x=y_col, y=x_col, height=height, width=width,
        color=x_col, cmap=cmap, colorbar=True, title=title,
        xlabel=y_col, ylabel=x_col,
        hover_cols=["PERCENTILE"] + [c for c in (hover or []) if c in filtered.columns and c not in (x_col, y_col)],
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
        
        
def plot_stat_over_time(df, players, stat, season_type="Regular Season", figsize=(11, 6)):
    """Line chart of one stat across seasons for one or more players.

    players: a name or list of names. stat: any column in df.
    Uses the TOT row for traded seasons so each player gets one point per year.
    Names must match the cleaned master ('Nikola Jokic', not 'Jokić').
    """
    if isinstance(players, str):
        players = [players]

    data = df.reset_index()
    data = data[data["SEASON_TYPE"] == season_type]

    if stat not in data.columns:
        print(f"⚠️ '{stat}' not found.")
        return

    fig, ax = plt.subplots(figsize=figsize)

    for player in players:
        pdf = data[data["PLAYER_NAME"] == player].copy()
        if pdf.empty:
            print(f"⚠️ No rows for {player} ({season_type}).")
            continue

        # one value per season: prefer the TOT (traded-season total) row
        pdf["_is_tot"] = pdf["TEAM_ABBREVIATION"] == "TOT"
        pdf = (pdf.sort_values(["SEASON", "_is_tot"], ascending=[True, False])
                  .drop_duplicates("SEASON", keep="first"))
        pdf = pdf.dropna(subset=[stat]).sort_values("SEASON")
        if pdf.empty:
            print(f"⚠️ {player} has no {stat} values.")
            continue

        ax.plot(pdf["SEASON"], pdf[stat], marker="o", label=player)

    ax.set_title(f"{stat.replace('_', ' ')} over time — {season_type}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel(stat.replace("_", " "))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# SECTION 10: BBREF SALARY SCRAPING
# =============================================================================

# 1) every player URL from the season totals pages (retired guys included)
def harvest_player_urls(end_years=range(2016, 2027), sleep=3.5):
    urls = set()
    for yr in end_years:
        r = requests.get(f"{BASE}/leagues/NBA_{yr}_totals.html", headers=HDR, timeout=30)
        if r.status_code != 200:
            print(f"  [{r.status_code}] {yr}"); time.sleep(sleep); continue
        hits = re.findall(r'/players/[a-z]/[a-z0-9]+\.html', r.text)
        urls.update(BASE + h for h in hits)
        print(f"  {yr-1}-{str(yr)[2:]}: {len(set(hits))} players (total unique {len(urls)})")
        time.sleep(sleep)
    return sorted(urls)


# 2) one player's salary history (name from the page <h1>)
def bbref_salary_history(url):
    slug = url.rstrip('/').split('/')[-1].replace('.html', '')
    cf = CACHE / f"{slug}.csv"
    if cf.exists(): return pd.read_csv(cf)
    r = requests.get(url, headers=HDR, timeout=30)
    if r.status_code != 200:
        print(f"  [{r.status_code}] {slug}"); return None
    soup = BeautifulSoup(r.content, "html.parser")
    h1 = soup.find("h1"); player = h1.get_text(strip=True) if h1 else slug
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):   # salaries table is comment-hidden
        if 'salaries' in c.lower() and '<table' in c.lower():
            soup.append(BeautifulSoup(c, "html.parser"))
    tbl = soup.find("table", id=re.compile("salar", re.I))
    if tbl is None: return None
    df = pd.read_html(StringIO(str(tbl)))[0]
    df = df[[c for c in df.columns if str(c) in ('Season', 'Salary')]].copy()
    df = df[df['Season'].astype(str).str.match(r'\d{4}-\d{2}$')]
    df['Salary'] = df['Salary'].map(lambda s: int(re.sub(r'[^0-9]', '', str(s)) or 0))
    df = df[df['Salary'] > 0]; df['Player'] = player
    df.to_csv(cf, index=False)
    return df


def scrape_all(urls, sleep=3.5):
    frames = []
    for i, u in enumerate(urls, 1):
        h = bbref_salary_history(u)
        if h is not None and not h.empty: frames.append(h)
        if i % 50 == 0: print(f"  {i}/{len(urls)}")
        time.sleep(sleep)
    out = pd.concat(frames, ignore_index=True).rename(columns={'Season': 'SEASON'})
    out['name_key'] = out['Player'].apply(norm_name)
    return out.drop_duplicates(['name_key', 'SEASON'])


def fetch_all_nba_raw(url="https://www.basketball-reference.com/awards/all_league.html"):
    r = requests.get(url, headers=HDR, timeout=30)
    print("status:", r.status_code)
    soup = BeautifulSoup(r.content, "html.parser")
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):     # BBRef comment-hides tables
        if 'all_league' in c.lower() and '<table' in c.lower():
            soup.append(BeautifulSoup(c, "html.parser"))
    tbl = soup.find("table", id=re.compile("all.?league", re.I)) or soup.find("table")
    df = pd.read_html(StringIO(str(tbl)))[0]
    print("shape:", df.shape, "| cols:", list(df.columns))
    return df


# =============================================================================
# SECTION 11: MODELING LAYER
# =============================================================================

def build_model_table(df_master, id_keys=ID_KEYS, descriptive=DESCRIPTIVE):
    df = df_master.reset_index()
    rs = df[df['SEASON_TYPE'] == 'Regular Season'].copy()
    po = df[df['SEASON_TYPE'] == 'Playoffs'].copy()

    # one row per player-season; if dupes exist, keep the most-populated row
    def _dedupe(d):
        return (d.assign(_n=d.notna().sum(axis=1))
                 .sort_values('_n', ascending=False)
                 .drop_duplicates(id_keys)
                 .drop(columns='_n'))
    rs, po = _dedupe(rs), _dedupe(po)

    # SEASON_TYPE is now redundant; descriptive cols come from RS only
    rs = rs.drop(columns=['SEASON_TYPE'], errors='ignore')
    po = po.drop(columns=['SEASON_TYPE'] + [c for c in descriptive if c in po.columns],
                 errors='ignore')

    # join on stable identity (PLAYER_ID + SEASON), NOT team —
    # a traded player's RS team can differ from his playoff team
    rs_i, po_i = rs.set_index(id_keys), po.set_index(id_keys)
    model = rs_i.join(po_i, lsuffix='_rs', rsuffix='_po', how='left')
    model['made_playoffs'] = model.index.isin(po_i.index).astype(int)
    return model.reset_index()
    
    
def attach_salaries(model, salaries, name_col='PLAYER_NAME'):
    model = model.copy()
    model['name_key'] = model[name_col].apply(norm_name)
    out = model.merge(salaries[['name_key', 'SEASON', 'Salary']],
                      on=['name_key', 'SEASON'], how='left')
    return out.drop(columns='name_key')


# =============================================================================
# SECTION 12: FEATURE SELECTION
# =============================================================================


def prefilter_features(df, target='pct_cap', drop_cols=None, nan_thresh=0.85, corr_thresh=0.95):
    import numpy as np
    if drop_cols is None:
        drop_cols = ['Salary','PLAYER_ID','PLAYER_NAME','name_key','SEASON',
                     'TEAM_ABBREVIATION','TEAM_ID','SEASON_TYPE','Player', target]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=[np.number])
    start = X.shape[1]

    const = X.nunique(dropna=True).pipe(lambda s: s[s <= 1].index.tolist())     # 1) near-constant
    empty = X.isna().mean().pipe(lambda s: s[s > nan_thresh].index.tolist())    # 2) near-empty (_po stays)

    X2 = X.drop(columns=set(const) | set(empty))
    corr  = X2.corr().abs()                                                     # 3) collinear pairs
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    collinear = [c for c in upper.columns if (upper[c] > corr_thresh).any()]

    keep = [c for c in X2.columns if c not in collinear]
    return keep, {'start': start, 'const': len(const), 'empty': len(empty),
                  'collinear': len(collinear), 'keep': len(keep)}


def explain_player(name, season, top_n=20):
    row = mdf[(mdf['PLAYER_NAME'] == name) & (mdf['SEASON'] == season)]
    if row.empty:
        print(f"no row for {name} {season}"); return
    i   = mdf.index.get_loc(row.index[0])
    cap = SALARY_CAP[season]

    contrib = pd.Series(sv[i], index=feats)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top_n)

    base = expl.expected_value
    pred = row['pred_pct_cap'].iloc[0]; actual = row['pct_cap'].iloc[0]
    print(f"\n{name} — {season}")
    print(f"  market baseline:  {base:6.1%}  (${base*cap:,.0f})")
    print(f"  model prediction: {pred:6.1%}  (${pred*cap:,.0f})")
    print(f"  actual salary:    {actual:6.1%}  (${actual*cap:,.0f})")
    print(f"  → {'OVER' if actual>pred else 'UNDER'}paid vs market by ${abs(actual-pred)*cap:,.0f}\n")
    print("  what pushed the market price (+ up / − down):")
    for f, v in top.items():
        print(f"    {'+' if v>0 else '−'} {f:32s} {v*cap:>+12,.0f}")


def prior_counts(name_key, year, decay=0.75):
    p = picks[picks['name_key'] == name_key]
    last3 = int(p[(p['yr'] < year) & (p['yr'] >= year - 3)]['n'].sum())
    past  = p[p['yr'] < year]
    # decayed career count: each past selection weighted by decay^(years_ago)
    ever  = float((past['n'] * (decay ** (year - past['yr']))).sum())
    return last3, ever

def prior_weighted(name_key, year, decay=0.75):
    p = aw[aw['name_key'] == name_key]
    last3 = int(p[(p['yr'] < year) & (p['yr'] >= year - 3)]['weight'].sum())
    past  = p[p['yr'] < year]
    ever  = float((past['weight'] * (decay ** (year - past['yr']))).sum())
    return ever, last3


# =============================================================================
# SECTION 13: LEBRON DATA RETRIEVAL
# =============================================================================

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
\
# Columns we keep from LEBRON. Drop Player/Team/Pos/MPG — they're either
# redundant with the master or would collide. Keep the impact metrics + roles.
LEBRON_VALUE_COLS = ["LEBRON", "O-LEBRON", "D-LEBRON", "WAR", "OffRole", "DefRole"]

# single-season files look like 'lebron-data-2019.csv'; the multi-season
# aggregate file (wrong grain) has multiple years and must be excluded.
_SINGLE_SEASON_RE = re.compile(r"lebron-data-(\d{4})\.csv$")


def _ending_year_to_season(year: int) -> str:
    """
    Convert a LEBRON file's year token to the NBA season string.

    CONFIRMED convention: the file year is the season's ENDING year.
      lebron-data-2026.csv -> '2025-26'   (current season)
      lebron-data-2025.csv -> '2024-25'
      lebron-data-2019.csv -> '2018-19'
    """
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


# =============================================================================
# LAG + ENCODE  (leak-safe feature construction)
# =============================================================================

LBR_NUMERIC = ["LBR_LEBRON", "LBR_O_LEBRON", "LBR_D_LEBRON", "LBR_WAR"]
LBR_ROLES   = ["LBR_OffRole", "LBR_DefRole"]


def _season_end_year(season: str) -> int:
    """'2015-16' -> 2016  (ending year, as int, for consecutiveness checks)."""
    start = int(str(season)[:4])
    return start + 1


def lag_and_encode_lebron(
    model_df: pd.DataFrame,
    numeric_lags=(1, 2),
    role_lag: int = 1,
    dummy_na: bool = True,
    drop_contemporaneous: bool = True,
):
    """
    Turn the merged (contemporaneous) LBR_* columns into LEAK-SAFE features:

      * numeric metrics (LEBRON / O / D / WAR) -> prior-season lags, gap-masked
      * roles (Off/Def) -> prior-season role, gap-masked, then one-hot encoded

    Gap-mask: a lag of k is only kept when the prior row is EXACTLY k seasons
    earlier (consecutive). If a player missed a season, the mispaired lag is
    nulled instead of silently gluing a 2-years-ago value into a 'prior year' slot.

    drop_contemporaneous=True removes the current-season LBR_* columns from the
    output so they can't accidentally be used as features (current-season impact
    predicting current-season salary is a leak).
    """
    idx_names = model_df.index.names if model_df.index.names[0] is not None else None
    df = model_df.reset_index() if idx_names else model_df.copy()

    for key in ("PLAYER_ID", "SEASON"):
        if key not in df.columns:
            raise KeyError(f"model_df missing '{key}' after reset_index().")

    df["_END"] = df["SEASON"].map(_season_end_year)
    df["PLAYER_ID"] = df["PLAYER_ID"].astype(str).str.strip()
    df = df.sort_values(["PLAYER_ID", "_END"]).reset_index(drop=True)

    g = df.groupby("PLAYER_ID", sort=False)
    end_by_player = g["_END"]

    new_cols = {}

    # ---- numeric lags, gap-masked ----
    for k in numeric_lags:
        gap_ok = (df["_END"] - end_by_player.shift(k)) == k   # prior row is exactly k seasons back
        for col in LBR_NUMERIC:
            if col not in df.columns:
                continue
            lagged = g[col].shift(k)
            lagged = lagged.where(gap_ok)                     # null where the season gap != k
            new_cols[f"{col}_prior{k}"] = lagged

    # ---- role lag, gap-masked, then one-hot ----
    role_frames = []
    gap_ok_role = (df["_END"] - end_by_player.shift(role_lag)) == role_lag
    for col in LBR_ROLES:
        if col not in df.columns:
            continue
        lagged_role = g[col].shift(role_lag).where(gap_ok_role)
        dummies = pd.get_dummies(
            lagged_role,
            prefix=f"{col}_prior{role_lag}",
            dummy_na=dummy_na,      # 'no prior-season role' becomes its own honest column
        )
        role_frames.append(dummies)

    out = pd.concat([df, pd.DataFrame(new_cols, index=df.index)] + role_frames, axis=1)

    if drop_contemporaneous:
        drop = [c for c in LBR_NUMERIC + LBR_ROLES if c in out.columns]
        out = out.drop(columns=drop)

    out = out.drop(columns=["_END"])

    n_num = len([c for c in out.columns if c.startswith("LBR_") and "_prior" in c])
    n_role = sum(fr.shape[1] for fr in role_frames)
    print(f"✅ Built {n_num} lagged numeric features + {n_role} lagged role one-hots.")
    if drop_contemporaneous:
        print("   Dropped contemporaneous LBR_* columns (leak-safe).")

    if idx_names:
        out = out.set_index(list(idx_names))
    return out
