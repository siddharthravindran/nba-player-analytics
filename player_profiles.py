"""Player style profiles: what a player tends to do, and how well he does it.

    from player_profiles import *
    profiles = build_profiles(df_master, "2025-26")

    player_profile(profiles, "Stephen Curry")          # one row per style group
    player_detail(profiles, "Stephen Curry", "Self-Creation")
    plot_player_profile(profiles, "Stephen Curry")
    compare_players(profiles, ["Stephen Curry", "Nikola Jokic"])
    similar_players(profiles, "Stephen Curry")
    style_leaders(profiles, "Rim Pressure")

Every group is split in two, because style and quality are different things:

  tendency   how much of it he does — a share of his own shots, a rate per 36
             minutes, a play-type frequency. This is the playstyle.
  skill      how well it goes — efficiency on those plays.

Each stat becomes a percentile among that season's qualified players, and a
group's score is the average of its stats. On the skill side a lower-is-better
stat (opponent FG% at the rim, turnover rate) is flipped, so 100 is always best.

Three choices worth knowing about:
  * Counting stats are per 36 minutes, not per game. Per game, a starter
    out-"tends" a reserve at everything just by being on the floor longer.
  * A skill stat only counts once the player has real volume behind it
    (SKILL_VOLUME): corner-3 accuracy on one attempt a week is noise.
  * A missing Synergy play type means too few possessions to be listed, so its
    frequency is treated as zero. Its efficiency stays missing.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nba_viz_utils import is_lower_better

# =============================================================================
# WHAT GOES IN EACH GROUP
# =============================================================================
# Names ending _PER36, _SHARE, _RATE, _FREQ are built by add_profile_columns();
# everything else is a df_master column as is.

PROFILE_GROUPS = {
    "Scoring Load": {
        "tendency": ["USG_PCT", "PTS_PER36", "FGA_PER36", "PCT_PTS", "TOUCHES_PER36"],
        "skill": ["TS_PCT", "EFG_PCT", "PTS_PER_TOUCH"],
    },
    "Rim Pressure": {
        "tendency": ["RIM_FGA_SHARE", "DRIVES_PER36", "FT_RATE", "PAINT_TOUCHES_PER36",
                     "PFD_PER36", "PCT_PTS_PAINT"],
        "skill": ["Restricted_Area_FG_PCT", "DRIVE_FG_PCT", "DRIVE_PTS_PCT"],
    },
    "Mid-Range Game": {
        "tendency": ["MID_FGA_SHARE", "PAINT_FGA_SHARE", "PCT_PTS_2PT_MR"],
        "skill": ["Mid_Range_FG_PCT", "In_The_Paint_Non_RA_FG_PCT"],
    },
    "Three-Point Shooting": {
        "tendency": ["FG3A_RATE", "FG3A_PER36", "PCT_PTS_3PT"],
        "skill": ["FG3_PCT", "CATCH_SHOOT_FG3_PCT", "PULL_UP_FG3_PCT", "Corner_3_FG_PCT",
                  "Above_the_Break_3_FG_PCT", "FT_PCT"],
    },
    "Self-Creation": {
        "tendency": ["PCT_UAST_FGM", "PULL_UP_FGA_SHARE", "DRIBB_3P_FGA_FREQ", "TOUCH_6P_FGA_FREQUENCY",
                     "AVG_SEC_PER_TOUCH", "AVG_DRIB_PER_TOUCH", "TIME_OF_POSS_PER36",
                     "ISO_FREQ", "PNRBH_FREQ", "DHO_FREQ", "LATE_CLOCK_FGA_FREQ", "TIGHT_FGA_FREQ"],
        "skill": ["PULL_UP_EFG_PCT", "ISO_PPP", "PNRBH_PPP", "DHO_PPP", "DRIBB_3_6_EFG_PCT", "TOUCH_6P_EFG_PCT"],
    },
    "Playmaking": {
        "tendency": ["AST_PCT", "AST_PER36", "POTENTIAL_AST_PER36", "PASSES_MADE_PER36",
                     "SECONDARY_AST_PER36", "FT_AST_PER36", "AST_POINTS_CREATED_PER36", "DRIVE_PASSES_PCT"],
        "skill": ["AST_TO", "AST_TO_PASS_PCT_ADJ", "TM_TOV_PCT", "DRIVE_AST_PCT"],
    },
    "Off-Ball Scoring": {
        "tendency": ["PCT_AST_FGM", "CATCH_SHOOT_FGA_SHARE", "DRIBB_0_FGA_FREQUENCY", "TOUCH_LT_2_FGA_FREQUENCY",
                     "SPOT_FREQ", "OFFSCREEN_FREQ", "CUT_FREQ", "CORNER_SHARE_OF_3S",
                     "DIST_MILES_OFF_PER36"],
        "skill": ["CATCH_SHOOT_EFG_PCT", "SPOT_PPP", "OFFSCREEN_PPP", "CUT_PPP"],
    },
    "Transition": {
        "tendency": ["TRANS_FREQ", "PCT_PTS_FB", "PTS_FB_PER36", "EARLY_CLOCK_FGA_FREQ", "AVG_SPEED_OFF"],
        "skill": ["TRANS_PPP"],
    },
    "Interior & Screening": {
        "tendency": ["POST_FREQ", "POST_TOUCHES_PER36", "ELBOW_TOUCHES_PER36", "RM_FREQ",
                     "PUTBACK_FREQ", "SCREEN_ASSISTS_PER36"],
        "skill": ["POST_PPP", "RM_PPP", "PUTBACK_PPP", "PTS_PER_POST_TOUCH", "PAINT_TOUCH_FG_PCT"],
    },
    "Rebounding": {
        "tendency": ["OREB_PCT", "DREB_PCT", "REB_CHANCES_PER36", "BOX_OUTS_PER36", "REB_CONTEST_PCT"],
        "skill": ["REB_CHANCE_PCT_ADJ", "OREB_CHANCE_PCT_ADJ", "DREB_CHANCE_PCT_ADJ"],
    },
    "Rim Protection": {
        "tendency": ["DEF_RIM_FGA_PER36", "BLK_PER36", "CONTESTED_SHOTS_2PT_PER36", "DEFLT6_FREQ"],
        "skill": ["DEF_RIM_FG_PCT", "DEFLT6_PLUSMINUS", "DEFLT10_PLUSMINUS"],
    },
    "Perimeter Defense": {
        "tendency": ["STL_PER36", "DEFLECTIONS_PER36", "CONTESTED_SHOTS_3PT_PER36", "DEF3_FREQ",
                     "DEFGT15_FREQ", "DIST_MILES_DEF_PER36"],
        "skill": ["DEF3_PLUSMINUS", "DEFGT15_PLUSMINUS", "PCT_PLUSMINUS", "PF_PER36"],
    },
    "Hustle": {
        "tendency": ["LOOSE_BALLS_RECOVERED_PER36", "CHARGES_DRAWN_PER36", "DEF_BOXOUTS_PER36",
                     "DIST_MILES_PER36", "AVG_SPEED"],
        "skill": [],
    },
}

# A skill stat is ranked only above this much per-game volume: {stat: (volume column, minimum)}
SKILL_VOLUME = {
    "Restricted_Area_FG_PCT": ("Restricted_Area_FGA", 1.0), "DRIVE_FG_PCT": ("DRIVE_FGA", 1.0),
    "DRIVE_PTS_PCT": ("DRIVES", 2.0), "DRIVE_AST_PCT": ("DRIVES", 2.0),
    "Mid_Range_FG_PCT": ("Mid_Range_FGA", 0.75), "In_The_Paint_Non_RA_FG_PCT": ("In_The_Paint_Non_RA_FGA", 0.75),
    "FG3_PCT": ("FG3A", 1.5), "CATCH_SHOOT_FG3_PCT": ("CATCH_SHOOT_FG3A", 1.0),
    "PULL_UP_FG3_PCT": ("PULL_UP_FG3A", 0.75), "Corner_3_FG_PCT": ("Corner_3_FGA", 0.5),
    "Above_the_Break_3_FG_PCT": ("Above_the_Break_3_FGA", 1.0), "FT_PCT": ("FTA", 1.0),
    "PULL_UP_EFG_PCT": ("PULL_UP_FGA", 1.0), "DRIBB_3_6_EFG_PCT": ("DRIBB_3_6_FGA", 0.75),
    "TOUCH_6P_EFG_PCT": ("TOUCH_6P_FGA", 0.75), "CATCH_SHOOT_EFG_PCT": ("CATCH_SHOOT_FGA", 1.0),
    "AST_TO": ("AST", 1.0), "AST_TO_PASS_PCT_ADJ": ("PASSES_MADE", 10.0),
    "PTS_PER_POST_TOUCH": ("POST_TOUCHES", 1.0), "PAINT_TOUCH_FG_PCT": ("PAINT_TOUCH_FGA", 1.0),
    "REB_CHANCE_PCT_ADJ": ("REB_CHANCES", 3.0), "OREB_CHANCE_PCT_ADJ": ("OREB_CHANCES", 1.0),
    "DREB_CHANCE_PCT_ADJ": ("DREB_CHANCES", 2.0),
    "DEF_RIM_FG_PCT": ("DEF_RIM_FGA", 2.0), "DEFLT6_PLUSMINUS": ("DEFLT6_FGA_LT_06", 1.5),
    "DEFLT10_PLUSMINUS": ("DEFLT10_FGA_LT_10", 2.0), "DEF3_PLUSMINUS": ("DEF3_FG3A", 1.5),
    "DEFGT15_PLUSMINUS": ("DEFGT15_FGA_GT_15", 2.0), "PCT_PLUSMINUS": ("D_FGA", 4.0),
}

PLAY_TYPE_PREFIXES = ["ISO", "PNRBH", "DHO", "CUT", "POST", "OFFSCREEN", "RM", "TRANS", "SPOT", "PUTBACK"]

LABELS = {
    "USG_PCT": "Usage rate", "PCT_PTS": "Share of team points (on court)", "TS_PCT": "True shooting %",
    "EFG_PCT": "Effective FG%", "PTS_PER_TOUCH": "Points per touch",
    "RIM_FGA_SHARE": "Share of shots at the rim", "FT_RATE": "Free-throw rate (FTA/FGA)",
    "PFD_PER36": "Fouls drawn per 36", "PCT_PTS_PAINT": "Share of points in the paint",
    "Restricted_Area_FG_PCT": "FG% at the rim", "DRIVE_FG_PCT": "FG% on drives",
    "DRIVE_PTS_PCT": "Points per drive rate",
    "MID_FGA_SHARE": "Share of shots from mid-range", "PAINT_FGA_SHARE": "Share of shots in the paint (non-rim)",
    "PCT_PTS_2PT_MR": "Share of points from mid-range", "Mid_Range_FG_PCT": "Mid-range FG%",
    "In_The_Paint_Non_RA_FG_PCT": "Paint (non-rim) FG%",
    "FG3A_RATE": "3-point attempt rate", "PCT_PTS_3PT": "Share of points from 3",
    "FG3_PCT": "3-point %", "CATCH_SHOOT_FG3_PCT": "Catch-and-shoot 3P%", "PULL_UP_FG3_PCT": "Pull-up 3P%",
    "Corner_3_FG_PCT": "Corner 3P%", "Above_the_Break_3_FG_PCT": "Above-the-break 3P%", "FT_PCT": "Free-throw %",
    "PCT_UAST_FGM": "Share of makes unassisted", "PULL_UP_FGA_SHARE": "Share of shots off the dribble (pull-up)",
    "DRIBB_3P_FGA_FREQ": "Share of shots after 3+ dribbles", "TOUCH_6P_FGA_FREQUENCY": "Share of shots after holding 6+ sec",
    "AVG_SEC_PER_TOUCH": "Seconds per touch", "AVG_DRIB_PER_TOUCH": "Dribbles per touch",
    "TIME_OF_POSS_PER36": "Time of possession per 36", "LATE_CLOCK_FGA_FREQ": "Share of shots late in the clock (<7s)",
    "TIGHT_FGA_FREQ": "Share of shots tightly contested (<4 ft)", "PULL_UP_EFG_PCT": "Pull-up eFG%",
    "DRIBB_3_6_EFG_PCT": "eFG% after 3-6 dribbles", "TOUCH_6P_EFG_PCT": "eFG% after holding 6+ sec",
    "AST_PCT": "Assist rate", "DRIVE_PASSES_PCT": "Pass rate on drives", "AST_TO": "Assist-to-turnover",
    "AST_TO_PASS_PCT_ADJ": "Assists per pass (adjusted)", "TM_TOV_PCT": "Turnover rate",
    "DRIVE_AST_PCT": "Assist rate on drives",
    "PCT_AST_FGM": "Share of makes assisted", "CATCH_SHOOT_FGA_SHARE": "Share of shots catch-and-shoot",
    "DRIBB_0_FGA_FREQUENCY": "Share of shots with no dribble", "TOUCH_LT_2_FGA_FREQUENCY": "Share of shots within 2 sec of the catch",
    "CORNER_SHARE_OF_3S": "Share of 3s from the corner", "CATCH_SHOOT_EFG_PCT": "Catch-and-shoot eFG%",
    "PCT_PTS_FB": "Share of points on the fast break", "EARLY_CLOCK_FGA_FREQ": "Share of shots early in the clock (18s+)",
    "AVG_SPEED_OFF": "Average speed on offense", "PTS_PER_POST_TOUCH": "Points per post touch",
    "PAINT_TOUCH_FG_PCT": "FG% on paint touches",
    "OREB_PCT": "Offensive rebound rate", "DREB_PCT": "Defensive rebound rate",
    "REB_CONTEST_PCT": "Share of rebounds contested", "REB_CHANCE_PCT_ADJ": "Rebound chances converted (adjusted)",
    "OREB_CHANCE_PCT_ADJ": "Off. rebound chances converted", "DREB_CHANCE_PCT_ADJ": "Def. rebound chances converted",
    "DEFLT6_FREQ": "Share of shots defended inside 6 ft", "DEF_RIM_FG_PCT": "Opponent FG% at the rim",
    "DEFLT6_PLUSMINUS": "Opp. FG% vs expected, inside 6 ft", "DEFLT10_PLUSMINUS": "Opp. FG% vs expected, inside 10 ft",
    "DEF3_FREQ": "Share of shots defended that are 3s", "DEFGT15_FREQ": "Share of shots defended beyond 15 ft",
    "DEF3_PLUSMINUS": "Opp. 3P% vs expected", "DEFGT15_PLUSMINUS": "Opp. FG% vs expected, beyond 15 ft",
    "PCT_PLUSMINUS": "Opp. FG% vs expected, all shots", "AVG_SPEED": "Average speed",
    "ISO_FREQ": "Isolation frequency", "PNRBH_FREQ": "P&R ball-handler frequency", "DHO_FREQ": "Dribble-handoff frequency",
    "CUT_FREQ": "Cut frequency", "POST_FREQ": "Post-up frequency", "OFFSCREEN_FREQ": "Off-screen frequency",
    "RM_FREQ": "Roll-man frequency", "TRANS_FREQ": "Transition frequency", "SPOT_FREQ": "Spot-up frequency",
    "PUTBACK_FREQ": "Putback frequency",
    "ISO_PPP": "Isolation points per possession", "PNRBH_PPP": "P&R ball-handler PPP", "DHO_PPP": "Dribble-handoff PPP",
    "CUT_PPP": "Cut PPP", "POST_PPP": "Post-up PPP", "OFFSCREEN_PPP": "Off-screen PPP", "RM_PPP": "Roll-man PPP",
    "TRANS_PPP": "Transition PPP", "SPOT_PPP": "Spot-up PPP", "PUTBACK_PPP": "Putback PPP",
}

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, MUTED, INK = "#e4e3de", "#8a897f", "#0b0b0b"


PER36_NAMES = {
    "PTS": "Points", "FGA": "Shot attempts", "TOUCHES": "Touches", "DRIVES": "Drives",
    "PAINT_TOUCHES": "Paint touches", "PFD": "Fouls drawn", "FG3A": "3-point attempts",
    "TIME_OF_POSS": "Time of possession", "AST": "Assists", "POTENTIAL_AST": "Potential assists",
    "PASSES_MADE": "Passes", "SECONDARY_AST": "Secondary (hockey) assists", "FT_AST": "Free-throw assists",
    "AST_POINTS_CREATED": "Points created by assists", "DIST_MILES_OFF": "Distance run on offense",
    "PTS_FB": "Fast-break points", "POST_TOUCHES": "Post touches", "ELBOW_TOUCHES": "Elbow touches",
    "SCREEN_ASSISTS": "Screen assists", "REB_CHANCES": "Rebound chances", "BOX_OUTS": "Box-outs",
    "DEF_RIM_FGA": "Shots defended at the rim", "BLK": "Blocks", "CONTESTED_SHOTS_2PT": "2-point shots contested",
    "STL": "Steals", "DEFLECTIONS": "Deflections", "CONTESTED_SHOTS_3PT": "3-point shots contested",
    "DIST_MILES_DEF": "Distance run on defense", "PF": "Fouls committed",
    "LOOSE_BALLS_RECOVERED": "Loose balls recovered", "CHARGES_DRAWN": "Charges drawn",
    "DEF_BOXOUTS": "Defensive box-outs", "DIST_MILES": "Distance run",
}


def label(stat):
    if stat in LABELS:
        return LABELS[stat]
    if stat.endswith("_PER36"):
        base = stat[:-6]
        return PER36_NAMES.get(base, base.replace("_", " ").capitalize()) + " per 36"
    return stat.replace("_", " ").capitalize()


# =============================================================================
# DERIVED COLUMNS
# =============================================================================

def add_profile_columns(df):
    """Add the rate and share columns the profile uses. Returns a copy.

    The new columns work with the other charts too, e.g.
    plot_interactive_barh(add_profile_columns(df_master), x_col="RIM_FGA_SHARE", ...).
    """
    out = df.copy()

    def num(col):
        return pd.to_numeric(out[col], errors="coerce") if col in out.columns else pd.Series(np.nan, index=out.index)

    def ratio(a, b):
        return a / b.where(b > 0)

    new = {}
    minutes = num("MIN")
    per36 = sorted({s[:-6] for g in PROFILE_GROUPS.values() for kind in g.values()
                    for s in kind if s.endswith("_PER36")})
    for col in per36:
        new[f"{col}_PER36"] = ratio(num(col) * 36, minutes)

    zones = {"RIM": "Restricted_Area_FGA", "PAINT": "In_The_Paint_Non_RA_FGA", "MID": "Mid_Range_FGA",
             "CORNER3": "Corner_3_FGA", "ATB3": "Above_the_Break_3_FGA"}
    zone_total = sum(num(c).fillna(0) for c in zones.values())
    for name, col in zones.items():
        new[f"{name}_FGA_SHARE"] = ratio(num(col), zone_total)
    new["CORNER_SHARE_OF_3S"] = ratio(num("Corner_3_FGA"), num("Corner_3_FGA").fillna(0) + num("Above_the_Break_3_FGA").fillna(0))

    fga = num("FGA")
    new["FT_RATE"] = ratio(num("FTA"), fga)
    new["FG3A_RATE"] = ratio(num("FG3A"), fga)
    new["PULL_UP_FGA_SHARE"] = ratio(num("PULL_UP_FGA"), fga)
    new["CATCH_SHOOT_FGA_SHARE"] = ratio(num("CATCH_SHOOT_FGA"), fga)

    # shot-context buckets come as shares of the player's own attempts
    new["DRIBB_3P_FGA_FREQ"] = num("DRIBB_3_6_FGA_FREQUENCY").fillna(0) + num("DRIBB_7P_FGA_FREQUENCY").fillna(0)
    new["LATE_CLOCK_FGA_FREQ"] = num("SC_7_4_FGA_FREQUENCY").fillna(0) + num("SC_4_0_FGA_FREQUENCY").fillna(0)
    new["EARLY_CLOCK_FGA_FREQ"] = num("SC_24_22_FGA_FREQUENCY").fillna(0) + num("SC_22_18_FGA_FREQUENCY").fillna(0)
    new["TIGHT_FGA_FREQ"] = num("CLOSESTDEF_0_2_FGA_FREQUENCY").fillna(0) + num("CLOSESTDEF_2_4_FGA_FREQUENCY").fillna(0)
    no_shots = fga.isna() | (fga <= 0)
    for col in ("DRIBB_3P_FGA_FREQ", "LATE_CLOCK_FGA_FREQ", "EARLY_CLOCK_FGA_FREQ", "TIGHT_FGA_FREQ"):
        new[col] = new[col].mask(no_shots)

    # Synergy lists a play type only above a minimum number of possessions
    for prefix in PLAY_TYPE_PREFIXES:
        new[f"{prefix}_FREQ"] = num(f"{prefix}_POSS_PCT").fillna(0)

    return pd.concat([out.drop(columns=[c for c in new if c in out.columns]), pd.DataFrame(new)], axis=1)


# =============================================================================
# BUILD
# =============================================================================

class Profiles:
    """What build_profiles returns. All frames are indexed by player name.

    pool      PLAYER_NAME, team, position, GP, MIN for everyone ranked
    values    the raw stat values
    pct       each stat as a percentile (skill stats: 100 = best)
    tendency  one column per group — how much he does it
    skill     one column per group — how well
    """
    def __init__(self, season, season_type, groups, pool, values, pct, tendency, skill, min_gp, min_min):
        self.season, self.season_type, self.groups = season, season_type, groups
        self.pool, self.values, self.pct, self.tendency, self.skill = pool, values, pct, tendency, skill
        self.min_gp, self.min_min = min_gp, min_min

    def __repr__(self):
        return (f"Profiles({self.season} {self.season_type}: {len(self.pool)} players with "
                f"{self.min_gp}+ games and {self.min_min}+ minutes, {len(self.groups)} groups, "
                f"{self.pct.shape[1]} stats)")


def build_profiles(df_master, season, season_type="Regular Season", min_gp=None, min_min=15.0,
                   groups=PROFILE_GROUPS):
    """Rank every qualified player on every profile stat for one season.

    min_gp defaults to 30% of the most games anyone has played, so it works in
    November as well as April (25 of 82 at season's end).
    """
    df = df_master.reset_index()
    df = df[(df["SEASON"] == season) & (df["SEASON_TYPE"] == season_type)].copy()
    if df.empty:
        raise ValueError(f"no rows for {season} {season_type}")
    gp, minutes = pd.to_numeric(df["GP"], errors="coerce"), pd.to_numeric(df["MIN"], errors="coerce")
    if min_gp is None:
        min_gp = max(1, int(round(0.3 * gp.max())))
    df = add_profile_columns(df[(gp >= min_gp) & (minutes >= min_min)])
    if len(df) < 20:
        raise ValueError(f"only {len(df)} players have {min_gp}+ games and {min_min}+ minutes")

    # one row per name: a second player with the same name gets his team appended
    dup = df["PLAYER_NAME"].duplicated(keep=False)
    df["PLAYER_NAME"] = df["PLAYER_NAME"].where(~dup, df["PLAYER_NAME"] + " (" + df["TEAM_ABBREVIATION"] + ")")
    df = df.drop_duplicates("PLAYER_NAME").set_index("PLAYER_NAME")

    stats = [s for g in groups.values() for kind in ("tendency", "skill") for s in g[kind]]
    missing = sorted({s for s in stats if s not in df.columns})
    if missing:
        print(f"⚠️ not in the master, skipped: {missing}")
    stats = list(dict.fromkeys(s for s in stats if s in df.columns))
    values = df[stats].apply(pd.to_numeric, errors="coerce")

    skill_stats = {s for g in groups.values() for s in g["skill"]}
    for stat, (volume_col, floor) in SKILL_VOLUME.items():
        if stat in values.columns and stat in skill_stats and volume_col in df.columns:
            values[stat] = values[stat].where(pd.to_numeric(df[volume_col], errors="coerce") >= floor)

    # best = 100 whichever way the stat runs; one concat, not a column at a time
    pct = pd.DataFrame({
        stat: values[stat].rank(pct=True, ascending=not (stat in skill_stats and is_lower_better(stat))) * 100
        for stat in stats
    }, index=values.index)

    def group_scores(kind):
        out = {}
        for group, spec in groups.items():
            cols = [s for s in spec[kind] if s in pct.columns]
            if not cols:
                out[group] = pd.Series(np.nan, index=pct.index)
                continue
            # averaged over whichever stats he has; SKILL_VOLUME has already
            # removed the ones without enough attempts behind them
            out[group] = pct[cols].mean(axis=1)
        return pd.DataFrame(out)

    pool_cols = [c for c in ("TEAM_ABBREVIATION", "PLAYER_POSITION", "GP", "MIN") if c in df.columns]
    return Profiles(season, season_type, groups, df[pool_cols], values, pct,
                    group_scores("tendency"), group_scores("skill"), min_gp, min_min)


def _find(profiles, name):
    names = profiles.pool.index
    if name in names:
        return name
    hits = [n for n in names if name.lower() in n.lower()]
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise KeyError(f"'{name}' matches several players: {hits[:8]}")
    raise KeyError(f"'{name}' is not in the {profiles.season} pool "
                   f"({profiles.min_gp}+ games, {profiles.min_min}+ minutes)")


# =============================================================================
# READ A PROFILE
# =============================================================================

def player_profile(profiles, name, verbose=True):
    """One row per group: tendency and skill percentiles, most-used style first."""
    name = _find(profiles, name)
    out = pd.DataFrame({"TENDENCY": profiles.tendency.loc[name], "SKILL": profiles.skill.loc[name]}).round(0)
    out = out.sort_values("TENDENCY", ascending=False)
    if verbose:
        t = out["TENDENCY"].dropna()
        leans = ", ".join(f"{g} ({v:.0f})" for g, v in t[t >= 70].items()) or "nothing above the 70th percentile"
        rarely = ", ".join(f"{g} ({v:.0f})" for g, v in t[t <= 30][::-1].items()) or "nothing below the 30th"
        info = profiles.pool.loc[name]
        print(f"{name} — {info.get('TEAM_ABBREVIATION', '')} · {profiles.season} · "
              f"{info['GP']:.0f} GP, {info['MIN']:.1f} min")
        print(f"  leans on: {leans}")
        print(f"  rarely:   {rarely}")
    return out


def player_detail(profiles, name, group=None):
    """Every stat behind the profile: raw value and percentile. `group` narrows it to one."""
    name = _find(profiles, name)
    rows = []
    for g, spec in profiles.groups.items():
        if group is not None and g != group:
            continue
        for kind in ("tendency", "skill"):
            for stat in spec[kind]:
                if stat in profiles.pct.columns:
                    rows.append((g, kind, label(stat), stat, profiles.values.at[name, stat],
                                 profiles.pct.at[name, stat]))
    out = pd.DataFrame(rows, columns=["GROUP", "KIND", "STAT", "COLUMN", "VALUE", "PERCENTILE"])
    out["PERCENTILE"] = out["PERCENTILE"].round(0)
    return out


def standouts(profiles, name, n=6):
    """The tendencies that most set a player apart: his n highest and n lowest."""
    detail = player_detail(profiles, name)
    t = detail[detail["KIND"] == "tendency"].dropna(subset=["PERCENTILE"]).sort_values("PERCENTILE", ascending=False)
    return pd.concat([t.head(n).assign(SIDE="does a lot"), t.tail(n)[::-1].assign(SIDE="does little")])[
        ["SIDE", "GROUP", "STAT", "VALUE", "PERCENTILE"]].reset_index(drop=True)


def style_leaders(profiles, group, kind="tendency", n=15):
    """Top n players in one group, on tendency (default) or skill."""
    scores = getattr(profiles, kind)[group].dropna().sort_values(ascending=False).head(n)
    out = profiles.pool.loc[scores.index, [c for c in ("TEAM_ABBREVIATION", "PLAYER_POSITION", "MIN") if c in profiles.pool.columns]].copy()
    out[kind.upper()] = scores.round(0)
    other = "skill" if kind == "tendency" else "tendency"
    out[other.upper()] = getattr(profiles, other).loc[scores.index, group].round(0)
    return out


def similar_players(profiles, name, n=10):
    """Players with the closest style: distance across every tendency percentile.

    Skill is left out on purpose — this finds players who play the same way,
    not players who are equally good. Each group counts the same however many
    stats it holds; a missing percentile sits at the middle (50).
    """
    name = _find(profiles, name)
    cols, weights = [], []
    for spec in profiles.groups.values():
        present = [s for s in spec["tendency"] if s in profiles.pct.columns]
        cols += present
        weights += [1 / len(present)] * len(present)
    x = profiles.pct[cols].fillna(50).to_numpy()
    w = np.array(weights) / np.sum(weights)
    dist = np.sqrt((((x - x[profiles.pct.index.get_loc(name)]) ** 2) * w).sum(axis=1))
    out = profiles.pool[[c for c in ("TEAM_ABBREVIATION", "PLAYER_POSITION", "MIN") if c in profiles.pool.columns]].copy()
    out["DISTANCE"] = dist.round(1)
    out = out.drop(index=name).sort_values("DISTANCE").head(n)
    mine = profiles.tendency.loc[name]
    # the groups where the match is closest, as a quick sanity read
    out["SHARED_STRENGTHS"] = [
        ", ".join(g for g in mine[mine >= 70].index if profiles.tendency.at[other, g] >= 70) for other in out.index]
    return out


# =============================================================================
# CHARTS
# =============================================================================

def _style_axes(ax, groups):
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ax.invert_yaxis()
    ax.set_xlim(-2, 102)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.axvline(50, color=MUTED, lw=1, ls=(0, (2, 3)), zorder=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED)


def plot_player_profile(profiles, name, figsize=(9, 6.5)):
    """Two dots per group: how much he does it (blue) and how well (orange)."""
    name = _find(profiles, name)
    groups = list(profiles.groups)
    tendency, skill = profiles.tendency.loc[name, groups], profiles.skill.loc[name, groups]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(groups))
    both = tendency.notna().to_numpy() & skill.notna().to_numpy()
    ax.hlines(y[both], tendency[both], skill[both], color=GRID, lw=3, zorder=1)
    ax.scatter(tendency, y, s=110, color=BLUE, zorder=3, label="How much he does it (tendency)")
    ax.scatter(skill, y, s=110, color=ORANGE, zorder=3, edgecolor="white", linewidth=1.5,
               label="How well (skill)")
    for i, v in enumerate(tendency):
        if pd.notna(v):
            ax.annotate(f"{v:.0f}", (v, i), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
    _style_axes(ax, groups)
    info = profiles.pool.loc[name]
    ax.set_title(f"{name} — style profile, {profiles.season}", loc="left", fontsize=13, fontweight="bold", pad=28)
    ax.text(0, 1.045, f"Percentile among {len(profiles.pool)} players with {profiles.min_gp}+ games and "
                      f"{profiles.min_min:g}+ minutes  ·  {info['GP']:.0f} GP, {info['MIN']:.1f} min",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    ax.set_xlabel("Percentile (50 = typical rotation player)", fontsize=9)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()


def compare_players(profiles, names, kind="tendency", figsize=(9, 6.5)):
    """One dot per player per group. Up to four players; tendency unless kind="skill"."""
    names = [_find(profiles, n) for n in names][:4]
    groups = list(profiles.groups)
    scores = getattr(profiles, kind)

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(groups))
    lo = scores.loc[names, groups].min()
    hi = scores.loc[names, groups].max()
    ok = lo.notna().to_numpy()
    ax.hlines(y[ok], lo[ok], hi[ok], color=GRID, lw=3, zorder=1)
    for i, (name, color) in enumerate(zip(names, (BLUE, ORANGE, AQUA, YELLOW))):
        # a small vertical step per player, so two players on the same percentile both show
        ax.scatter(scores.loc[name, groups], y + (i - (len(names) - 1) / 2) * 0.16, s=110, color=color,
                   zorder=3, edgecolor="white", linewidth=1.5, label=name)
    _style_axes(ax, groups)
    what = "how much each does it" if kind == "tendency" else "how well each does it"
    ax.set_title(f"Style comparison — {what}, {profiles.season}", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Percentile (50 = typical rotation player)", fontsize=9)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(names), fontsize=9)
    plt.tight_layout()
    plt.show()
