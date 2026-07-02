"""
feature_glossary.py — compositional decoder for NBA model feature names.

Drop-in replacement for the DEFS-only lookup in app.py. Instead of relying on a
hand-written dict that only covers ~35 of ~245 surfacing features (everything
else falling back to "A tracked stat in the feature set"), this parses a feature
name into its parts and composes a plain-English definition:

    (play-type / context prefix)(bucket range)(base stat)(season + lag suffix)

    e.g. CLOSESTDEF_6P_FG3_PCT_rs
       -> "3-pt field goal % on shots where the closest defender was 6+ ft away
           (wide open), regular season."

Your curated DEFS entries still win when present (they read better than the
composed version), so this only *fills the gaps* rather than replacing your
good hand-written definitions.
"""

# ---------------------------------------------------------------------------
# 1. PLAY-TYPE / CONTEXT PREFIXES  (mirrors PREFIXES in nba_viz_utils.py)
#    raw prefix -> (friendly name, short gloss of what the context means)
# ---------------------------------------------------------------------------
PREFIX_DEFS = {
    "PNRBH_":      ("Pick & roll (ball handler)", "as the ball handler in pick-and-roll"),
    "RM_":         ("Pick & roll (roll man)", "as the roll man in pick-and-roll"),
    "ISO_":        ("Isolation", "in isolation possessions"),
    "TRANS_":      ("Transition", "in transition / fast-break possessions"),
    "POST_":       ("Post-up", "on post-up possessions"),
    "SPOT_":       ("Spot-up", "on spot-up possessions"),
    "SPOTUP_":     ("Spot-up", "on spot-up possessions"),
    "DRIVE_":      ("Drives", "on drives to the basket"),
    "DHO_":        ("Hand-off", "on dribble hand-off possessions"),
    "CUT_":        ("Cuts", "on cut possessions"),
    "ELBOW_":      ("Elbow touch", "on touches from the elbow"),
    "POSTT_":      ("Post touch", "on touches from the post"),
    "PAINT_":      ("Paint touch", "on touches in the paint"),
    "SHOT_":       ("Shooting", "on general shot attempts"),
    "SC_":         ("Shot clock", "by shot-clock situation"),
    "DRIBB_":      ("Dribbles", "by number of dribbles before shooting"),
    "TOUCH_":      ("Touch time", "by how long the ball was held before shooting"),
    "CLOSESTDEF_": ("Closest defender", "by how far away the nearest defender was"),
    "OFFSCREEN_":  ("Off screen", "on off-screen possessions"),
    "PUTBACK_":    ("Put-back", "on offensive-rebound put-back possessions"),
}

# ---------------------------------------------------------------------------
# 2. BUCKET RANGES  (mirrors BUCKET_CONFIG in nba_viz_utils.py, plus the
#    cleaned forms clean_bucket() produces, e.g. '4_6', '6P', '0_2', 'LT_2')
#    token -> human phrase
# ---------------------------------------------------------------------------
BUCKET_DEFS = {
    # ClosestDefender (feet)
    "0_2":  "closest defender 0–2 ft away (very tight)",
    "2_4":  "closest defender 2–4 ft away (tight)",
    "4_6":  "closest defender 4–6 ft away (open)",
    "6P":   "closest defender 6+ ft away (wide open)",
    # Dribble counts
    "0":    "off zero dribbles (catch-and-shoot)",
    "1":    "off one dribble",
    "2":    "off two dribbles",
    "3_6":  "off 3–6 dribbles",
    "7P":   "off 7+ dribbles",
    # Shot clock (seconds remaining)
    "24_22": "with 24–22 sec on the shot clock (very early)",
    "22_18": "with 22–18 sec on the shot clock (early)",
    "18_15": "with 18–15 sec on the shot clock (early)",
    "15_7":  "with 15–7 sec on the shot clock (average)",
    "7_4":   "with 7–4 sec on the shot clock (late)",
    "4_0":   "with 4–0 sec on the shot clock (very late)",
    # Touch time (seconds holding the ball)
    "LT_2": "holding the ball <2 sec (quick)",
    "LT2":  "holding the ball <2 sec (quick)",
    "6P_":  "holding the ball 6+ sec (long)",
}

# ---------------------------------------------------------------------------
# 2b. COMPOUND TOUCH CONTEXTS — checked before the plain prefixes above.
#     These generate names like PAINT_TOUCH_PASSES_PCT where the context token
#     is longer than the bare PAINT_ prefix.
# ---------------------------------------------------------------------------
TOUCH_CONTEXT_DEFS = {
    "PAINT_TOUCH_": "on touches in the paint",
    "POST_TOUCH_":  "on touches from the post",
    "ELBOW_TOUCH_": "on touches from the elbow",
    "FRONT_CT_TOUCHES": "front-court touches — a usage / on-ball role proxy",
}

# Stat tokens specific to touch contexts
TOUCH_STAT_DEFS = [
    ("PASSES_PCT", "share of those touches a player passes out of"),
    ("AST_PCT",    "share of those touches that lead to an assist"),
    ("TOV_PCT",    "turnover rate on those touches"),
    ("SF_PCT",     "shooting-foul-drawn rate on those touches"),
    ("FG_PCT",     "field goal % out of those touches"),
    ("PTS",        "points generated from those touches"),
    ("TOUCHES",    "number of such touches"),
]


def _try_touch_context(core, season_phrase, time_phrase):
    """Handle PAINT_TOUCH_ / POST_TOUCH_ / ELBOW_TOUCH_ / FRONT_CT_TOUCHES."""
    # FRONT_CT_TOUCHES is a standalone stat, not a context+stat
    if core.startswith("FRONT_CT_TOUCHES"):
        return "Front-court touches — a usage / on-ball role proxy" + season_phrase + time_phrase + "."
    for ctx, gloss in TOUCH_CONTEXT_DEFS.items():
        if ctx.endswith("_") and core.startswith(ctx):
            rem = core[len(ctx):]
            for tok, phrase in TOUCH_STAT_DEFS:
                if rem == tok or rem.startswith(tok):
                    s = f"{phrase.capitalize()}, {gloss}"
                    return s + season_phrase + time_phrase + "."
            # context matched but stat unknown
            return f"A measurement {gloss}" + season_phrase + time_phrase + "."
    return None
#    Order matters: check longer/compound tokens before short ones.
#    token -> human phrase
# ---------------------------------------------------------------------------
BASE_STAT_DEFS = [
    ("FG3A_FREQUENCY", "3-pt attempt rate (share of shots that are 3s)"),
    ("FG2A_FREQUENCY", "2-pt attempt rate (share of shots that are 2s)"),
    ("FGA_FREQUENCY",  "shot-attempt frequency"),
    ("FG3_PCT",        "3-pt field goal %"),
    ("FG2_PCT",        "2-pt field goal %"),
    ("FG_PCT",         "field goal %"),
    ("FG3M",           "3-pt field goals made"),
    ("FG3A",           "3-pt field goals attempted"),
    ("FG2M",           "2-pt field goals made"),
    ("FG2A",           "2-pt field goals attempted"),
    ("FGM",            "field goals made"),
    ("FGA",            "field goals attempted"),
    ("FTM",            "free throws made"),
    ("FTA",            "free throws attempted"),
    ("FT_PCT",         "free throw %"),
    ("EFG_PCT",        "effective field goal %"),
    ("TS_PCT",         "true shooting %"),
    ("POSS_PCT",       "share of offensive possessions"),
    ("POSS",           "possessions"),
    ("FREQUENCY",      "frequency (share of possessions)"),
    ("PPP",            "points per possession"),
    ("PTS",            "points"),
    ("AST_PCT",        "assist share"),
    ("AST",            "assists"),
    ("TOV_POSS_PCT",   "turnover rate on those possessions"),
    ("TOV",            "turnovers"),
    ("SF_POSS_PCT",    "shooting-foul-drawn rate on those possessions"),
    ("PASSES_PCT",     "pass-out rate on those touches"),
    ("PASSES",         "passes"),
    ("REB",            "rebounds"),
    ("OREB",           "offensive rebounds"),
    ("DREB",           "defensive rebounds"),
    ("BLK",            "blocks"),
    ("STL",            "steals"),
    ("PCT",            "percentage"),
    ("FREQ",           "frequency"),
    ("PF",             "personal fouls"),
    ("PLUSMINUS",      "plus-minus"),
]

# ---------------------------------------------------------------------------
# 3b. EXACT-MATCH CORE STATS — checked against the time/season-stripped core.
#     Covers the tail the compositional parser doesn't reach: defensive
#     tracking, hustle, clutch, team ratings, speed, and plain box-score stats.
#     Keys are the stripped core (no _rs/_po/_lag).
# ---------------------------------------------------------------------------
CORE_STAT_DEFS = {
    # --- star credential / eligibility / draft ---
    "ALLNBA_WT_EVER":    "Weighted career All-NBA selections (1st=3, 2nd=2, 3rd=1), prior seasons only — the star credential",
    "ALLNBA_PRIOR_EVER": "Career All-NBA selections before this season — the star credential",
    "ALLNBA_PRIOR3":     "All-NBA selections in the prior three seasons — a recent-star signal",
    "ALLNBA_WT3":        "Weighted All-NBA selections over the prior three seasons (1st=3, 2nd=2, 3rd=1) — recent-star weight",
    "MAX_PCT_ELIGIBLE":  "The CBA salary ceiling a player qualifies for by experience (25 / 30 / 35% of cap)",
    "DRAFT_POSITION":    "Where a player was drafted — pedigree the market keeps paying for years later",
    "EXPERIENCE":        "Seasons in the league — one of the single biggest salary drivers",

    # --- core role / availability ---
    "MIN":        "Per-game minutes — the model's proxy for role and trust",
    "TOTAL_MIN":  "Total season minutes (per-game × games) — a durability / availability signal",
    "GP":         "Games played — availability",
    "L":          "Team losses in the player's games — a team-context signal",
    "TOUCHES":    "Total times a player touches the ball — overall involvement",
    "FT_AST":     "Passes that lead to a teammate's free throws",
    "DD2":        "Double-doubles on the season",
    "USG_PCT":    "Usage rate — share of team possessions a player ends",

    # --- impact / team ratings ---
    "PIE":         "Player Impact Estimate — a box-score composite of a player's share of the game's positive events",
    "OFF_RATING":  "Team points scored per 100 possessions while the player is on the floor",
    "DEF_RATING":  "Team points allowed per 100 possessions while the player is on the floor (lower = better)",
    "NET_RATING":  "Team point differential per 100 possessions while on the floor",
    "PLUS_MINUS":  "On-court point differential",
    "E_PACE":      "Estimated possessions per 48 minutes while on the floor (team tempo)",

    # --- speed / movement (SportVU) ---
    "AVG_SPEED":     "Average movement speed while on the floor (mph)",
    "AVG_SPEED_OFF": "Average speed on offense (mph)",
    "AVG_SPEED_DEF": "Average speed on defense (mph)",

    # --- shooting: catch-and-shoot ---
    "CATCH_SHOOT_FG_PCT":  "Field goal % on catch-and-shoot attempts (no dribbles)",
    "CATCH_SHOOT_EFG_PCT": "Effective FG% on catch-and-shoot attempts (weights 3s)",

    # --- hustle stats ---
    "LOOSE_BALLS_RECOVERED":     "Loose balls recovered — a hustle stat",
    "OFF_LOOSE_BALLS_RECOVERED": "Loose balls recovered on offense — a hustle stat",
    "DEF_BOXOUTS":               "Defensive box-outs — sealing an opponent off the glass",
    "OFF_BOXOUTS":               "Offensive box-outs — sealing for an offensive rebound",
    "SCREEN_ASSISTS":            "Screens that directly lead to a teammate's made basket",
    "DEFLECTIONS":               "Times a player gets a hand on the ball on defense",
    "CONTESTED_SHOTS":           "Opponent shots a player contested",
    "CONTESTED_SHOTS_3PT":       "Opponent 3-pt shots a player contested",

    # --- defensive tracking: opponent shooting when this player defends ---
    "D_FGM":            "Field goals made by opponents this player defended",
    "DEF_RIM_FGM":      "Made field goals allowed at the rim when this player defends",
    "DEF2_FREQ":        "How often this player defends 2-pt attempts",
    "DEF2_NS_FG2_PCT":  "Opponent 2-pt % on clean attempts a player defended (lower = better interior deterrence)",
    "DEF3_FG3_PCT":     "Opponent 3-pt % on attempts this player defended (lower = better perimeter defense)",
    "DEF3_NS_FG3_PCT":  "Opponent 3-pt % on clean attempts this player defended (lower = better)",
    "DEFGT15_FGA_GT_15":   "Opponent attempts a player defends from 15+ ft away",
    "DEFGT15_FGM_GT_15":   "Opponent makes a player defends from 15+ ft away",
    "DEFGT15_NS_GT_15_PCT":"Opponent % on clean 15+ ft shots a player defends (lower = better)",
    "DEFLT10_FREQ":     "How often a player defends shots inside 10 ft",
    "DEFLT10_LT_10_PCT":"Opponent % on shots inside 10 ft a player defends (lower = better)",
    "DEFLT6_FREQ":      "How often a player defends shots inside 6 ft",
    "DEFLT6_LT_06_PCT": "Opponent % on shots inside 6 ft a player defends (lower = rim protection)",
    "DEF_WS":           "Defensive win shares — an estimate of wins from a player's defense",

    # --- opponent points allowed on-court ---
    "OPP_PTS_OFF_TOV":     "Opponent points off turnovers allowed while on the floor (lower = better)",
    "OPP_PTS_2ND_CHANCE":  "Opponent second-chance points allowed while on the floor (lower = better)",
    "OPP_PTS_FB":          "Opponent fast-break points allowed while on the floor (lower = better)",

    # --- shot-location makes/pcts (from the shot-location dashboard) ---
    "In_The_Paint_Non_RA_FGM": "Field goals made in the paint but outside the restricted area (floaters / short mid-range)",
    "Mid_Range_FGM":           "Mid-range field goals made",
    "Mid_Range_FG_PCT":        "Mid-range field goal %",
}

# Clutch stats share a pattern: "<stat>, in clutch minutes (last 5 min, margin ≤5)"
CLUTCH_BASE = {
    "BLK": "blocks", "BLKA": "shots blocked by opponents (blocked attempts)",
    "DEF_RATING": "team defensive rating", "DREB_PCT": "defensive rebound %",
    "E_PACE": "estimated pace", "PACE": "pace", "FG3M": "3-pt field goals made",
    "FTA": "free throws attempted", "PF": "personal fouls", "POSS": "possessions played",
    "REB": "rebounds", "TM_TOV_PCT": "team turnover %",
}


def _try_core_stat(core, season_phrase, time_phrase):
    """Exact-match the stripped core against CORE_STAT_DEFS and clutch stats."""
    # clutch family: CLUTCH_<STAT>
    if core.startswith("CLUTCH_"):
        rest = core[len("CLUTCH_"):]
        phrase = CLUTCH_BASE.get(rest)
        if phrase:
            s = phrase[0].upper() + phrase[1:]
            return f"{s}, in clutch minutes (last 5 min, margin ≤ 5)" + season_phrase + time_phrase + "."
        # unknown clutch stat — still better than "tracked stat"
        return "A clutch-situation stat (last 5 min, margin ≤ 5)" + season_phrase + time_phrase + "."
    if core in CORE_STAT_DEFS:
        return CORE_STAT_DEFS[core] + season_phrase + time_phrase + "."
    return None


# ---------------------------------------------------------------------------
# 4. SEASON / LAG SUFFIXES
# ---------------------------------------------------------------------------
def _strip_time(f):
    """Return (core, time_phrase, season_phrase) after peeling _rs/_po/_lag."""
    time_phrase = ""
    if "_lag1" in f:
        time_phrase = " (from the prior season)"
    elif "_lag2" in f:
        time_phrase = " (from two seasons ago)"

    season_phrase = ""
    if "_rs" in f:
        season_phrase = ", regular season"
    elif "_po" in f:
        season_phrase = ", playoffs"

    core = (f.replace("_rs", "").replace("_po", "")
             .replace("_lag1", "").replace("_lag2", ""))
    return core, time_phrase, season_phrase


# ---------------------------------------------------------------------------
# 5. THE COMPOSITIONAL DECODER
# ---------------------------------------------------------------------------
def compose_definition(feature, curated=None):
    """
    Build a plain-English definition for a raw feature name.

    curated: your existing DEFS dict. If the time/season-stripped core is a key
    there, that definition wins (it's hand-tuned). Otherwise we compose one.
    """
    curated = curated or {}
    core, time_phrase, season_phrase = _strip_time(feature)

    # 1) Curated override on the stripped core (your good hand-written entries)
    if core in curated:
        return curated[core] + time_phrase

    # 2) Curated override on the FULL name (some DEFS keys include _rs etc.)
    if feature in curated:
        return curated[feature]

    # 2b) Compound touch contexts (PAINT_TOUCH_ / POST_TOUCH_ / ELBOW_TOUCH_ / FRONT_CT)
    touch = _try_touch_context(core, season_phrase, time_phrase)
    if touch:
        return touch

    # 2c) Exact-match core stats (defensive tracking, hustle, clutch, ratings, box score)
    core_hit = _try_core_stat(core, season_phrase, time_phrase)
    if core_hit:
        return core_hit

    # 3) Compose from parts
    remainder = core
    context_phrase = ""     # from prefix
    bucket_phrase = ""      # from bucket range

    # -- play-type / context prefix
    for pfx, (name, gloss) in PREFIX_DEFS.items():
        if remainder.startswith(pfx):
            context_phrase = gloss
            remainder = remainder[len(pfx):]
            break

    # -- bucket range (may sit between prefix and base stat)
    #    try the longest matching bucket token at the start of remainder
    for tok in sorted(BUCKET_DEFS, key=len, reverse=True):
        if remainder.startswith(tok + "_") or remainder == tok:
            bucket_phrase = BUCKET_DEFS[tok]
            remainder = remainder[len(tok):].lstrip("_")
            break

    # -- base stat (measurement)
    stat_phrase = ""
    for tok, phrase in BASE_STAT_DEFS:
        if remainder == tok or remainder.startswith(tok):
            stat_phrase = phrase
            break

    # -- assemble
    if stat_phrase:
        parts = [stat_phrase]
        if bucket_phrase:
            # bucket already conveys the context (e.g. defender distance), so it
            # replaces the generic prefix gloss rather than doubling up
            parts.append(bucket_phrase)
        elif context_phrase:
            parts.append(context_phrase)
        sentence = ", ".join(parts)
        sentence = sentence[0].upper() + sentence[1:]
        return sentence + season_phrase + time_phrase + "."

    # -- prefix but unknown stat
    if context_phrase:
        name = next(n for p, (n, g) in PREFIX_DEFS.items() if g == context_phrase)
        return (f"A {context_phrase.replace('as ', '').replace('on ', '').replace('by ', '')} "
                f"measurement" + season_phrase + time_phrase + ".")

    # -- last resort (should be rare now)
    return "A tracked stat in the feature set" + season_phrase + time_phrase + "."


# ---------------------------------------------------------------------------
# 6. QUICK SELF-TEST — run `python feature_glossary.py` to eyeball coverage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pprint import pprint
    samples = [
        "CLOSESTDEF_6P_FG3_PCT_rs",
        "CLOSESTDEF_0_2_FGA_FREQUENCY_rs",
        "PNRBH_PPP_po",
        "RM_POSS_PCT_rs",
        "DRIBB_0_FG3A_FREQUENCY_rs",
        "SC_22_18_FG2A_FREQUENCY_rs",
        "TOUCH_LT2_FG_PCT_rs",
        "ISO_FGA_rs_lag1",
        "PAINT_TOUCH_PASSES_PCT_rs",
        "POST_TOUCH_AST_PCT_po",
        "FRONT_CT_TOUCHES_rs",
        "POST_SF_POSS_PCT_po",
        "SPOT_FG3_PCT_rs",
        "TRANS_PPP_rs",
        "MIN_rs",
        "FGM_rs_lag1",
    ]
    curated = {  # a few of your real DEFS entries, to show override behavior
        "MIN": "Per-game minutes — the model's proxy for role and trust.",
        "CLOSESTDEF_4_6_FGM": "Field goals made with the closest defender 4–6 ft away — open-shot makes.",
    }
    for s in samples:
        print(f"{s:38s} -> {compose_definition(s, curated)}")
