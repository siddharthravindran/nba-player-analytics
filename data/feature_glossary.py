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
