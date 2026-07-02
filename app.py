"""
Market Value — NBA salary vs. the market's price for a player's profile.
Dark broadcast-graphics build.

Reads precomputed artifacts (no model at runtime):
  data/app_meta_v1.parquet  — per player-season: actual %, predicted %, residual $
  data/shap_v1.parquet      — long: per (player-season, feature) SHAP dollar contribution

Run:  python3 -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from feature_glossary import compose_definition

st.set_page_config(page_title="Market Value · NBA", page_icon="●", layout="wide")

SALARY_CAP = {'2015-16':70_000_000,'2016-17':94_143_000,'2017-18':99_093_000,
              '2018-19':101_869_000,'2019-20':109_140_000,'2020-21':109_140_000,
              '2021-22':112_414_000,'2022-23':123_655_000,'2023-24':136_021_000,
              '2024-25':140_588_000,'2025-26':154_647_000}
CAP_2526 = SALARY_CAP['2025-26']
SEASON_DEFAULT = "2025-26"

BG, CARD, INK, MUTE = "#0d1014", "#15191f", "#f2f4f7", "#7d8694"
TEAL, AMBER, HAIR = "#2dd4bf", "#fbbf24", "#262c35"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;600&family=Oswald:wght@500;600;700&display=swap');
  .stApp {{ background:{BG}; }}
  .block-container {{ padding-top:1.6rem; padding-bottom:3rem; max-width:1180px; }}
  #MainMenu, footer, header {{ visibility:hidden; }}
  html, body, [class*="css"], p, span, div {{ color:{INK}; font-family:'Archivo',-apple-system,sans-serif; }}
  .eyebrow {{ font-family:'Oswald'; font-size:.78rem; letter-spacing:.24em; text-transform:uppercase; color:{TEAL}; font-weight:600; }}
  .hero {{ font-family:'Oswald'; font-weight:700; font-size:4rem; line-height:.92; letter-spacing:.01em; text-transform:uppercase; margin:.1rem 0 .5rem; }}
  .sub {{ color:{MUTE}; font-size:.96rem; line-height:1.5; max-width:780px; }}
  .sub i {{ color:{INK}; font-style:italic; }}
  .bignum {{ font-family:'Oswald'; font-weight:700; font-size:3.2rem; line-height:1; letter-spacing:.005em; }}
  .verdict-sub {{ color:{MUTE}; font-size:1.05rem; margin-top:.35rem; }}
  .verdict-sub b {{ color:{INK}; }}
  hr.hair {{ border:none; border-top:1px solid {HAIR}; margin:1.3rem 0; }}
  div[role="radiogroup"] {{ gap:.4rem; background:{CARD}; padding:.3rem; border-radius:10px; border:1px solid {HAIR}; width:fit-content; }}
  div[role="radiogroup"] label {{ margin:0; padding:.4rem 1.1rem; border-radius:7px; cursor:pointer; font-family:'Oswald'; font-weight:500; letter-spacing:.04em; text-transform:uppercase; font-size:.85rem; color:{MUTE}; }}
  div[role="radiogroup"] label:has(input:checked) {{ background:{TEAL}; color:{BG}; }}
  div[role="radiogroup"] label > div:first-child {{ display:none; }}
  div[data-baseweb="select"] > div {{ background:{CARD}; border-color:{HAIR}; }}
  label[data-testid="stWidgetLabel"] p {{ color:{MUTE}; font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; }}
  .stDataFrame {{ border:1px solid {HAIR}; border-radius:10px; }}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load():
    return pd.read_parquet("data/app_meta_v2.parquet"), pd.read_parquet("data/shap_v2.parquet")

def nice_feature(f):
    return ((f.replace("_rs", " (reg. season)").replace("_po", " (playoffs)")
              .replace("_lag1", " · prior yr").replace("_lag2", " · 2 yrs prior")
              .replace("ALLNBA_WT_EVER", "All-NBA résumé").replace("ALLNBA_PRIOR_EVER", "All-NBA selections")
              .replace("MAX_PCT_ELIGIBLE", "Max-tier eligibility").replace("FRONT_CT_TOUCHES", "Front-court touches")
              .replace("CLOSESTDEF_4_6_FGM", "Open-shot makes").replace("TOTAL_MIN", "Total minutes")
              .replace("MIN", "Minutes").replace("FGM", "Field goals made").replace("FTM", "Free throws made")
              .replace("EXPERIENCE", "Experience").replace("DRAFT_POSITION", "Draft slot")
              .replace("DRAFT_YR", "Draft year").replace("PIE", "Player Impact Est.")
              .replace("POST_TOUCHES", "Post touches").replace("OPP_PTS_OFF_TOV", "Opp. pts off TO")
              .replace("USG_PCT", "Usage rate").replace("_", " ").strip()))

def money(x):
    return f"{'−' if x < 0 else ''}${abs(x)/1e6:,.1f}M"

def color_name(c):
    return "teal" if c == TEAL else "amber"

def cap_dollars(pct, season):
    return pct * SALARY_CAP.get(season, CAP_2526)

meta, shap = load()

st.markdown('<div class="eyebrow">NBA · pay vs. the market price of a profile</div>', unsafe_allow_html=True)
st.markdown('<div class="hero">Market Value</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">A model trained on a decade of contracts learns how the league <i>prices</i> a '
            'statistical profile, then flags who sits off the curve and which skills move them there. '
            'This is market price, not on-court worth: <i>underpaid</i> means paid below comparable profiles.</div>',
            unsafe_allow_html=True)
st.markdown('<hr class="hair">', unsafe_allow_html=True)

view = st.radio("View", ["Player", "League ladder"], horizontal=True, label_visibility="collapsed")
st.write("")

if view == "Player":
    seasons = sorted(meta["SEASON"].unique(), reverse=True)
    c1, c2 = st.columns([3, 1])
    with c2:
        season = st.selectbox("Season", seasons, index=seasons.index(SEASON_DEFAULT) if SEASON_DEFAULT in seasons else 0)
    roster = meta[meta["SEASON"] == season].sort_values("PLAYER_NAME")
    with c1:
        player = st.selectbox("Player", roster["PLAYER_NAME"].tolist(),
                              index=0, placeholder="Type to search a player…",
                              help="Click and type to filter by name")

    row = roster[roster["PLAYER_NAME"] == player].iloc[0]
    actual, pred, resid = row["pct_cap"], row["pred_pct_cap"], row["resid_usd"]
    under = resid < 0
    color, word = (TEAL, "UNDERPAID") if under else (AMBER, "OVERPAID")
    season_cap = SALARY_CAP.get(season, CAP_2526)

    paid_usd = cap_dollars(actual, season)
    mkt_usd  = cap_dollars(pred, season)
    st.markdown(f'<div class="bignum" style="color:{color}">{money(abs(resid))} {word}</div>'
                f'<div class="verdict-sub">{player.title()} is paid '
                f'<b>${paid_usd/1e6:,.1f}M</b> <span style="color:{MUTE}">({actual:.1%} of cap)</span> · '
                f'the market values this profile at '
                f'<b style="color:{color}">${mkt_usd/1e6:,.1f}M</b> '
                f'<span style="color:{MUTE}">({pred:.1%})</span></div>',
                unsafe_allow_html=True)
    _err = row['pred_pct_cap']*season_cap - row['pct_cap']*season_cap
    st.caption(f"{season} salary cap: \\${season_cap/1e6:,.0f}M  ·  "
               f"prediction error vs. actual: {money(_err).replace('$', chr(92)+'$')}")
    st.write("")

    # ---- price-summary anchor: baseline → market value (predicted) → actual salary ----
    # makes "overpaid" legible even when every force-bar is teal: the bars build the market
    # price up FROM the baseline, and the actual-salary marker sits above/below that price.
    base_usd = float(meta["baseline_pct"].iloc[0]) * season_cap
    summary = go.Figure()
    # market value bar (what the profile is worth), colored by verdict
    summary.add_trace(go.Bar(
        x=[mkt_usd], y=["price"], orientation="h", width=0.5,
        marker_color=color, marker_line_width=0,
        hovertemplate=f"Market value: {money(mkt_usd)}<extra></extra>", showlegend=False))
    # baseline marker (where every player starts)
    summary.add_vline(x=base_usd, line_width=1.5, line_dash="dot", line_color=MUTE,
                      annotation_text=f"league baseline {money(base_usd)}",
                      annotation_position="top left",
                      annotation_font=dict(size=10, color=MUTE))
    # actual-salary marker — the line the bar must clear to be "fairly paid"
    summary.add_vline(x=paid_usd, line_width=2.5, line_color=INK,
                      annotation_text=f"actually paid {money(paid_usd)}",
                      annotation_position="bottom right",
                      annotation_font=dict(size=11, color=INK))
    summary.update_layout(
        height=120, margin=dict(l=8, r=16, t=26, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family="Archivo"),
        xaxis=dict(tickprefix="$", tickformat=".2s", gridcolor=HAIR, zeroline=False,
                   color=MUTE, range=[0, max(mkt_usd, paid_usd) * 1.18]),
        yaxis=dict(visible=False), showlegend=False, bargap=0.4)
    st.plotly_chart(summary, use_container_width=True, config={"displayModeBar": False})
    _verb = "falls short of" if under else "overshoots"
    st.caption(f"The {color_name(color)} bar is the market price the model builds for {player.split()[0].title()} "
               f"(starting from the league baseline). The black line is the salary actually paid — "
               f"when the bar {'doesn’t reach' if under else 'falls below'} the line, the player is {word.lower()}.")

    if actual > 0.355:  # paid above the highest CBA tier (35% of cap, 10+ yr vets)
        st.info(
            f"**{player.split()[0].title()} is paid above the standard CBA maximum (35% of cap).** "
            "The individual max is tiered by experience — 25% (0–6 yrs), 30% (7–9 yrs), 35% (10+ yrs) — "
            "and the model's training data tops out at the league max, so it never predicts above ~35%. "
            "A contract above that (via super-max or veteran provisions) will read as 'overpaid' by rule, "
            "not because anything in the profile lowers the player's value.")

    psh = shap[(shap["PLAYER_NAME"] == player) & (shap["SEASON"] == season)]
    psh = psh.reindex(psh["shap_usd"].abs().sort_values(ascending=False).index).head(15).iloc[::-1]

    fig = go.Figure(go.Bar(
        x=psh["shap_usd"], y=[nice_feature(f) for f in psh["feature"]], orientation="h",
        marker_color=[TEAL if v > 0 else AMBER for v in psh["shap_usd"]],
        customdata=[("raises" if v > 0 else "lowers") + f" price by {money(abs(v))}" for v in psh["shap_usd"]],
        hovertemplate="%{y}<br>%{customdata}<extra></extra>"))
    fig.add_vline(x=0, line_width=1.5, line_color=MUTE)
    fig.update_layout(
        title=dict(text="WHAT MOVES THIS PLAYER'S PRICE", font=dict(size=14, color=MUTE, family="Oswald"), x=0, xanchor="left"),
        height=460, margin=dict(l=8, r=16, t=44, b=28), autosize=True,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=12, family="Archivo"),
        xaxis=dict(title="Contribution to predicted salary", tickprefix="$", tickformat=".2s", gridcolor=HAIR, zeroline=False, color=MUTE),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", color=INK), bargap=0.34)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"Each bar is one factor's dollar contribution for {player.split()[0].title()} specifically, "
               f"summing from the baseline toward the market price above. "
               f"Teal raises the price, amber lowers it. This player's top 15 factors.")

    with st.expander("Show all factors"):
        allf = shap[(shap["PLAYER_NAME"] == player) & (shap["SEASON"] == season)].copy()
        allf = allf.reindex(allf["shap_usd"].abs().sort_values(ascending=False).index)
        tbl = pd.DataFrame({
            "Factor": [nice_feature(f) for f in allf["feature"]],
            "Effect on price": [money(v) for v in allf["shap_usd"]],
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True, height=320)

    with st.expander("Factor glossary — look up any stat"):
        DEFS = {
            "EXPERIENCE": "Seasons in the league. The single biggest salary driver.",
            "MIN": "Per-game minutes — the model's proxy for role and trust.",
            "FGM": "Field goals made — scoring volume.",
            "FTM": "Free throws made — rim pressure / foul-drawing.",
            "FRONT_CT_TOUCHES": "How often a player handles the ball in the front court; a usage/role proxy.",
            "ALLNBA_WT_EVER": "Weighted career All-NBA selections (1st=3, 2nd=2, 3rd=1), prior seasons only — the star credential.",
            "ALLNBA_PRIOR_EVER": "Career All-NBA selections before this season — the star credential.",
            "ALLNBA_PRIOR3": "All-NBA selections in the prior three seasons — recent-star signal.",
            "MAX_PCT_ELIGIBLE": "The CBA salary ceiling a player qualifies for by experience (25 / 30 / 35% of cap).",
            "DRAFT_POSITION": "Where a player was drafted; pedigree the market pays for years later.",
            "PIE": "Player Impact Estimate — a box-score composite of a player's share of the game's positive events.",
            "POST_TOUCHES": "Possessions a player operates from the post.",
            "CLOSESTDEF_4_6_FGM": "Field goals made with the closest defender 4–6 ft away — open-shot makes.",
            "DEF2_NS_FG2_PCT": "Opponent 2-pt % on clean attempts a player defended; lower = better interior deterrence.",
            "OPP_PTS_OFF_TOV": "Opponent points off turnovers allowed while on the floor; lower = better transition defense.",
            "USG_PCT": "Usage rate — share of team possessions a player ends.",
            "TOTAL_MIN": "Total season minutes (per-game × games) — a durability/availability signal.",
            "GP": "Games played — availability.",
            "SCREEN_ASSISTS": "Screens that directly lead to a teammate's made basket.",
            "DEFLECTIONS": "Times a player gets a hand on the ball on defense.",
            "POST_TOV_POSS_PCT": "Share of post-up possessions ending in a turnover.",
            "TOUCHES": "Total times a player touches the ball.",
            "DREB": "Defensive rebounds.", "OREB": "Offensive rebounds.", "REB": "Total rebounds.",
            "AST": "Assists.", "AST_TO_PASS_PCT": "Share of a player's passes that become assists.",
            "FT_AST": "Passes that lead to a teammate's free throws.",
            "DD2": "Double-doubles.", "TD3": "Triple-doubles.",
            "D_FGM": "Field goals made by opponents a player defended.",
            "DEF_RIM_FG_PCT": "Opponent FG% at the rim when this player defends — true rim protection.",
            "CLOSESTDEF_0_2_FGA_FREQUENCY": "How often a player takes shots with a defender 0–2 ft away — contested-shot willingness.",
            "CLOSESTDEF_6P_FG2_PCT": "2-pt% defended with the closest defender 6+ ft away.",
            "CLUTCH_E_PACE": "Team pace in clutch minutes while on the floor.",
            "CLUTCH_REB": "Rebounds in clutch situations.",
            "PAINT_TOUCH_AST_PCT": "Share of paint touches that lead to an assist.",
            "PAINT_TOUCH_PASSES_PCT": "Share of paint touches a player passes out of.",
            "POST_SF_POSS_PCT": "Share of post-ups drawing a shooting foul.",
            "RM_POSS_PCT": "Share of offense as the roll man in pick-and-roll.",
            "SHOT_FG2M": "Two-point field goals made.",
            "SC_22_18_FG2A_FREQUENCY": "Shot-attempt rate with 18–22 seconds on the shot clock (early offense).",
            "DRIBB_2_FG_PCT": "FG% on shots after exactly two dribbles.",
            "DRIBB_0_FG3A_FREQUENCY": "Catch-and-shoot 3-point attempt rate (zero dribbles).",
            "DEFGT15_FGM_GT_15": "Opponent makes a player defends from 15+ ft away.",
        }
        # def define(f):
        #     base = f.replace("_rs","").replace("_po","").replace("_lag1","").replace("_lag2","")
        #     suffix = (" (this season, reg.)" if f.endswith("_rs") else
        #               " (playoffs)" if "_po" in f else "")
        #     time = (" — from the prior season" if "_lag1" in f else
        #             " — from two seasons ago" if "_lag2" in f else "")
        #     d = DEFS.get(base, "A tracked stat in the feature set.")
        #     return d + time

        def define(f):
            return compose_definition(f, DEFS)
        all_feats = sorted(shap["feature"].unique())
        pick = st.selectbox("Look up a factor", all_feats, key="gloss")
        st.markdown(f"**{nice_feature(pick)}**  \n{define(pick)}")
else:
    seasons = sorted(meta["SEASON"].unique(), reverse=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        season = st.selectbox("Season", seasons, index=seasons.index(SEASON_DEFAULT) if SEASON_DEFAULT in seasons else 0)
    with c2:
        sort_by = st.selectbox("Rank by", ["Most underpaid", "Most overpaid", "Highest market value"])
    d = meta[meta["SEASON"] == season].copy()
    _cap = SALARY_CAP.get(season, CAP_2526)
    d["Market $"]     = d["pred_pct_cap"] * _cap
    d["Actual $"]     = d["pct_cap"] * _cap
    d["Market value"] = d["pred_pct_cap"]; d["Actual"] = d["pct_cap"]; d["Gap ($)"] = -d["resid_usd"]
    d = (d.sort_values("resid_usd") if sort_by == "Most underpaid"
         else d.sort_values("resid_usd", ascending=False) if sort_by == "Most overpaid"
         else d.sort_values("pred_pct_cap", ascending=False))
    show = d[["PLAYER_NAME", "Market $", "Actual $", "Gap ($)", "Market value", "Actual"]].head(40).reset_index(drop=True)
    show.index = show.index + 1
    show["PLAYER_NAME"] = show["PLAYER_NAME"].str.title()
    st.dataframe(
        show.style.format({"Market $": lambda v: f"${v/1e6:,.1f}M", "Actual $": lambda v: f"${v/1e6:,.1f}M",
                           "Gap ($)": money, "Market value": "{:.1%}", "Actual": "{:.1%}"})
            .map(lambda v: f"color:{TEAL}" if isinstance(v, (int, float)) and v > 0 else
                 (f"color:{AMBER}" if isinstance(v, (int, float)) and v < 0 else ""), subset=["Gap ($)"]),
        use_container_width=True, height=600)
    st.caption("Gap = market value − actual pay. Teal = market prices the profile above the salary.")


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