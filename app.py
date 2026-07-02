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
