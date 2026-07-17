"""
Market Value — NBA salary vs. the market's price for a player's profile.
Dark broadcast-graphics build.

Two models, one comparison:
  Production model — prices players on individual on-court output (box + tracking + play-type).
  Impact model     — the production model PLUS LEBRON, a plus-minus impact metric. Built on top.

Reads precomputed artifacts (no model at runtime):
  data/app_meta_v2.parquet         — production: actual %, predicted %, residual $, baseline
  data/shap_v2.parquet             — production long SHAP (per player-season, per feature)
  data/app_meta_v2_lebron.parquet  — impact model predictions
  data/shap_v2_lebron.parquet      — impact long SHAP (includes LBR_* features)

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
VIOLET = "#a78bfa"   # highlight color for LEBRON/impact features in the impact waterfall

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;600&family=Oswald:wght@500;600;700&display=swap');
  .stApp {{ background:{BG}; }}
  .block-container {{ padding-top:1.6rem; padding-bottom:3rem; max-width:1180px; }}
  #MainMenu, footer, header {{ visibility:hidden; }}
  html, body, [class*="css"], p, span, div {{ color:{INK}; font-family:'Archivo',-apple-system,sans-serif; }}
  .eyebrow {{ font-family:'Oswald'; font-size:.78rem; letter-spacing:.24em; text-transform:uppercase; color:{TEAL}; font-weight:600; }}
  .hero {{ font-family:'Oswald'; font-weight:700; font-size:4rem; line-height:.92; letter-spacing:.01em; text-transform:uppercase; margin:.1rem 0 .5rem; }}
  .sub {{ color:{MUTE}; font-size:.96rem; line-height:1.5; max-width:820px; }}
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


# ------------------------------------------------------------------ data
@st.cache_data
def load_production():
    return (pd.read_parquet("data/app_meta_v2.parquet"),
            pd.read_parquet("data/shap_v2.parquet"))

@st.cache_data
def load_impact():
    return (pd.read_parquet("data/app_meta_v2_lebron.parquet"),
            pd.read_parquet("data/shap_v2_lebron.parquet"))

@st.cache_data
def load_comparison():
    box = (pd.read_parquet("data/app_meta_v2.parquet")
             [["PLAYER_NAME", "SEASON", "pred_pct_cap"]]
             .rename(columns={"pred_pct_cap": "pred_boxonly"}))
    leb = (pd.read_parquet("data/app_meta_v2_lebron.parquet")
             [["PLAYER_NAME", "SEASON", "pred_pct_cap", "pct_cap"]]
             .rename(columns={"pred_pct_cap": "pred_lebron", "pct_cap": "actual"}))
    cmp = box.merge(leb, on=["PLAYER_NAME", "SEASON"])
    cmp["lebron_effect"] = cmp["pred_lebron"] - cmp["pred_boxonly"]
    return cmp


# ------------------------------------------------------------------ helpers
def nice_feature(f):
    return ((f.replace("_rs", " (reg. season)").replace("_po", " (playoffs)")
              .replace("_prior1", " · prior yr").replace("_prior2", " · 2 yrs prior")
              .replace("_lag1", " · prior yr").replace("_lag2", " · 2 yrs prior")
              .replace("LBR_D_LEBRON", "Defensive LEBRON").replace("LBR_O_LEBRON", "Offensive LEBRON")
              .replace("LBR_WAR", "LEBRON WAR").replace("LBR_LEBRON", "LEBRON (overall)")
              .replace("ALLNBA_WT_EVER", "All-NBA résumé").replace("ALLNBA_PRIOR_EVER", "All-NBA selections")
              .replace("MAX_PCT_ELIGIBLE", "Max-tier eligibility").replace("FRONT_CT_TOUCHES", "Front-court touches")
              .replace("CLOSESTDEF_4_6_FGM", "Open-shot makes").replace("TOTAL_MIN", "Total minutes")
              .replace("MIN", "Minutes").replace("FGM", "Field goals made").replace("FTM", "Free throws made")
              .replace("EXPERIENCE", "Experience").replace("DRAFT_POSITION", "Draft slot")
              .replace("DRAFT_YR", "Draft year").replace("PIE", "Player Impact Est.")
              .replace("POST_TOUCHES", "Post touches").replace("OPP_PTS_OFF_TOV", "Opp. pts off TO")
              .replace("USG_PCT", "Usage rate").replace("_", " ").strip()))

def is_impact_feature(f):
    return "LBR_" in f or "LEBRON" in f

def money(x):
    return f"{'−' if x < 0 else ''}${abs(x)/1e6:,.1f}M"

def color_name(c):
    return "teal" if c == TEAL else "amber"

def cap_dollars(pct, season):
    return pct * SALARY_CAP.get(season, CAP_2526)


def render_player_view(meta, shap, highlight_impact=False):
    """Shared per-player view. If highlight_impact, LEBRON bars are drawn in violet."""
    seasons = sorted(meta["SEASON"].unique(), reverse=True)
    c1, c2 = st.columns([3, 1])
    with c2:
        season = st.selectbox("Season", seasons,
                              index=seasons.index(SEASON_DEFAULT) if SEASON_DEFAULT in seasons else 0,
                              key=f"season_{'imp' if highlight_impact else 'prod'}")
    roster = meta[meta["SEASON"] == season].sort_values("PLAYER_NAME")
    with c1:
        player = st.selectbox("Player", roster["PLAYER_NAME"].tolist(), index=0,
                              placeholder="Type to search a player...",
                              key=f"player_{'imp' if highlight_impact else 'prod'}")

    row = roster[roster["PLAYER_NAME"] == player].iloc[0]
    actual, pred, resid = row["pct_cap"], row["pred_pct_cap"], row["resid_usd"]
    under = resid < 0
    color, word = (TEAL, "UNDERPAID") if under else (AMBER, "OVERPAID")
    season_cap = SALARY_CAP.get(season, CAP_2526)
    paid_usd = cap_dollars(actual, season)
    mkt_usd = cap_dollars(pred, season)

    st.markdown(f'<div class="bignum" style="color:{color}">{money(abs(resid))} {word}</div>'
                f'<div class="verdict-sub">{player.title()} is paid '
                f'<b>${paid_usd/1e6:,.1f}M</b> <span style="color:{MUTE}">({actual:.1%} of cap)</span> · '
                f'the market values this profile at '
                f'<b style="color:{color}">${mkt_usd/1e6:,.1f}M</b> '
                f'<span style="color:{MUTE}">({pred:.1%})</span></div>',
                unsafe_allow_html=True)
    _err = pred * season_cap - actual * season_cap
    st.caption(f"{season} salary cap: \\${season_cap/1e6:,.0f}M  ·  "
               f"prediction error vs. actual: {money(_err).replace('$', chr(92)+'$')}")
    st.write("")

    # price-summary anchor
    base_usd = float(meta["baseline_pct"].iloc[0]) * season_cap
    summary = go.Figure()
    summary.add_trace(go.Bar(x=[mkt_usd], y=["price"], orientation="h", width=0.5,
                             marker_color=color, marker_line_width=0,
                             hovertemplate=f"Market value: {money(mkt_usd)}<extra></extra>", showlegend=False))
    summary.add_vline(x=base_usd, line_width=1.5, line_dash="dot", line_color=MUTE,
                      annotation_text=f"league baseline {money(base_usd)}",
                      annotation_position="top left", annotation_font=dict(size=10, color=MUTE))
    summary.add_vline(x=paid_usd, line_width=2.5, line_color=INK,
                      annotation_text=f"actually paid {money(paid_usd)}",
                      annotation_position="bottom right", annotation_font=dict(size=11, color=INK))
    summary.update_layout(height=120, margin=dict(l=8, r=16, t=26, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=INK, family="Archivo"),
                          xaxis=dict(tickprefix="$", tickformat=".2s", gridcolor=HAIR, zeroline=False,
                                     color=MUTE, range=[0, max(mkt_usd, paid_usd) * 1.18]),
                          yaxis=dict(visible=False), showlegend=False, bargap=0.4)
    st.plotly_chart(summary, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"The {color_name(color)} bar is the market price the model builds for {player.split()[0].title()} "
               f"(from the league baseline). The black line is the salary actually paid — "
               f"when the bar {'doesn’t reach' if under else 'falls below'} the line, the player is {word.lower()}.")

    if actual > 0.355:
        st.info(f"**{player.split()[0].title()} is paid above the standard CBA maximum (35% of cap).** "
                "The individual max is tiered by experience — 25% (0–6 yrs), 30% (7–9 yrs), 35% (10+ yrs) — "
                "and the model's training data tops out at the league max, so it never predicts above ~35%. "
                "A contract above that reads as 'overpaid' by rule, not because the profile lowers the value.")

    # waterfall — top 15 by |shap_usd|
    psh = shap[(shap["PLAYER_NAME"] == player) & (shap["SEASON"] == season)]
    psh = psh.reindex(psh["shap_usd"].abs().sort_values(ascending=False).index).head(15).iloc[::-1]

    if highlight_impact:
        bar_colors = [VIOLET if is_impact_feature(f) else (TEAL if v > 0 else AMBER)
                      for f, v in zip(psh["feature"], psh["shap_usd"])]
    else:
        bar_colors = [TEAL if v > 0 else AMBER for v in psh["shap_usd"]]

    fig = go.Figure(go.Bar(
        x=psh["shap_usd"], y=[nice_feature(f) for f in psh["feature"]], orientation="h",
        marker_color=bar_colors,
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
    if highlight_impact:
        st.caption(f"Each bar is one factor's dollar contribution for {player.split()[0].title()}. "
                   f"<span style='color:{VIOLET}'>Violet bars are the impact (LEBRON) features</span> — "
                   f"the value the production model can't see. Teal raises price, amber lowers it. Top 15 factors.",
                   unsafe_allow_html=True)
    else:
        st.caption(f"Each bar is one factor's dollar contribution for {player.split()[0].title()} specifically. "
                   f"Teal raises the price, amber lowers it. This player's top 15 factors.")

    with st.expander("Show all factors"):
        allf = shap[(shap["PLAYER_NAME"] == player) & (shap["SEASON"] == season)].copy()
        allf = allf.reindex(allf["shap_usd"].abs().sort_values(ascending=False).index)
        tbl = pd.DataFrame({"Factor": [nice_feature(f) for f in allf["feature"]],
                            "Effect on price": [money(v) for v in allf["shap_usd"]]})
        st.dataframe(tbl, use_container_width=True, hide_index=True, height=320)


def render_ladder_view(meta, key):
    seasons = sorted(meta["SEASON"].unique(), reverse=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        season = st.selectbox("Season", seasons,
                              index=seasons.index(SEASON_DEFAULT) if SEASON_DEFAULT in seasons else 0,
                              key=f"ladder_season_{key}")
    with c2:
        sort_by = st.selectbox("Rank by", ["Most underpaid", "Most overpaid", "Highest market value"],
                               key=f"ladder_sort_{key}")
    d = meta[meta["SEASON"] == season].copy()
    _cap = SALARY_CAP.get(season, CAP_2526)
    d["Market $"] = d["pred_pct_cap"] * _cap
    d["Actual $"] = d["pct_cap"] * _cap
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


# ------------------------------------------------------------------ header
st.markdown('<div class="eyebrow">NBA · pay vs. the market price of a profile</div>', unsafe_allow_html=True)
st.markdown('<div class="hero">Market Value</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">A model trained on a decade of contracts learns how the league <i>prices</i> a '
            'statistical profile, then flags who sits off the curve and which skills move them there. '
            'This is market price, not on-court worth: <i>underpaid</i> means paid below comparable profiles.</div>',
            unsafe_allow_html=True)
st.markdown('<hr class="hair">', unsafe_allow_html=True)

tab = st.radio("Model", ["Production", "Impact", "Impact lens"], horizontal=True, label_visibility="collapsed")
st.write("")

if tab == "Production":
    meta, shap = load_production()
    st.markdown('<div class="sub" style="margin-bottom:1rem">The <b>production model</b> — the primary model — '
                'prices players entirely on individual on-court output: box score, tracking, and play-type stats. '
                'No impact metric, fully reproducible from raw production.</div>', unsafe_allow_html=True)
    sub = st.radio("View", ["Player", "League ladder"], horizontal=True, label_visibility="collapsed", key="prod_sub")
    st.write("")
    if sub == "Player":
        render_player_view(meta, shap, highlight_impact=False)
    else:
        render_ladder_view(meta, key="prod")

elif tab == "Impact":
    meta, shap = load_impact()
    st.markdown('<div class="sub" style="margin-bottom:1rem">The <b>impact model</b> is the production model '
                '<i>plus</i> LEBRON, a plus-minus impact metric — built on top of the production model to fold in '
                'value the box score can\'t see (defense, spacing, connective play). '
                '<span style="color:#7d8694">Experimental, layered on the primary production model.</span></div>',
                unsafe_allow_html=True)
    sub = st.radio("View", ["Player", "League ladder"], horizontal=True, label_visibility="collapsed", key="imp_sub")
    st.write("")
    if sub == "Player":
        render_player_view(meta, shap, highlight_impact=True)
    else:
        render_ladder_view(meta, key="imp")

else:  # Impact lens — the disagreement comparison
    st.markdown('<div class="sub" style="margin-bottom:1rem">Where the two models <i>disagree</i> on a player\'s '
                'fair price is the invisible value, in dollars — what the impact metric sees that production alone '
                'misses.<br><br><span style="color:#7d8694">Across seasons the driver is WAR (total winning '
                'contribution), not the offense/defense split — and it\'s individual, not archetypal.</span></div>',
                unsafe_allow_html=True)
    cmp = load_comparison()
    seasons = sorted(cmp["SEASON"].unique(), reverse=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        season = st.selectbox("Season", seasons, index=seasons.index(SEASON_DEFAULT) if SEASON_DEFAULT in seasons else 0, key="lens_season")
    with c2:
        n = st.selectbox("Players per list", [10, 15, 20, 30], index=2, key="lens_n")

    df = cmp[cmp["SEASON"] == season].copy()
    _cap = SALARY_CAP.get(season, CAP_2526)
    df["effect_usd"] = df["lebron_effect"] * _cap

    def _display(frame):
        out = frame[["PLAYER_NAME", "pred_boxonly", "pred_lebron", "lebron_effect", "effect_usd", "actual"]].copy()
        out["PLAYER_NAME"] = out["PLAYER_NAME"].str.title()
        out.columns = ["Player", "Production", "Impact", "Shift", "Shift $", "Actual"]
        return out

    raised = _display(df.nlargest(n, "lebron_effect"))
    lowered = _display(df.nsmallest(n, "lebron_effect"))
    fmt = {"Production": "{:.1%}", "Impact": "{:.1%}", "Shift": "{:+.1%}",
           "Shift $": lambda v: f"${v/1e6:+.1f}M", "Actual": "{:.1%}"}

    left, right = st.columns(2)
    with left:
        st.markdown("#### Impact raises them")
        st.caption("Production underrates their winning impact")
        st.dataframe(raised.style.format(fmt).map(lambda v: f"color:{TEAL}", subset=["Shift", "Shift $"]),
                     use_container_width=True, hide_index=True, height=560)
    with right:
        st.markdown("#### Impact lowers them")
        st.caption("Production overrates them — volume without impact")
        st.dataframe(lowered.style.format(fmt).map(lambda v: f"color:{AMBER}", subset=["Shift", "Shift $"]),
                     use_container_width=True, hide_index=True, height=560)
    st.caption("Sorted by how much the impact metric moves each player's predicted price — the disagreement "
               "between the two models, the value the box score can't see.")
