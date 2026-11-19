"""Team-level stats: where teams take and give up shots, what they run and what
gets run against them, and the possession battle (rebounding, turnovers, transition).

The player master can't answer these. Summing per-game player lines back up to a
team double-counts traded players, and nothing in it describes what a team
*allows* — so this pulls the team endpoints directly, both sides of the ball.

    data = build_team_data("2025-26")          # ~35 requests the first time, cached after
    plot_shot_profile(data["zones"], side="offense")
    plot_zone_stack(data["zones"], side="defense", metric="POINTS")
    plot_contest_stack(data["contest"], side="offense", shot_type="3")
    plot_play_types(data["play_types"], side="defense", metric="PPP")
    plot_rebounding(data["teams"])

Open / contested shots for players come straight from the master (the
CLOSESTDEF_ columns), via player_contest(df_master, season).

Importing nba_viz_utils installs the x-nba-stats headers, so these calls get them too.
"""
import time

import holoviews as hv
import hvplot.pandas  # noqa: F401
import pandas as pd
from nba_api.stats.endpoints import (
    LeagueDashOppPtShot,
    LeagueDashTeamPtShot,
    LeagueDashTeamShotLocations,
    LeagueDashTeamStats,
    SynergyPlayTypes,
)
from nba_api.stats.static import teams as static_teams

from nba_viz_utils import DATA_DIR, cached_fetch, safe_cache_name, should_refresh

TEAM_ABBR = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}
TEAM_ID = {abbr: team_id for team_id, abbr in TEAM_ABBR.items()}


def logo_url(team):
    """Team abbreviation -> its logo on the NBA's CDN (loaded by the browser, not by Python)."""
    return f"https://cdn.nba.com/logos/nba/{TEAM_ID[team]}/primary/L/logo.svg"


def _all_teams(names):
    return all(str(n) in TEAM_ID for n in names)


def _logo_axis_hook(teams, x, size=21):
    """Swap the team tick labels on the y axis for logos.

    bokeh can't put an image in a tick label, so the labels are hidden and the
    logos are drawn just inside the frame, at `x` (in the x axis's own units).
    """
    def hook(plot, element):
        fig = plot.state
        fig.yaxis.major_label_text_font_size = "0pt"
        fig.yaxis.major_tick_line_color = None
        fig.image_url(url=[logo_url(t) for t in teams], x=[x] * len(teams), y=list(teams),
                      w=size, h=size, w_units="screen", h_units="screen", anchor="center")
    return hook


def _logo_point_hook(df, x, y, size=34):
    """Draw each team's logo at its point."""
    def hook(plot, element):
        plot.state.image_url(url=[logo_url(t) for t in df["TEAM"]], x=list(df[x]), y=list(df[y]),
                             w=size, h=size, w_units="screen", h_units="screen", anchor="center")
    return hook

# One request each. Opponent is what the team allows, in the same units as Base.
TEAM_MEASURES = ["Base", "Advanced", "Four Factors", "Misc", "Opponent"]

ZONES = ["Restricted Area", "In The Paint (Non-RA)", "Mid-Range", "Corner 3", "Above the Break 3"]
ZONE_LABELS = {
    "Restricted Area": "Rim", "In The Paint (Non-RA)": "Paint (non-rim)",
    "Mid-Range": "Mid-range", "Corner 3": "Corner 3", "Above the Break 3": "Above-break 3",
}
ZONE_POINTS = {"Corner 3": 3, "Above the Break 3": 3}

PLAY_TYPES = {
    "PRBallHandler": "P&R handler", "PRRollman": "P&R roll man", "Isolation": "Isolation",
    "Transition": "Transition", "Postup": "Post up", "Spotup": "Spot up",
    "Handoff": "Handoff", "Cut": "Cut", "OffScreen": "Off screen", "OffRebound": "Putback",
}

SIDES = {"offense": "Offensive", "defense": "Defensive"}

# Closest-defender distance at the release, tightest first. Keys are the API's
# bucket names; the master stores the same four as CLOSESTDEF_{code}_*.
CONTEST = {
    "0-2 Feet - Very Tight": ("0_2", "Very tight (0-2 ft)"),
    "2-4 Feet - Tight": ("2_4", "Tight (2-4 ft)"),
    "4-6 Feet - Open": ("4_6", "Open (4-6 ft)"),
    "6+ Feet - Wide Open": ("6P", "Wide open (6+ ft)"),
}
CONTEST_LABELS = [label for _, label in CONTEST.values()]
OPEN_LABELS = CONTEST_LABELS[2:]

# Zones are identities -> categorical hues in a fixed order. Contest level is an
# ordered scale with two poles -> two reds (tight) and two blues (open).
ZONE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
CONTEST_COLORS = ["#b2332f", "#eb9c8e", "#86b6ef", "#1c5cab"]
INK, PAPER = "#0b0b0b", "#ffffff"


# =============================================================================
# FETCH
# =============================================================================

def _fetch(cache_name, fetch_raw, season, season_type, force_refresh):
    """cached_fetch that pauses after every attempt and never raises.

    One dead endpoint shouldn't sink the other 26, and the pause has to run on
    the failures too — skipping it there is how a slow run turns into hammering.
    """
    try:
        return cached_fetch(
            cache_name, fetch_raw,
            force_refresh=force_refresh or should_refresh(season, season_type),
        )
    except Exception as e:
        print(f"   ⚠️ {cache_name}: {e}")
        return None
    finally:
        time.sleep(0.8)


def fetch_team_dash(season, season_type, per_mode, timeout, force_refresh):
    """Base / Advanced / Four Factors / Misc / Opponent, merged to one row per team."""
    wide = None
    for measure in TEAM_MEASURES:
        def fetch_raw(measure=measure):
            return LeagueDashTeamStats(
                measure_type_detailed_defense=measure, per_mode_detailed=per_mode,
                season=season, season_type_all_star=season_type, timeout=timeout,
            ).get_data_frames()[0]

        df = _fetch(safe_cache_name("team_dash", measure, season, season_type, per_mode),
                    fetch_raw, season, season_type, force_refresh)
        if df is None or df.empty:
            continue

        df = df.drop(columns=[c for c in df.columns if c.endswith("_RANK") or c.startswith("CF")])
        if wide is None:
            wide = df
        else:
            # the measure types repeat GP, W, MIN and each other's headline columns
            new = [c for c in df.columns if c not in wide.columns]
            wide = wide.merge(df[["TEAM_ID"] + new], on="TEAM_ID", how="outer")

    if wide is None:
        return pd.DataFrame()

    wide.insert(0, "TEAM", wide["TEAM_ID"].map(TEAM_ABBR))
    wide["SEASON"], wide["SEASON_TYPE"] = season, season_type
    return wide


def fetch_team_zones(season, season_type, per_mode, timeout, force_refresh):
    """Shot attempts and accuracy by court zone, taken and allowed. Long format."""
    frames = []
    for side, measure in (("offense", "Base"), ("defense", "Opponent")):
        def fetch_raw(measure=measure):
            raw = LeagueDashTeamShotLocations(
                distance_range="By Zone", measure_type_simple=measure,
                per_mode_detailed=per_mode, season=season,
                season_type_all_star=season_type, timeout=timeout,
            ).get_data_frames()[0]
            # zone names sit in a second header row; flatten before caching, as
            # fetch_nba_shot_locations() does, so the parquet never holds a MultiIndex
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = ["_".join(str(x) for x in col if str(x) != "") for col in raw.columns]
            return raw

        raw = _fetch(safe_cache_name("team_zones", measure, season, season_type, per_mode),
                     fetch_raw, season, season_type, force_refresh)
        if raw is None or raw.empty:
            continue
        frames.append(_tidy_zones(raw, side))

    if not frames:
        return pd.DataFrame()

    zones = pd.concat(frames, ignore_index=True)
    zones["SEASON"], zones["SEASON_TYPE"] = season, season_type
    return zones


def _tidy_zones(raw, side):
    def col(zone, stat):
        # the Opponent pull names its columns OPP_FGM / OPP_FGA
        for name in (f"{zone}_{stat}", f"{zone}_OPP_{stat}"):
            if name in raw.columns:
                return pd.to_numeric(raw[name], errors="coerce")
        return None

    rows = []
    for zone in ZONES:
        fgm, fga = col(zone, "FGM"), col(zone, "FGA")
        if fga is None and zone == "Corner 3":
            # older seasons only report the two corners separately
            fgm = col("Left Corner 3", "FGM") + col("Right Corner 3", "FGM")
            fga = col("Left Corner 3", "FGA") + col("Right Corner 3", "FGA")
        if fga is None:
            continue
        rows.append(pd.DataFrame({
            "TEAM_ID": raw["TEAM_ID"], "ZONE": ZONE_LABELS[zone], "FGM": fgm, "FGA": fga,
            "_pts": ZONE_POINTS.get(zone, 2),
        }))

    long = pd.concat(rows, ignore_index=True)
    long["TEAM"] = long["TEAM_ID"].map(TEAM_ABBR)
    long["SIDE"] = side
    long["FG_PCT"] = long["FGM"] / long["FGA"]
    long["PPS"] = long["FG_PCT"] * long.pop("_pts")                    # points per shot
    long["SHARE"] = long["FGA"] / long.groupby("TEAM_ID")["FGA"].transform("sum")
    return long


def fetch_team_play_types(season, season_type, per_mode, timeout, force_refresh):
    """Synergy play types run by each team and run against it. Long format."""
    frames = []
    for side, grouping in SIDES.items():
        for play_type, label in PLAY_TYPES.items():
            def fetch_raw(play_type=play_type, grouping=grouping):
                return SynergyPlayTypes(
                    play_type_nullable=play_type, player_or_team_abbreviation="T",
                    type_grouping_nullable=grouping, per_mode_simple=per_mode,
                    season=season, season_type_all_star=season_type, timeout=timeout,
                ).get_data_frames()[0]

            df = _fetch(
                safe_cache_name("team_synergy", play_type, grouping, season, season_type, per_mode),
                fetch_raw, season, season_type, force_refresh,
            )
            if df is None or df.empty:
                continue
            df = df[["TEAM_ID", "POSS_PCT", "PPP", "POSS", "PERCENTILE", "EFG_PCT", "TOV_POSS_PCT"]].copy()
            df["PLAY_TYPE"], df["SIDE"] = label, side
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    pt = pd.concat(frames, ignore_index=True)
    pt.insert(0, "TEAM", pt["TEAM_ID"].map(TEAM_ABBR))
    pt["SEASON"], pt["SEASON_TYPE"] = season, season_type
    return pt


def _tidy_contest(wide, name_col):
    """Wide (one row per entity, {code}_{stat} columns) -> long, with shares.

    Shares are of that entity's own attempts, separately for all shots, 2s and
    3s. The split matters: most "very tight" attempts are at the rim, so the
    all-shots mix says as much about where a team shoots as how open it gets.
    The 3-point mix is the cleaner read on openness.
    """
    rows = []
    for code, label in CONTEST.values():
        def num(stat, code=code):
            c = f"{code}_{stat}"
            return pd.to_numeric(wide[c], errors="coerce") if c in wide.columns else float("nan")

        part = wide[name_col].copy()
        part["CONTEST"] = label
        for stat in ("FGA", "FGM", "FG2A", "FG2M", "FG3A", "FG3M", "EFG_PCT"):
            part[stat] = num(stat)
        rows.append(part)

    long = pd.concat(rows, ignore_index=True)
    long[["FGA", "FG2A", "FG3A"]] = long[["FGA", "FG2A", "FG3A"]].fillna(0)
    key = [c for c in name_col if c in ("TEAM", "PLAYER_NAME", "TEAM_ABBREVIATION", "SIDE")]
    for att, share in (("FGA", "SHARE"), ("FG2A", "SHARE_2"), ("FG3A", "SHARE_3")):
        total = long.groupby(key)[att].transform("sum")
        long[share] = (long[att] / total).where(total > 0)
    long["FG2_PCT"] = (long["FG2M"] / long["FG2A"]).where(long["FG2A"] > 0)
    long["FG3_PCT"] = (long["FG3M"] / long["FG3A"]).where(long["FG3A"] > 0)
    return long


def fetch_team_contest(season, season_type, per_mode, timeout, force_refresh):
    """Shots by closest-defender distance, taken and allowed. Long format."""
    frames = []
    for side, endpoint in (("offense", LeagueDashTeamPtShot), ("defense", LeagueDashOppPtShot)):
        wide = None
        for bucket, (code, _) in CONTEST.items():
            def fetch_raw(bucket=bucket, endpoint=endpoint):
                return endpoint(
                    close_def_dist_range_nullable=bucket, per_mode_simple=per_mode,
                    season=season, season_type_all_star=season_type, timeout=timeout,
                ).get_data_frames()[0]

            df = _fetch(safe_cache_name("team_contest", side, code, season, season_type, per_mode),
                        fetch_raw, season, season_type, force_refresh)
            if df is None or df.empty:
                continue
            keep = [c for c in ("FGA", "FGM", "FG2A", "FG2M", "FG3A", "FG3M", "EFG_PCT") if c in df.columns]
            df = df[["TEAM_ID"] + keep].rename(columns={c: f"{code}_{c}" for c in keep})
            wide = df if wide is None else wide.merge(df, on="TEAM_ID", how="outer")

        if wide is None:
            continue
        wide["TEAM"], wide["SIDE"] = wide["TEAM_ID"].map(TEAM_ABBR), side
        frames.append(_tidy_contest(wide, ["TEAM_ID", "TEAM", "SIDE"]))

    if not frames:
        return pd.DataFrame()

    contest = pd.concat(frames, ignore_index=True)
    contest["SEASON"], contest["SEASON_TYPE"] = season, season_type
    return contest


def _season_gp_floor(df, share=0.3):
    """A games-played floor that scales with the season so far.

    A fixed floor of 20 games empties the pool in November. This is 30% of the
    most games anyone has played: 25 of 82 at the end, 3 or 4 two weeks in.
    """
    most = pd.to_numeric(df["GP"], errors="coerce").max()
    return 1 if pd.isna(most) else max(1, int(round(share * most)))


def player_contest(df_master, season, season_type="Regular Season", players=None,
                   top_n=25, min_gp=None, min_fga=5.0):
    """Open / contested shot mix per player, from the master's CLOSESTDEF_ columns.

    Same long shape as the team `contest` frame, so plot_contest_stack and
    plot_openness take either. `players` picks names; otherwise the top_n by
    attempts among players with min_gp games and min_fga attempts a game.
    min_gp defaults to 30% of the most games anyone has played that season.
    """
    df = df_master.reset_index()
    df = df[(df["SEASON"] == season) & (df["SEASON_TYPE"] == season_type)]
    if min_gp is None:
        min_gp = _season_gp_floor(df)
    wide = df[["PLAYER_NAME", "TEAM_ABBREVIATION", "GP"]].copy()
    for code, _ in CONTEST.values():
        for stat in ("FGA", "FGM", "FG2A", "FG2M", "FG3A", "FG3M", "EFG_PCT"):
            src = f"CLOSESTDEF_{code}_{stat}"
            if src in df.columns:
                wide[f"{code}_{stat}"] = pd.to_numeric(df[src], errors="coerce")
    wide["_fga"] = wide[[f"{code}_FGA" for code, _ in CONTEST.values()]].sum(axis=1)

    if players is not None:
        wide = wide[wide["PLAYER_NAME"].isin(players)]
    else:
        wide = wide[(wide["GP"] >= min_gp) & (wide["_fga"] >= min_fga)].nlargest(top_n, "_fga")

    # a traded player has one row per team; the label keeps them apart
    dup = wide["PLAYER_NAME"].duplicated(keep=False)
    wide["PLAYER"] = wide["PLAYER_NAME"].where(~dup, wide["PLAYER_NAME"] + " (" + wide["TEAM_ABBREVIATION"] + ")")
    long = _tidy_contest(wide, ["PLAYER_NAME", "TEAM_ABBREVIATION", "PLAYER"])
    long["SIDE"] = "offense"
    return long


def build_team_data(season, season_type="Regular Season", per_mode="PerGame",
                    timeout=30, force_refresh=False, save=True):
    """Fetch everything for one season. Returns {'teams', 'zones', 'play_types', 'contest'}."""
    print(f"🏀 Team data — {season} {season_type}")
    args = (season, season_type, per_mode, timeout, force_refresh)
    data = {
        "teams": fetch_team_dash(*args),
        "zones": fetch_team_zones(*args),
        "play_types": fetch_team_play_types(*args),
        "contest": fetch_team_contest(*args),
    }

    tag = f"{season.replace('-', '_')}_{season_type.lower().replace(' ', '_')}"
    for name, df in data.items():
        if df.empty:
            print(f"   ⚠️ {name}: nothing came back")
            continue
        print(f"   ✅ {name}: {df.shape[0]} rows, {df['TEAM'].nunique()} teams")
        if save:
            df.to_parquet(DATA_DIR / f"team_{name}_{tag}.parquet", index=False)
    return data


def load_team_data(season, season_type="Regular Season"):
    tag = f"{season.replace('-', '_')}_{season_type.lower().replace(' ', '_')}"
    paths = {name: DATA_DIR / f"team_{name}_{tag}.parquet"
             for name in ("teams", "zones", "play_types", "contest")}
    # a season fetched before `contest` existed has no file for it yet
    return {name: pd.read_parquet(path) if path.exists() else pd.DataFrame()
            for name, path in paths.items()}


# =============================================================================
# CHARTS
# =============================================================================

def _team_heatmap(long, x, value, *, relative, lower_is_better, fmt, title, x_order,
                  sort_by=None, also=None, also_fmt=None, hover=(), scale=1, clabel="",
                  logos=True, height=760, width=800):
    """Team x category heatmap. `relative` recentres each column on the league average.

    Rows are sorted on `sort_by` (default: the first column), best or biggest
    on top; sort_by="team" is alphabetical. `also` prints a second column after
    the value in each cell; `hover` lists extra tooltip columns. `scale` and
    `clabel` put the colour bar in readable units.
    """
    df = long.dropna(subset=[value]).copy()
    df["_v"] = (df[value] - df.groupby(x)[value].transform("mean") if relative else df[value]) * scale
    df["_label"] = df[value].map(fmt)
    if sort_by is None:
        sort_by = next(c for c in x_order if c in set(df[x]))
    if also is not None:
        df["_label"] = df["_label"] + "  ·  " + df[also].map(also_fmt)

    # bokeh draws the first factor at the bottom, so reverse to read top-down —
    # and on a lower-is-better metric the smallest value is the one that leads
    if sort_by == "team":
        order = sorted(df["TEAM"].unique(), reverse=True)
    else:
        order = (df[df[x] == sort_by]
                 .sort_values("_v", ascending=not lower_is_better)["TEAM"].tolist())
    df["TEAM"] = pd.Categorical(df["TEAM"], categories=order, ordered=True)
    df[x] = pd.Categorical(df[x], categories=[c for c in x_order if c in set(df[x])], ordered=True)
    df = df.sort_values([x, "TEAM"])

    # Warm is always "good" on a quality metric, so a lower-is-better one flips
    # the map. On a share there's no good or bad; warm just means more than usual.
    cmap = "RdBu" if lower_is_better else "RdBu_r"
    opts = dict(cmap=cmap, colorbar=True, height=height, width=width, title=title,
                xrotation=35, tools=["hover"], xlabel="", ylabel="", clabel=clabel)
    columns = [c for c in x_order if c in set(df[x])]
    if logos and _all_teams(order):
        # room for the logos left of the first column, in column widths
        def pad(plot, element):
            plot.state.x_range.range_padding = 0.3
            plot.state.x_range.range_padding_units = "absolute"
        opts["hooks"] = [pad, _logo_axis_hook(order, (columns[0], -0.66))]
    if relative:
        m = df["_v"].abs().max()
        opts["clim"] = (-m, m)

    vdims = ["_v", value] + [c for c in hover if c in df.columns and c != value]
    heat = hv.HeatMap(df, kdims=[x, "TEAM"], vdims=vdims).opts(**opts)
    text = hv.Labels(df, kdims=[x, "TEAM"], vdims="_label").opts(text_font_size="7.5pt", text_color="#222")
    return heat * text


def plot_shot_profile(zones, side="offense", metric="SHARE", relative=True, sort_by=None,
                      with_share=True, **kw):
    """Where a team's shots come from (offense) or where it gives them up (defense).

    metric: SHARE (of attempts), FG_PCT, or PPS (points per shot). With
    relative=True the colour is the gap to league average and the cell text is
    the raw value. On FG_PCT / PPS the cell also shows that zone's share of the
    team's attempts (with_share=False to drop it) — efficiency on 5% of the
    shots and on 30% of them are different facts.
    """
    quality = metric in ("FG_PCT", "PPS")
    points = metric != "PPS"                            # shares and FG% read best in percentage points
    kw.setdefault("scale", 100 if points else 1)
    if relative:
        kw.setdefault("clabel", "percentage points vs league average" if points
                      else "points per shot vs league average")
    if quality and with_share:
        kw.setdefault("also", "SHARE")
        kw.setdefault("also_fmt", lambda v: f"{v * 100:.0f}%")
    kw.setdefault("hover", ("SHARE", "FG_PCT", "PPS", "FGA"))
    verb = {"offense": "take" if not quality else "make", "defense": "allow"}[side]
    what = {"SHARE": "share of attempts", "FG_PCT": "FG%", "PPS": "points per shot"}[metric]
    return _team_heatmap(
        zones[zones["SIDE"] == side], "ZONE", metric,
        relative=relative,
        lower_is_better=quality and side == "defense",
        fmt=(lambda v: f"{v:.2f}") if metric == "PPS" else (lambda v: f"{v * 100:.0f}"),
        title=f"Shots teams {verb} by zone — {what}" + (" vs league avg" if relative else ""),
        x_order=list(ZONE_LABELS.values()), sort_by=sort_by, **kw,
    )


def plot_play_types(play_types, side="offense", metric="POSS_PCT", relative=True, sort_by=None, **kw):
    """What a team runs (offense) or has run against it (defense).

    metric: POSS_PCT (how often) or PPP (points per possession — how well).
    """
    quality = metric == "PPP"
    kw.setdefault("hover", ("POSS_PCT", "PPP", "POSS", "PERCENTILE"))
    kw.setdefault("scale", 1 if quality else 100)
    if relative:
        kw.setdefault("clabel", "points per possession vs league average" if quality
                      else "percentage points vs league average")
    whose = {"offense": "Play types run", "defense": "Play types faced"}[side]
    what = {"POSS_PCT": "frequency", "PPP": "points per possession"}[metric]
    return _team_heatmap(
        play_types[play_types["SIDE"] == side], "PLAY_TYPE", metric,
        relative=relative,
        lower_is_better=quality and side == "defense",
        fmt=(lambda v: f"{v:.2f}") if quality else (lambda v: f"{v * 100:.0f}"),
        title=f"{whose} — {what}" + (" vs league avg" if relative else ""),
        x_order=list(PLAY_TYPES.values()), sort_by=sort_by, **kw,
    )


def plot_team_quadrant(teams, x, y, *, x_lower_better=False, y_lower_better=False,
                       title=None, xlabel=None, ylabel=None, label_max=40, logos=True,
                       height=620, width=760):
    """Two team stats against each other, with league-average crosshairs.

    A lower-is-better stat gets its axis inverted, so up and to the right is
    always the good corner whichever stats are plotted. Past label_max points
    (a player pool, not 30 teams) only the ones furthest from the middle are
    named; the rest are on hover.
    """
    df = teams.dropna(subset=[x, y])
    logos = logos and _all_teams(df["TEAM"])
    # with logos the dots stay, invisible, so the tooltip still has something to hit
    points = df.hvplot.scatter(x=x, y=y, size=600 if logos else 70, color="#3b6ea5",
                               alpha=0 if logos else 0.75, hover_cols=["TEAM"])
    named = df
    if len(df) > label_max:
        dist = ((df[x] - df[x].mean()) / df[x].std()) ** 2 + ((df[y] - df[y].mean()) / df[y].std()) ** 2
        named = df.loc[dist.nlargest(label_max).index]
    labels = named.hvplot.labels(x=x, y=y, text="TEAM", text_baseline="bottom",
                                 text_font_size="8pt", text_color="#222")
    lines = (hv.VLine(df[x].mean()) * hv.HLine(df[y].mean())).opts(
        hv.opts.VLine(color="#999", line_dash="dashed", line_width=1),
        hv.opts.HLine(color="#999", line_dash="dashed", line_width=1),
    )
    chart = lines * points.opts(hooks=[_logo_point_hook(df, x, y)]) if logos else lines * points * labels
    return chart.opts(
        height=height, width=width, title=title or f"{y} vs {x}",
        # the axis is flipped, so the low (good) values sit right / up
        xlabel=(xlabel or x) + ("  (axis flipped: lower is better →)" if x_lower_better else ""),
        ylabel=(ylabel or y) + ("  (axis flipped: lower is better →)" if y_lower_better else ""),
        invert_xaxis=x_lower_better, invert_yaxis=y_lower_better,
    )


def plot_rebounding(teams, **kw):
    return plot_team_quadrant(
        teams, "OREB_PCT", "DREB_PCT", title="Rebounding — both ends",
        xlabel="Offensive rebound %", ylabel="Defensive rebound %", **kw)


def plot_turnovers(teams, **kw):
    return plot_team_quadrant(
        teams, "TM_TOV_PCT", "OPP_TOV_PCT", x_lower_better=True,
        title="Turnovers — protect it vs force them",
        xlabel="Own turnover %", ylabel="Opponent turnover % (forced)", **kw)


def plot_transition(teams, **kw):
    return plot_team_quadrant(
        teams, "PTS_FB", "OPP_PTS_FB", y_lower_better=True,
        title="Transition — fast-break points scored vs allowed",
        xlabel="Fast-break points per game", ylabel="Opponent fast-break points", **kw)


def plot_ratings(teams, **kw):
    return plot_team_quadrant(
        teams, "OFF_RATING", "DEF_RATING", y_lower_better=True,
        title="Offense vs defense — points per 100 possessions",
        xlabel="Offensive rating", ylabel="Defensive rating", **kw)


def _stacked_barh(long, entity, segment, value, *, segments, colors, order, title, xlabel,
                  fmt, hover=(), label_min=0.06, logos=True, height=None, width=860):
    """One horizontal bar per entity, split into ordered segments.

    `order` lists the entities top to bottom. Segments wider than label_min of
    the longest bar get their value printed; the rest are left to the tooltip.
    """
    df = long[long[segment].isin(segments)].dropna(subset=[value]).copy()
    df = df[df[entity].isin(order)]
    df[segment] = pd.Categorical(df[segment], categories=segments, ordered=True)
    # bokeh draws the first factor at the bottom
    df[entity] = pd.Categorical(df[entity], categories=order[::-1], ordered=True)
    df = df.sort_values([entity, segment])

    df["_end"] = df.groupby(entity, observed=True)[value].cumsum()
    df["_mid"] = df["_end"] - df[value] / 2
    df["_label"] = df[value].map(fmt)
    longest = df.groupby(entity, observed=True)[value].sum().max()

    bars = hv.Bars(df, kdims=[entity, segment],
                   vdims=[value] + [c for c in hover if c in df.columns and c != value]).opts(
        # cmap alone colours by the stack dimension. Passing color=segment as
        # well builds a second colour column in the wrong row order.
        stacked=True, invert_axes=True, cmap=dict(zip(segments, colors)),
        line_color=PAPER, line_width=1.5, bar_width=0.82, tools=["hover"],
        legend_position="top", show_legend=True,
    )
    # dark ink on the light fills, white on the dark ones
    light_fill = {seg: _luminance(col) > 0.36 for seg, col in zip(segments, colors)}
    overlay = bars
    for seg in segments:
        part = df[(df[segment] == seg) & (df[value] >= label_min * longest)]
        if part.empty:
            continue
        overlay = overlay * hv.Labels(part, kdims=[entity, "_mid"], vdims="_label").opts(
            invert_axes=True, text_font_size="7.5pt", text_color=INK if light_fill[seg] else PAPER)

    n = df[entity].nunique()
    left = 0
    if logos and _all_teams(order):
        left = -0.045 * longest                         # a strip left of zero for the logos
        overlay = overlay.opts(hv.opts.Bars(hooks=[_logo_axis_hook(order[::-1], left / 2)]))
    return overlay.opts(
        height=height or max(260, 24 * n + 130), width=width, title=title,
        # options name the axes before the flip: "y" is the value axis here
        xlabel="", ylabel=xlabel, ylim=(left, longest * 1.02), show_grid=False,
    )


def _luminance(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def plot_zone_stack(zones, side="offense", metric="SHARE", sort_by=None, **kw):
    """Stacked bar per team by court zone.

    metric="SHARE"   each segment is that zone's share of the team's attempts;
                     every bar is 100%.
    metric="POINTS"  each segment is the points that zone produces per 100
                     shots (share x points per shot). FG% itself can't be
                     stacked — percentages don't add — but this can, and the bar
                     length is the team's overall points per 100 shots, so the
                     chart shows where the efficiency comes from.

    sort_by: a zone label, or "total" (POINTS only). Defaults to "Rim" for
    SHARE and "total" for POINTS. On defense, POINTS sorts stingiest first.
    """
    df = zones[zones["SIDE"] == side].copy()
    segments = [z for z in ZONE_LABELS.values() if z in set(df["ZONE"])]
    df["SHARE_PCT"] = df["SHARE"] * 100
    df["POINTS"] = df["SHARE"] * df["PPS"] * 100
    value = {"SHARE": "SHARE_PCT", "POINTS": "POINTS"}[metric]

    sort_by = sort_by or ("Rim" if metric == "SHARE" else "total")
    key = (df.groupby("TEAM")[value].sum() if sort_by == "total"
           else df[df["ZONE"] == sort_by].set_index("TEAM")[value])
    best_low = metric == "POINTS" and side == "defense"
    order = key.sort_values(ascending=best_low).index.tolist()

    verb = {"offense": "take", "defense": "allow"}[side]
    scored = {"offense": "score", "defense": "allow"}[side]
    title, xlabel = {
        "SHARE": (f"Where teams {verb} their shots — share of attempts by zone", "% of field-goal attempts"),
        "POINTS": (f"Where the points come from — points teams {scored} per 100 shots, by zone",
                   "Points per 100 field-goal attempts"),
    }[metric]
    return _stacked_barh(
        df, "TEAM", "ZONE", value, segments=segments, colors=ZONE_COLORS[:len(segments)],
        order=order, title=title, xlabel=xlabel, fmt=lambda v: f"{v:.0f}",
        hover=("SHARE_PCT", "FG_PCT", "PPS", "FGA"), **kw)


PLAYER_ZONES = {          # master column prefix -> (zone label, points per make)
    "Restricted_Area": ("Rim", 2), "In_The_Paint_Non_RA": ("Paint (non-rim)", 2),
    "Mid_Range": ("Mid-range", 2), "Corner_3": ("Corner 3", 3), "Above_the_Break_3": ("Above-break 3", 3),
}


def player_zones(df_master, season, season_type="Regular Season", min_gp=None, min_fga=8.0):
    """Shots by court zone, one row per player per zone — the player version of the team zones table.

    Columns: PLAYER, TEAM, GP, ZONE, FGA, FGM (per game), FG_PCT, PPS (points per
    shot), SHARE (of his own attempts), TOTAL_FGA. min_gp=None uses 30% of the
    most games anyone has played; min_fga is zone attempts per game summed over
    the five zones (heaves from the backcourt aren't counted).
    """
    df = df_master.reset_index()
    df = df[(df["SEASON"] == season) & (df["SEASON_TYPE"] == season_type)].copy()
    if df.empty:
        raise ValueError(f"no rows for {season} {season_type}")
    if min_gp is None:
        min_gp = _season_gp_floor(df)
    dup = df["PLAYER_NAME"].duplicated(keep=False)
    df["PLAYER"] = df["PLAYER_NAME"].where(~dup, df["PLAYER_NAME"] + " (" + df["TEAM_ABBREVIATION"] + ")")

    parts = []
    for prefix, (zone, points) in PLAYER_ZONES.items():
        fga = pd.to_numeric(df[f"{prefix}_FGA"], errors="coerce").fillna(0)
        fgm = pd.to_numeric(df[f"{prefix}_FGM"], errors="coerce").fillna(0)
        parts.append(pd.DataFrame({
            "PLAYER": df["PLAYER"], "TEAM": df["TEAM_ABBREVIATION"],
            "GP": pd.to_numeric(df["GP"], errors="coerce"), "ZONE": zone, "FGA": fga, "FGM": fgm,
            "FG_PCT": fgm / fga.where(fga > 0), "PPS": points * fgm / fga.where(fga > 0),
        }))
    out = pd.concat(parts, ignore_index=True)
    out["TOTAL_FGA"] = out.groupby("PLAYER")["FGA"].transform("sum")
    out["SHARE"] = out["FGA"] / out["TOTAL_FGA"].where(out["TOTAL_FGA"] > 0)
    out = out[(out["GP"] >= min_gp) & (out["TOTAL_FGA"] >= min_fga)].reset_index(drop=True)
    out.attrs.update(season=season, season_type=season_type, min_gp=min_gp, min_fga=min_fga)
    return out


def plot_player_zone_stack(df_master, season, season_type="Regular Season", metric="POINTS",
                           players=None, top_n=20, sort_by=None, ascending=False,
                           min_gp=None, min_fga=8.0, **kw):
    """Stacked bar per player by court zone — the player version of plot_zone_stack.

    metric="POINTS"  each segment is the points that zone gives him per 100
                     shots (share of his shots x points per shot there). The bar
                     length is his overall points per 100 shots, so it shows how
                     efficient he is and where the efficiency comes from.
    metric="SHARE"   each segment is that zone's share of his attempts; every
                     bar is 100%.

    players: a list of names to show exactly those (the volume floors are
        dropped for them). Otherwise the top_n of the qualified pool, ranked by
        sort_by: a zone label ("Rim", "Mid-range", "Above-break 3", ...) or
        "total" (POINTS only). Defaults to "total" for POINTS and "Rim" for
        SHARE. ascending=True shows the bottom of the pool instead.
    """
    if players is not None:
        zones = player_zones(df_master, season, season_type, min_gp=0, min_fga=0)
        wanted = [players] if isinstance(players, str) else list(players)
        base = zones["PLAYER"].str.replace(r" \([A-Z]{3}\)$", "", regex=True)
        missing = [p for p in wanted if p not in set(base)]
        if missing:
            print(f"⚠️ not in {season} {season_type}: {missing}")
        zones = zones[base.isin(wanted)]
    else:
        zones = player_zones(df_master, season, season_type, min_gp=min_gp, min_fga=min_fga)
    if zones.empty:
        raise ValueError("no players left to plot — lower min_gp / min_fga or check the names")

    df = zones.copy()
    segments = [zone for zone, _ in PLAYER_ZONES.values()]
    df["SHARE_PCT"] = df["SHARE"] * 100
    df["POINTS"] = (df["SHARE"] * df["PPS"] * 100).fillna(0)
    value = {"SHARE": "SHARE_PCT", "POINTS": "POINTS"}[metric]

    sort_by = sort_by or ("Rim" if metric == "SHARE" else "total")
    if sort_by == "total" and metric == "SHARE":
        raise ValueError('every SHARE bar totals 100 — sort_by a zone, e.g. sort_by="Rim"')
    key = (df.groupby("PLAYER")[value].sum() if sort_by == "total"
           else df[df["ZONE"] == sort_by].set_index("PLAYER")[value])
    if key.empty:
        raise ValueError(f"sort_by must be 'total' or one of {segments}")
    order = key.sort_values(ascending=ascending).index.tolist()
    pool = len(order)
    if players is None and top_n:
        order = order[:top_n]

    by = "total" if sort_by == "total" else sort_by.lower()
    who = (f"{len(order)} selected players" if players is not None else
           f"{'bottom' if ascending else 'top'} {len(order)} of {pool} by {by}")
    if players is None:
        print(f"pool: {pool} players with {zones.attrs['min_gp']:g}+ games and {min_fga:g}+ shots a game")
    title, xlabel = {
        "SHARE": (f"Where they shoot from, {season} — {who}", "% of field-goal attempts"),
        "POINTS": (f"Points per 100 shots by zone, {season} — {who}", "Points per 100 field-goal attempts"),
    }[metric]
    return _stacked_barh(
        df, "PLAYER", "ZONE", value, segments=segments, colors=ZONE_COLORS[:len(segments)],
        order=order, title=title, xlabel=xlabel, fmt=lambda v: f"{v:.0f}",
        hover=("SHARE_PCT", "FG_PCT", "PPS", "FGA", "TEAM"), logos=False, **kw)


def _contest_cols(shot_type):
    return {"all": ("SHARE", "FGA", "EFG_PCT", "shots"), "2": ("SHARE_2", "FG2A", "FG2_PCT", "2-pointers"),
            "3": ("SHARE_3", "FG3A", "FG3_PCT", "3-pointers")}[str(shot_type)]


def _contest_entity(contest):
    return "PLAYER" if "PLAYER" in contest.columns else "TEAM"


def plot_contest_stack(contest, side="offense", shot_type="all", sort_by="open", **kw):
    """How contested the shots are: share of attempts by closest-defender distance.

    Takes the team `contest` frame or the output of player_contest().
    shot_type: "all", "2" or "3". sort_by: "open" (open + wide open share,
    most open on top; on defense the stingiest on top) or "tight".
    """
    share, att, _, noun = _contest_cols(shot_type)
    entity = _contest_entity(contest)
    df = contest[contest["SIDE"] == side].copy()
    df["SHARE_PCT"] = df[share] * 100

    is_open = df["CONTEST"].isin(OPEN_LABELS)
    open_share = df[is_open].groupby(entity)["SHARE_PCT"].sum().dropna()
    most_open_first = (sort_by == "open") != (side == "defense")
    order = open_share.sort_values(ascending=not most_open_first).index.tolist()

    who = "players" if entity == "PLAYER" else "teams"
    verb = {"offense": "take", "defense": "allow"}[side]
    return _stacked_barh(
        df, entity, "CONTEST", "SHARE_PCT", segments=CONTEST_LABELS, colors=CONTEST_COLORS,
        order=order, title=f"How open are the {noun} {who} {verb}? — by closest defender",
        xlabel=f"% of {noun}", fmt=lambda v: f"{v:.0f}", hover=(att, "EFG_PCT", "FG3_PCT"), **kw)


def contest_summary(contest, side="offense", shot_type="all", min_attempts=None):
    """One row per team/player: open share and efficiency on open vs tight shots.

    min_attempts drops anyone below that many attempts a game of this shot type.
    It defaults to 2 for players — a big man's handful of threes would otherwise
    top the chart — and to no floor for teams.
    """
    share, att, _, _ = _contest_cols(shot_type)
    made, pts = {"all": (None, None), "2": ("FG2M", 2), "3": ("FG3M", 3)}[str(shot_type)]
    entity = _contest_entity(contest)
    df = contest[contest["SIDE"] == side].copy()
    # points per shot in each bucket; EFG_PCT x 2 is the same thing for all shots
    df["_pts"] = df["EFG_PCT"] * 2 * df[att] if made is None else df[made] * pts
    df["_open"] = df["CONTEST"].isin(OPEN_LABELS)

    g = df.groupby([entity, "_open"]).agg(att=(att, "sum"), pts=("_pts", "sum")).unstack("_open")
    out = pd.DataFrame({
        "OPEN_SHARE": g[("att", True)] / (g[("att", True)] + g[("att", False)]),
        "OPEN_PPS": g[("pts", True)] / g[("att", True)],
        "TIGHT_PPS": g[("pts", False)] / g[("att", False)],
        "ATTEMPTS": g[("att", True)] + g[("att", False)],
    }).reset_index()
    wide_open = df[df["CONTEST"] == CONTEST_LABELS[-1]].set_index(entity)[share]
    out["WIDE_OPEN_SHARE"] = out[entity].map(wide_open)
    if min_attempts is None:
        min_attempts = 2.0 if entity == "PLAYER" else 0.0
    return out[out["ATTEMPTS"] >= min_attempts].reset_index(drop=True)


def plot_openness(contest, side="offense", shot_type="3", min_attempts=None, **kw):
    """Shot creation vs shot making: how many looks are open, and what they're worth.

    x is the share of attempts with the defender 4+ feet away, y is points per
    shot on those attempts. On defense both axes flip, so up and to the right
    is still the good corner (few open looks allowed, and little damage on them).
    """
    _, _, _, noun = _contest_cols(shot_type)
    entity = _contest_entity(contest)
    summary = contest_summary(contest, side, shot_type, min_attempts).rename(columns={entity: "TEAM"})
    defense = side == "defense"
    verb = "allowed" if defense else "taken"
    return plot_team_quadrant(
        summary, "OPEN_SHARE", "OPEN_PPS", x_lower_better=defense, y_lower_better=defense,
        title=f"Open {noun} {verb} — how many vs how well",
        xlabel=f"Share of {noun} that are open (defender 4+ ft)",
        ylabel=f"Points per shot on open {noun}", **kw)


# =============================================================================
# LEFT OPEN, OR GOT OPEN?
# =============================================================================

def open_look_profile(df_master, season, season_type="Regular Season", min_gp=None, min_fg3a=2.0):
    """Who shoots wide-open threes because the defense lets them?

    Tracking data records how far the defender was, not why. The why has to be
    inferred from two things together: how often a player's threes are wide
    open (6+ ft), and how well he shoots them overall. Each is cut into thirds
    among qualified shooters:

      left open    most wide-open looks, worst accuracy   -> defenses help off him
      earns it     most wide-open looks, best accuracy    -> open despite being a threat
                   (movement, scheme, or a teammate drawing the help)
      guarded      fewest wide-open looks, best accuracy  -> defenses stay attached

    It is a read on the pattern, not a measurement of intent. min_gp defaults
    to 30% of the most games anyone has played that season; early in a season
    the accuracy half of this is a few dozen shots per player, so expect names
    to move.
    """
    df = df_master.reset_index()
    df = df[(df["SEASON"] == season) & (df["SEASON_TYPE"] == season_type)]
    if min_gp is None:
        min_gp = _season_gp_floor(df)
    codes = [code for code, _ in CONTEST.values()]
    att = df[[f"CLOSESTDEF_{c}_FG3A" for c in codes]].apply(pd.to_numeric, errors="coerce").fillna(0)
    made = df[[f"CLOSESTDEF_{c}_FG3M" for c in codes]].apply(pd.to_numeric, errors="coerce").fillna(0)

    out = df[["PLAYER_NAME", "TEAM_ABBREVIATION", "GP"]].copy()
    out["FG3A"] = att.sum(axis=1)
    out["FG3_PCT"] = made.sum(axis=1) / out["FG3A"].where(out["FG3A"] > 0)
    out["WIDE_OPEN_SHARE"] = att["CLOSESTDEF_6P_FG3A"] / out["FG3A"].where(out["FG3A"] > 0)
    out["WIDE_OPEN_FG3_PCT"] = made["CLOSESTDEF_6P_FG3M"] / att["CLOSESTDEF_6P_FG3A"].where(att["CLOSESTDEF_6P_FG3A"] > 0)
    out = out[(out["GP"] >= min_gp) & (out["FG3A"] >= min_fg3a)].copy()
    out = out.dropna(subset=["WIDE_OPEN_SHARE", "FG3_PCT"])
    if len(out) < 9:
        raise ValueError(
            f"Only {len(out)} players in {season} {season_type} have {min_gp}+ games and "
            f"{min_fg3a}+ threes a game; lower min_gp or min_fg3a.")

    share_third = pd.qcut(out["WIDE_OPEN_SHARE"].rank(method="first"), 3, labels=["low", "mid", "high"])
    pct_third = pd.qcut(out["FG3_PCT"].rank(method="first"), 3, labels=["low", "mid", "high"])
    out["READ"] = ""
    out.loc[(share_third == "high") & (pct_third == "low"), "READ"] = "left open"
    out.loc[(share_third == "high") & (pct_third == "high"), "READ"] = "earns it"
    out.loc[(share_third == "low") & (pct_third == "high"), "READ"] = "guarded"
    return out.sort_values("WIDE_OPEN_SHARE", ascending=False).reset_index(drop=True)


def plot_open_looks(profile, **kw):
    """Wide-open share of a player's threes against how well he shoots them.

    Bottom right is left open on purpose; top right gets open anyway; top left
    is guarded tightly because he can shoot.
    """
    return plot_team_quadrant(
        profile.rename(columns={"PLAYER_NAME": "TEAM"}), "WIDE_OPEN_SHARE", "FG3_PCT",
        title="Left open, or got open? — wide-open threes vs accuracy",
        xlabel="Share of 3-point attempts that are wide open (defender 6+ ft)",
        ylabel="3-point %", **kw)


# =============================================================================
# REPORT CARD
# =============================================================================
# Each skill is graded against the other 29 teams: A = top six, B = 7-12, C = 13-18, D = 19-24, F = bottom six.
# Lower-is-better stats (points allowed, opponent percentages, turnover rate) are flipped so the best team ranks 1st.

_SET_PLAYS = ("Cut", "Handoff", "Isolation", "Off screen", "P&R handler", "P&R roll man", "Post up", "Spot up")
_LOWER_BETTER = {"HALF_COURT_DEF", "ISO_DEF", "PNR_DEF", "SPOTUP_DEF", "TRANSITION_DEF", "RIM_DEF", "THREE_DEF",
                 "THREE_RATE_DEF", "TM_TOV", "CONTEST_EFG_DEF"}


def _skill_values(data):
    """Raw value per team for every skill (see report_card for the grading)."""
    T = data["teams"].set_index("TEAM")
    Z, P, C = data["zones"], data["play_types"], data["contest"]

    def ppp(side, types=None, exclude=()):
        x = P[P.SIDE == side]
        x = x[x.PLAY_TYPE.isin(types)] if types else x[~x.PLAY_TYPE.isin(exclude)]
        return (x.PPP * x.POSS).groupby(x.TEAM).sum() / x.POSS.groupby(x.TEAM).sum()

    rim = Z[Z.ZONE == "Rim"].set_index(["TEAM", "SIDE"]).FG_PCT
    th = Z[Z.ZONE.isin(["Corner 3", "Above-break 3"])]
    tight = C[C.SIDE == "defense"].assign(TIGHT=lambda x: x.CONTEST.isin(["Very tight (0-2 ft)", "Tight (2-4 ft)"]))
    t = tight[tight.TIGHT]
    vals = {
        "HALF_COURT_OFF": ppp("offense", _SET_PLAYS), "HALF_COURT_DEF": ppp("defense", _SET_PLAYS),
        "ISO_OFF": ppp("offense", ("Isolation",)), "ISO_DEF": ppp("defense", ("Isolation",)),
        "PNR_OFF": ppp("offense", ("P&R handler", "P&R roll man")), "PNR_DEF": ppp("defense", ("P&R handler", "P&R roll man")),
        "SPOTUP_OFF": ppp("offense", ("Spot up",)), "SPOTUP_DEF": ppp("defense", ("Spot up",)),
        "TRANSITION_OFF": ppp("offense", ("Transition",)), "TRANSITION_DEF": ppp("defense", ("Transition",)),
        "RIM_OFF": rim.xs("offense", level=1), "RIM_DEF": rim.xs("defense", level=1),
        "THREE_OFF": th[th.SIDE == "offense"].groupby("TEAM")[["FGM", "FGA"]].sum().pipe(lambda x: x.FGM / x.FGA),
        "THREE_DEF": th[th.SIDE == "defense"].groupby("TEAM")[["FGM", "FGA"]].sum().pipe(lambda x: x.FGM / x.FGA),
        "THREE_RATE_DEF": th[th.SIDE == "defense"].groupby("TEAM").FGA.sum() / Z[Z.SIDE == "defense"].groupby("TEAM").FGA.sum(),
        "CONTEST_RATE_DEF": t.groupby("TEAM").FGA.sum() / tight.groupby("TEAM").FGA.sum(),
        "CONTEST_EFG_DEF": t.groupby("TEAM").apply(lambda x: (x.EFG_PCT * x.FGA).sum() / x.FGA.sum()),
        "AST_RATIO": T.AST_RATIO, "TM_TOV": T.TM_TOV_PCT, "OREB": T.OREB_PCT, "DREB": T.DREB_PCT,
    }
    return pd.DataFrame(vals).dropna(), T


def report_card(season="2026-27", data=None):
    """Letter grades per team and skill, plus net rating for context. Best team on each skill ranks first."""
    data = data or load_team_data(season)
    raw, T = _skill_values(data)
    grades = pd.DataFrame(index=raw.index)
    for c in raw.columns:
        ranked = raw[c].rank(ascending=c in _LOWER_BETTER, method="first")       # lower is better: smallest value ranks 1st
        grades[c] = ranked.map(lambda r: "A" if r <= 6 else "B" if r <= 12 else "C" if r <= 18 else "D" if r <= 24 else "F")
    grades["NET_RATING"] = T.loc[grades.index, "NET_RATING"].round(1)
    grades = grades.sort_values("NET_RATING", ascending=False)
    grades.attrs["season"] = season
    return grades


def report_card_raw(season="2026-27", data=None):
    """The numbers behind report_card, for the same teams."""
    data = data or load_team_data(season)
    raw, _ = _skill_values(data)
    return raw.round(3)
