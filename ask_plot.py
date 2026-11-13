"""Ask for a chart in plain English.

    from ask_plot import PlotAssistant
    nba = PlotAssistant(df_master)

    nba.ask("top 15 players by drives per 36 this season")
    nba.ask("who allows the lowest FG% at the rim? at least 3 rim shots defended a game")
    nba.ask("usage vs true shooting for 20-point scorers last season")
    nba.ask("how has Curry's three-point attempt rate changed over his career")
    nba.ask("now do the same for Jokic")                   # follow-ups see the previous chart

How it works, and why it is built this way:

  1. Claude reads the request and returns a small JSON *spec* — which chart, which
     columns, which season, which filters. It never writes or runs code.
  2. The spec is validated here: every column must exist, every filter is a
     (column, operator, number) triple, every season must be in the data.
  3. The spec is handed to the plotting functions already in this project.

So a wrong answer shows up as a wrong-looking spec you can read (it is printed
with every chart), not as arbitrary code running against your data. If the
spec fails validation the error is sent back once for a repair.

Needs the Anthropic SDK and a key in the environment:
    pip install anthropic            # 0.125.0 is the newest release for Python 3.9
    export ANTHROPIC_API_KEY=...     # in the shell that launches Jupyter, not in a notebook cell
"""
import difflib
import json

import pandas as pd

import bbref_stats
import monthly_splits
import player_profiles
import team_stats
from nba_viz_utils import ACTIVE_SEASON, DATA_DIR, is_lower_better, plot_interactive_barh, plot_stat_over_time

MODEL = "claude-opus-5-5"

CHARTS = {
    "player_bar": "Ranked horizontal bars of one stat across players. Uses: stat, season, season_type, filters, top_n, direction.",
    "player_scatter": "Two stats against each other, one dot per player. Uses: stat (x), stat_y, season, season_type, filters.",
    "player_trend_seasons": "One stat across seasons for named players (a career line). Uses: stat, players, season_type.",
    "player_trend_months": "One stat month by month within a season for named players. Uses: stat, players, season. MONTHLY columns only.",
    "player_month_change": "Biggest risers/fallers from the previous month to the latest. Uses: stat, season, top_n, direction. MONTHLY columns only.",
    "player_season_change": "Biggest risers/fallers between two seasons. Uses: stat, season_from, season, top_n, direction, filters.",
    "player_profile": "One player's style profile (tendency and skill across 13 groups). Uses: players (one name), season.",
    "player_compare": "Style profiles of 2-4 players side by side. Uses: players, season.",
    "player_similar": "Table of players with the most similar playing style. Uses: players (one name), season, top_n.",
    "team_bar": "Ranked bars of one team stat. Uses: stat, season, top_n, direction. TEAM columns only.",
    "team_scatter": "Two team stats against each other, logos as markers. Uses: stat (x), stat_y, season. TEAM columns only.",
    "team_shot_zones": "Heatmap, team x court zone. Uses: side, metric (SHARE | FG_PCT | PPS), season.",
    "team_zone_stack": "Stacked bar per team by court zone. Uses: side, metric (SHARE | POINTS), season.",
    "team_play_types": "Heatmap, team x play type. Uses: side, metric (POSS_PCT | PPP), season.",
    "team_contest": "Stacked bar per team: shots by closest-defender distance. Uses: side, shot_type (all | 2 | 3), season.",
    "none": "The request can't be answered with these charts or columns. Say why in `explanation`.",
}

_NULLABLE_STR = {"type": ["string", "null"]}
SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "chart": {"type": "string", "enum": list(CHARTS)},
        "stat": _NULLABLE_STR,
        "stat_y": _NULLABLE_STR,
        "season": _NULLABLE_STR,
        "season_from": _NULLABLE_STR,
        "season_type": {"type": "string", "enum": ["Regular Season", "Playoffs"]},
        "players": {"type": "array", "items": {"type": "string"}},
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "op": {"type": "string", "enum": [">=", ">", "<=", "<", "==", "!="]},
                    "value": {"type": "number"},
                },
                "required": ["column", "op", "value"],
                "additionalProperties": False,
            },
        },
        "top_n": {"type": ["integer", "null"]},
        "direction": {"type": "string", "enum": ["top", "bottom", "both"]},
        "side": {"type": "string", "enum": ["offense", "defense"]},
        "metric": _NULLABLE_STR,
        "shot_type": {"type": "string", "enum": ["all", "2", "3"]},
        "title": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["chart", "stat", "stat_y", "season", "season_from", "season_type", "players", "filters",
                 "top_n", "direction", "side", "metric", "shot_type", "title", "explanation"],
    "additionalProperties": False,
}

INSTRUCTIONS = """You turn a basketball fan's request into a chart specification for an NBA stats notebook.
You do not write code. You return one JSON object that matches the schema; a program validates it
and draws the chart with existing functions.

Choosing well:
- Use column names exactly as they appear in the catalogue below. Never invent one. If nothing fits,
  return chart "none" and say in `explanation` what is missing and the closest thing that exists.
- Player counting stats in the master are PER GAME. For "per 36" or to compare players with different
  minutes, use the *_PER36 columns. For "how often" questions prefer the share, rate and frequency columns.
- Rates and percentages need a sample. Unless the user sets their own, add filters so the chart isn't
  topped by a player with ten minutes: a games floor, a minutes floor (MIN is minutes per game), and a
  volume floor on the matching attempts column when there is one (FG3_PCT -> FG3A, DEF_RIM_FG_PCT ->
  DEF_RIM_FGA, Restricted_Area_FG_PCT -> Restricted_Area_FGA). The user message says how many games the
  current season has, so scale the games floor to that; a full season is 82.
- Lower-is-better stats (opponent FG% allowed, turnovers) are handled by the charts: "top" always means
  best. Use direction "bottom" only when the user asks for the worst.
- "This season" is the active season given in the user message; "last season" is the one before.
- Filters are numeric comparisons only. To restrict to named players use `players`, not a filter.
- Write player names in full as they'd appear on a roster ("Stephen Curry", not "Steph").
- `title` is a short chart title in plain English. `explanation` is one or two sentences saying what you
  chose and any assumption you made (the floor you picked, which column you read "X" as).
- Fields a chart doesn't use: null for the nullable ones, [] for lists, any allowed value for the enums.
"""

PREFIX_LEGEND = """Column prefixes in the player master:
  DRIVE_ drives · PULL_UP_ / CATCH_SHOOT_ shot types · PAINT_TOUCH_ ELBOW_TOUCH_ POST_TOUCH_ touches there
  DRIBB_{0,1,2,3_6,7P}_ shots by dribbles before · TOUCH_{LT_2,2_6,6P}_ by seconds held · SC_{24_22..4_0}_ by shot clock
  CLOSESTDEF_{0_2,2_4,4_6,6P}_ by closest defender in feet · SHOT_ overall shot split
  Synergy play types, each with _POSS_PCT (frequency) _PPP (points per possession) _PERCENTILE _EFG_PCT:
    ISO_ isolation · PNRBH_ P&R ball handler · RM_ roll man · POST_ post up · SPOT_ spot up · DHO_ handoff
    CUT_ cut · OFFSCREEN_ off screen · TRANS_ transition · PUTBACK_ putback
  D_FG_PCT / PCT_PLUSMINUS overall defended FG% and vs expected · DEF_RIM_ at the rim
  DEF2_ DEF3_ DEFLT6_ DEFLT10_ DEFGT15_ defended shots by type/distance (_PLUSMINUS = vs expected, lower is better)
  CLUTCH_ last five minutes within five points · PCT_ share of team total while on court
  ONCOURT_ / OFFCOURT_ / ONOFF_ team rating with him on or off the floor (if present)
  BBR_ Basketball-Reference advanced: BBR_PER, BBR_BPM, BBR_OBPM, BBR_DBPM, BBR_VORP, BBR_WS, BBR_WS_48"""


class SpecError(ValueError):
    """The spec named something that isn't in the data. The message is written for the model to act on."""


def _suggest(name, options, n=4):
    close = difflib.get_close_matches(str(name).upper(), [str(o).upper() for o in options], n=n, cutoff=0.5)
    lookup = {str(o).upper(): o for o in options}
    return [lookup[c] for c in close]


class PlotAssistant:
    """Holds the data and the conversation. One per notebook is enough."""

    def __init__(self, df_master, model=MODEL, client=None, verbose=True):
        self.model, self.client, self.verbose = model, client, verbose
        master = df_master.reset_index() if isinstance(df_master.index, pd.MultiIndex) else df_master
        master = player_profiles.add_profile_columns(master)
        bbref_file = DATA_DIR / "bbref_advanced.parquet"
        self.advanced = pd.read_parquet(bbref_file) if bbref_file.exists() else None
        if self.advanced is not None and "BBR_BPM" not in master.columns:
            master = _quiet(bbref_stats.add_bbref_advanced, master, self.advanced)
        self.master = master
        self.seasons = sorted(master["SEASON"].unique())
        self._monthly, self._teams, self._profiles = None, {}, {}
        self.last = None                                # (question, spec) of the previous chart
        self._system = None

    # ------------------------------------------------------------------ data
    @property
    def monthly(self):
        if self._monthly is None:
            try:
                self._monthly = monthly_splits.load_monthly("player")
            except FileNotFoundError:
                raise SpecError("No monthly data on disk. Run monthly_splits.build_monthly([season]) first.")
        return self._monthly

    def team_data(self, season):
        if season not in self._teams:
            try:
                data = team_stats.load_team_data(season)
            except FileNotFoundError:
                data = {}
            if not data or data["teams"].empty:
                raise SpecError(f"No team data on disk for {season}. Run team_stats.build_team_data('{season}') first.")
            self._teams[season] = data
        return self._teams[season]

    def profiles(self, season):
        if season not in self._profiles:
            self._profiles[season] = player_profiles.build_profiles(self.master, season)
        return self._profiles[season]

    # ------------------------------------------------------------------ prompt
    def catalogue(self):
        """Everything the model may refer to. Stable between calls, so it can be cached."""
        identity = {"SEASON", "SEASON_TYPE", "PLAYER_NAME", "PLAYER_ID", "TEAM_ABBREVIATION", "TEAM_ID",
                    "SEASON_ID", "PLAY_TYPE", "TYPE_GROUPING", "PLAYER_POSITION", "BBR_ID", "BBR_TEAM", "BBR_POS"}
        derived = [c for c in self.master.columns if c in player_profiles.LABELS or c.endswith("_PER36")]
        lines = ["CHARTS"] + [f"  {name}: {desc}" for name, desc in CHARTS.items()]
        lines += ["", "PLAYER MASTER columns (one row per player-season):",
                  ", ".join(c for c in self.master.columns if c not in identity and c not in derived)]
        lines += ["", "PLAYER derived columns, with meanings:"]
        lines += [f"  {c}: {player_profiles.label(c)}" for c in sorted(derived)]
        lines += ["", PREFIX_LEGEND]
        try:
            lines += ["", "MONTHLY columns (player_trend_months, player_month_change):",
                      ", ".join(c for c in self.monthly.columns if c not in identity and not c.startswith("MONTH"))]
        except SpecError:
            lines += ["", "MONTHLY: not available."]
        team_cols = next((list(d["teams"].columns) for d in self._all_team_data()), None)
        lines += ["", "TEAM columns (team_bar, team_scatter):",
                  ", ".join(c for c in team_cols if c not in ("TEAM", "TEAM_ID", "TEAM_NAME", "SEASON", "SEASON_TYPE"))
                  if team_cols else "not available."]
        lines += ["", f"Seasons in the data: {', '.join(self.seasons)}"]
        return "\n".join(lines)

    def _all_team_data(self):
        for season in reversed(self.seasons):
            try:
                yield self.team_data(season)
            except SpecError:
                continue

    def _context(self):
        """The parts that change from day to day. Kept out of the system prompt so it stays cacheable."""
        active = self.master[(self.master["SEASON"] == ACTIVE_SEASON) & (self.master["SEASON_TYPE"] == "Regular Season")]
        games = int(pd.to_numeric(active["GP"], errors="coerce").max()) if len(active) else 0
        team_seasons = [s for s in self.seasons if (DATA_DIR / f"team_teams_{s.replace('-', '_')}_regular_season.parquet").exists()]
        text = (f"Active season: {ACTIVE_SEASON} (most games any player has played so far: {games}). "
                f"Team charts have data for: {', '.join(team_seasons) or 'no season yet'}.")
        if self.last:
            text += (f"\nPrevious request: {self.last[0]!r}\nPrevious spec: {json.dumps(self.last[1])}\n"
                     f"If the new request is a follow-up, start from that spec and change only what was asked.")
        return text

    # ------------------------------------------------------------------ model
    def plan(self, question, repair=None):
        """Ask Claude for a spec. `repair` is (bad_spec, error) from a failed attempt."""
        client = self.client or _default_client()
        if self._system is None:
            self._system = [{"type": "text", "text": INSTRUCTIONS + "\n" + self.catalogue(),
                             "cache_control": {"type": "ephemeral"}}]
        messages = [{"role": "user", "content": f"{self._context()}\n\nRequest: {question}"}]
        if repair:
            bad_spec, error = repair
            messages += [{"role": "assistant", "content": json.dumps(bad_spec)},
                         {"role": "user", "content": f"That spec could not be drawn: {error}\nReturn a corrected spec."}]
        response = client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=self._system,
            messages=messages,
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SPEC_SCHEMA}},
            # if a safety classifier declines the request, the API re-runs it on a fallback model
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("The model declined this request.")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("The reply was cut off before the spec was complete.")
        text = next(block.text for block in response.content if block.type == "text")
        return json.loads(text)

    # ------------------------------------------------------------------ validation
    def _column(self, name, frame, what):
        if name is None:
            raise SpecError(f"`{what}` is required for this chart.")
        if name not in frame.columns:
            raise SpecError(f"column {name!r} does not exist here. Closest: {_suggest(name, frame.columns)}")
        if not pd.api.types.is_numeric_dtype(pd.to_numeric(frame[name], errors="coerce")):
            raise SpecError(f"column {name!r} is not numeric.")
        return name

    def _season(self, season, what="season"):
        if season not in self.seasons:
            raise SpecError(f"{what} {season!r} is not in the data. Seasons: {self.seasons}")
        return season

    def _names(self, names, pool, at_least=1, at_most=None):
        if len(names) < at_least:
            raise SpecError(f"this chart needs at least {at_least} name(s) in `players`.")
        out = []
        for name in names[:at_most]:
            exact = [p for p in pool if p.lower() == name.lower()]
            if exact:
                out.append(exact[0])
                continue
            inside = [p for p in pool if name.lower() in p.lower()]
            close = inside if len(inside) == 1 else difflib.get_close_matches(name, list(pool), n=1, cutoff=0.75)
            if not close:
                raise SpecError(f"no player named {name!r} in that data. Closest: "
                                f"{difflib.get_close_matches(name, list(pool), n=4, cutoff=0.4)}")
            out.append(close[0])
        return out

    def _filtered(self, frame, spec):
        ops = {">=": "ge", ">": "gt", "<=": "le", "<": "lt", "==": "eq", "!=": "ne"}
        for f in spec["filters"]:
            col = self._column(f["column"], frame, "filter column")
            frame = frame[getattr(pd.to_numeric(frame[col], errors="coerce"), ops[f["op"]])(f["value"])]
        if frame.empty:
            raise SpecError("no rows pass those filters; loosen them.")
        return frame

    # ------------------------------------------------------------------ render
    def render(self, spec):
        """Validate a spec and draw it. Raises SpecError with a message the model can act on."""
        chart = spec["chart"]
        if chart == "none":
            return None
        if chart not in CHARTS:
            raise SpecError(f"unknown chart {chart!r}")
        season = self._season(spec["season"] or ACTIVE_SEASON)
        season_type, title, top_n = spec["season_type"], spec["title"], spec["top_n"] or 15
        if chart.startswith("team_"):
            return self._render_team(chart, spec, season, title, top_n)

        if chart in ("player_trend_months", "player_month_change"):
            stat = self._column(spec["stat"], self.monthly, "stat")
            if season not in set(self.monthly["SEASON"]):
                raise SpecError(f"monthly data only covers {sorted(set(self.monthly['SEASON']))}")
            if chart == "player_trend_months":
                names = self._names(spec["players"], set(self.monthly["PLAYER_NAME"]), at_most=6)
                return monthly_splits.plot_monthly_trend(self.monthly, names, stat, season)
            return monthly_splits.plot_month_change(self.monthly, stat, season, top_n=top_n,
                                                    direction=_direction(spec, both_ok=True))

        if chart in ("player_profile", "player_compare", "player_similar"):
            profiles = self.profiles(season)
            names = self._names(spec["players"], profiles.pool.index, at_least=2 if chart == "player_compare" else 1,
                                at_most=4)
            if chart == "player_profile":
                player_profiles.player_profile(profiles, names[0])
                return player_profiles.plot_player_profile(profiles, names[0])
            if chart == "player_compare":
                return player_profiles.compare_players(profiles, names)
            return player_profiles.similar_players(profiles, names[0], n=top_n)

        stat = self._column(spec["stat"], self.master, "stat")
        if chart == "player_trend_seasons":
            names = self._names(spec["players"], set(self.master["PLAYER_NAME"]), at_most=6)
            return plot_stat_over_time(self.master, names, stat, season_type=season_type)

        if chart == "player_season_change":
            season_from = self._season(spec["season_from"] or self.seasons[self.seasons.index(season) - 1], "season_from")
            frame = self.master[(self.master["SEASON_TYPE"] == "Regular Season")
                                & self.master["SEASON"].isin([season_from, season])]
            if not spec["filters"]:
                # a floor in each season: 30% of that season's games so far, 15 minutes a game
                floor = 0.3 * frame.groupby("SEASON")["GP"].transform("max")
                frame = frame[(frame["GP"] >= floor) & (frame["MIN"] >= 15)]
            frame = self._filtered(frame, spec).drop(columns=["BBR_MP"], errors="ignore")
            return bbref_stats.plot_season_change(frame, stat, season_from, season, top_n=top_n,
                                                  direction=_direction(spec, both_ok=True), name_col="PLAYER_NAME")

        frame = self.master[(self.master["SEASON"] == season) & (self.master["SEASON_TYPE"] == season_type)]
        frame = self._filtered(frame, spec)
        if spec["players"]:
            frame = frame[frame["PLAYER_NAME"].isin(self._names(spec["players"], set(frame["PLAYER_NAME"])))]
        if chart == "player_bar":
            worst = spec["direction"] == "bottom"
            return plot_interactive_barh(frame, x_col=stat, y_col="PLAYER_NAME", season=season, season_type=season_type,
                                         title=title, top_n=top_n, height=max(320, 24 * min(top_n, len(frame)) + 120),
                                         ascending=is_lower_better(stat) if worst else None)
        if chart == "player_scatter":
            y = self._column(spec["stat_y"], self.master, "stat_y")
            return team_stats.plot_team_quadrant(
                frame.rename(columns={"PLAYER_NAME": "TEAM"}), stat, y, title=title, logos=False, label_max=35,
                x_lower_better=is_lower_better(stat), y_lower_better=is_lower_better(y),
                xlabel=player_profiles.label(stat), ylabel=player_profiles.label(y))
        raise SpecError(f"chart {chart!r} is not implemented")

    def _render_team(self, chart, spec, season, title, top_n):
        data = self.team_data(season)
        side, metric = spec["side"], spec["metric"]

        def pick(allowed, default):
            if metric is None:
                return default
            if metric not in allowed:
                raise SpecError(f"metric for {chart} must be one of {allowed}, not {metric!r}")
            return metric

        if chart == "team_shot_zones":
            return team_stats.plot_shot_profile(data["zones"], side, metric=pick(("SHARE", "FG_PCT", "PPS"), "SHARE"))
        if chart == "team_zone_stack":
            return team_stats.plot_zone_stack(data["zones"], side, metric=pick(("SHARE", "POINTS"), "SHARE"))
        if chart == "team_play_types":
            return team_stats.plot_play_types(data["play_types"], side, metric=pick(("POSS_PCT", "PPP"), "POSS_PCT"))
        if chart == "team_contest":
            if data["contest"].empty:
                raise SpecError(f"No open/contested data on disk for {season}. Re-run team_stats.build_team_data('{season}').")
            return team_stats.plot_contest_stack(data["contest"], side, shot_type=spec["shot_type"])
        teams = data["teams"]
        stat = self._column(spec["stat"], teams, "stat")
        if chart == "team_bar":
            worst = spec["direction"] == "bottom"
            return plot_interactive_barh(teams, x_col=stat, y_col="TEAM", season=season, title=title,
                                         top_n=spec["top_n"] or 30, height=760,
                                         ascending=is_lower_better(stat) if worst else None)
        y = self._column(spec["stat_y"], teams, "stat_y")
        return team_stats.plot_team_quadrant(teams, stat, y, title=title, x_lower_better=is_lower_better(stat),
                                             y_lower_better=is_lower_better(y))

    # ------------------------------------------------------------------ the one call
    def ask(self, question):
        """Plain-English request -> chart. Returns whatever the plotting function returns."""
        spec = self.plan(question)
        try:
            result = self.render(spec)
        except SpecError as error:
            if self.verbose:
                print(f"↻ first spec didn't validate ({error}); asking for a fix")
            spec = self.plan(question, repair=(spec, str(error)))
            result = self.render(spec)                  # a second failure is raised to you
        self.last = (question, spec)
        if self.verbose:
            print(f"💬 {spec['explanation']}")
            print(f"   spec: {describe(spec)}")
        return result

    def reset(self):
        """Forget the previous chart, so the next request isn't read as a follow-up."""
        self.last = None


def _direction(spec, both_ok=False):
    d = spec["direction"]
    if d == "both":
        return "both" if both_ok else "up"
    return "up" if d == "top" else "down"


def describe(spec):
    """The spec on one line, leaving out what the chart doesn't use."""
    parts = [spec["chart"]]
    for key in ("stat", "stat_y", "season_from", "season", "metric"):
        if spec.get(key):
            parts.append(f"{key}={spec[key]}")
    if spec["chart"].startswith("team_") and spec["chart"] not in ("team_bar", "team_scatter"):
        parts.append(f"side={spec['side']}")
    if spec.get("players"):
        parts.append(f"players={spec['players']}")
    if spec.get("filters"):
        parts.append("filters: " + ", ".join(f"{f['column']} {f['op']} {f['value']:g}" for f in spec["filters"]))
    if spec.get("top_n"):
        parts.append(f"{spec['direction']} {spec['top_n']}")
    return " · ".join(parts)


def _quiet(fn, *args, **kwargs):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _default_client():
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("The Anthropic SDK isn't installed. Run:  pip install anthropic") from None
    try:
        return anthropic.Anthropic()                    # reads ANTHROPIC_API_KEY from the environment
    except anthropic.AnthropicError as error:
        raise RuntimeError(
            "No Anthropic credentials found. Set ANTHROPIC_API_KEY in the shell that starts Jupyter "
            f"(not in a notebook cell, which would save the key in the file). Details: {error}") from None
