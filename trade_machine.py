"""Trade machine and value chart: what each team gives and gets, in the same dollars as trade_value.

    from trade_value import *
    from draft_picks import *
    from trade_machine import *
    tv = trade_value(model, df_master)
    curve = slot_values(model, df_master)
    odds = pick_odds(proj, df_master, absences=KNOWN_ABSENCES)

    trade({"NOP": ["Trey Murphy III"], "MEM": ["Jaylen Wells", Pick("MEM", 2027), Pick("MEM", 2029)]},
          tv, curve=curve, odds=odds, proj=proj, df_master=df_master, absences=KNOWN_ABSENCES)
    value_chart(tv, top=30)              # the most valuable contracts
    value_chart(tv, team="MEM")          # one team's

Two teams: each list is what that team SENDS, and the other team gets it.
Three or more: give each asset its destination, ("Trey Murphy III", "MEM").

What each column means
  VALUE_OUT_M / VALUE_IN_M   trade value sent and received ($M, trade_value's
      TRADE_VALUE_M for players, pick_value for picks); NET_M = in - out
  SALARY_OUT_M / SALARY_IN_M   this season's salaries
  SALARY_CHECK   a rough salary-matching test: a team taking in more salary than
      it sends may take in up to SALARY_MATCH x outgoing + SALARY_CUSHION_M. The
      real rules depend on cap room and the aprons, which this doesn't know.
  WINS_BEFORE / WINS_AFTER, PLAYOFFS_BEFORE / _AFTER   this season's projection
      with the players moved (team_projection's moves=), when proj is given.

A value is what the contract is worth to an average team; it doesn't know fit,
need or timeline (a rebuilding team values picks and young players more).

Two ways to value a player (basis=):
  "wins"    trade_value's TRADE_VALUE_M: wins x one price per win, plus title
            odds and re-signing rights. Every win costs the same, so it misses
            what teams pay extra for (high-usage scoring) and sees value above
            the max contract (Wembanyama).
  "market"  MARKET_VALUE_M from add_market_value: what the market salary model
            says he'd be paid over the rest of his deal minus what he's owed.
            It pays what teams pay, scorers included, but can't see past the max
            (stars look fairly paid) and has no re-signing or title value.
"""
import difflib
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nba_viz_utils import norm_name

__all__ = ["Pick", "trade", "value_chart", "add_market_value", "SALARY_MATCH", "SALARY_CUSHION_M"]

BLUE, ORANGE, GRID, MUTED, INK = "#2a78d6", "#eb6834", "#e4e3de", "#8a897f", "#0b0b0b"   # as in salaries.py
SALARY_MATCH = 1.25
SALARY_CUSHION_M = 0.25


@dataclass(frozen=True)
class Pick:
    """A draft pick: whose pick it is (team), the draft year, the round and an optional top-N protection."""
    team: str
    year: int
    rnd: int = 1
    protect: Optional[int] = None

    def __str__(self):
        s = f"{self.year} {self.team} {'1st' if self.rnd == 1 else '2nd'}"
        return s + (f" (top-{self.protect} protected)" if self.protect else "")


_BASIS = {"wins": "TRADE_VALUE_M", "market": "MARKET_VALUE_M"}


def add_market_value(tv, deals):
    """tv with MARKET_VALUE_M added: the market rate over the rest of his deal minus what he's owed, in $M.

    deals: salary_market.contracts_vs_market(score_market(df_master)). Players
    the market model didn't score (too few games) are left blank.
    """
    k = deals.assign(_k=deals["Player"].map(norm_name)).drop_duplicates("_k").set_index("_k")
    out = tv.copy()
    keys = out["Player"].map(norm_name)
    out["MARKET_VALUE_M"] = keys.map(k["MARKET_M"] - k["OWED_M"])
    return out


def _value(row, basis):
    col = _BASIS[basis]
    if col not in row.index:
        raise ValueError("basis='market' needs tv from add_market_value(tv, deals)")
    v = row[col]
    if pd.isna(v):
        print(f"⚠️ no market value for {row['Player']} (not scored by the market model); using the wins-based value")
        return float(row["TRADE_VALUE_M"])
    return float(v)


def _player(tv, name):
    hit = tv[tv["Player"].map(norm_name) == norm_name(name)]
    if hit.empty:
        close = difflib.get_close_matches(name, tv["Player"].tolist(), n=3, cutoff=0.6)
        raise ValueError(f"{name!r} isn't in the trade-value table (players under contract)."
                         + (f" Did you mean {', '.join(close)}?" if close else ""))
    return hit.iloc[0]


def _moves(sends):
    """(asset, from, to) for every asset."""
    teams = list(sends)
    out = []
    for frm, assets in sends.items():
        for a in assets:
            if isinstance(a, tuple):
                asset, to = a
            elif len(teams) == 2:
                asset, to = a, teams[1 - teams.index(frm)]
            else:
                raise ValueError(f"with {len(teams)} teams, give each asset a destination: ({a!r}, 'TEAM')")
            if to not in sends:
                sends_to = ", ".join(teams)
                raise ValueError(f"{asset} goes to {to}, which isn't in the trade ({sends_to}); add it with an empty list")
            out.append((asset, frm, to))
    return out


def trade(sends, tv, curve=None, odds=None, proj=None, df_master=None, absences=None, sims=10_000, verbose=True,
          basis="wins"):
    """One row per team: value and salary out and in, a rough salary check, and the projection before and after.

    sends: {team: [what it sends]}; players by name, picks as Pick(team, year, rnd, protect).
    curve, odds: from draft_picks.slot_values / pick_odds, needed when picks are in the deal.
    proj, df_master: to add WINS_BEFORE / WINS_AFTER (team_projection). absences: KNOWN_ABSENCES.
    basis: "wins" (TRADE_VALUE_M) or "market" (MARKET_VALUE_M, needs add_market_value); picks are
    valued the same way either way. The per-asset table is in .attrs["assets"].
    """
    rows = []
    for asset, frm, to in _moves(sends):
        if isinstance(asset, Pick):
            if curve is None or odds is None:
                raise ValueError("picks need curve= (slot_values) and odds= (pick_odds)")
            from draft_picks import pick_value
            value = pick_value(asset.team, asset.year, asset.rnd, curve=curve, odds=odds, protect=asset.protect, proj=proj)
            rows.append({"ASSET": str(asset), "FROM": frm, "TO": to, "VALUE_M": value, "SALARY_M": 0.0, "THROUGH": "", "KIND": "pick"})
        else:
            p = _player(tv, asset)
            if p["TEAM"] != frm and verbose:
                print(f"⚠️ {p['Player']} is listed on {p['TEAM']}, not {frm}")
            rows.append({"ASSET": p["Player"], "FROM": frm, "TO": to, "VALUE_M": _value(p, basis),
                         "SALARY_M": float(p["SALARY_NOW_M"]), "THROUGH": p["THROUGH"], "KIND": "player"})
    assets = pd.DataFrame(rows)

    out = []
    for t in sends:
        gives, gets = assets[assets["FROM"] == t], assets[assets["TO"] == t]
        s_out, s_in = gives["SALARY_M"].sum(), gets["SALARY_M"].sum()
        room = SALARY_MATCH * s_out + SALARY_CUSHION_M
        check = "ok" if s_in <= s_out or s_in <= room else f"takes in ${s_in - room:.1f}M too much"
        out.append({"TEAM": t, "GIVES": ", ".join(gives["ASSET"]), "GETS": ", ".join(gets["ASSET"]),
                    "VALUE_OUT_M": gives["VALUE_M"].sum(), "VALUE_IN_M": gets["VALUE_M"].sum(),
                    "SALARY_OUT_M": s_out, "SALARY_IN_M": s_in, "SALARY_CHECK": check})
    out = pd.DataFrame(out)
    out.insert(out.columns.get_loc("VALUE_IN_M") + 1, "NET_M", out["VALUE_IN_M"] - out["VALUE_OUT_M"])

    if proj is not None and df_master is not None:
        from team_projection import project_season
        import contextlib, io
        moved = {a: to for a, k, to in zip(assets["ASSET"], assets["KIND"], assets["TO"]) if k == "player"}
        with contextlib.redirect_stdout(io.StringIO()):
            before = project_season(proj, df_master, absences=absences, sims=sims).set_index("TEAM")
            after = project_season(proj, df_master, absences=absences, sims=sims, moves=moved).set_index("TEAM")
        out["WINS_BEFORE"] = out["TEAM"].map(before["PROJ_W"])
        out["WINS_AFTER"] = out["TEAM"].map(after["PROJ_W"])
        out["PLAYOFFS_BEFORE"] = out["TEAM"].map(before["PLAYOFFS"])
        out["PLAYOFFS_AFTER"] = out["TEAM"].map(after["PLAYOFFS"])

    out = out.round(2)
    out.attrs["assets"] = assets.round(2)
    if verbose:
        for r in out.itertuples():
            net = ("+" if r.NET_M >= 0 else "") + _money(r.NET_M)
            line = f"{r.TEAM}: gets {_money(r.VALUE_IN_M)}, gives {_money(r.VALUE_OUT_M)}, net {net}"
            if "WINS_AFTER" in out.columns:
                line += f" | projected wins {r.WINS_BEFORE:.1f} → {r.WINS_AFTER:.1f}"
            print(line + ("" if r.SALARY_CHECK == "ok" else f" | salary: {r.SALARY_CHECK}"))
    return out


def _money(v):
    v = round(float(v))
    return f"{'-' if v < 0 else ''}${abs(v):,.0f}M"


def value_chart(tv, top=30, team=None, worst=False, title=None, ax=None, basis="wins"):
    """Horizontal bars of trade value ($M): the most valuable contracts (or one team's, or the worst with worst=True).

    basis: "wins", "market" (needs add_market_value) or "both", which draws the two
    side by side for each player, ordered by the wins-based value.
    """
    if basis != "wins" and "MARKET_VALUE_M" not in tv.columns:
        raise ValueError("basis='market' or 'both' needs tv from add_market_value(tv, deals)")
    sort_col = _BASIS.get(basis, "TRADE_VALUE_M")
    d = tv[tv["TEAM"] == team] if team else tv
    d = d.dropna(subset=[sort_col]).sort_values(sort_col, ascending=worst)
    if top and not team:
        d = d.head(top)
    d = d.iloc[::-1]                                              # largest at the top of the chart
    if d.empty:
        raise ValueError(f"no players for {team!r}")
    labels = [f"{p}  ({t})" if not team else p for p, t in zip(d["Player"], d["TEAM"])]
    if ax is None:
        _, ax = plt.subplots(figsize=(9, (0.42 if basis == "both" else 0.32) * len(d) + 1.2))
    y = np.arange(len(d))
    if basis == "both":
        series = [("TRADE_VALUE_M", "Wins-based", BLUE, 0.19), ("MARKET_VALUE_M", "Market-based", ORANGE, -0.19)]
        vals = np.concatenate([d[c].to_numpy(float) for c, *_ in series])
        span = max(np.nanmax(np.abs(vals)), 1)
        for col, name, color, off in series:
            v = d[col].to_numpy(float)
            ax.barh(y + off, np.nan_to_num(v), height=0.36, color=color, label=name)
            for yi, vi in zip(y + off, v):
                if np.isnan(vi):                                  # not scored by the market model
                    ax.text(0.01 * span, yi, "n/a", va="center", ha="left", fontsize=7, color=MUTED)
                else:
                    ax.text(vi + (0.01 if vi >= 0 else -0.01) * span, yi, _money(vi), va="center",
                            ha="left" if vi >= 0 else "right", fontsize=7, color=INK)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
        v = vals[~np.isnan(vals)]
    else:
        v = d[sort_col].to_numpy(float)
        span = max(np.abs(v).max(), 1)
        ax.barh(y, v, height=0.6, color=np.where(v >= 0, BLUE, ORANGE))
        for yi, vi in zip(y, v):
            ax.text(vi + (0.01 if vi >= 0 else -0.01) * span, yi, _money(vi), va="center",
                    ha="left" if vi >= 0 else "right", fontsize=8, color=INK)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, labels, fontsize=9, color=INK)
    left = v.min() - 0.14 * span if v.min() < 0 else 0          # room for a negative bar's label
    ax.set_xlim(left, max(v.max() * 1.15, 0) + 0.02 * span)
    xlab = {"wins": "Trade value, $M: worth over the contract (wins x price per win) minus what he's owed, plus re-signing rights",
            "market": "Market value, $M: what the market salary model would pay him over the deal minus what he's owed",
            "both": "$M over the rest of the contract"}[basis]
    ax.set_xlabel(xlab, color=MUTED, fontsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    what = {"wins": "trade value", "market": "market value", "both": "trade value (wins-based and market-based)"}[basis]
    default = (f"{team} contracts by {what}" if team
               else f"{'Worst' if worst else 'Most valuable'} {len(d)} contracts by {what}")
    ax.set_title(title or default, loc="left", fontsize=11, color=INK)
    plt.tight_layout()
    return ax
