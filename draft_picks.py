"""What a draft pick is worth, in the same dollars as trade_value's player values.

    from draft_picks import *
    curve = slot_values(model, df_master)                     # what each pick slot (1-60) has been worth
    odds = pick_odds(proj, df_master, absences=KNOWN_ABSENCES)    # this season's pick: each team's chance of every slot
    pick_value("GSW", 2027, curve=curve, odds=odds)           # one pick, in $M
    pick_value("WAS", 2027, curve=curve, odds=odds, protect=8)    # top-8 protected: the value if it conveys
    all_picks(curve, odds)                                    # every team's 1st and 2nd, this year and the next few

What a slot is worth: for every draft in DRAFTS, each pick's surplus over his
first CONTROL_YEARS seasons (the rookie deal): wins (player_ranking) x the
price of a win (trade_value.price_of_a_win: what teams pay veterans per win)
minus his salary, as a share of the cap, plus the title odds he adds as in
trade_value (TITLE_VALUE). A pick who never plays is worth 0.
Averaged by slot and smoothed so value never rises with a later pick.

Which slot a team will get: this season's projection (team_projection) plays
the season out thousands of times; every simulated season gets a lottery
(2019 odds for the 14 non-playoff teams, worst record first) and the rest of
the order by record. Later drafts are less certain: a team's chances are mixed
with an average team's by how much team strength carries over from one season
to the next (fitted from past seasons), once per year ahead.

Value is in $M at the cap of the seasons the player would play (cap growing
CAP_GROWTH a year), like trade_value. Not counted: what the player is worth
after his rookie deal (re-signing rights), and that a contender may prefer
help now (pass discount= to mark future value down).
"""
import contextlib
import io

import numpy as np
import pandas as pd

from nba_viz_utils import SALARY_CAP, SEASONS, cached_fetch, norm_name

__all__ = ["slot_values", "pick_odds", "pick_value", "all_picks", "carry_over"]

DRAFTS = range(2015, 2023)            # drafts whose rookie deals have played out in the data
CONTROL_YEARS = 4                     # rookie-scale seasons; second-rounders are counted over the same span
CAP_GROWTH = 0.07
CURRENT = max(SEASONS)
DRAFT_NOW = int(CURRENT[:4]) + 1      # the draft this season's results decide
LOTTERY_ODDS = np.array([140, 140, 140, 125, 105, 90, 75, 60, 45, 30, 20, 15, 10, 5], float)   # chances of #1, per 1000
LOTTERY_DRAWS = 4


def _draft_history():
    from nba_api.stats.endpoints import DraftHistory
    d = cached_fetch("draft_history", lambda: DraftHistory(league_id="00", timeout=60).get_data_frames()[0])
    d["SEASON"] = d["SEASON"].astype(int)
    return d


def slot_values(model, df_master, drafts=DRAFTS, years=CONTROL_YEARS, price=None, verbose=True, title_value=None, title_odds=None):
    """What each pick slot (1-60) has been worth: surplus over the first `years` seasons, as a share of the cap.

    Columns: RAW (that slot's average), VALUE (smoothed, never rising with a
    later pick), N (picks averaged), HIT (share who produced any surplus).
    """
    from sklearn.isotonic import IsotonicRegression
    from player_ranking import rank_players
    from salaries import load_salaries
    from trade_value import TITLE_VALUE, _title_gain, fit_title_odds, price_of_a_win
    title_value = TITLE_VALUE if title_value is None else title_value
    with contextlib.redirect_stdout(io.StringIO()):
        price = price_of_a_win(model, df_master) if price is None else price
        beta, field = (title_odds or fit_title_odds()) if title_value else (None, None)
        _, replacement = model.war
        wins = {}
        for s in SEASONS[:-1]:
            r = rank_players(model, df_master, s, min_min=0, min_gp=0).assign(PID=lambda t: t["PLAYER_ID"].astype(int))
            g = r.groupby("PID")
            value = g["WINS"].sum() * price                               # wins, as a share of the cap
            if title_value:                                               # plus the title odds he adds (as in trade_value)
                pts = ((r["RATING"] + replacement).clip(lower=0) * r["MIN"] / 48)
                gain = _title_gain(pts.to_numpy(), beta, field) * (r["GP"] / 82).clip(upper=1).to_numpy() * title_value
                value = value + pd.Series(gain, index=r.index).groupby(r["PID"]).sum()
            wins[s] = value
        pay = load_salaries(save=False)
    paid = pay.set_index(["name_key", "SEASON"])["pct_cap"]
    dh = _draft_history()
    dh = dh[dh["SEASON"].isin(list(drafts)) & dh["OVERALL_PICK"].between(1, 60)]
    rows = []
    for _, p in dh.iterrows():
        total = 0.0
        for k in range(years):
            s = f"{p['SEASON'] + k}-{str(p['SEASON'] + k + 1)[2:]}"
            if s not in wins:
                continue
            w = float(wins[s].get(int(p["PERSON_ID"]), 0.0))             # already priced, share of the cap
            cost = float(paid.get((norm_name(p["PLAYER_NAME"]), s), 0.0)) if w else 0.0
            total += max(w, 0.0) - cost
        rows.append({"PICK": int(p["OVERALL_PICK"]), "DRAFT": p["SEASON"], "Player": p["PLAYER_NAME"], "SURPLUS": total})
    picks = pd.DataFrame(rows)
    g = picks.groupby("PICK")["SURPLUS"]
    curve = pd.DataFrame({"RAW": g.mean(), "N": g.size(), "HIT": picks.assign(H=picks["SURPLUS"] > 0).groupby("PICK")["H"].mean()})
    curve = curve.reindex(range(1, 61))
    smooth = curve["RAW"].rolling(5, center=True, min_periods=1).mean()
    iso = IsotonicRegression(increasing=False).fit(smooth.index, smooth.fillna(smooth.mean()), sample_weight=curve["N"].fillna(1))
    curve["VALUE"] = iso.predict(curve.index)
    curve.attrs.update(price=price, drafts=f"{min(drafts)}-{max(drafts)}", years=years, picks=picks, title_value=title_value)
    if verbose:
        cap = SALARY_CAP[CURRENT]
        print(f"Pick values from the {min(drafts)}-{max(drafts)} drafts, surplus over {years} seasons "
              f"(a win priced at {price:.2%} of the cap): " + ", ".join(
                  f"#{k} ${curve.at[k, 'VALUE'] * cap / 1e6:.0f}M" for k in (1, 3, 5, 10, 15, 20, 30, 45, 60)) + " at this season's cap")
    return curve


def carry_over(proj):
    """How much of a team's strength carries into the next season: correlation of team ratings, season to season."""
    seasons = sorted(proj.final)
    pairs = [pd.concat([proj.final[a], proj.final[b]], axis=1, join="inner").dropna() for a, b in zip(seasons, seasons[1:])]
    return float(np.mean([p.corr().iloc[0, 1] for p in pairs]))


def _lottery(order, rng):
    """order: the 14 lottery teams, worst first. Returns them in pick order."""
    left, weights, top = list(order), list(LOTTERY_ODDS), []
    for _ in range(LOTTERY_DRAWS):
        w = np.array(weights) / np.sum(weights)
        i = rng.choice(len(left), p=w)
        top.append(left.pop(i))
        weights.pop(i)
    return top + left


def pick_odds(proj, df_master, absences=None, sims=10_000, seed=4):
    """This season's draft (DRAFT_NOW): each team's chance of each first-round slot, from the season projection.

    Returns a table indexed by team with columns 1..30 (first round) and the
    expected second-round slot (31-60 by record, no lottery).
    """
    from team_projection import project_season
    with contextlib.redirect_stdout(io.StringIO()):
        _, s = project_season(proj, df_master, sims=sims, absences=absences, return_sims=True)
    teams, wins, playoffs = np.array(s["teams"]), s["wins"], s["playoffs"]
    rng = np.random.default_rng(seed)
    first = np.zeros((len(teams), 30))
    second = np.zeros((len(teams), 30))
    noise = rng.random(wins.shape) * 1e-3
    for i in range(len(wins)):
        w = wins[i] + noise[i]
        out = np.flatnonzero(~playoffs[i])
        lottery = out[np.argsort(w[out])][:14]
        rest = np.setdiff1d(np.arange(len(teams)), lottery)
        rest = rest[np.argsort(w[rest])]
        order = _lottery(lottery, rng) + list(rest)
        first[order, np.arange(len(order))] += 1
        second[np.argsort(w), np.arange(len(teams))] += 1
    out = pd.DataFrame(first / len(wins), index=teams, columns=range(1, 31))
    out["SECOND_ROUND"] = (second / len(wins)) @ np.arange(31, 61)
    out.attrs["second"] = pd.DataFrame(second / len(wins), index=teams, columns=range(31, 61))
    out.attrs["draft"] = DRAFT_NOW
    return out


def _slot_probs(team, year, rnd, odds, carry):
    """A team's chance of each slot in `rnd` of the `year` draft, mixed toward an average team for later drafts."""
    now = odds.attrs["second"].loc[team].to_numpy() if rnd == 2 else odds.loc[team, list(range(1, 31))].to_numpy(float)
    average = np.full(30, 1 / 30)
    keep = carry ** max(year - DRAFT_NOW, 0)
    return keep * now + (1 - keep) * average


def pick_value(team, year, rnd=1, curve=None, odds=None, protect=None, carry=None, discount=0.0, proj=None, detail=False):
    """A team's pick in $M: expected surplus over the rookie deal, at the cap of the seasons he'd play.

    protect=N: top-N protected, so only the slots after N count (and the chance it
    conveys is returned with detail=True). discount: yearly rate to mark down
    value that arrives later (0 = none).
    """
    carry = carry_over(proj) if carry is None and proj is not None else (0.6 if carry is None else carry)
    p = _slot_probs(team, year, rnd, odds, carry)
    slots = np.arange(1, 31) + (30 if rnd == 2 else 0)
    vals = curve.loc[slots, "VALUE"].to_numpy()
    if protect:
        p = np.where(slots > protect, p, 0.0)
    ahead = year - DRAFT_NOW + 1                                  # seasons until his rookie year
    caps = SALARY_CAP[CURRENT] * (1 + CAP_GROWTH) ** (ahead + np.arange(CONTROL_YEARS))
    per_slot = vals * caps.mean() * (1 + discount) ** -ahead / 1e6
    value = float(p @ per_slot)
    if detail:
        exp_slot = float((p * slots).sum() / max(p.sum(), 1e-9))
        return {"TEAM": team, "YEAR": year, "ROUND": rnd, "VALUE_M": round(value, 1), "CONVEYS": round(float(p.sum()), 2),
                "EXPECTED_SLOT": round(exp_slot, 1), "TOP_5": round(float(p[:5].sum()), 2) if rnd == 1 else 0.0}
    return value


def all_picks(curve, odds, years=3, carry=0.6, discount=0.0):
    """Every team's own 1st and 2nd for this draft and the next `years - 1`, in $M, most valuable first.

    Ignores who owns each pick now (swaps and trades): it's the value of that
    team's pick, wherever it ends up.
    """
    rows = []
    for t in odds.index:
        for y in range(DRAFT_NOW, DRAFT_NOW + years):
            for rnd in (1, 2):
                rows.append(pick_value(t, y, rnd, curve, odds, carry=carry, discount=discount, detail=True))
    out = pd.DataFrame(rows)
    wide = out.pivot_table(index="TEAM", columns=["YEAR", "ROUND"], values="VALUE_M")
    wide.columns = [f"{y}_{'1st' if r == 1 else '2nd'}" for y, r in wide.columns]
    first_now = f"{DRAFT_NOW}_1st"
    wide = wide.sort_values(first_now, ascending=False)
    wide.insert(0, "EXP_SLOT_NOW", out[(out["YEAR"] == DRAFT_NOW) & (out["ROUND"] == 1)].set_index("TEAM")["EXPECTED_SLOT"])
    return wide.round(1)
