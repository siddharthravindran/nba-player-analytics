"""Trade value: what a player should be worth over the rest of his contract, minus what he is owed.

    from player_ranking import *
    from trade_value import *
    model = fit_rating_model(df_master)
    tv = trade_value(model, df_master)                     # best contracts first
    tv.tail(15)                                            # the worst
    contract_outlook(model, df_master, "Cooper Flagg")     # the same sum, year by year
    fit_aging_curve(model, df_master)                      # expected change in rating at each age

For every remaining year of a contract:
    worth  = projected wins x the going price of a win x that year's cap
    surplus = worth - salary
TRADE_VALUE_M is the surplus added up. Positive means the contract is an asset
another team would give something for; negative means it costs something to move.

Where each piece comes from
  - Today's level: the rating model over the last three seasons, this season
    weighted most per minute, playoffs included, pulled toward average when
    there is little to go on.
  - Aging: how ratings have changed from one season to the next at each age,
    2015-16 on. Players who fell out of the league aren't in it, so decline at
    older ages is understated.
  - Wins: rating above replacement x minutes a game x expected games, from the
    wins formula in player_ranking. Expected games come from how much of the
    schedule he has played lately, pulled toward normal by as much as history
    says missed games fail to repeat (fit_availability).
  - Price of a win: what teams paid veterans per win over the last three
    finished seasons, as a share of the cap, so it scales with cap_growth.
    Rookie contracts are left out because their pay is fixed by draft slot.
    The price is the same for every win, a star's or a backup's — real teams
    pay more for wins that come from one player, so stars are undervalued here
    and cheap role players overvalued.
  - Options: a player option counts only if it would overpay him (he'd take
    it); a team option only if it's a bargain (the team would).

What it leaves out: draft-pick value, what a team keeps after the contract
ends (Bird rights, restricted free agency), no-trade clauses and whether the
salaries can legally be matched. It prices production, so a teenager with a
high ceiling and poor numbers is undervalued, and so is defense the rating
model can't see.
"""
import numpy as np
import pandas as pd

from bbref_stats import fetch_contract_years, name_key
from nba_viz_utils import SALARY_CAP
from player_ranking import _playoff_ratings, _regular_season, rank_players
from salaries import load_salaries

SEASON_GAMES = 82
RECENCY = (1, 3, 10)                 # per-minute weight on two seasons ago, last season, this season
PRIOR_MINUTES = 1500                 # weighted minutes of "league average" mixed into every estimate
AVAILABILITY = (0.30, 0.95)          # nobody is projected to play fewer or more of the schedule than this
BBR_TEAM = {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX"}
AGES = range(19, 42)
# Title value: a win isn't worth the same to everyone. Wins concentrated in one player move a team's title odds far
# more than the same wins spread over a bench, and salaries can't show it (the max contract caps what stars are paid).
TITLE_VALUE = 2.0                    # what a title is worth to a team, in salary caps: a judgment, not fitted
CONTEXT_RATING = 2.0                 # the team he's valued on: a solid playoff team (points per game better than average)
# Re-signing rights: when a contract ends, the team holding him can re-sign him first. If he's worth more than the most
# he can be paid, that edge is worth something. Counted for RESIGN_YEARS more seasons at the max (MAX_SHARE of the cap,
# by experience as the CBA sets it), only when he's worth more than that, at RESIGN_SHARE of the surplus because he
# could still leave.
RESIGN_YEARS = 3
RESIGN_SHARE = 0.5


def _games_tables(df):
    """Games played and games possible, one row per player and one column per season."""
    gp = df.pivot_table(index="PLAYER_ID", columns="SEASON", values="GP", aggfunc="max")
    season_games = gp.max().clip(upper=SEASON_GAMES)
    games = pd.DataFrame(np.tile(season_games.to_numpy(float), (len(gp), 1)), index=gp.index, columns=gp.columns)
    latest = gp.columns[-1]
    if season_games[latest] < 0.8 * SEASON_GAMES:            # in progress: every team has played a different number
        cur = df[df["SEASON"] == latest]
        team = pd.Series(cur.groupby("TEAM_ABBREVIATION")["GP"].transform("max").to_numpy(float), index=cur["PLAYER_ID"].to_numpy())
        team = team[~team.index.duplicated()]
        games[latest] = team.reindex(gp.index).fillna(team.median())
    return gp.mask(gp > games, games), games


def _recent_share(gp, games, window, weights):
    """Share of his team's games a player appeared in over `window`, later seasons weighted more.

    A season with no row counts as zero games once he has entered the league,
    so a year lost to injury is not skipped.
    """
    g, n = gp[list(window)], games[list(window)]
    started = g.notna().cummax(axis=1)
    w = pd.Series(list(weights), index=list(window), dtype=float)
    played = (g.fillna(0).where(started) * w).sum(axis=1)
    possible = (n.where(started) * w).sum(axis=1)
    return (played / possible).where(possible > 0)


def fit_availability(df_master, min_min=20, verbose=False):
    """(floor, carry): next season's share of games played = floor + carry x recent share.

    Fitted on every past rotation player. Missed games carry over only partly,
    so a player who has just lost a season is not written off for his whole contract.
    """
    df = _regular_season(df_master)
    gp, games = _games_tables(df)
    seasons = list(gp.columns)
    past, came = [], []
    for i in range(1, len(seasons) - 1):
        target, after = seasons[i], seasons[i + 1]
        if games[target].max() < 0.8 * SEASON_GAMES:
            continue
        window = seasons[max(0, i - len(RECENCY)):i]
        share = _recent_share(gp, games, window, RECENCY[-len(window):])
        prev = df[(df["SEASON"] == seasons[i - 1]) & (df["MIN"] >= min_min)]["PLAYER_ID"].unique()
        outcome = gp[target] / games[target]
        outcome = outcome.where(outcome.notna() | gp[after].isna(), 0.0)    # sat out the whole year, then came back
        both = pd.DataFrame({"past": share, "next": outcome}).reindex(prev).dropna()
        past.append(both["past"]); came.append(both["next"])
    past, came = pd.concat(past), pd.concat(came)
    carry, floor = np.polyfit(past, came, 1)
    if verbose:
        print(f"Availability from {len(past):,} player-seasons: next season = {floor:.2f} + {carry:.2f} x recent share "
              f"(someone who played half his games is expected to play {floor + carry * 0.5:.0%} of the next season)")
    return float(floor), float(carry)


def current_level(model, df_master, playoffs=True, availability=None):
    """Where each player stands today, for everyone who has played in the last three seasons.

    LEVEL (rating), MPG, RECENT_SHARE (share of games played lately), AVAILABILITY
    (share of games he is projected to play, see fit_availability), AGE this season.
    """
    df = _regular_season(df_master)
    seasons = sorted(df["SEASON"].unique())[-len(RECENCY):]
    weights = RECENCY[-len(seasons):]
    weight = dict(zip(seasons, weights))
    latest = seasons[-1]
    parts = []
    for season in seasons:
        one = rank_players(model, df_master, season, min_min=0, min_gp=0)[["PLAYER_ID", "GP", "MIN", "RATING"]]
        parts.append(one.assign(SEASON=season))
        post = _playoff_ratings(model, df_master, season) if playoffs else None
        if post is not None:
            parts.append(post[["PLAYER_ID", "GP", "MIN", "RATING"]].assign(SEASON=season))
    long = pd.concat(parts, ignore_index=True)
    long["W"] = long["GP"] * long["MIN"] * long["SEASON"].map(weight)
    level = (long["RATING"] * long["W"]).groupby(long["PLAYER_ID"]).sum() / (long.groupby("PLAYER_ID")["W"].sum() + PRIOR_MINUTES)

    rows = df[df["SEASON"].isin(seasons)]
    gw = rows["GP"] * rows["SEASON"].map(weight)
    mpg = (rows["MIN"] * gw).groupby(rows["PLAYER_ID"]).sum() / gw.groupby(rows["PLAYER_ID"]).sum()

    gp, games = _games_tables(df)
    recent = _recent_share(gp, games, seasons, weights)
    floor, carry = fit_availability(df_master) if availability is None else availability

    last = rows.sort_values("SEASON").drop_duplicates("PLAYER_ID", keep="last").set_index("PLAYER_ID")
    behind = int(latest[:4]) - last["SEASON"].str[:4].astype(int)
    out = pd.DataFrame({"Player": last["PLAYER_NAME"], "TEAM": last["TEAM_ABBREVIATION"],
                        "BBR_ID": last["BBR_ID"] if "BBR_ID" in last.columns else None,
                        "AGE": (last["BBR_AGE"] if "BBR_AGE" in last.columns else np.nan) + behind,
                        "PLAYED_THIS_SEASON": behind == 0})
    out["TEAM_GAMES"] = games[latest].reindex(out.index)
    out["LEVEL"], out["MPG"], out["RECENT_SHARE"] = level, mpg, recent
    out["AVAILABILITY"] = (floor + carry * out["RECENT_SHARE"]).clip(*AVAILABILITY)
    return out.dropna(subset=["LEVEL", "MPG"])


def fit_aging_curve(model, df_master, min_minutes=1000, verbose=False):
    """Expected change in rating from each age to the next, as a Series indexed by age.

    Every pair of back-to-back seasons with min_minutes in both is one
    observation; a straight line in age is fitted through the changes, with the
    player's starting rating held fixed so that "good seasons are followed by
    worse ones" isn't mistaken for aging.
    """
    df = _regular_season(df_master)
    if "BBR_AGE" not in df.columns:
        raise ValueError("BBR_AGE not in the master — run add_bbref_advanced(df_master) first")
    ages = df.set_index(["PLAYER_ID", "SEASON"])["BBR_AGE"]
    seasons = sorted(df["SEASON"].unique())
    rated = {}
    for s in seasons:
        one = rank_players(model, df_master, s, min_min=0, min_gp=0)
        one = one[one["GP"] * one["MIN"] >= min_minutes].set_index("PLAYER_ID")
        rated[s] = one
    rows = []
    for a, b in zip(seasons[:-1], seasons[1:]):
        if rated[b]["GP"].max() < 0.5 * SEASON_GAMES:            # the season in progress isn't a finished outcome
            continue
        both = rated[a].join(rated[b][["RATING", "GP", "MIN"]], rsuffix="_NEXT", how="inner")
        for pid, r in both.iterrows():
            rows.append((ages.get((pid, a), np.nan), r["RATING"], r["RATING_NEXT"] - r["RATING"],
                         min(r["GP"] * r["MIN"], r["GP_NEXT"] * r["MIN_NEXT"])))
    obs = pd.DataFrame(rows, columns=["AGE", "RATING", "CHANGE", "W"]).dropna()
    w = np.sqrt(obs["W"].to_numpy())
    A = np.column_stack([np.ones(len(obs)), obs["AGE"], obs["RATING"]])
    (a0, a1, a2), *_ = np.linalg.lstsq(A * w[:, None], obs["CHANGE"].to_numpy() * w, rcond=None)
    typical = np.average(obs["RATING"], weights=obs["W"])
    curve = pd.Series({age: a0 + a1 * age + a2 * typical for age in AGES}, name="CHANGE_NEXT_SEASON")
    curve.index.name = "AGE"
    if verbose:
        peak = -(a0 + a2 * typical) / a1
        print(f"Aging curve from {len(obs):,} season pairs: players improve until about {peak:.0f}, "
              f"{curve[21]:+.2f} a year at 21, {curve[30]:+.2f} at 30, {curve[35]:+.2f} at 35")
    return curve


def price_of_a_win(model, df_master, last=3, veteran_after=4):
    """Share of the salary cap teams pay per win above replacement, over the last finished seasons.

    Only players past their first `veteran_after` seasons are counted: rookie
    contracts are price-controlled, and the alternative to trading for a player
    is buying wins on the open market. veteran_after=0 gives the average over
    every contract instead (about a third lower).
    """
    pay = load_salaries(save=False)
    df = _regular_season(df_master)
    finished = [s for s in sorted(df["SEASON"].unique()) if df.loc[df["SEASON"] == s, "GP"].max() >= 0.8 * SEASON_GAMES]
    debut = df.groupby("PLAYER_ID")["SEASON"].min().str[:4].astype(int)
    shares = []
    for s in finished[-last:]:
        rated = rank_players(model, df_master, s, min_min=0, min_gp=0)
        rated = rated[int(s[:4]) - rated["PLAYER_ID"].map(debut) >= veteran_after]
        rated = rated.assign(name_key=rated["Player"].map(name_key)).drop_duplicates("name_key")
        paid = pay[pay["SEASON"] == s].merge(rated[["name_key", "WINS"]], on="name_key", how="inner")
        shares.append(paid["pct_cap"].sum() / paid["WINS"].clip(lower=0).sum())
    return float(np.mean(shares))


def fit_title_odds(seasons=None):
    """How a team's rating turns into title odds: a softmax over each season's teams, fitted to the champions.

    Returns (beta, field): beta per point of rating, and the last finished
    season's team ratings to measure a team against.
    """
    from scipy.optimize import minimize_scalar
    from nba_api.stats.endpoints import LeagueGameFinder
    from nba_viz_utils import SEASONS, cached_fetch
    import team_projection as T
    rows = []
    for s in (seasons or SEASONS[:-1]):
        g = cached_fetch(f"team_games_{s}_playoffs", lambda s=s: LeagueGameFinder(
            player_or_team_abbreviation="T", season_nullable=s, season_type_nullable="Playoffs",
            league_id_nullable="00", timeout=60).get_data_frames()[0])
        champ = g[g["WL"] == "W"].groupby("TEAM_ABBREVIATION").size().idxmax()
        reg = T.load_games(s)
        teams = sorted(set(reg["HOME"]) | set(reg["AWAY"]))
        r, _, _ = T._ratings(reg, teams, prior_games=0.5)
        rows.append((r.to_numpy(), teams.index(champ)))

    def nll(b):
        return -sum(b * r[c] - np.log(np.exp(b * r).sum()) for r, c in rows)
    beta = float(minimize_scalar(nll, bounds=(0.05, 3.0), method="bounded").x)
    return beta, rows[-1][0]


def _title_gain(points, beta, field, base=CONTEXT_RATING):
    """Extra title chance from adding `points` per game to a team rated `base`, against `field` (the other teams)."""
    others = np.exp(beta * np.sort(field)[1:]).sum()                  # the field minus one average-ish slot he joins
    f = lambda r: np.exp(beta * r) / (np.exp(beta * r) + others)
    return f(base + np.asarray(points, float)) - f(base)


def _years(model, df_master, cap_growth, contracts, price, curve, availability=None, title=None):
    level = current_level(model, df_master, availability=availability)
    curve = fit_aging_curve(model, df_master) if curve is None else curve
    price = price_of_a_win(model, df_master) if price is None else price
    contracts = fetch_contract_years() if contracts is None else contracts
    this = sorted(_regular_season(df_master)["SEASON"].unique())[-1]
    if this not in SALARY_CAP:
        raise ValueError(f"no SALARY_CAP entry for {this}")
    per_point, replacement = model.war

    level = level.reset_index()
    level["name_key"] = level["Player"].map(name_key)
    by_id = contracts.merge(level.dropna(subset=["BBR_ID"]), on="BBR_ID", how="inner", suffixes=("_C", ""))
    rest = contracts[~contracts["BBR_ID"].isin(by_id["BBR_ID"])]
    by_name = rest.merge(level[~level["PLAYER_ID"].isin(by_id["PLAYER_ID"])], on="name_key", how="inner", suffixes=("_C", ""))
    y = pd.concat([by_id, by_name], ignore_index=True)
    y["TEAM"] = y["TEAM"].where(y["PLAYED_THIS_SEASON"], y["Tm"].replace(BBR_TEAM))      # hasn't played yet: the contract says where he is

    y["AHEAD"] = y["SEASON"].str[:4].astype(int) - int(this[:4])
    y = y[y["AHEAD"] >= 0].sort_values(["PLAYER_ID", "AHEAD"])
    drift = curve.reindex(range(min(AGES), max(AGES) + 1)).ffill().bfill()
    age0 = y["AGE"].fillna(27).clip(min(AGES), max(AGES)).astype(int)
    gain = np.array([drift.loc[a:a + k - 1].sum() if k else 0.0 for a, k in zip(age0, y["AHEAD"])])
    y["AGE_THEN"] = y["AGE"] + y["AHEAD"]
    y["RATING"] = y["LEVEL"] + gain
    y["WINS"] = per_point * (y["RATING"] + replacement).clip(lower=0) * y["MPG"] * SEASON_GAMES * y["AVAILABILITY"]
    cap = SALARY_CAP[this] * (1 + cap_growth) ** y["AHEAD"]
    left = np.where(y["AHEAD"] == 0, ((SEASON_GAMES - y["TEAM_GAMES"]) / SEASON_GAMES).clip(0, 1), 1.0)
    y["WORTH_M"] = y["WINS"] * price * cap * left / 1e6
    y["TITLE_M"] = 0.0
    if title is not None and title[2]:                 # (beta, field, title value in caps)
        beta, field, caps = title
        points = (y["RATING"] + replacement).clip(lower=0) * y["MPG"] / 48           # points per game over a replacement player
        y["TITLE_M"] = _title_gain(points, beta, field) * y["AVAILABILITY"] * caps * cap * left / 1e6
        y["WORTH_M"] = y["WORTH_M"] + y["TITLE_M"]
    y["SALARY_M"] = y["Salary"] * left / 1e6
    y["SURPLUS_M"] = y["WORTH_M"] - y["SALARY_M"]
    # an option is exercised by whoever it favours
    y["COUNTED"] = np.select([y["OPTION"] == "player", y["OPTION"] == "team"],
                             [y["SURPLUS_M"] < 0, y["SURPLUS_M"] > 0], default=True)
    y.loc[y["AHEAD"] == 0, "COUNTED"] = True          # this season's options were settled last summer
    # once one year is declined, the contract is over
    y["COUNTED"] = y.groupby("PLAYER_ID")["COUNTED"].cummin().astype(bool)
    y["RESIGN"] = False
    if title is not None and RESIGN_YEARS:
        y = pd.concat([y, _resign_years(y, per_point, replacement, price, this, cap_growth, title, drift)], ignore_index=True)
    return y, price


def _resign_years(y, per_point, replacement, price, this, cap_growth, title, drift):
    """Seasons after the contract, worth counting only if he'd be worth more than the max: see RESIGN_YEARS."""
    last = y[y["COUNTED"]].sort_values("AHEAD").groupby("PLAYER_ID").tail(1)
    rows = []
    for k in range(1, RESIGN_YEARS + 1):
        r = last.copy()
        r["AHEAD"] = last["AHEAD"] + k
        r["SEASON"] = [f"{int(this[:4]) + a}-{str(int(this[:4]) + a + 1)[2:]}" for a in r["AHEAD"]]
        age0 = last["AGE"].fillna(27).clip(min(AGES), max(AGES)).astype(int)
        r["AGE_THEN"] = r["AGE"] + r["AHEAD"]
        r["RATING"] = last["LEVEL"] + np.array([drift.loc[a:a + n - 1].sum() for a, n in zip(age0, r["AHEAD"])])
        r["WINS"] = per_point * (r["RATING"] + replacement).clip(lower=0) * r["MPG"] * SEASON_GAMES * r["AVAILABILITY"]
        cap = SALARY_CAP[this] * (1 + cap_growth) ** r["AHEAD"]
        beta, field, caps = title
        points = (r["RATING"] + replacement).clip(lower=0) * r["MPG"] / 48
        r["TITLE_M"] = _title_gain(points, beta, field) * r["AVAILABILITY"] * caps * cap / 1e6
        r["WORTH_M"] = r["WINS"] * price * cap / 1e6 + r["TITLE_M"]
        exp = (r["AGE_THEN"] - 19).clip(lower=0)                      # rough years in the league, for the max tier
        max_share = np.where(exp <= 6, 0.25, np.where(exp <= 9, 0.30, 0.35))
        r["SALARY_M"] = max_share * cap / 1e6
        r["SURPLUS_M"] = (r["WORTH_M"] - r["SALARY_M"]).clip(lower=0) * RESIGN_SHARE
        r["WORTH_M"] = r["SALARY_M"] + r["SURPLUS_M"]                  # so WORTH - OWED is the re-signing edge
        r["TITLE_M"] = r["TITLE_M"] * RESIGN_SHARE * (r["SURPLUS_M"] > 0)
        r["Salary"] = r["SALARY_M"] * 1e6
        r["OPTION"], r["COUNTED"], r["RESIGN"] = "", r["SURPLUS_M"] > 0, True
        rows.append(r)
    out = pd.concat(rows, ignore_index=True)
    return out[out["COUNTED"]]


def trade_value(model, df_master, cap_growth=0.07, discount=0.0, contracts=None, price=None, curve=None, verbose=True,
                title_value=TITLE_VALUE, title_odds=None):
    """Every player under contract, most valuable contract first.

    TRADE_VALUE_M   projected worth minus pay over the years he is likely to play out, in $ millions
    WORTH_M / OWED_M  the two sides of that sum (this season counts only for the games left)
    YEARS           contract seasons counted, this one included
    THROUGH         the last of them
    LEVEL           his rating today (see current_level); AGE, MPG and AVAILABILITY feed the projection
    WORTH_NOW_M / SALARY_NOW_M   full-season rates for this season, for a quick read

    TITLE_M         the part of WORTH_M that is title value (see TITLE_VALUE)
    RESIGN_M        the edge of holding his rights when the deal ends (see RESIGN_YEARS); in TRADE_VALUE_M

    cap_growth: yearly rise in the cap, which is what a win costs (the CBA
    allows up to 10%). discount: yearly discount on future surplus; 0 treats a
    dollar in 2029 like a dollar today.
    title_value: what a title is worth to a team, in caps. Each season he adds
    the title chance his points per game give a solid playoff team
    (CONTEXT_RATING), times that value: concentrated talent moves title odds
    more than the same wins spread around, which flat pricing misses.
    0 turns it off (the old, flat price for every win). title_odds: (beta,
    field) from fit_title_odds(), to skip refitting.
    """
    title = None
    if title_value:
        beta, field = title_odds if title_odds is not None else fit_title_odds()
        title = (beta, field, title_value)
    y, price = _years(model, df_master, cap_growth, contracts, price, curve, title=title)
    y["DISCOUNT"] = (1 + discount) ** -y["AHEAD"]
    kept = y[y["COUNTED"]]
    resign = kept[kept["RESIGN"]]
    kept_contract = kept[~kept["RESIGN"]]
    g = kept_contract.groupby("PLAYER_ID")
    first = y[y["AHEAD"] == 0].set_index("PLAYER_ID")
    out = pd.DataFrame({
        "Player": first["Player"], "TEAM": first["TEAM"], "AGE": first["AGE"],
        "LEVEL": first["LEVEL"], "MPG": first["MPG"], "AVAILABILITY": first["AVAILABILITY"],
        "YEARS": g.size(), "THROUGH": g["SEASON"].max(),
        "WORTH_M": (kept["WORTH_M"] * kept["DISCOUNT"]).groupby(kept["PLAYER_ID"]).sum(),
        "TITLE_M": (kept["TITLE_M"] * kept["DISCOUNT"]).groupby(kept["PLAYER_ID"]).sum(),
        "OWED_M": (kept["SALARY_M"] * kept["DISCOUNT"]).groupby(kept["PLAYER_ID"]).sum(),
        "WORTH_NOW_M": first["WINS"] * price * SALARY_CAP[first["SEASON"].iloc[0]] / 1e6,
        "SALARY_NOW_M": first["Salary"] / 1e6,
    })
    out["RESIGN_M"] = ((resign["WORTH_M"] - resign["SALARY_M"]) * resign["DISCOUNT"]).groupby(resign["PLAYER_ID"]).sum()
    out["RESIGN_M"] = out["RESIGN_M"].fillna(0.0)
    out["WORTH_M"] = out["WORTH_M"] - (resign["WORTH_M"] * resign["DISCOUNT"]).groupby(resign["PLAYER_ID"]).sum().reindex(out.index).fillna(0)
    out["OWED_M"] = out["OWED_M"] - (resign["SALARY_M"] * resign["DISCOUNT"]).groupby(resign["PLAYER_ID"]).sum().reindex(out.index).fillna(0)
    out["TRADE_VALUE_M"] = out["WORTH_M"] - out["OWED_M"] + out["RESIGN_M"]
    out = out.dropna(subset=["TRADE_VALUE_M"]).sort_values("TRADE_VALUE_M", ascending=False).reset_index(drop=True)
    out.insert(0, "RANK", np.arange(1, len(out) + 1))
    cols = ["RANK", "Player", "TEAM", "AGE", "TRADE_VALUE_M", "WORTH_M", "TITLE_M", "OWED_M", "RESIGN_M", "YEARS", "THROUGH", "LEVEL",
            "MPG", "AVAILABILITY", "WORTH_NOW_M", "SALARY_NOW_M"]
    out = out[cols]
    out.attrs.update(price_of_a_win=price, cap_growth=cap_growth, discount=discount, n=len(out), title_value=title_value)
    if verbose:
        this_cap = SALARY_CAP[first["SEASON"].iloc[0]]
        print(f"Trade value for {len(out)} players: a win costs {price:.1%} of the cap (${price * this_cap / 1e6:.1f}M this season), "
              f"cap growing {cap_growth:.0%} a year")
    return out


def contract_outlook(model, df_master, player, cap_growth=0.07, contracts=None, price=None, curve=None):
    """One player's contract year by year: projected rating, wins, worth, salary and whether the year is counted."""
    y, _ = _years(model, df_master, cap_growth, contracts, price, curve)
    mine = y[y["Player"].str.lower() == player.lower()]
    if mine.empty:
        mine = y[y["Player"].str.contains(player, case=False, regex=False)]
    if mine.empty:
        raise ValueError(f"no contract found for {player!r}")
    if mine["PLAYER_ID"].nunique() > 1:
        raise ValueError(f"{player!r} matches {sorted(mine['Player'].unique())} — be more specific")
    out = mine[["SEASON", "AGE_THEN", "RATING", "WINS", "WORTH_M", "SALARY_M", "SURPLUS_M", "OPTION", "COUNTED"]]
    out = out.rename(columns={"AGE_THEN": "AGE"}).reset_index(drop=True)
    out.attrs.update(player=mine["Player"].iloc[0], trade_value=float(out.loc[out["COUNTED"], "SURPLUS_M"].sum()))
    return out
