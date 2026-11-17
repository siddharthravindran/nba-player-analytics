"""Where will every team finish? Final records from point margins and the schedule still to play.

    from team_projection import *
    proj = fit_projection(df_master, model)       # fits on past seasons, prints how it would have done
    project_season(proj, df_master, absences=KNOWN_ABSENCES)   # this season, with known injuries
    proj.check                                    # every past season frozen at the same point, vs what happened

    fit_projection() with no arguments leaves the players out: margins and schedule only.

How it works
  - Each team's rating is points per game better than average, after adjusting
    for opponents and home court: a least-squares fit of every game's margin.
  - Early on that is pulled toward a prior built from the roster: each
    player's rating from player_ranking (last season's, so it is known before
    the games), weighted by his share of the team's minutes so far. Trades and
    signings count from day one. Some of last season's team rating can be kept
    too. How hard to pull (the prior counts like PRIOR_GAMES games), the weight
    on the roster and the carry-over are fitted on past seasons, frozen at the
    same point, by how well each would have predicted the final records.
  - Going forward each player is expected to play his usual minutes x his
    availability (his last three seasons' share of games, pulled toward the
    usual 82%): an iron man counts more than a player who misses a third of
    every season. Known absences (project_season's `absences`, e.g.
    KNOWN_ABSENCES from awards.py: games still to miss) take him out for that
    stretch. Minutes go to teammates, and the change in roster strength is
    applied at the weight fitted above. Rookies' ratings come from this
    season once they have games (ROOKIE_GAMES). This part can't be tested on past
    seasons (there are no old injury reports), so treat it as a reasoned
    adjustment, not a measured one.
  - Every remaining game gets a home win chance from the two ratings, home
    court and the spread of real game margins. The season is played out 10,000
    times, each time with ratings drawn around the estimates (they are only
    estimates), so the ranges stay honest.

What it can't see: injuries nobody has reported, future trades, rest days,
and players who haven't played this season at all (they aren't on any roster
in the data).
"""
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from nba_viz_utils import cached_fetch

__all__ = ["fit_projection", "project_season", "load_games", "player_ratings"]

SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2015, 2027)]
CURRENT = SEASONS[-1]
CUTOFFS = (10, 15, 20, 30, 41)                 # games per team at which past seasons are frozen for fitting
PRIOR_GAMES = (2, 4, 6, 8, 10, 14, 20, 30)
CARRY = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SIMS = 10_000
SEASON_GAMES = 82
SD_SCALES = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5)   # widen the rating uncertainty: injuries and trades are still to come
ROSTER_WEIGHT = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
CARRY_WITH_ROSTER = (0.0, 0.2, 0.4, 0.6)
ROOKIE_RATING = -1.5                           # a player with no reliable past season: below average, a judgment
RELIABLE_MIN = 500                             # minutes in a season before his rating there is used
SETTLED_GP = 5                                 # games this season before his minutes per game are trusted
ROOKIE_GAMES = 10                              # a new player's own rating this season counts like this many games against ROOKIE_RATING
USUAL_AVAILABILITY = 0.82                      # share of games the typical rotation player plays
AVAILABILITY_SEASONS = (0.5, 0.3, 0.2)         # weights on the last three seasons' share of games played
AVAILABILITY_PRIOR = 0.5                       # the usual rate counts like half a season against his own history
ROTATION_MPG = 20                              # rotation players' minutes going forward: minutes per game x availability
EAST = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"}


def _fetch_games(season):
    from nba_api.stats.endpoints import LeagueGameFinder
    return cached_fetch(f"team_games_{season}_regular",
                        lambda: LeagueGameFinder(player_or_team_abbreviation="T", season_nullable=season,
                                                 season_type_nullable="Regular Season", league_id_nullable="00",
                                                 timeout=60).get_data_frames()[0],
                        force_refresh=(season == CURRENT))


def load_games(season):
    """One row per game played: GAME_ID, DATE, HOME, AWAY (abbreviations), HOME_PTS, AWAY_PTS, MARGIN (home minus away)."""
    g = _fetch_games(season)
    home = g[g["MATCHUP"].str.contains(" vs. ")].set_index("GAME_ID")
    away = g[g["MATCHUP"].str.contains(" @ ")].set_index("GAME_ID")
    both = home.join(away, lsuffix="_H", rsuffix="_A", how="inner")
    out = pd.DataFrame({"DATE": pd.to_datetime(both["GAME_DATE_H"]), "HOME": both["TEAM_ABBREVIATION_H"],
                        "AWAY": both["TEAM_ABBREVIATION_A"], "HOME_PTS": both["PTS_H"], "AWAY_PTS": both["PTS_A"]})
    out["MARGIN"] = out["HOME_PTS"] - out["AWAY_PTS"]
    return out.sort_values("DATE").reset_index().rename(columns={"index": "GAME_ID"})


def _remaining_schedule(season=CURRENT):
    """Regular-season games not yet played (both teams known): HOME, AWAY, DATE."""
    from nba_api.stats.endpoints import ScheduleLeagueV2
    sch = cached_fetch(f"schedule_{season}", lambda: ScheduleLeagueV2(season=season, timeout=60).get_data_frames()[0],
                       force_refresh=True)
    sch = sch[sch["gameId"].astype(str).str.startswith("002") & (sch["gameStatus"] != 3)]
    sch = sch[sch["homeTeam_teamTricode"].notna() & sch["awayTeam_teamTricode"].notna()
              & (sch["homeTeam_teamTricode"] != "") & (sch["awayTeam_teamTricode"] != "")]
    return pd.DataFrame({"HOME": sch["homeTeam_teamTricode"].to_numpy(), "AWAY": sch["awayTeam_teamTricode"].to_numpy(),
                         "DATE": pd.to_datetime(sch["gameDate"]).to_numpy()})


def _ratings(games, teams, prior=None, prior_games=0.0):
    """Ratings (points better than average) and home court, by least squares pulled toward `prior`."""
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(games), len(teams)
    X = np.zeros((n, k + 1))
    X[:, 0] = 1.0
    X[np.arange(n), 1 + games["HOME"].map(idx).to_numpy()] = 1.0
    X[np.arange(n), 1 + games["AWAY"].map(idx).to_numpy()] = -1.0
    y = games["MARGIN"].to_numpy(float)
    prior = np.zeros(k) if prior is None else prior
    lam = np.sqrt(max(prior_games, 1e-6))                                # each team's prior counts like `prior_games` games
    P = np.hstack([np.zeros((k, 1)), np.eye(k) * lam])
    sumzero = np.hstack([[0.0], np.ones(k)]) * 10.0                      # ratings average zero
    A = np.vstack([X, P, sumzero])
    b = np.concatenate([y, prior * lam, [0.0]])
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    played = np.bincount(np.concatenate([games["HOME"].map(idx), games["AWAY"].map(idx)]), minlength=k)
    return pd.Series(sol[1:], index=teams), float(sol[0]), played


def _cut(games, per_team):
    """Games played up to the date when the median team had played `per_team` games."""
    counts = pd.concat([games[["DATE", "HOME"]].rename(columns={"HOME": "T"}), games[["DATE", "AWAY"]].rename(columns={"AWAY": "T"})])
    counts = counts.sort_values("DATE")
    counts["N"] = counts.groupby("T").cumcount() + 1
    day = counts[counts["N"] == per_team].groupby("T")["DATE"].first().median()
    return games[games["DATE"] <= day], games[games["DATE"] > day], day


def _record(games, teams):
    w = pd.Series(0, index=teams, dtype=float)
    hw = games["MARGIN"] > 0
    w = w.add(games.loc[hw, "HOME"].value_counts(), fill_value=0).add(games.loc[~hw, "AWAY"].value_counts(), fill_value=0)
    gp = games["HOME"].value_counts().add(games["AWAY"].value_counts(), fill_value=0).reindex(teams).fillna(0)
    return w.reindex(teams).fillna(0), gp


def _expected_wins(rest, r, home, sigma, teams):
    p = norm.cdf((home + rest["HOME"].map(r).to_numpy() - rest["AWAY"].map(r).to_numpy()) / sigma)
    return (pd.Series(p, index=rest["HOME"].to_numpy()).groupby(level=0).sum().reindex(teams).fillna(0)
            + pd.Series(1 - p, index=rest["AWAY"].to_numpy()).groupby(level=0).sum().reindex(teams).fillna(0))


def player_ratings(df_master, rating_model):
    """Every season's player ratings from player_ranking: {season: DataFrame PLAYER_ID, RATING, MINUTES, MPG}."""
    import contextlib, io
    from player_ranking import rank_players
    out = {}
    for s in SEASONS:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                r = rank_players(rating_model, df_master, s, min_min=0, min_gp=0)
        except ValueError:
            continue
        r = r.assign(PLAYER_ID=r["PLAYER_ID"].astype(int), MINUTES=r["MIN"] * r["GP"])
        g = r.groupby("PLAYER_ID")
        out[s] = pd.DataFrame({"RATING": g.apply(lambda t: np.average(t["RATING"], weights=t["MINUTES"].clip(lower=1e-6))),
                               "MINUTES": g["MINUTES"].sum(), "GP": g["GP"].sum()})
        out[s]["MPG"] = out[s]["MINUTES"] / out[s]["GP"].clip(lower=1)
    return out


def _known_rating(pids, season, ratings, current=False):
    """Each player's rating going into `season`: last season, else the one before, else ROOKIE_RATING.

    current=True (this season's projection only, never the backtest): a player
    with no reliable past season gets his own rating this season, weighed
    against ROOKIE_RATING by games: after 10 games it is half his own.
    """
    vals = pd.Series(ROOKIE_RATING, index=pd.Index(pids, dtype=int), dtype=float)
    known = pd.Series(False, index=vals.index)
    i = SEASONS.index(season)
    for back in (2, 1):                                   # the nearer season wins
        if i - back >= 0 and SEASONS[i - back] in ratings:
            r = ratings[SEASONS[i - back]]
            r = r[r["MINUTES"] >= RELIABLE_MIN]["RATING"].reindex(vals.index).dropna()
            vals.update(r)
            known[r.index] = True
    if current and season in ratings:
        now = ratings[season].reindex(vals.index)
        new = ~known & now["RATING"].notna()
        gp = now.loc[new, "GP"]
        vals[new] = (gp * now.loc[new, "RATING"] + ROOKIE_GAMES * ROOKIE_RATING) / (gp + ROOKIE_GAMES)
    return vals


def _availability(pids, season, ratings):
    """Share of games each player is expected to play: his last three seasons, pulled toward the usual rate."""
    i = SEASONS.index(season)
    num = pd.Series(AVAILABILITY_PRIOR * USUAL_AVAILABILITY, index=pd.Index(pids, dtype=int))
    den = pd.Series(AVAILABILITY_PRIOR, index=num.index)
    for back, w in zip((1, 2, 3), AVAILABILITY_SEASONS):
        if i - back < 0 or SEASONS[i - back] not in ratings:
            continue
        r = ratings[SEASONS[i - back]]
        share = (r["GP"] / r["GP"].max()).clip(upper=1).reindex(num.index)
        seen = share.notna()
        num[seen] += w * share[seen]
        den[seen] += w
    return (num / den).clip(0.05, 1.0)


def _asof_minutes(season, day):
    """Each player's minutes for each team up to `day`: PLAYER_ID, TEAM, GP, MINUTES."""
    from nba_api.stats.endpoints import LeagueDashPlayerStats
    df = cached_fetch(f"player_minutes_{season}_to_{day:%Y%m%d}",
                      lambda: LeagueDashPlayerStats(season=season, season_type_all_star="Regular Season", per_mode_detailed="Totals",
                                                    date_to_nullable=day.strftime("%m/%d/%Y"), timeout=60).get_data_frames()[0])
    time.sleep(0.5)
    return pd.DataFrame({"PLAYER_ID": df["PLAYER_ID"].astype(int), "TEAM": df["TEAM_ABBREVIATION"],
                         "GP": df["GP"], "MINUTES": df["MIN"].astype(float)})


def _roster_raw(minutes, season, ratings, teams, current=False):
    """Minutes-weighted player rating x 5 per team: roughly points per game better than average."""
    m = minutes[minutes["MINUTES"] > 0].copy()
    m["R"] = _known_rating(m["PLAYER_ID"].to_numpy(), season, ratings, current).to_numpy()
    s = (m["R"] * m["MINUTES"]).groupby(m["TEAM"]).sum() / m.groupby("TEAM")["MINUTES"].sum() * 5
    return s.reindex(teams)


def _centered(s):
    return (s - s.mean()).fillna(0.0)


def _simulate(rest, r, home, sigma, sd, teams, wins_now, sims=SIMS, seed=0, unscheduled=None, adj=None):
    """Final wins for every team in every simulated season: a sims x teams array.

    unscheduled: games per team not on the schedule yet (the NBA Cup's last two
    are set in December), played against an average opponent on a neutral floor.
    adj: (home_adj, away_adj, end_adj) lineup changes in points, per remaining
    game for each side, and per team for the unscheduled games.
    """
    rng = np.random.default_rng(seed)
    idx = {t: i for i, t in enumerate(teams)}
    h, a = rest["HOME"].map(idx).to_numpy(), rest["AWAY"].map(idx).to_numpy()
    draws = r.to_numpy()[None, :] + rng.standard_normal((sims, len(teams))) * sd.to_numpy()[None, :]
    ha, aa, end = (np.zeros(len(rest)), np.zeros(len(rest)), np.zeros(len(teams))) if adj is None else adj
    p = norm.cdf((home + draws[:, h] + ha[None, :] - draws[:, a] - aa[None, :]) / sigma)
    home_won = rng.random(p.shape) < p
    wins = np.tile(wins_now.to_numpy(float), (sims, 1))
    np.add.at(wins.T, h, home_won.T)
    np.add.at(wins.T, a, ~home_won.T)
    if unscheduled is not None and np.any(unscheduled > 0):
        wins += rng.binomial(np.asarray(unscheduled, int)[None, :].repeat(sims, 0), norm.cdf((draws + end[None, :]) / sigma))
    return wins


@dataclass
class Projection:
    prior_games: float
    carry: float
    sigma: float                   # spread of real game margins around the rating difference
    sd_scale: float                # how much wider than sampling error the ratings' uncertainty is, fitted for 80% ranges
    final: dict = field(repr=False)            # each past season's full-season ratings, for priors
    check: pd.DataFrame = field(default=None, repr=False)
    roster_weight: float = 0.0     # points of team margin per point of roster strength; 0 when players are left out
    ratings: dict = field(default=None, repr=False)   # player_ratings(), when players are used


def fit_projection(df_master=None, rating_model=None, verbose=True, cutoff_now=None):
    """Fit the prior (roster and/or last season) and how hard to pull toward it on past seasons; test at today's point.

    With df_master and rating_model the prior uses the roster (player_ratings);
    without them it is last season's team rating only. check has both, so the
    roster's value is measured, not assumed.
    """
    ratings = player_ratings(df_master, rating_model) if df_master is not None else None
    games = {s: load_games(s) for s in SEASONS[:-1]}
    final, resid = {}, []
    for s, g in games.items():
        teams = sorted(set(g["HOME"]) | set(g["AWAY"]))
        r, h, _ = _ratings(g, teams, prior_games=0.5)
        final[s] = r
        resid.append(g["MARGIN"] - (h + g["HOME"].map(r) - g["AWAY"].map(r)))
    sigma = float(pd.concat(resid).std())

    roster = {}

    def strength(s, per_team, teams):
        if ratings is None:
            return np.zeros(len(teams))
        if (s, per_team) not in roster:
            day = _cut(games[s], per_team)[2]
            roster[(s, per_team)] = _centered(_roster_raw(_asof_minutes(s, day), s, ratings, teams)).to_numpy()
        return roster[(s, per_team)]

    def frozen(s, per_team, prior_games, carry, weight=0.0):
        g = games[s]
        teams = sorted(final[s].index)
        prior = final[SEASONS[SEASONS.index(s) - 1]].reindex(teams).fillna(0).to_numpy() * carry
        if weight:
            prior = prior + weight * strength(s, per_team, teams)
        played, rest, _ = _cut(g, per_team)
        r, h, n = _ratings(played, teams, prior, prior_games)
        wins_now, gp = _record(played, teams)
        exp = wins_now + _expected_wins(rest, r, h, sigma, teams)
        actual, total = _record(g, teams)
        return played, rest, r, h, n, wins_now, gp, exp, actual, total, teams, prior

    test_seasons = SEASONS[1:-1]

    def search(carries, weights):
        best, err = None, np.inf
        for pg in PRIOR_GAMES:
            for c in carries:
                for w in weights:
                    e = []
                    for s in test_seasons:
                        for cut in CUTOFFS:
                            *_, exp, actual, total, teams, _ = frozen(s, cut, pg, c, w)
                            e.append((((exp - actual) / total * 82) ** 2).mean())
                    if np.mean(e) < err:
                        err, best = np.mean(e), (pg, c, w)
        return best

    plain = search(CARRY, (0.0,))
    pg, c, w = search(CARRY_WITH_ROSTER, ROSTER_WEIGHT) if ratings is not None else plain

    now = cutoff_now or int(np.median(_record(load_games(CURRENT), sorted(final[SEASONS[-2]].index))[1]))
    frozen_now = {s: frozen(s, now, pg, c, w) for s in test_seasons}
    plain_now = {s: frozen(s, now, *plain) for s in test_seasons} if ratings is not None else None

    def coverage(scale):
        inside = []
        for s, (played, rest, r, h, n, wins_now, gp, exp, actual, total, teams, prior) in frozen_now.items():
            sd = pd.Series(scale * sigma / np.sqrt(n + pg), index=teams)
            sim = _simulate(rest, r, h, sigma, sd, teams, wins_now, sims=2000, seed=1)
            inside.append(((actual >= np.percentile(sim, 10, axis=0)) & (actual <= np.percentile(sim, 90, axis=0))).to_numpy())
        return float(np.concatenate(inside).mean())
    cover = {k: coverage(k) for k in SD_SCALES}
    scale = min(cover, key=lambda k: abs(cover[k] - 0.8))

    rows = []
    for s in test_seasons:
        played, rest, r, h, n, wins_now, gp, exp, actual, total, teams, prior = frozen_now[s]
        sd = scale * sigma / np.sqrt(n + pg)
        sim = _simulate(rest, r, h, sigma, pd.Series(sd, index=teams), teams, wins_now, sims=2000, seed=1)
        lo, hi = np.percentile(sim, 10, axis=0), np.percentile(sim, 90, axis=0)
        pace = wins_now / gp.replace(0, np.nan) * total
        diff = played.groupby("HOME")["MARGIN"].sum().add(-played.groupby("AWAY")["MARGIN"].sum(), fill_value=0).reindex(teams) / gp
        pyth = wins_now + norm.cdf(diff / sigma) * (total - gp)                  # point differential, no schedule
        last = final[SEASONS[SEASONS.index(s) - 1]].reindex(teams).fillna(0)
        last_wins = norm.cdf(last / sigma) * total                               # last season's level
        per82 = SEASON_GAMES / total
        methods = [("MODEL", exp), ("PACE", pace), ("POINT_DIFF", pyth), ("LAST_SEASON", last_wins)]
        if plain_now is not None:
            methods.append(("MODEL_NO_PLAYERS", plain_now[s][7]))
        for name, guess in methods:
            rows.append({"SEASON": s, "METHOD": name, "MAE_WINS": float(((guess - actual).abs() * per82).mean())})
        rows.append({"SEASON": s, "METHOD": "IN_80_RANGE", "MAE_WINS": float(((actual >= lo) & (actual <= hi)).mean())})
    check = pd.DataFrame(rows).pivot(index="SEASON", columns="METHOD", values="MAE_WINS")
    if verbose:
        print(f"\nSeason projection: prior counts like {pg} games, carry-over {c:.1f} of last season, "
              f"roster weight {w:g}, game spread {sigma:.1f} pts")
        print(f"Past seasons frozen {now} games in, average miss on final wins (per 82 games):")
        m = check.drop(columns="IN_80_RANGE").mean()
        print("  " + ", ".join(f"{k.lower().replace('_', ' ')} {v:.1f}" for k, v in m.sort_values().items()))
        print(f"  share of teams finishing inside the 80% range: {check['IN_80_RANGE'].mean():.0%} "
              f"(rating uncertainty widened {scale:g}x; at 1x it was {cover[1.0]:.0%})")
    return Projection(pg, c, sigma, scale, final, check, w, ratings)


def _lineup_adjustments(rest, teams, df_master, proj, season, absences, now_raw, team_gp):
    """Per remaining game, each side's change in strength from who is available, against the lineup so far.

    A rotation player's share going forward (ROTATION_MPG+ when he plays) is
    his minutes per game x his expected availability (_availability: his last
    three seasons' games played, pulled toward the usual rate), so a player who
    often misses games counts for less and his minutes go to teammates. Bench
    players keep the minutes per team game they have actually been getting
    (their games missed are mostly the coach's choice). Players in `absences` are
    out for that many of their team's next games.
    """
    from player_ranking import _regular_season
    cur = _regular_season(df_master)
    cur = cur[cur["SEASON"] == season].sort_values("GP").drop_duplicates("PLAYER_ID", keep="last")   # his main team
    pids = cur["PLAYER_ID"].astype(int).to_numpy()
    rating = _known_rating(pids, season, proj.ratings, current=True).to_numpy()
    last = proj.ratings.get(SEASONS[SEASONS.index(season) - 1], pd.DataFrame(columns=["MPG"]))["MPG"]
    mpg = np.where(cur["GP"].to_numpy() >= SETTLED_GP, cur["MIN"].to_numpy(float),
                   pd.Series(pids).map(last).fillna(pd.Series(cur["MIN"].to_numpy(float))).to_numpy())
    avail = _availability(pids, season, proj.ratings).to_numpy()
    tgp = cur["TEAM_ABBREVIATION"].map(team_gp).clip(lower=1).to_numpy(float)
    so_far = cur["MIN"].to_numpy(float) * cur["GP"].to_numpy(float) / tgp           # minutes per team game so far
    share = np.where(mpg >= ROTATION_MPG, mpg * avail, so_far)
    roster = pd.DataFrame({"TEAM": cur["TEAM_ABBREVIATION"].to_numpy(), "NAME": cur["PLAYER_NAME"].to_numpy(),
                           "R": rating, "MPG": share, "AVAIL": avail, "OUT": cur["PLAYER_NAME"].map(absences or {}).fillna(0).to_numpy()})
    long = pd.concat([pd.DataFrame({"G": np.arange(len(rest)), "T": rest["HOME"].to_numpy(), "SIDE": "H", "DATE": rest["DATE"].to_numpy()}),
                      pd.DataFrame({"G": np.arange(len(rest)), "T": rest["AWAY"].to_numpy(), "SIDE": "A", "DATE": rest["DATE"].to_numpy()})])
    long = long.sort_values(["DATE", "G"])
    long["J"] = long.groupby("T").cumcount()                       # 0 = his team's next game

    def strength(team, j):
        t = roster[(roster["TEAM"] == team) & (roster["OUT"] <= j)]
        return float((t["R"] * t["MPG"]).sum() / t["MPG"].sum() * 5) if len(t) else np.nan

    cache = {}
    vals = []
    for t, j in zip(long["T"], long["J"]):
        k = (t, min(j, int(roster.loc[roster["TEAM"] == t, "OUT"].max() or 0)))
        if k not in cache:
            cache[k] = strength(*k)
        vals.append(cache[k])
    long["ADJ"] = proj.roster_weight * (np.array(vals) - long["T"].map(now_raw).to_numpy())
    long["ADJ"] = long["ADJ"].fillna(0.0)
    ha = long[long["SIDE"] == "H"].set_index("G")["ADJ"].reindex(range(len(rest))).fillna(0).to_numpy()
    aa = long[long["SIDE"] == "A"].set_index("G")["ADJ"].reindex(range(len(rest))).fillna(0).to_numpy()
    healthy = pd.Series({t: strength(t, 10**6) for t in teams})
    end = (proj.roster_weight * (healthy - now_raw.reindex(teams))).fillna(0.0).to_numpy()
    avg = long.groupby("T")["ADJ"].mean().reindex(teams).fillna(0.0)
    return (ha, aa, end), avg


def project_season(proj, df_master=None, season=CURRENT, sims=SIMS, absences=None):
    """This season: wins so far, expected final wins, an 80% range, and the chance of the top 6 / top 10 in the conference.

    With df_master (and a proj fitted with players), the prior uses the roster
    and `absences` ({player name: games still to miss}, e.g. KNOWN_ABSENCES)
    adjusts each remaining game. LINEUP_ADJ is the average adjustment over the
    remaining games, in points: negative for a team with players out or prone
    to missing games, positive for one getting players back.
    """
    played = load_games(season)
    teams = sorted(proj.final[SEASONS[-2]].index)
    prior = proj.final[SEASONS[-2]].reindex(teams).fillna(0).to_numpy() * proj.carry
    use_players = proj.roster_weight > 0 and df_master is not None
    if use_players:
        from player_ranking import _regular_season
        cur = _regular_season(df_master)
        cur = cur[cur["SEASON"] == season]
        minutes = pd.DataFrame({"PLAYER_ID": cur["PLAYER_ID"].astype(int).to_numpy(), "TEAM": cur["TEAM_ABBREVIATION"].to_numpy(),
                                "GP": cur["GP"].to_numpy(), "MINUTES": (cur["MIN"] * cur["GP"]).astype(float).to_numpy()})
        now_raw = _roster_raw(minutes, season, proj.ratings, teams, current=True)
        prior = prior + proj.roster_weight * _centered(now_raw).to_numpy()
    r, h, n = _ratings(played, teams, prior, proj.prior_games)
    wins_now, gp = _record(played, teams)
    rest = _remaining_schedule(season)
    adj, lineup = (None, pd.Series(0.0, index=teams))
    if use_players:
        adj, lineup = _lineup_adjustments(rest, teams, df_master, proj, season, absences, now_raw, gp)
    sd = pd.Series(proj.sd_scale * proj.sigma / np.sqrt(n + proj.prior_games), index=teams)
    scheduled = gp + rest["HOME"].value_counts().add(rest["AWAY"].value_counts(), fill_value=0).reindex(teams).fillna(0)
    unscheduled = (SEASON_GAMES - scheduled).clip(lower=0)
    sim = _simulate(rest, r, h, proj.sigma, sd, teams, wins_now, sims=sims, unscheduled=unscheduled.to_numpy(), adj=adj)
    conf = np.array(["East" if t in EAST else "West" for t in teams])
    seed = np.zeros_like(sim)
    noise = np.random.default_rng(2).random(sim.shape) * 1e-3                   # random tie-breaks
    for cf in ("East", "West"):
        cols = np.flatnonzero(conf == cf)
        order = np.argsort(-(sim[:, cols] + noise[:, cols]), axis=1)
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.arange(1, len(cols) + 1)[None, :].repeat(len(sim), 0), axis=1)
        seed[:, cols] = ranks
    out = pd.DataFrame({
        "TEAM": teams, "CONF": conf, "W": wins_now.astype(int).to_numpy(), "L": (gp - wins_now).astype(int).to_numpy(),
        "RATING": r.round(1).to_numpy(), "LINEUP_ADJ": lineup.round(1).to_numpy(), "PROJ_W": sim.mean(axis=0).round(1),
        "LOW_80": np.percentile(sim, 10, axis=0).round(0).astype(int), "HIGH_80": np.percentile(sim, 90, axis=0).round(0).astype(int),
        "UNSCHEDULED": unscheduled.astype(int).to_numpy(), "TOP_6": (seed <= 6).mean(axis=0).round(2), "TOP_10": (seed <= 10).mean(axis=0).round(2),
    })
    out = out.sort_values(["CONF", "PROJ_W"], ascending=[True, False]).reset_index(drop=True)
    out.attrs.update(season=season, home_court=round(h, 2), remaining=len(rest))
    return out
