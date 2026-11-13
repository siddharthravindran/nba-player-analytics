"""Rank players: a ridge regression from the master's stats onto LEBRON.

    from player_ranking import *
    model = fit_rating_model(df_master)                # trains, prints the 2025-26 holdout check
    rank = rank_players(model, df_master)              # 2026-27, best first
    rank.head(20)
    rank_players(model, df_master, "2023-24")          # any season in the master
    rank_multi_year(model, df_master, last=3)          # the last three seasons combined
    rank_multi_year(model, df_master, playoffs=True)   # ... with playoff games counted too
    rank_players(model, df_master, by="D_RATING")      # best defenders (or "O_RATING", "WINS")
    most_improved(model, df_master)                    # biggest jumps from last season to this one
    compare_seasons(model, df_master, "Cade Cunningham", "2024-25", "2025-26")   # what changed in his game
    fit_rating_model(df_master, target="blend")        # a second opinion: LEBRON and BPM averaged
    rating_drivers(model, df_master, "Stephen Curry")            # which part of his game carries the rating
    rating_breakdown(model, df_master, "Stephen Curry")          # the same, stat by stat
    plot_interactive_barh(rank, x_col="RATING", y_col="Player", top_n=20, title="Estimated impact")

RATING is an estimate of LEBRON: points per 100 possessions a player adds over
an average player, offense (O_RATING) plus defense (D_RATING). LEBRON itself
needs lineup data and isn't published during the season in the files this
project has; this gives a live stand-in built from stats that refresh daily.

How it works
  - Counting stats become per-36 rates, so minutes played are not an input:
    the rating says how well a player plays, not how much his coach uses him.
  - Every stat is measured against the league that season (z-scores), so the
    three-point boom and pace changes don't shift the scale.
  - Two ridge regressions (offense, defense), weighted by minutes. Ridge keeps
    the weights stable when hundreds of stats overlap.
  - The penalty is chosen by leaving out one season at a time; a final season
    is held out entirely to report how well it works, then the model is refit
    on everything.

What it is not: an independent measure of impact. It learns to imitate LEBRON,
so it inherits LEBRON's view of the game, and part of LEBRON is itself built
from similar stats.
"""
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from nba_viz_utils import DATA_DIR, load_lebron_seasons

# Left out of the inputs: playing time and totals (the rating is per possession),
# other people's all-in-one metrics, clutch splits (tiny samples), ranks and ids.
_DROP = re.compile(r"^(GP|W|L|MIN|TEAM_COUNT|DD2|TD3|POSS|DEF_WS|FGM_PG|FGA_PG|ONCOURT_MIN|OFFCOURT_MIN|FP_HIGH_SCORE)$"
                   r"|^BBR_|^CLUTCH_|^LBR_|PERCENTILE|_RANK$|_ID$")
# Already a rate; everything else is a per-game count and is put on a per-36 basis.
_RATE = re.compile(r"PCT|FREQ|PPP|RATING|RATIO|AVG_|_PER_|PIE|PACE|USG|AST_TO|EFG|SPEED|PLUSMINUS|^AGE$|HEIGHT|WEIGHT|INCHES")
_COUNT = re.compile(r"_POSS$|^TIME_OF_POSS$|^AST_POINTS_CREATED$|^AST_ADJ$|^(OPP_)?PTS_(OFF_TOV|2ND_CHANCE|FB|PAINT)$"
                    r"|^DEFGT15_FG[MA]_GT_15$|^CONTESTED_SHOTS_[23]PT$")
TARGETS = {"O_RATING": "_T_O", "D_RATING": "_T_D"}
# What the model learns to reproduce: (offense column, defense column) per target name
TARGET_SOURCES = {"lebron": ("LBR_O_LEBRON", "LBR_D_LEBRON"), "bpm": ("BBR_OBPM", "BBR_DBPM")}
ALPHAS = (300, 1000, 3000, 10000, 30000, 100000)
POOL_MIN, POOL_GP_SHARE = 10.0, 0.3
PLAYOFF_MIN_GP = 4                  # the shortest possible series


@dataclass
class RatingModel:
    features: list
    counts: list
    coef: dict                       # {"O_RATING": Series, "D_RATING": Series}
    intercept: dict
    alpha: float
    seasons: list
    holdout: pd.DataFrame = field(default=None, repr=False)
    war: tuple = None                # (wins per rating-point-minute, replacement level)
    target: str = "lebron"
    lebron: pd.DataFrame = field(default=None, repr=False)     # the published values, for comparison


def _season_rows(df_master, season_type):
    df = df_master.reset_index() if isinstance(df_master.index, pd.MultiIndex) else df_master.copy()
    df = df[df["SEASON_TYPE"] == season_type].copy()
    df["PLAYER_ID"] = df["PLAYER_ID"].astype(str)
    # one row per player-season: the one with the most games
    return (df.sort_values("GP", ascending=False).drop_duplicates(["PLAYER_ID", "SEASON"])
              .sort_values(["SEASON", "PLAYER_NAME"]).reset_index(drop=True))


def _regular_season(df_master):
    return _season_rows(df_master, "Regular Season")


def _pool(df):
    """Rotation players: 10+ minutes and at least 30% of the most games anyone has played that season."""
    most = df.groupby("SEASON")["GP"].transform("max")
    return df[(df["MIN"] >= POOL_MIN) & (df["GP"] >= POOL_GP_SHARE * most)].copy()


def _feature_names(df):
    cols = [c for c in df.select_dtypes("number").columns if not _DROP.search(c)]
    counts = [c for c in cols if _COUNT.search(c) or not _RATE.search(c)]
    return cols, counts


def _rates(pool, features, counts):
    """The model's inputs before scaling: counting stats per 36 minutes, rates as they are."""
    X = pool.reindex(columns=features).astype(float)
    play_types = [c for c in features if c.endswith("_POSS_PCT") or c.endswith("_POSS")]
    X[play_types] = X[play_types].fillna(0)                 # Synergy omits a play type a player barely runs
    X[counts] = X[counts].div(pool["MIN"], axis=0) * 36
    return X


def _standardize(pool, features, counts, reference=None):
    """Per-36 for counts, then each stat as standard deviations from that season's average.

    reference: another pool from the same season whose averages to measure
    against instead of the pool's own — how playoff stats are put on the
    regular-season scale.
    """
    X = _rates(pool, features, counts)
    if reference is None:
        by_season = X.groupby(pool["SEASON"])
        mean, std = by_season.transform("mean"), by_season.transform("std")
    else:
        R = _rates(reference, features, counts)
        mean, std = R.mean(), R.std()
    z = (X - mean) / std.replace(0, np.nan)
    if reference is not None:                                # a stat this pool doesn't have at all is no evidence
        z.loc[:, pool.reindex(columns=features).isna().all().to_numpy()] = np.nan
    return z.clip(-5, 5).fillna(0)                           # a missing stat counts as league average


def _fit(Z, y, weights, alpha):
    fits = {name: Ridge(alpha=alpha).fit(Z, y[target], sample_weight=weights) for name, target in TARGETS.items()}
    return ({k: pd.Series(m.coef_, index=Z.columns) for k, m in fits.items()},
            {k: float(m.intercept_) for k, m in fits.items()})


def _predict(Z, coef, intercept):
    out = pd.DataFrame({k: Z[coef[k].index].to_numpy() @ coef[k].to_numpy() + intercept[k] for k in TARGETS},
                       index=Z.index)
    out["RATING"] = out["O_RATING"] + out["D_RATING"]
    return out


def fit_rating_model(df_master, test_season="2025-26", alpha=None, verbose=True, target="lebron"):
    """Train the rating model. Returns a RatingModel; .holdout has the test-season comparison.

    target: whose idea of "good" the model learns.
        "lebron"  LEBRON (default). Rewards rim protection and low-usage
                  role players more.
        "bpm"     Basketball-Reference BPM. Box-score only, so it rewards
                  counting-stat production more; its scale is wider.
        "blend"   the average of the two — a second opinion built in.
    The top of the ranking barely depends on this; the middle does. Fit two
    and compare if a particular player's rank matters.
    WINS always uses LEBRON's wins formula, so with "bpm" or "blend" read the
    wins columns as an ordering, not as a count of wins.

    test_season is kept out of training to measure accuracy, then the model is
    refit including it. alpha=None picks the ridge penalty by leaving out one
    training season at a time.
    """
    df = _regular_season(df_master)
    lebron = load_lebron_seasons(str(DATA_DIR)) if verbose else _quiet(load_lebron_seasons, str(DATA_DIR))
    lebron["PLAYER_ID"] = lebron["PLAYER_ID"].astype(str)
    df = df.merge(lebron, on=["PLAYER_ID", "SEASON"], how="left")

    pool = _pool(df)
    features, counts = _feature_names(pool.drop(columns=[c for c in pool.columns if c.startswith("LBR_")]))
    Z = _standardize(pool, features, counts)
    weights = (pool["MIN"] * pool["GP"]).to_numpy()
    if target not in ("lebron", "bpm", "blend"):
        raise ValueError('target must be "lebron", "bpm" or "blend"')
    sources = list(TARGET_SOURCES.values()) if target == "blend" else [TARGET_SOURCES[target]]
    absent = [c for pair in sources for c in pair if c not in pool.columns]
    if absent:
        raise ValueError(f"{absent} not in the master — run add_bbref_advanced(df_master) first")
    # a plain mean, so a blend needs both sources for the row
    pool["_T_O"] = sum(pool[o].astype(float) for o, _ in sources) / len(sources)
    pool["_T_D"] = sum(pool[d].astype(float) for _, d in sources) / len(sources)
    pool["_T"] = pool["_T_O"] + pool["_T_D"]
    name = target.upper()
    y = pool[["_T", "_T_O", "_T_D"]]
    labelled = y["_T"].notna().to_numpy()
    test = labelled & (pool["SEASON"] == test_season).to_numpy()
    train = labelled & ~test
    if not test.any():
        raise ValueError(f"no LEBRON values for {test_season} to test against")

    if alpha is None:
        seasons = sorted(pool.loc[train, "SEASON"].unique())
        errors = {}
        for a in ALPHAS:
            err = []
            for s in seasons:
                held = train & (pool["SEASON"] == s).to_numpy()
                rest = train & ~held
                coef, icpt = _fit(Z[rest], y[rest], weights[rest], a)
                pred = _predict(Z[held], coef, icpt)
                err.append(np.average((pred["RATING"] - y.loc[held, "_T"]) ** 2, weights=weights[held]))
            errors[a] = float(np.mean(err))
        alpha = min(errors, key=errors.get)

    coef, icpt = _fit(Z[train], y[train], weights[train], alpha)
    pred = _predict(Z[test], coef, icpt)
    holdout = pd.concat([pool.loc[test, ["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN"]], pred,
                         y[test].rename(columns={"_T": name, "_T_O": f"O_{name}", "_T_D": f"D_{name}"})], axis=1)
    for col in ("BBR_BPM", "BBR_OBPM", "BBR_DBPM"):
        if col in pool.columns and target == "lebron":
            holdout[col] = pool.loc[test, col]
    holdout["MISS"] = holdout["RATING"] - holdout[name]
    holdout = holdout.sort_values("RATING", ascending=False).reset_index(drop=True)

    # wins: LEBRON's WAR is (rating above replacement) x minutes, so learn those two numbers from its own files
    known = pool[pool["LBR_LEBRON"].notna().to_numpy() & pool["LBR_WAR"].notna().to_numpy()]
    minutes = (known["MIN"] * known["GP"]).to_numpy()
    A = np.column_stack([known["LBR_LEBRON"].to_numpy() * minutes, minutes])
    (per_point, per_minute), *_ = np.linalg.lstsq(A, known["LBR_WAR"].to_numpy(), rcond=None)
    war = (float(per_point), float(per_minute / per_point))

    if verbose:
        print(f"\nRating model: {len(features)} stats, {int(train.sum())} player-seasons, ridge penalty {alpha:g}")
        print(f"Held-out {test_season} ({int(test.sum())} players), estimate vs actual {name}:")
        for mine, theirs, bench in (("RATING", name, "BBR_BPM"), ("O_RATING", f"O_{name}", "BBR_OBPM"),
                                    ("D_RATING", f"D_{name}", "BBR_DBPM")):
            line = (f"  {mine:9s} correlation {holdout[mine].corr(holdout[theirs]):.3f}, "
                    f"rank correlation {holdout[mine].corr(holdout[theirs], method='spearman'):.3f}, "
                    f"typical miss {(holdout[mine] - holdout[theirs]).abs().median():.2f}")
            if bench in holdout.columns:
                line += f"   ({bench[4:]} manages {holdout[bench].corr(holdout[theirs]):.3f})"
            print(line)
        top = set(holdout.nlargest(20, name)["PLAYER_NAME"])
        print(f"  top-20 overlap: {len(top & set(holdout.nlargest(20, 'RATING')['PLAYER_NAME']))} of 20")

    final = labelled                                           # refit on every season that has LEBRON
    coef, icpt = _fit(Z[final], y[final], weights[final], alpha)
    return RatingModel(features, counts, coef, icpt, alpha, sorted(pool.loc[final, "SEASON"].unique()), holdout, war,
                       target, lebron[["PLAYER_ID", "SEASON", "LBR_LEBRON"]])


def _quiet(fn, *args):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def _scored(model, df_master, season):
    pool = _pool(_regular_season(df_master))
    pool = pool[pool["SEASON"] == season]
    if pool.empty:
        raise ValueError(f"no regular-season rows for {season}")
    absent = [f for f in model.features if f not in pool.columns]
    if absent:
        print(f"⚠️ {len(absent)} of the model's stats are not in the master (treated as league average), e.g. {absent[:5]}")
    return pool, _standardize(pool, model.features, model.counts)


def rank_players(model, df_master, season="2026-27", min_min=15.0, min_gp=None, by="RATING"):
    """Everyone in the rotation pool for a season, best first.

    by: what "best" means — "RATING" (default), "O_RATING", "D_RATING" for the
    best offensive or defensive players, or "WINS" / "WINS_82" for total value.
    RANK follows it.

    RATING     estimated LEBRON: points per 100 possessions over an average player
    O_RATING / D_RATING   the offensive and defensive halves
    WINS       wins above a replacement-level player so far: rating x minutes played
    WINS_82    the same at his current minutes over a full season of his team's games
    RANK / WINS_RANK      position by rate and by total
    LEBRON     the published value, for seasons that have one. The model was
               trained on those seasons, so there RATING tracks LEBRON more
               closely than it would for a season it hasn't seen.

    min_gp=None asks for half of the most games anyone has played (7 of 13 in
    mid-November, 41 of 82 at the end), so a hot four-game stretch can't top
    the list. Early in a season the ratings are still noisy — check GP.
    """
    pool, Z = _scored(model, df_master, season)
    pred = _predict(Z, model.coef, model.intercept)
    out = pd.DataFrame({"Player": pool["PLAYER_NAME"], "PLAYER_ID": pool["PLAYER_ID"],
                        "TEAM": pool["TEAM_ABBREVIATION"], "GP": pool["GP"].astype(int), "MIN": pool["MIN"]})
    out = pd.concat([out, pred[["RATING", "O_RATING", "D_RATING"]]], axis=1)
    if model.lebron is not None:
        known = model.lebron[model.lebron["SEASON"] == season].set_index("PLAYER_ID")["LBR_LEBRON"]
        if len(known):
            out["LEBRON"] = out["PLAYER_ID"].map(known)
    per_point, replacement = model.war
    team_games = pool.groupby("TEAM_ABBREVIATION")["GP"].transform("max").clip(lower=1)
    wins_per_minute = per_point * (out["RATING"] + replacement)
    out["WINS"] = wins_per_minute * pool["MIN"] * pool["GP"]
    out["WINS_82"] = out["WINS"] * (82.0 / team_games).clip(lower=1.0)
    if min_gp is None:
        min_gp = int(np.ceil(pool["GP"].max() / 2))
    out = out[(out["MIN"] >= min_min) & (out["GP"] >= min_gp)]
    if by not in ("RATING", "O_RATING", "D_RATING", "WINS", "WINS_82"):
        raise ValueError('by must be "RATING", "O_RATING", "D_RATING", "WINS" or "WINS_82"')
    out = out.sort_values(by, ascending=False).reset_index(drop=True)
    out.insert(0, "RANK", np.arange(1, len(out) + 1))
    out["WINS_RANK"] = out["WINS"].rank(ascending=False, method="min").astype(int)
    out.attrs.update(season=season, min_min=min_min, min_gp=min_gp, n=len(out), by=by)
    return out


def _playoff_ratings(model, df_master, season):
    """One season's playoff rows rated on the regular-season scale; None if the season has no playoffs yet.

    Playoff stats are measured against that regular season's averages, so the
    same production earns the same rating. Defenses are tougher in the playoffs
    and nearly everyone's numbers dip, so the whole field is then shifted to
    match its own regular-season level (weighted by playoff minutes). A
    player's playoff rating therefore says whether he held up better or worse
    than the field did, not whether his raw numbers fell.
    """
    regular = _pool(_regular_season(df_master))
    regular = regular[regular["SEASON"] == season]
    po = _season_rows(df_master, "Playoffs")
    po = po[(po["SEASON"] == season) & (po["MIN"] >= POOL_MIN) & (po["GP"] >= PLAYOFF_MIN_GP)]
    if regular.empty or po.empty:
        return None
    pred = _predict(_standardize(po, model.features, model.counts, reference=regular), model.coef, model.intercept)
    base = _predict(_standardize(regular, model.features, model.counts), model.coef, model.intercept)
    base.index = regular["PLAYER_ID"].to_numpy()
    base = base.reindex(po["PLAYER_ID"].to_numpy())
    minutes = (po["MIN"] * po["GP"]).to_numpy()
    both = base["RATING"].notna().to_numpy()
    for side in TARGETS:
        if both.any():
            pred[side] += np.average((base[side].to_numpy() - pred[side].to_numpy())[both], weights=minutes[both])
    pred["RATING"] = pred["O_RATING"] + pred["D_RATING"]
    out = pd.DataFrame({"Player": po["PLAYER_NAME"], "PLAYER_ID": po["PLAYER_ID"], "TEAM": po["TEAM_ABBREVIATION"],
                        "GP": po["GP"].astype(int), "MIN": po["MIN"]})
    out = pd.concat([out, pred[["RATING", "O_RATING", "D_RATING"]]], axis=1)
    per_point, replacement = model.war
    out["WINS"] = per_point * (out["RATING"] + replacement) * minutes
    return out.reset_index(drop=True)


def rank_multi_year(model, df_master, seasons=None, last=3, min_minutes=1500, playoffs=False, playoff_weight=1.0):
    """Several seasons combined: rating averaged by minutes played, wins added up.

    seasons: an explicit list, or None for the most recent `last` seasons in
    the master (the one in progress included). min_minutes is total
    regular-season minutes across the window. One column per season shows the
    regular-season trend; a blank means he wasn't in that season's rotation pool.

    playoffs=True also counts each season's playoff games (4+ games at 10+
    minutes). playoff_weight is how much one playoff minute counts against one
    regular-season minute: 1 treats them alike, 2 makes a 20-game run worth
    about half a season. Extra columns:
        REG_RATING   the regular-season-only rating (what playoffs=False gives)
        PO_RATING    his playoff rating across the window
        PO_GP / PO_MINUTES / PO_WINS   how much playoff basketball that is
        PO_SHIFT     RATING minus REG_RATING: how far the playoffs moved him
    A single series is a small sample against one opponent — the weighting by
    minutes is what keeps a five-game exit from counting like a Finals run.
    The default of 1 is the tested one: over ten seasons, counting playoff
    minutes at 0.5 to 1 predicted the next season's LEBRON slightly better
    than ignoring them, and at 2 or more slightly worse. That test scores
    against a regular-season measure, so raise the weight if playoff
    performance matters to you for its own sake.
    The season in progress has no playoffs yet and is simply skipped.
    """
    every = sorted(_regular_season(df_master)["SEASON"].unique())
    seasons = list(seasons) if seasons is not None else every[-last:]
    parts = []
    for season in seasons:
        one = rank_players(model, df_master, season, min_min=0, min_gp=0)
        one["SEASON"], one["PLAYOFFS"] = season, False
        parts.append(one)
        post = _playoff_ratings(model, df_master, season) if playoffs else None
        if post is not None:
            post["SEASON"], post["PLAYOFFS"] = season, True
            parts.append(post)
    long = pd.concat(parts, ignore_index=True)
    long["MINUTES"] = long["MIN"] * long["GP"]
    long["WEIGHT"] = long["MINUTES"] * np.where(long["PLAYOFFS"], playoff_weight, 1.0)
    reg, post = long[~long["PLAYOFFS"]], long[long["PLAYOFFS"]]

    def weighted(rows, col, by="WEIGHT"):
        return (rows[col] * rows[by]).groupby(rows["PLAYER_ID"]).sum() / rows.groupby("PLAYER_ID")[by].sum()

    latest = reg.sort_values("SEASON").groupby("PLAYER_ID").last()
    out = pd.DataFrame({
        "Player": latest["Player"], "TEAM": latest["TEAM"],
        "SEASONS": reg.groupby("PLAYER_ID")["SEASON"].nunique(), "GP": reg.groupby("PLAYER_ID")["GP"].sum(),
        "MINUTES": reg.groupby("PLAYER_ID")["MINUTES"].sum().round(0),
        "RATING": weighted(long, "RATING"), "O_RATING": weighted(long, "O_RATING"), "D_RATING": weighted(long, "D_RATING"),
        "WINS": reg.groupby("PLAYER_ID")["WINS"].sum(),
    })
    if playoffs:
        out["REG_RATING"] = weighted(reg, "RATING")
        out["PO_RATING"] = weighted(post, "RATING", by="MINUTES")
        out["PO_GP"] = post.groupby("PLAYER_ID")["GP"].sum().reindex(out.index).fillna(0).astype(int)
        out["PO_MINUTES"] = post.groupby("PLAYER_ID")["MINUTES"].sum().reindex(out.index).fillna(0).round(0)
        out["PO_WINS"] = post.groupby("PLAYER_ID")["WINS"].sum().reindex(out.index).fillna(0.0)
        out["PO_SHIFT"] = out["RATING"] - out["REG_RATING"]
    by_season = reg.pivot(index="PLAYER_ID", columns="SEASON", values="RATING").reindex(columns=seasons)
    out = out.join(by_season)
    out = out[out["MINUTES"] >= min_minutes].sort_values("RATING", ascending=False).reset_index(drop=True)
    out.insert(0, "RANK", np.arange(1, len(out) + 1))
    out["WINS_RANK"] = out["WINS"].rank(ascending=False, method="min").astype(int)
    out.attrs.update(seasons=seasons, min_minutes=min_minutes, n=len(out), playoffs=playoffs, playoff_weight=playoff_weight)
    return out


def _previous_season(season):
    start = int(season[:4]) - 1
    return f"{start}-{str(start + 1)[-2:]}"


def most_improved(model, df_master, season="2026-27", season_from=None, by="RATING",
                  min_minutes_from=1000, min_min=20.0, min_gp=None):
    """Biggest changes in rating from one season to the next, largest rise first (.tail() for the fallers).

    season_from defaults to the season before `season`. by: "RATING",
    "O_RATING" or "D_RATING" for how well he plays per possession, or
    "WINS_82" for total value at an 82-game pace. The ratings ignore minutes on
    purpose, so a player who holds his efficiency in a much bigger role barely
    moves on "RATING" and jumps on "WINS_82" — closer to how the award is voted.
    A player needs min_minutes_from total minutes in
    the earlier season, and min_min minutes a game plus the usual games floor
    in the later one — a jump from a 300-minute season is mostly noise.

    CHANGE is the headline; O_CHANGE and D_CHANGE split it, and MIN_CHANGE
    shows whether his role grew. While `season` is still being played, a short
    hot or cold stretch reads as a change in level: treat the list as players
    to watch until the season has more games.
    """
    season_from = season_from or _previous_season(season)
    now = rank_players(model, df_master, season, min_min=min_min, min_gp=min_gp)
    was = rank_players(model, df_master, season_from, min_min=0, min_gp=0)
    was = was[was["MIN"] * was["GP"] >= min_minutes_from]
    both = now.merge(was, on="PLAYER_ID", suffixes=("", "_FROM"))
    out = pd.DataFrame({
        "Player": both["Player"], "TEAM": both["TEAM"], "GP": both["GP"], "MIN": both["MIN"],
        "FROM": both[f"{by}_FROM"], "TO": both[by], "CHANGE": both[by] - both[f"{by}_FROM"],
        "O_CHANGE": both["O_RATING"] - both["O_RATING_FROM"], "D_CHANGE": both["D_RATING"] - both["D_RATING_FROM"],
        "MIN_CHANGE": both["MIN"] - both["MIN_FROM"], "RANK_FROM": both["RANK_FROM"], "RANK_TO": both["RANK"],
    })
    out = out.sort_values("CHANGE", ascending=False).reset_index(drop=True)
    out.attrs.update(season=season, season_from=season_from, by=by, n=len(out))
    return out


def compare_seasons(model, df_master, player, season_from, season_to="2026-27"):
    """Which parts of a player's game moved his rating between two seasons.

    One row per stat family: its contribution in each season and the
    difference, biggest gain first. The CHANGE column adds up to the change in
    his rating.
    """
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        was = rating_drivers(model, df_master, player, season_from)
        now = rating_drivers(model, df_master, player, season_to)
    out = pd.DataFrame({"FROM": was.set_index("FAMILY")["TOTAL"], "TO": now.set_index("FAMILY")["TOTAL"]})
    out["CHANGE"] = out["TO"] - out["FROM"]
    out = out.sort_values("CHANGE", ascending=False).reset_index()
    print(f"{player}: {was.attrs['rating']:+.2f} in {season_from} -> {now.attrs['rating']:+.2f} in {season_to} "
          f"({now.attrs['rating'] - was.attrs['rating']:+.2f})")
    print(f"  biggest gain: {out.iloc[0]['FAMILY']} ({out.iloc[0]['CHANGE']:+.2f})    "
          f"biggest loss: {out.iloc[-1]['FAMILY']} ({out.iloc[-1]['CHANGE']:+.2f})")
    return out.round(2)


# Which part of the game a stat belongs to. First match wins, so order matters.
# All-in-one numbers (PIE, net rating) share a family with scoring volume and
# usage on purpose: the model pays for production through the first and charges
# for shots used through the second, so apart they show two large numbers that
# cancel, and together they show the net.
FAMILIES = [
    ("Overall production and usage", r"^(PIE|NET_RATING|OFF_RATING|DEF_RATING|PLUS_MINUS|W_PCT|E_PACE|PACE|PACE_PER40)$|^ONCOURT_|^OFFCOURT_|^ONOFF_"),
    ("Defense", r"^(STL|BLK|PF|DEFLECTIONS|CHARGES_DRAWN|CONTESTED_SHOTS.*|D_FG.*|PCT_PLUSMINUS|PCT_STL|PCT_BLK|PCT_PF)$|^DEF|^OPP_|LOOSE_BALLS|DIST_MILES_DEF|AVG_SPEED_DEF"),
    ("Rebounding", r"REB|BOX_?OUT|PUTBACK|PTS_2ND_CHANCE"),
    ("Playmaking and ball security", r"AST|PASSES|TOV|^TIME_OF_POSS$|^AVG_SEC_PER_TOUCH$|^AVG_DRIB_PER_TOUCH$|^TOUCHES$|^FRONT_CT_TOUCHES$"),
    ("Free throws and drawing fouls", r"^(FTM|FTA|FT_PCT|PFD|PCT_FTM|PCT_FTA|PCT_PFD|PCT_PTS_FT)$|_FT_POSS_PCT$|_SF_POSS_PCT$|_PLUSONE_POSS_PCT$|_FT[MA]$|_FT_PCT$|_FOULS"),
    ("Transition", r"^TRANS_|PTS_FB|^SC_24_22_|^SC_22_18_"),
    ("Post and elbow", r"^POST|^ELBOW|PTS_PER_(POST|ELBOW)"),
    ("Rim and paint", r"Restricted_Area|In_The_Paint|^DRIVE|^PAINT|^CUT_|^RM_|PCT_PTS_PAINT|PTS_PER_PAINT|^(BLKA|PCT_BLKA)$"),
    ("Three-point shooting", r"FG3|Corner_3|Above_the_Break_3|3PT|3PM|^SPOT_|^OFFSCREEN_|^CATCH_SHOOT"),
    ("Self-created scoring", r"^ISO_|^PNRBH_|^DHO_|^PULL_UP|^DRIBB_|^TOUCH_|Mid_Range|PCT_UAST|PCT_AST_|PCT_PTS_2PT_MR|^SC_|^CLOSESTDEF_"),
    ("Overall production and usage", r"^(PTS|POINTS|FGM|FGA|FG_PCT|EFG_PCT|TS_PCT|USG_PCT|PCT_FGM|PCT_FGA|PCT_PTS|PTS_PER_TOUCH|PTS_OFF_TOV)$|^SHOT_|2PT|2PM|PCT_PTS"),
    ("Movement and screens", r"DIST_|AVG_SPEED|SCREEN_"),
]


def stat_family(stat):
    for name, pattern in FAMILIES:
        if re.search(pattern, stat):
            return name
    return "Other"


def rating_drivers(model, df_master, player, season="2026-27", top_stats=3):
    """Which part of a player's game carries his rating.

    Every stat's contribution (its weight x how far he is from the season
    average) is added up within a family: scoring, playmaking, defense, and so
    on. Grouping matters because overlapping stats split and trade credit under
    ridge — usage gets a negative weight that scoring volume offsets — so
    family totals are steadier than any single line.

    Returns one row per family, biggest lift first: its share of his offensive
    rating (O), defensive rating (D), the TOTAL, and the stats inside it that
    move the number most. The contributions add up to how far he is from an
    average player (BASELINE is the average player's rating).
    """
    pool, Z = _scored(model, df_master, season)
    hit = pool["PLAYER_NAME"] == player
    if not hit.any():
        raise ValueError(f"{player!r} is not in the {season} pool ({POOL_MIN:g}+ minutes and enough games)")
    z = Z[hit].iloc[0]
    stats = pd.DataFrame({"O": model.coef["O_RATING"] * z, "D": model.coef["D_RATING"] * z})
    stats["TOTAL"] = stats["O"] + stats["D"]
    stats["FAMILY"] = [stat_family(c) for c in stats.index]

    def leaders(group):
        top = group["TOTAL"].reindex(group["TOTAL"].abs().sort_values(ascending=False).index).head(top_stats)
        return ", ".join(f"{name} ({value:+.2f})" for name, value in top.items())

    out = stats.groupby("FAMILY")[["O", "D", "TOTAL"]].sum()
    out["TOP_STATS"] = stats.groupby("FAMILY").apply(leaders)
    out = out.sort_values("TOTAL", ascending=False).reset_index()
    baseline = model.intercept["O_RATING"] + model.intercept["D_RATING"]
    rating = out["TOTAL"].sum() + baseline
    lead, drag = out.iloc[0], out.iloc[-1]
    print(f"{player}, {season}: rating {rating:+.2f} = average player {baseline:+.2f} + his stats {out['TOTAL'].sum():+.2f}")
    print(f"  carried by: {lead['FAMILY']} ({lead['TOTAL']:+.2f})    held back by: {drag['FAMILY']} ({drag['TOTAL']:+.2f})")
    out.attrs.update(player=player, season=season, rating=float(rating), baseline=float(baseline))
    return out.round(2)


def rating_breakdown(model, df_master, player, season="2026-27", n=8):
    """Which stats push a player's rating up or down the most.

    One row per stat: VALUE (per-36 for counts), Z (standard deviations from
    the season average), and its contribution to O_RATING, D_RATING and the
    total. Returns the n biggest pushes up and the n biggest pushes down.
    Overlapping stats share credit under ridge, so read families of rows
    (several rim stats, several passing stats), not single lines.
    """
    pool, Z = _scored(model, df_master, season)
    hit = pool["PLAYER_NAME"] == player
    if not hit.any():
        raise ValueError(f"{player!r} is not in the {season} pool ({POOL_MIN:g}+ minutes and enough games)")
    row, z = pool[hit].iloc[0], Z[hit].iloc[0]
    value = pd.to_numeric(row.reindex(model.features), errors="coerce")
    value[model.counts] = value[model.counts] / row["MIN"] * 36
    table = pd.DataFrame({"VALUE": value, "Z": z, "O": model.coef["O_RATING"] * z, "D": model.coef["D_RATING"] * z})
    table["TOTAL"] = table["O"] + table["D"]
    table = table.sort_values("TOTAL", ascending=False)
    total = table["TOTAL"].sum() + model.intercept["O_RATING"] + model.intercept["D_RATING"]
    print(f"{player}, {season}: rating {total:+.2f} "
          f"(offense {table['O'].sum() + model.intercept['O_RATING']:+.2f}, "
          f"defense {table['D'].sum() + model.intercept['D_RATING']:+.2f})")
    out = pd.concat([table.head(n), table.tail(n)]).rename_axis("STAT").reset_index()
    return out.round(3)


def top_weights(model, side="O_RATING", n=15):
    """The stats the model leans on most for offense or defense (weight per standard deviation)."""
    coef = model.coef[side]
    return coef.reindex(coef.abs().sort_values(ascending=False).index).head(n).round(3)
