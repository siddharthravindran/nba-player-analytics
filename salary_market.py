"""The market salary model (01_modeling.ipynb) for the season in progress, and whole contracts against it.

    from salary_market import *
    scored = score_market(df_master)                 # every player-season, rebuilt as the notebook does, scored by both saved models
    this_season(scored)                              # paid vs market rate this season
    contracts_vs_market(scored)                      # what he's owed over the rest of his deal vs what the market would pay

The notebook trains the models and builds the features; this file only reruns
those feature steps (checked: 2025-26 comes out identical to the app) and adds
two things for a season in progress:

  - Season totals (games, wins, losses, possessions, double-doubles...) are put
    on an 82-game pace, found automatically by comparing this season's values
    with last season's. Otherwise 13 games looks like a player who missed most
    of the year.
  - Whole contracts. The market model prices one season. contracts_vs_market
    prices every season left on the deal by scoring him again for each one,
    a year more experienced each time and with this season as his last season
    (so a rookie-scale player's rate jumps when his rookie deal would end, and
    a veteran's falls the way the market's experience effect says), times a
    cap growing CAP_GROWTH a year. It assumes he keeps playing like this season.
    A player option counts if he'd be paid more than his market rate that
    year (he'd pick it up), a team option if he'd be paid less; anything else
    is guaranteed.
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from nba_viz_utils import (INDEX_COLS, SALARY_CAP, SEASONS, attach_salaries, build_model_table, fetch_all_nba_raw,
                           get_draft_table, lag_and_encode_lebron, load_lebron_seasons, merge_lebron, norm_name)

__all__ = ["score_market", "this_season", "contracts_vs_market", "market_ahead"]

CURRENT = max(SEASONS)
VERSIONS = {
    # v2: the notebook's models. Undrafted players have no EXPERIENCE (it is counted from draft year).
    "v2": {"box": ("data/salary_model_v2.txt", "data/feats_v2.json"),
           "lebron": ("data/salary_model_v2_lebron.txt", "data/feats_v2_lebron.json")},
    # v3: retrained the same way, with undrafted players' EXPERIENCE counted from their first NBA season.
    "v3": {"box": ("data/salary_model_v3.txt", "data/feats_v2.json"),
           "lebron": ("data/salary_model_v3_lebron.txt", "data/feats_v2_lebron.json")},
}
MODELS = VERSIONS["v2"]
SALARY_FILE = "data/bbref_salaries_2015-2027.csv"
CAP_GROWTH = 0.07
MIN_PCT_CAP = 0.008                       # roughly a minimum contract, as a share of the cap
MIN_GP_NOW = 5                            # games this season before a player is scored


def _season_totals(model, feats):
    """Inputs that are season totals: this season's values are a small fraction of last season's for the same players."""
    cur, prev = model[model["SEASON"] == CURRENT], model[model["SEASON"] == SEASONS[-2]]
    both = cur.merge(prev, on="PLAYER_ID", suffixes=("", "_prev"))
    both = both[(both["GP_rs"] >= 8) & (both["GP_rs_prev"] >= 40)]
    out = []
    for f in feats:
        if f.endswith("_rs") and f in model.columns and pd.api.types.is_numeric_dtype(model[f]) \
                and not any(k in f for k in ("RATING", "PCT", "PIE")):
            a, b = pd.to_numeric(both[f], errors="coerce"), pd.to_numeric(both[f + "_prev"], errors="coerce")
            ok = (a.abs() > 0) & (b.abs() > 0)
            if ok.sum() >= 30 and (a[ok] / b[ok]).median() < 0.4:
                out.append(f)
    return out


def _all_nba(model):
    """ALLNBA_PRIOR3 / _PRIOR_EVER / _WT_EVER / _WT3, as in the notebook (cells 30-32)."""
    raw = fetch_all_nba_raw()
    an = raw[raw["Lg"] == "NBA"].melt(id_vars=["Season", "Tm"], value_vars=[f"Unnamed: {i}" for i in range(4, 9)],
                                      value_name="player_raw").dropna(subset=["player_raw"])
    an["name_key"] = an["player_raw"].str.replace(r"\s+[CFG](-[CFG])?$", "", regex=True).apply(norm_name)
    an = an.rename(columns={"Season": "SEASON"})
    an = an[an["SEASON"].isin(SEASONS)][["name_key", "SEASON", "Tm"]]
    an["yr"] = an["SEASON"].str[:4].astype(int)
    picks = an.groupby(["name_key", "yr"]).size().rename("n").reset_index()
    an["weight"] = an["Tm"].map({"1st": 3, "2nd": 2, "3rd": 1})
    aw = an.dropna(subset=["weight"])

    def counts(p, col, year, decay=0.75):
        last3 = p[(p["yr"] < year) & (p["yr"] >= year - 3)][col].sum()
        past = p[p["yr"] < year]
        return last3, float((past[col] * (decay ** (year - past["yr"]))).sum())
    by_n, by_w = dict(tuple(picks.groupby("name_key"))), dict(tuple(aw.groupby("name_key")))
    empty_n, empty_w = picks.iloc[:0], aw.iloc[:0]
    keys, yrs = model["PLAYER_NAME"].apply(norm_name), model["SEASON"].str[:4].astype(int)
    c = [counts(by_n.get(k, empty_n), "n", y) for k, y in zip(keys, yrs)]
    w = [counts(by_w.get(k, empty_w), "weight", y) for k, y in zip(keys, yrs)]
    model["ALLNBA_PRIOR3"] = [int(x[0]) for x in c]
    model["ALLNBA_PRIOR_EVER"] = [x[1] for x in c]
    model["ALLNBA_WT_EVER"] = [x[1] for x in w]
    model["ALLNBA_WT3"] = [int(x[0]) for x in w]
    return model


def _debut_years():
    """Each player's first NBA season (start year), from the NBA's all-players list: {PLAYER_ID (str): year}."""
    from nba_viz_utils import cached_fetch
    from nba_api.stats.endpoints import CommonAllPlayers
    p = cached_fetch("common_all_players", lambda: CommonAllPlayers(is_only_current_season=0, league_id="00",
                                                                    timeout=60).get_data_frames()[0])
    return dict(zip(p["PERSON_ID"].astype(str), pd.to_numeric(p["FROM_YEAR"], errors="coerce")))


# Early in a season each stat is mixed with last season's: this season counts GP / (GP + BLEND_GAMES). Tested on past
# seasons frozen at 16, 30, 45 and 60 games, against the rate the full season ended up giving: the average miss at 16
# games drops 12% (box) and 9% (LEBRON); 4-6 was best, and by game 60 it makes no difference. Only box and advanced
# stats could be frozen, so the tracking stats are blended the same way untested.
BLEND_GAMES = 5
BLEND_PRIOR_GP = 15                  # a last season shorter than this is not used


def _blend_last_season(mdf, model, features):
    """This season's inputs mixed with the player's last season, in place of a small sample on its own."""
    now = np.flatnonzero((mdf["SEASON"] == CURRENT).to_numpy())
    last = model[(model["SEASON"] == SEASONS[-2]) & (pd.to_numeric(model["GP_rs"], errors="coerce") >= BLEND_PRIOR_GP)]
    last = last.drop_duplicates("PLAYER_ID").set_index("PLAYER_ID").reindex(mdf["PLAYER_ID"].to_numpy()[now])
    gp = pd.to_numeric(mdf["GP_ACTUAL"], errors="coerce").to_numpy(float)[now]
    w = gp / (gp + BLEND_GAMES)
    cols = [f for f in features if f.endswith("_rs") and f in mdf.columns and f in last.columns]
    new = {}
    for f in cols:
        full = pd.to_numeric(mdf[f], errors="coerce").to_numpy(float).copy()
        a, b = full[now], pd.to_numeric(last[f], errors="coerce").to_numpy(float)
        full[now] = np.where(np.isnan(b), a, np.where(np.isnan(a), np.nan, w * a + (1 - w) * b))
        new[f] = full
    return mdf.assign(**new)


def score_market(df_master, version="v3", verbose=True, blend=True):
    """Every scoreable player-season with the market rate from both saved models: MARKET_BOX / MARKET_LEBRON (share of cap).

    Scoreable as in the notebook: a salary, 20+ games (this season: on pace)
    and 10+ minutes; this season also needs MIN_GP_NOW real games.
    blend=True mixes this season's stats with last season's (BLEND_GAMES).
    version "v3" (default) counts undrafted players' experience from their first
    NBA season and scores with the v3 models; "v2" reproduces the notebook/app.
    """
    t0 = time.time()
    models = VERSIONS[version]
    feats = {k: json.load(open(f)) for k, (_, f) in models.items()}
    model = build_model_table(df_master)
    now = model["SEASON"] == CURRENT
    totals = _season_totals(model, sorted(set(feats["box"]) | set(feats["lebron"])))
    team_games = model.loc[now].groupby("TEAM_ABBREVIATION")["GP_rs"].max()
    pace = 82 / model.loc[now, "TEAM_ABBREVIATION"].map(team_games).clip(lower=1)
    model["GP_ACTUAL"] = model["GP_rs"]
    for f in set(totals) | {"GP_rs"}:
        model[f] = pd.to_numeric(model[f], errors="coerce").astype(float)      # whole-number columns can't hold the paced values
        model.loc[now, f] = model.loc[now, f] * pace

    model = attach_salaries(model, pd.read_csv(SALARY_FILE))
    model["PLAYER_ID"] = model["PLAYER_ID"].astype(str)
    bio = get_draft_table(SEASONS)
    bio["PLAYER_ID"] = bio["PLAYER_ID"].astype(str)
    model = model.drop(columns=["DRAFT_POSITION", "DRAFT_YR"], errors="ignore").merge(bio, on="PLAYER_ID", how="left")
    model["IS_UNDRAFTED"] = model["DRAFT_POSITION"].isna().astype(int)
    model["EXPERIENCE"] = (model["SEASON"].str[:4].astype(int) - model["DRAFT_YR"]).clip(lower=0)
    if version != "v2":                                  # undrafted: count from the first NBA season instead
        debut = model["PLAYER_ID"].map(_debut_years())
        missing = model["EXPERIENCE"].isna()
        model.loc[missing, "EXPERIENCE"] = (model["SEASON"].str[:4].astype(int) - debut)[missing].clip(lower=0)
    model["MAX_PCT_ELIGIBLE"] = model["EXPERIENCE"].map(
        lambda e: np.nan if pd.isna(e) else (0.25 if e <= 6 else 0.30 if e <= 9 else 0.35))
    model["pct_cap"] = model["Salary"] / model["SEASON"].map(SALARY_CAP)
    model = _all_nba(model)

    model["GP_rs"] = model["GP_rs"].fillna(0)
    model["AVAILABILITY"] = model["GP_rs"] / 82.0
    model["TOTAL_MIN_rs"] = model["MIN_rs"] * model["GP_rs"]
    spine = [c for c in ["MIN_rs", "FGM_rs", "FTM_rs", "PIE_rs", "USG_PCT_rs", "FRONT_CT_TOUCHES_rs", "POST_TOUCHES_rs",
                         "CLOSESTDEF_4_6_FGM_rs", "GP_rs", "TOTAL_MIN_rs", "ALLNBA_WT_EVER"] if c in model.columns]
    model["_yr"] = model["SEASON"].str[:4].astype(int)
    model = model.sort_values(["PLAYER_ID", "_yr"])
    for k in (1, 2):
        lagged = model.groupby("PLAYER_ID")[spine].shift(k)
        lagged = lagged.where((model["_yr"] - model.groupby("PLAYER_ID")["_yr"].shift(k)) == k)
        lagged.columns = [f"{c}_lag{k}" for c in spine]
        model = pd.concat([model, lagged], axis=1)
    model = model.drop(columns="_yr").loc[:, lambda m: ~m.columns.duplicated()]

    real_gp = model["GP_rs"].mask((model["SEASON"] == CURRENT).astype(bool), pd.to_numeric(model["GP_ACTUAL"], errors="coerce"))
    mdf = model[model["Salary"].notna() & (model["GP_rs"] >= 20) & (model["MIN_rs"] >= 10) & (real_gp >= MIN_GP_NOW)
                & model["pct_cap"].notna()].copy()
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        mdf = lag_and_encode_lebron(merge_lebron(mdf, load_lebron_seasons("data")))
    mdf = mdf.reset_index(drop=True)
    if blend:
        mdf = _blend_last_season(mdf, model, sorted(set(feats["box"]) | set(feats["lebron"])))
    mdf["index"] = mdf.index                       # the saved models carry a leftover 'index' input (it has no weight)
    import os
    for k, (path, _) in models.items():
        if os.path.exists(path):
            X = mdf.reindex(columns=feats[k]).apply(pd.to_numeric, errors="coerce")
            mdf[f"MARKET_{k.upper()}"] = lgb.Booster(model_file=path).predict(X)
    if verbose:
        print(f"Market model scored {len(mdf)} player-seasons ({int((mdf['SEASON'] == CURRENT).sum())} this season) "
              f"in {time.time() - t0:.0f}s; on an 82-game pace this season: {sorted(set(totals) | {'GP_rs'})}")
    mdf.attrs["paced"] = sorted(set(totals) | {"GP_rs"})
    mdf.attrs["version"] = version
    return mdf


def this_season(scored, season=CURRENT, model="box"):
    """Paid vs market rate for one season, in $M: OVER_M > 0 is paid above the market rate."""
    cap = SALARY_CAP[season]
    s = scored[scored["SEASON"] == season]
    out = pd.DataFrame({"Player": s["PLAYER_NAME"], "TEAM": s["TEAM_ABBREVIATION"], "GP": s["GP_ACTUAL"].astype(int),
                        "AGE_EXP": s["EXPERIENCE"], "PAID_M": s["Salary"] / 1e6,
                        "MARKET_M": s[f"MARKET_{model.upper()}"] * cap / 1e6})
    out["OVER_M"] = out["PAID_M"] - out["MARKET_M"]
    return out.sort_values("OVER_M", ascending=False).reset_index(drop=True)


def market_ahead(rows, years_ahead, model="box", version="v3"):
    """The market rate (share of the cap) for these player-seasons, `years_ahead` seasons on.

    Rescored with EXPERIENCE + years_ahead (and its max-contract tier), and
    this season standing in as last season (and the one before), i.e. if he
    keeps playing like this. 0 years ahead is the plain market rate.
    """
    path, fpath = VERSIONS[version][model]
    feats = json.load(open(fpath))
    X = rows.reindex(columns=feats).apply(pd.to_numeric, errors="coerce").copy()
    if years_ahead:
        X["EXPERIENCE"] = X["EXPERIENCE"] + years_ahead
        X["MAX_PCT_ELIGIBLE"] = X["EXPERIENCE"].map(lambda e: np.nan if pd.isna(e) else (0.25 if e <= 6 else 0.30 if e <= 9 else 0.35))
        for f in [f[:-5] for f in feats if f.endswith("_lag1")]:
            if f in X.columns:
                X[f"{f}_lag2"] = X[f"{f}_lag1"] if years_ahead == 1 else X[f]
                X[f"{f}_lag1"] = X[f]
    return pd.Series(lgb.Booster(model_file=path).predict(X), index=rows.index)


def contracts_vs_market(scored, model="box", cap_growth=CAP_GROWTH, contracts=None):
    """Every scored player's remaining contract against the market rate for each of those seasons, in $M.

    OWED_M      salary over the seasons that count (options as described above)
    MARKET_M    what the market would pay him over the same seasons (market_ahead)
    OVER_M      OWED_M - MARKET_M: overpaid over the rest of the deal (> 0)
    OVER_NOW_M  the same for this season alone, for contrast
    YEARS / THROUGH   seasons counted and the last one
    """
    from bbref_stats import fetch_contract_years
    c = (contracts if contracts is not None else fetch_contract_years()).copy()
    c = c[c["SEASON"] >= CURRENT]
    now = scored[scored["SEASON"] == CURRENT].copy()
    now["name_key"] = now["PLAYER_NAME"].map(norm_name)
    span = int(c["SEASON"].str[:4].astype(int).max()) - int(CURRENT[:4]) + 1
    version = scored.attrs.get("version", "v3")
    rate = pd.DataFrame({k: market_ahead(now, k, model, version) for k in range(span)})
    cap0 = SALARY_CAP[CURRENT]
    rows = []
    for i, p in now.iterrows():
        mine = c[c["name_key"] == p["name_key"]].sort_values("SEASON")
        if mine.empty:
            continue
        owed = market = 0.0
        counted = []
        for _, y in mine.iterrows():
            ahead = int(y["SEASON"][:4]) - int(CURRENT[:4])
            # real terms: divide pay by the cap growth to that season (contracts are fixed dollars), worth at today's cap
            pay, worth = float(y["Salary"]) / (1 + cap_growth) ** ahead, max(float(rate.at[i, ahead]), MIN_PCT_CAP) * cap0
            opt = str(y.get("OPTION", "") or "").lower()
            if ahead == 0 or "player" not in opt and "team" not in opt or ("player" in opt and pay > worth) \
                    or ("team" in opt and pay < worth):
                owed += pay
                market += worth
                counted.append(y["SEASON"])
        rows.append({"Player": p["PLAYER_NAME"], "TEAM": p["TEAM_ABBREVIATION"], "EXP": p["EXPERIENCE"],
                     "YEARS": len(counted), "THROUGH": counted[-1] if counted else None,
                     "OWED_M": owed / 1e6, "MARKET_M": market / 1e6, "OVER_M": (owed - market) / 1e6,
                     "OVER_NOW_M": (p["Salary"] - p[f"MARKET_{model.upper()}"] * cap0) / 1e6})
    out = pd.DataFrame(rows).sort_values("OVER_M", ascending=False).reset_index(drop=True)
    out.attrs.update(model=model, cap_growth=cap_growth)
    return out
