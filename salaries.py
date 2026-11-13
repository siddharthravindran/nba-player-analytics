"""Salaries: what each playing style gets paid, and this season's pay against the model.

    from salaries import *
    sal = load_salaries()                                  # 2015-16 .. this season, with % of cap

    profiles = build_profiles(df_master, "2025-26")        # from player_profiles
    salary_by_profile(profiles, sal)                       # which styles the league pays for
    plot_salary_by_profile(profiles, sal)

    score_season(df_master)                                # 2026-27 pay vs the model, on this season's stats
    pay_check()                                            # the same question from LAST season's stats

Pay is compared as a share of the salary cap, so seasons with different caps
line up; dollar figures use the cap of the season in question.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bbref_stats import DATA_DIR, fetch_contracts, name_key
from nba_viz_utils import (SALARY_CAP, SEASONS, attach_salaries, build_model_table, fetch_all_nba_raw,
                           get_draft_table, get_player_registry, norm_name, prefilter_features)

HISTORY_FILE = DATA_DIR / "bbref_salaries_2015-2026.csv"       # written by 01_modeling.ipynb
BLUE, ORANGE, GRID, MUTED, INK = "#2a78d6", "#eb6834", "#e4e3de", "#8a897f", "#0b0b0b"


def load_salaries(refresh_current=False, save=True):
    """Salary history plus this season's contracts: SEASON, Player, name_key, Salary, pct_cap.

    The history file stops at 2025-26; this season comes from Basketball-Reference's
    contracts page. With save=True the combined table is written next to the
    history file as bbref_salaries_2015-2027.csv, in the same four columns, so
    the modeling notebook can read it in place of the old one.
    """
    history = pd.read_csv(HISTORY_FILE)
    current = fetch_contracts(force=refresh_current)[["SEASON", "Salary", "Player", "name_key"]]
    history = history[~history["SEASON"].isin(current["SEASON"].unique())]
    out = pd.concat([history, current], ignore_index=True)
    out["name_key"] = out["name_key"].fillna(out["Player"].map(name_key))
    out = out.drop_duplicates(["name_key", "SEASON"])
    missing = sorted(set(out["SEASON"]) - set(SALARY_CAP))
    if missing:
        print(f"⚠️ no SALARY_CAP entry for {missing}; pct_cap is empty there")
    if save:
        last = out["SEASON"].max()
        out.to_csv(DATA_DIR / f"bbref_salaries_2015-{int(last[:4]) + 1}.csv", index=False)
    out["pct_cap"] = out["Salary"] / out["SEASON"].map(SALARY_CAP)
    return out


def _on_rookie_deal(names, season):
    """First-round picks in their first four seasons: paid by draft slot, not by how they play."""
    try:
        bio = pd.read_parquet(DATA_DIR / "bio_registry.parquet")
    except FileNotFoundError:
        return pd.Series(False, index=names)
    bio = bio.drop_duplicates("PLAYER_NAME", keep="last").copy()
    bio["_key"] = bio["PLAYER_NAME"].map(name_key)
    bio = bio.drop_duplicates("_key", keep="last")       # the same player spelled with and without accents
    year = pd.to_numeric(bio.set_index("_key")["DRAFT_YR"], errors="coerce")
    first_round = pd.to_numeric(bio.set_index("_key")["DRAFT_ROUND"], errors="coerce") == 1
    keys = pd.Index(names).map(name_key)
    years_in = int(season[:4]) - year.reindex(keys).to_numpy()
    return pd.Series((years_in < 4) & first_round.reindex(keys).fillna(False).to_numpy(), index=names)


def salary_by_profile(profiles, salaries=None, top_share=0.25, exclude_rookie_deals=True):
    """What each style is paid: one row per profile group.

    PAY_IF_HIGH_TENDENCY   median % of cap among the top quarter by tendency (does it most)
    PAY_IF_HIGH_SKILL      the same for the top quarter by skill (does it best)
    TENDENCY_CORR / SKILL_CORR   rank correlation of the score with pay, everyone in the pool

    Rookie-scale players are left out by default: their pay follows draft slot,
    so they would make any style popular with young players look cheap.

    These are descriptions, not prices. The groups overlap heavily — high-usage
    scorers also self-create and pass — so a group's pay reflects everything
    those players do, not that one skill in isolation.
    """
    salaries = load_salaries(save=False) if salaries is None else salaries
    pay = salaries[salaries["SEASON"] == profiles.season].set_index("name_key")["pct_cap"]
    pct = pd.Series(pd.Index(profiles.pool.index).map(name_key).map(pay), index=profiles.pool.index)
    keep = pct.notna()
    if exclude_rookie_deals:
        keep &= ~_on_rookie_deal(profiles.pool.index, profiles.season)
    pct = pct[keep]
    cap = SALARY_CAP[profiles.season]

    rows = []
    for group in profiles.groups:
        row = {"GROUP": group}
        for kind in ("tendency", "skill"):
            score = getattr(profiles, kind).loc[pct.index, group].dropna()
            if len(score) < 20:
                continue
            top = score[score >= score.quantile(1 - top_share)].index
            row[f"PAY_IF_HIGH_{kind.upper()}"] = pct[top].median()
            row[f"{kind.upper()}_CORR"] = score.rank().corr(pct[score.index].rank())
            row[f"N_{kind.upper()}"] = len(top)
        rows.append(row)
    out = pd.DataFrame(rows).set_index("GROUP")
    for col in ("PAY_IF_HIGH_TENDENCY", "PAY_IF_HIGH_SKILL"):
        if col in out.columns:
            out[col.replace("PAY", "USD_M")] = out[col] * cap / 1e6
    out.attrs.update(season=profiles.season, n=len(pct), median_pct=float(pct.median()),
                     median_usd_m=float(pct.median() * cap / 1e6), exclude_rookie_deals=exclude_rookie_deals)
    return out.sort_values("PAY_IF_HIGH_TENDENCY", ascending=False)


def plot_salary_by_profile(profiles, salaries=None, figsize=(9, 6.5), **kw):
    """Median salary of the players who do each thing most (blue) and best (orange)."""
    table = salary_by_profile(profiles, salaries, **kw)
    a = table.attrs
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(table))
    t, s = table["USD_M_IF_HIGH_TENDENCY"], table.get("USD_M_IF_HIGH_SKILL", pd.Series(np.nan, index=table.index))
    both = (t.notna() & s.notna()).to_numpy()
    ax.hlines(y[both], t[both], s[both], color=GRID, lw=3, zorder=1)
    ax.scatter(t, y, s=110, color=BLUE, zorder=3, label="Top quarter by tendency (do it most)")
    ax.scatter(s, y, s=110, color=ORANGE, zorder=3, edgecolor="white", linewidth=1.5,
               label="Top quarter by skill (do it best)")
    ax.axvline(a["median_usd_m"], color=MUTED, lw=1, ls=(0, (2, 3)), zorder=0)
    ax.text(a["median_usd_m"], -0.75, f"  pool median ${a['median_usd_m']:.1f}M", fontsize=8, color=MUTED, va="center")
    for i, v in enumerate(t):
        if pd.notna(v):
            ax.annotate(f"${v:.0f}M", (v, i), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(table.index)
    ax.invert_yaxis()
    ax.set_xlim(0, None)
    ax.set_title(f"What each style gets paid — median salary, {a['season']}", loc="left", fontsize=13,
                 fontweight="bold", pad=24)
    who = "veterans only (rookie-scale deals excluded)" if a["exclude_rookie_deals"] else "all players"
    ax.text(0, 1.03, f"{a['n']} players with a profile and a salary, {who}", transform=ax.transAxes,
            fontsize=9, color=MUTED)
    ax.set_xlabel("Median salary, $M", fontsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()


def pay_check(season="2026-27", profile_season="2025-26", model="v2", salaries=None):
    """This season's pay against the market price of LAST season's profile.

    Takes what the model said last season's production was worth (as a share
    of the cap), applies it to this season's cap, and compares with this
    season's salary. It needs no games from the new season, so it works on
    opening night; score_season() takes over once players have games.

        GAP_USD > 0   paid less than last season's profile priced at  -> underpaid
        GAP_USD < 0   paid more                                       -> overpaid

    It can't see anything that changed since April: age, injuries, a new role.
    WAS_ROOKIE_DEAL marks players who were on a rookie-scale contract last
    season; their "market price" was held down by that contract, so filter them
    out (out[~out.WAS_ROOKIE_DEAL]) before reading the overpaid end.
    Players who weren't in last season's model table (rookies, under 20 games)
    aren't here. score_season() is the version that uses this season's stats.
    model="v2_lebron" uses the impact model.
    """
    salaries = load_salaries(save=False) if salaries is None else salaries
    meta = pd.read_parquet(DATA_DIR / f"app_meta_{model}.parquet")
    last = meta[meta["SEASON"] == profile_season].copy()
    last["name_key"] = last["PLAYER_NAME"].map(name_key)
    now = salaries[salaries["SEASON"] == season]
    cap, old_cap = SALARY_CAP[season], SALARY_CAP[profile_season]

    out = now.merge(last[["name_key", "pct_cap", "pred_pct_cap"]], on="name_key", suffixes=("", "_last"))
    out = out.rename(columns={"pct_cap": "PCT_CAP", "pct_cap_last": "PCT_CAP_LAST", "pred_pct_cap": "MARKET_PCT"})
    out["SALARY_M"] = out["Salary"] / 1e6
    out["MARKET_M"] = out["MARKET_PCT"] * cap / 1e6
    out["GAP_M"] = out["MARKET_M"] - out["SALARY_M"]
    out["LAST_SALARY_M"] = out["PCT_CAP_LAST"] * old_cap / 1e6
    out["RAISE_M"] = out["SALARY_M"] - out["LAST_SALARY_M"]
    out["VERDICT"] = np.where(out["GAP_M"] > 0, "underpaid", "overpaid")
    # The model knows a rookie-scale player is cheap by rule, so it priced him low
    # last season. Once his extension starts, that old price says nothing about him.
    out["WAS_ROOKIE_DEAL"] = _on_rookie_deal(out["Player"], profile_season).to_numpy()
    cols = ["Player", "SALARY_M", "MARKET_M", "GAP_M", "VERDICT", "WAS_ROOKIE_DEAL",
            "PCT_CAP", "MARKET_PCT", "LAST_SALARY_M", "RAISE_M"]
    out = out[cols].sort_values("GAP_M", ascending=False).reset_index(drop=True)
    out.attrs.update(season=season, profile_season=profile_season, n=len(out), model=model)
    return out


# =============================================================================
# SCORE A SEASON WITH THE SALARY MODEL
# =============================================================================

TRAIN_TABLE = DATA_DIR / "transposed_df_master_with_bio_v2.parquet"    # the table 01_modeling.ipynb trained v2 on
SPINE = ["MIN_rs", "FGM_rs", "FTM_rs", "PIE_rs", "USG_PCT_rs", "FRONT_CT_TOUCHES_rs", "POST_TOUCHES_rs",
         "CLOSESTDEF_4_6_FGM_rs", "GP_rs", "TOTAL_MIN_rs", "ALLNBA_WT_EVER"]
ALLNBA_WEIGHT = {"1st": 3, "2nd": 2, "3rd": 1}


def _allnba_selections(refresh=False):
    """All-NBA teams from Basketball-Reference, one row per selection: name_key, yr, weight."""
    cache = DATA_DIR / "allnba_raw.parquet"
    if cache.exists() and not refresh:
        raw = pd.read_parquet(cache)
    else:
        raw = fetch_all_nba_raw()
        raw.columns = [str(c) for c in raw.columns]
        raw = raw.astype(object).where(raw.notna(), None).astype("string")
        raw.to_parquet(cache)
    raw = raw[raw["Lg"] == "NBA"]
    long = raw.melt(id_vars=["Season", "Tm"], value_vars=[f"Unnamed: {i}" for i in range(4, 9)],
                    value_name="player").dropna(subset=["player"])
    long["name_key"] = long["player"].str.replace(r"\s+[CFG](-[CFG])?$", "", regex=True).map(norm_name)
    long = long[long["Season"].isin(SEASONS)]
    long["yr"] = long["Season"].str[:4].astype(int)
    long["weight"] = long["Tm"].map(ALLNBA_WEIGHT)
    return long[["name_key", "yr", "weight"]]


def _allnba_features(model, selections, decay=0.75):
    """ALLNBA_PRIOR3 / PRIOR_EVER (counts) and ALLNBA_WT3 / WT_EVER (1st=3, 2nd=2, 3rd=1).

    Only seasons BEFORE the row's season count; the career versions fade by
    `decay` per year. Same definitions as the notebook's prior_counts / prior_weighted.
    """
    keys = model["PLAYER_NAME"].map(norm_name).to_numpy()
    years = model["SEASON"].str[:4].astype(int).to_numpy()
    by_player = {k: g for k, g in selections.groupby("name_key")}
    out = np.zeros((len(model), 4))
    for i, (key, year) in enumerate(zip(keys, years)):
        picks = by_player.get(key)
        if picks is None:
            continue
        past = picks[picks["yr"] < year]
        recent = past[past["yr"] >= year - 3]
        ranked = past.dropna(subset=["weight"])
        out[i] = (len(recent), (decay ** (year - past["yr"])).sum(), recent["weight"].sum(),
                  (ranked["weight"] * decay ** (year - ranked["yr"])).sum())
    return pd.DataFrame(out, columns=["ALLNBA_PRIOR3", "ALLNBA_PRIOR_EVER", "ALLNBA_WT3", "ALLNBA_WT_EVER"],
                        index=model.index)


def _draft_table(player_ids, season):
    """Draft year and slot per player; fetches this season's registry if players are missing from it."""
    bio = get_draft_table(SEASONS)
    missing = set(map(str, player_ids)) - set(bio["PLAYER_ID"])
    if missing:
        try:
            fresh = get_player_registry(season, retries=2)
            path = DATA_DIR / "bio_registry.parquet"
            old = pd.read_parquet(path)
            pd.concat([old[old["SEASON"] != season], fresh], ignore_index=True).to_parquet(path)
            bio = get_draft_table(SEASONS)
            print(f"bio registry: added {season} ({len(fresh)} players)")
        except Exception as e:
            print(f"⚠️ {len(missing)} players have no draft info and the registry fetch failed ({type(e).__name__}); "
                  "they are marked NO_BIO and left out of the verdicts")
    return bio


def build_model_features(df_master, salaries=None, season=None, prorate=True, refresh_allnba=False):
    """The modeling table rebuilt from the master: one row per player-season, every feature.

    Follows 01_modeling.ipynb step for step (regular season + playoffs side by
    side, salaries, draft and experience, All-NBA credentials, availability,
    one- and two-season lags), so its rows can go straight into the saved model.

    season + prorate: that season is still being played, so its games-played
    features are scaled to an 82-game pace using each team's games so far.
    Without it a healthy player with 12 of 12 games looks like one who missed
    70. Per-game stats are untouched.
    """
    salaries = load_salaries(save=False) if salaries is None else salaries
    model = build_model_table(df_master)
    model = model.loc[:, ~model.columns.duplicated()]
    model = attach_salaries(model, salaries)

    model["PLAYER_ID"] = model["PLAYER_ID"].astype(str)
    latest = season or model["SEASON"].max()
    bio = _draft_table(model.loc[model["SEASON"] == latest, "PLAYER_ID"], latest)
    model = model.drop(columns=["DRAFT_POSITION", "DRAFT_YR"], errors="ignore").merge(bio, on="PLAYER_ID", how="left")
    model["NO_BIO"] = ~model["PLAYER_ID"].isin(set(bio["PLAYER_ID"]))
    model["IS_UNDRAFTED"] = model["DRAFT_POSITION"].isna().astype(int)
    model["EXPERIENCE"] = (model["SEASON"].str[:4].astype(int) - model["DRAFT_YR"]).clip(lower=0)
    model["MAX_PCT_ELIGIBLE"] = model["EXPERIENCE"].map(
        lambda e: np.nan if pd.isna(e) else (0.25 if e <= 6 else 0.30 if e <= 9 else 0.35))
    model["pct_cap"] = model["Salary"] / model["SEASON"].map(SALARY_CAP)

    model = pd.concat([model, _allnba_features(model, _allnba_selections(refresh_allnba))], axis=1)

    model["GP_rs"] = model["GP_rs"].fillna(0)
    model["GP_PLAYED"] = model["GP_rs"]
    if season and prorate:
        now = model["SEASON"] == season
        team_games = model[now].groupby("TEAM_ABBREVIATION")["GP_rs"].transform("max").clip(lower=1)
        pace = (82.0 / team_games).clip(lower=1.0)
        for col in ("GP_rs", "W_rs", "L_rs"):
            if col in model.columns:
                model.loc[now, col] = (model.loc[now, col] * pace).clip(upper=82.0)
    model["AVAILABILITY"] = model["GP_rs"] / 82.0
    model["TOTAL_MIN_rs"] = model["MIN_rs"] * model["GP_rs"]

    spine = [c for c in SPINE if c in model.columns]
    model["_yr"] = model["SEASON"].str[:4].astype(int)
    model = model.sort_values(["PLAYER_ID", "_yr"])
    for k in (1, 2):
        lagged = model.groupby("PLAYER_ID")[spine].shift(k)
        gap_ok = (model["_yr"] - model.groupby("PLAYER_ID")["_yr"].shift(k)) == k     # consecutive seasons only
        lagged = lagged.where(gap_ok)
        lagged.columns = [f"{c}_lag{k}" for c in spine]
        model = pd.concat([model, lagged], axis=1)
    return model.drop(columns="_yr").reset_index(drop=True)


def salary_model(retrain=False):
    """The v2 box-score salary model and its feature list.

    The notebook saved v2's predictions but not the model itself, so this
    retrains it the way the notebook does — same table, same features, same
    settings, trained through 2021-22 — and checks the result against the
    saved predictions before keeping it (data/salary_model_v2.txt).
    """
    import lightgbm as lgb
    model_path, feats_path = DATA_DIR / "salary_model_v2.txt", DATA_DIR / "feats_v2.json"
    if model_path.exists() and feats_path.exists() and not retrain:
        return lgb.Booster(model_file=str(model_path)), json.load(open(feats_path))

    table = pd.read_parquet(TRAIN_TABLE)
    keep, _ = prefilter_features(table)
    feats = keep + [c for c in ("AVAILABILITY", "TOTAL_MIN_rs") if c not in keep]
    feats.remove("DRAFT_YR")
    train, valid = table[table["SEASON"] <= "2021-22"], table[table["SEASON"] == "2022-23"]
    fit = lgb.LGBMRegressor(
        n_estimators=3000, learning_rate=0.02, num_leaves=31, min_child_samples=30, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)
    fit.fit(train[feats], train["pct_cap"], eval_set=[(valid[feats], valid["pct_cap"])],
            callbacks=[lgb.early_stopping(150, verbose=False)])

    saved = pd.read_parquet(DATA_DIR / "app_meta_v2.parquet").set_index(["PLAYER_NAME", "SEASON"])["pred_pct_cap"]
    mine = pd.Series(fit.predict(table[feats]), index=pd.MultiIndex.from_frame(table[["PLAYER_NAME", "SEASON"]]))
    worst = float((mine - saved.reindex(mine.index)).abs().max())
    if not worst < 1e-6:
        raise RuntimeError(f"retrained model is off the saved v2 predictions by up to {worst:.4f} of the cap; not saved")
    fit.booster_.save_model(str(model_path))
    json.dump(feats, open(feats_path, "w"))
    print(f"salary model v2 rebuilt: {len(feats)} features, matches the saved predictions")
    return fit.booster_, feats


def score_season(df_master, season="2026-27", min_gp=10, min_min=10.0, salaries=None, prorate=True):
    """This season's pay against what the model says this season's play is worth.

        pay = score_season(df_master)                 # 2026-27, players with 10+ games
        pay[pay.VERDICT_OK].head(20)                  # most underpaid
        pay[pay.VERDICT_OK].tail(20)                  # most overpaid

    The model was trained on finished seasons (20+ games). Scoring a season in
    progress works because it leans on per-game stats, experience and prior
    seasons rather than totals — but a 10-game sample is noisy, so read big
    gaps as leads to re-check at 20 games, not as findings.

        GAP_M > 0   paid less than the model's price  -> underpaid
        GAP_M < 0   paid more                         -> overpaid

    VERDICT_OK is False for players without draft info (NO_BIO): the model
    can't tell a rookie-scale deal from a veteran one without it.
    ROOKIE_SCALE marks first-round picks in their first four seasons. The model
    knows about those contracts and prices them as such, so they belong in the
    table, but "underpaid" means little when the pay is set by draft slot.
    """
    salaries = load_salaries(save=False) if salaries is None else salaries
    booster, feats = salary_model()
    in_progress = season == max(SALARY_CAP) and prorate
    model = build_model_features(df_master, salaries, season=season if in_progress else None)

    rows = model[(model["SEASON"] == season) & model["Salary"].notna()
                 & (model["GP_PLAYED"] >= min_gp) & (model["MIN_rs"] >= min_min)].copy()
    if rows.empty:
        raise ValueError(f"no {season} players with a salary, {min_gp}+ games and {min_min}+ minutes")
    # "index" is a leftover row counter that got into the notebook's feature list; it carries no weight
    absent = [f for f in feats if f not in rows.columns and f != "index"]
    if absent:
        print(f"⚠️ {len(absent)} model features are not in the master and were left empty, e.g. {absent[:5]}")
    X = rows.reindex(columns=feats).apply(pd.to_numeric, errors="coerce")
    cap = SALARY_CAP[season]
    rows["MARKET_PCT"] = booster.predict(X)

    out = pd.DataFrame({
        "Player": rows["PLAYER_NAME"], "TEAM": rows["TEAM_ABBREVIATION"],
        "GP": rows["GP_PLAYED"].astype(int), "MIN": rows["MIN_rs"],
        "SALARY_M": rows["Salary"] / 1e6, "MARKET_M": rows["MARKET_PCT"] * cap / 1e6,
        "PCT_CAP": rows["pct_cap"], "MARKET_PCT": rows["MARKET_PCT"],
        "EXPERIENCE": rows["EXPERIENCE"], "NO_BIO": rows["NO_BIO"],
    })
    out["GAP_M"] = out["MARKET_M"] - out["SALARY_M"]
    out["VERDICT"] = np.where(out["GAP_M"] > 0, "underpaid", "overpaid")
    out["ROOKIE_SCALE"] = _on_rookie_deal(out["Player"], season).to_numpy()
    out["VERDICT_OK"] = ~out["NO_BIO"]
    cols = ["Player", "TEAM", "GP", "MIN", "SALARY_M", "MARKET_M", "GAP_M", "VERDICT", "VERDICT_OK",
            "ROOKIE_SCALE", "EXPERIENCE", "PCT_CAP", "MARKET_PCT", "NO_BIO"]
    out = out[cols].sort_values("GAP_M", ascending=False).reset_index(drop=True)
    out.attrs.update(season=season, min_gp=min_gp, min_min=min_min, n=len(out), prorated=in_progress)
    return out
