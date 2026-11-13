"""
REPLACEMENT for the SHAP-export cell in 01_modeling.ipynb.

OLD behavior: stored SHAP for the GLOBAL top-25 features (same 25 for every player).
NEW behavior: stores each player's OWN top-N contributors (by |SHAP| for THAT player),
so the app shows what actually drives each player — including idiosyncratic features
(e.g. Wemby's clutch playoff possessions) that aren't league-wide top features.

Run this in the notebook AFTER `sv` (SHAP values), `feats`, `mdf`, `SALARY_CAP`,
and `expl` are defined. Then re-upload data/shap_v1.parquet to GitHub.
"""
import numpy as np, pandas as pd

PER_PLAYER_N = 18   # how many top contributors to store per player-season (app shows top 15)

cap_by_season = mdf['SEASON'].map(SALARY_CAP).values
feat_index = {f: j for j, f in enumerate(feats)}
feats_arr = np.array(feats)

rows = []
for r in range(len(mdf)):
    cap = cap_by_season[r]
    row_sv = sv[r]                                   # SHAP values for this player-season, all feats
    # this player's OWN top-N features by |SHAP|
    top_idx = np.argsort(np.abs(row_sv))[::-1][:PER_PLAYER_N]
    name = mdf['PLAYER_NAME'].iloc[r]
    season = mdf['SEASON'].iloc[r]
    for j in top_idx:
        f = feats_arr[j]
        rows.append((name, season, f, row_sv[j], row_sv[j] * cap, mdf[f].iloc[r]))

shap_long = pd.DataFrame(rows, columns=['PLAYER_NAME','SEASON','feature',
                                        'shap','shap_usd','feat_value'])
shap_long.to_parquet("data/shap_v1.parquet")

# meta unchanged (baseline + prediction anchors)
base = float(expl.expected_value)
meta = mdf[['PLAYER_NAME','SEASON','pct_cap','pred_pct_cap','resid_usd']].copy()
meta['baseline_pct'] = base
meta.to_parquet("data/app_meta_v1.parquet")

n_unique_feats = shap_long['feature'].nunique()
print(f"stored top {PER_PLAYER_N} features PER player-season")
print(f"  {len(mdf)} player-seasons × {PER_PLAYER_N} = {len(shap_long):,} rows")
print(f"  {n_unique_feats} distinct features now appear across the app "
      f"(vs 25 before) — players keep their idiosyncratic drivers")
print(f"  file size ~{shap_long.memory_usage(deep=True).sum()/1e6:.1f} MB")
