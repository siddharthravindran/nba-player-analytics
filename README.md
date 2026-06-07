# nba-player-analytics

An end-to-end pipeline for NBA player data — assembling a wide, multi-source feature set from the NBA's official stats API and modeling player salary to surface over- and under-paid players.

## Overview

This project pulls player statistics from dozens of NBA Stats endpoints, reconciles them into a single player-season feature table, and uses that table to model player compensation. The goal isn't the salary prediction itself — it's the **residual**: the gap between what a model says a player should earn and what they actually earn, which is a proxy for market mispricing (team surplus value, underpaid players, etc.).

## What it does

- **Ingests** player stats across many NBA Stats endpoints: traditional and advanced box score, tracking (drives, touches, speed/distance, rebounding), Synergy play types (pick-and-roll, isolation, post-up, spot-up, and more), shot dashboards (by defender distance, dribbles, shot clock, touch time), shot locations, defensive matchup data, hustle, and clutch.
- **Reconciles** them into one wide table indexed by season, season type, and player — handling the messy parts: traded players (TOT rows), unicode/whitespace in names, and team-column naming differences across endpoints.
- **Caches** every API pull to parquet, with selective refresh so only the in-progress season is re-fetched while completed seasons stay frozen.
- **Updates incrementally** — new stat families and new seasons patch into the master without re-fetching or clobbering existing columns.
- **Visualizes** distributions, player comparisons, shot mix, and stat changes (e.g., regular season vs playoffs).

## Data sources

- **[nba_api](https://github.com/swar/nba_api)** — official NBA Stats endpoints. Player tracking is available from ~2013-14 and Synergy play types from ~2015-16, which sets the usable history floor.
- **Salary data** — player salary by season (the target), used to compute salary as a share of that season's cap. (to be retrieved)

## Approach / design notes (planned)

A few deliberate choices, since the *why* matters more than the code:

- **Target = % of the salary cap, not raw dollars.** The cap rises every year, so $20M in 2016 and $20M in 2025 aren't comparable. Percent-of-cap makes salaries comparable across a decade.
- **Residuals are the product.** Predicted minus actual identifies mispriced players; the raw prediction is just the means to that end.
- **Time-ordered validation, not random k-fold.** Player-seasons leak across time — a player's adjacent seasons are near-duplicates, and random folds let the future inform the past — so the model trains on earlier seasons and tests on strictly later ones.
- **Gradient-boosted trees, not deep learning.** On small/medium tabular data, GBDTs (XGBoost/LightGBM) are the stronger, faster choice.
- **Salary-regime awareness.** NBA salaries come from different processes — rule-set rookie scale, veteran minimums, and open-market deals — which are handled separately rather than lumped together.

## Tech stack

Python · pandas · NumPy · nba_api · matplotlib · seaborn · hvPlot / HoloViews · parquet (PyArrow) · (modeling: XGBoost / LightGBM)

## Repo structure

```
nba-player-analytics/
├── nba_viz_utils.py     # data pipeline: fetch, cache, consolidate, refresh, visualize
├── notebooks/           # exploration, feature building, modeling
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/siddharthravindran/nba-player-analytics.git
cd nba-player-analytics
pip install -r requirements.txt
```

The pipeline writes its parquet cache to a data directory (defaults to `./nba_data`). Point it elsewhere with an environment variable:

```bash
export NBA_DATA_DIR=/path/to/your/data
```

## Usage

```python
from nba_viz_utils import fetch_nba_stats, scatter_nba_stats

# Pull advanced stats for a season
df = fetch_nba_stats("Advanced", ["2024-25"], season_type="Regular Season")

# Plot two stats against each other
scatter_nba_stats(
    df, x_stat="USG_PCT", y_stat="TS_PCT",
    season="2024-25", min_gp=40,
)
```

## Status

- [x] Multi-source feature pipeline (hundreds of engineered features per player-season)
- [x] Caching, selective refresh, and incremental updates
- [ ] Backfill to 2016 *(in progress)*
- [ ] Salary data ingestion + player-identity join
- [ ] Salary model + residual analysis
- [ ] Results write-up

## Notes

- The NBA Stats API rate-limits aggressively; the pipeline retries failed calls and caches successes, so re-runs only fetch the gaps.
- Tracking and Synergy availability limit usable history to ~2015-16 onward.

## License

MIT
