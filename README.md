# nba-player-analytics

An end-to-end pipeline for NBA player data — assembling a wide, multi-source feature set from the NBA's
official stats API, then modeling player salary to surface over- and under-paid players relative to the
market.

**Live app:** https://siddharthravindran-nba-player-analytics-app-1rfmm9.streamlit.app/

## Overview

This project pulls player statistics from dozens of NBA Stats endpoints, reconciles them into a single
player-season feature table, and uses that table to model player compensation. The goal isn't the salary
prediction itself — it's the **residual**: the gap between what the model says a player's profile should
earn and what they actually earn, which is a proxy for market mispricing.

Importantly, this is a **market-pricing** model, not a player-value model. It answers *"is this player paid
what the market pays players with his profile?"* — not *"is he worth it?"* Most public NBA salary work is
normative (compute what a player *should* earn from an impact metric like DARKO, then call the gap
over/underpaid). This one is descriptive: it learns the league's actual pricing function from a decade of
real contracts, with **no impact metric in the loop** — built entirely from raw production I sourced myself.
That keeps it fully reproducible end to end, and lets it answer a question impact models can't: *net of
minutes and scoring, which skills does the market pay a premium for?*

## What it does

- **Ingests player stats across many NBA Stats endpoints:** traditional and advanced box score, tracking
  (drives, touches, speed/distance, rebounding), Synergy play types (pick-and-roll, isolation, post-up,
  spot-up, and more), shot dashboards (by defender distance, dribbles, shot clock, touch time), shot
  locations, defensive matchup data, hustle, and clutch.
- **Reconciles them into one wide table** indexed by season, season type, and player — handling the messy
  parts: traded players (TOT rows), unicode/whitespace in names, and team-column naming differences across
  endpoints.
- **Caches every API pull to parquet**, with selective refresh so only the in-progress season is re-fetched
  while completed seasons stay frozen.
- **Updates incrementally** — new stat families and new seasons patch into the master without re-fetching or
  clobbering existing columns.
- **Sources accurate historical salaries** via a custom Basketball-Reference scrape (the NBA Stats API has
  no salary data), and All-NBA selections parsed from BBR's award tables.
- **Models salary as a share of the cap**, then surfaces and explains the residuals.
- **Visualizes** distributions, player comparisons, shot mix, and stat changes, plus a deployed Streamlit
  app for the model results.

## Data sources

- **nba_api** — official NBA Stats endpoints. Player tracking is available from ~2013-14 and Synergy play
  types from ~2015-16, which sets the usable history floor.
- **Basketball-Reference** — historical player salaries (the target) and All-NBA selections, scraped
  (including comment-hidden tables), used to compute salary as a share of that season's cap.

## Modeling approach / design notes

A few deliberate choices, since the why matters more than the code:

- **Target = % of the salary cap, not raw dollars.** The cap rises every year, so $20M in 2016 and $20M in
  2025 aren't comparable. Percent-of-cap makes salaries comparable across a decade.
- **Residuals are the product.** Predicted minus actual identifies mispriced players; the raw prediction is
  just the means to that end.
- **Time-ordered validation, not random k-fold.** Player-seasons leak across time — a player's adjacent
  seasons are near-duplicates, and random folds let the future inform the past — so the model trains on
  earlier seasons (≤ 2021-22) and tests on strictly later ones (2023-24 onward).
- **Gradient-boosted trees, not deep learning.** On small/medium tabular data, GBDTs (LightGBM) are the
  stronger, faster choice, and they handle the missing values that come from sparse tracking columns
  natively.
- **Temporal lag features.** Each player's prior-1 and prior-2 season production, with a consecutive-season
  gap-mask so non-adjacent seasons aren't falsely paired. This was the single biggest accuracy gain
  (R² 0.79 → 0.85); the model learns the decay rate itself (prior year ≈ 3× the weight of two-years-ago).
  It lets the model distinguish a rising player from a fading one, and an injured star from a declined one.
- **All-NBA credential, weighted.** Career All-NBA selections (1st team = 3, 2nd = 2, 3rd = 1), prior
  seasons only — the signal that separates max-tier stars from merely good players, and which mirrors CBA
  supermax logic.
- **Salary-regime awareness.** NBA salaries come from different processes — rule-set rookie scale, veteran
  minimums, and open-market deals — and the feature set (experience, max-tier eligibility, availability)
  gives the model the context to tell them apart.

## Results

- **R² = 0.845** on the out-of-time test (train ≤ 2021-22, test 2023-24+) — forecasting unseen seasons, not
  fitting in-sample.
- **MAE ≈ $3.73M** on salaries that range to ~$60M.
- The top features are the *obvious* ones — experience, minutes, scoring, draft pedigree, All-NBA credential.
  That's the evidence the model learned the real market: a salary model whose top drivers *weren't* the
  obvious ones would be broken. The novel findings live in the tail (interior defense, shot creation,
  trajectory) and in the per-player residuals.

**Giannis Antetokounmpo, 2025-26** is the model in one chart. Every factor says max-tier — experience, prior
healthy minutes, All-NBA résumé, scoring — all pushing his price up. A single bar pulls it down: this
season's minutes, because he played 36 games. The model correctly reads "elite player, injury-discounted
season." The injury, isolated in one bar.

<img width="2288" height="1592" alt="8EE7920E-113D-4DF1-8993-D3E01174000F" src="https://github.com/user-attachments/assets/474e58ab-4a9b-46d2-b3ce-ea5be77d5ad1" />

## Limitations (by design, not bugs)

A market model built on individual box-score production has structural blind spots, and naming them is part
of the analysis:

- **Role / relational value is invisible.** Glue defenders and spacers get paid for fit and scheme value
  that doesn't appear in their own stat line, so they can read as overpaid.
- **Injury seasons read as overpaid.** A contract signed on prior healthy years meets a current injured
  season; the lag features soften this but a contemporaneous model can't fully resolve it.
- **Breakout seasons read as underpaid** — the exact flip side of the injury fix. The lag features anchor a
  player to recent history, which a breakout by definition contradicts.
- **The very top is set by credentials and scarcity**, not just production — which is why the All-NBA
  credential feature exists, and why it narrows but doesn't fully close the gap at the ceiling.

## Measuring the blind spot: an impact-metric variant

The limitations above aren't hypothetical — the biggest one (value the box score can't see) is measurable.
To measure it, I built a **second model identical to the first, plus one addition: LEBRON**, a plus-minus
impact metric (BBall Index), added as leak-safe prior-season features. Where the two models *disagree* on a
player's fair price is the invisible value, quantified — the box score's blind spot, in dollars.

This is deliberately a **study, not a replacement**: the deployed model stays box-only and fully reproducible,
consistent with the market-pricing thesis. The impact model is a lens applied to it.

**Method**
- Two-model A/B — box-only vs. box + LEBRON, same target and population; the prediction gap is the signal.
- Cohort SHAP analysis to decompose *which* impact signal drives the disagreement.
- Cross-season validation across four seasons (2022-23 → 2025-26).

**Findings**
- **The disagreement is basketball-coherent.** LEBRON raises players whose winning impact outstrips their
  box score — invisible-value defenders (Alex Caruso, Derrick White, Ausar Thompson, Lu Dort, Jaden
  McDaniels), spacing and rim-protecting bigs (Myles Turner, Al Horford, Kristaps Porzingis, Lauri Markkanen,
  Brook Lopez), and high-impact stars whose effect tops even their box score (Giannis, Curry) — and lowers
  empty-volume scorers (Jordan Poole, Devin Vassell, Jalen Green, Kyle Kuzma). These lists fell out of the
  model-to-model disagreement; they weren't hand-picked.
- **WAR is the signal carrier — not the offense/defense split.** Decomposing the LEBRON features via cohort
  SHAP, the offense/defense components contribute almost nothing; it's *WAR* (wins above replacement,
  combining per-possession impact with availability) doing the work. The market underprices total winning
  contribution, not "defense" as a category.
- **The mispricing is individual, not archetypal.** Grouping by LEBRON role or offense/defense tilt washed
  out entirely — two players of the same type sit at opposite ends of the disagreement. The inefficiency
  persists *because* it's idiosyncratic; if it reduced to a type, teams would have arbitraged it away.
- **A correction, not a revolution.** Adding a strong impact metric barely moved aggregate accuracy
  (R2 0.840 vs. 0.845 box-only). Impact refines market pricing at the margins — the market prices production
  first. That modest effect is itself the finding.

**Alex Caruso, 2025-26:** the box-only model prices him at 4.8% of the cap (~$7.4M); adding impact raises
that to 6.8% (~$10.5M). His defensive LEBRON is elite (+1.9) while his offensive LEBRON is slightly negative
— the box score can't see why he's worth paying, but the impact model can. He actually earns 11.7%, so both
models flag him underpaid; impact closes roughly a third of the gap the box score left open.

## Tech stack

Python · pandas · NumPy · nba_api · BeautifulSoup · LightGBM · SHAP · Streamlit · Plotly ·
matplotlib · seaborn · hvPlot / HoloViews · parquet (PyArrow)

## Repo structure

```
nba-player-analytics/
├── nba_viz_utils.py     # data pipeline: fetch, cache, consolidate, refresh, visualize
├── app.py               # Streamlit app (reads precomputed model artifacts)
├── notebooks/           # exploration, feature building, modeling
├── data/                # precomputed artifacts the app reads
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/siddharthravindran/nba-player-analytics.git
cd nba-player-analytics
pip install -r requirements.txt
```

The pipeline writes its parquet cache to a data directory. Run the Streamlit app with:

```bash
python3 -m streamlit run app.py
```

## Usage

```python
from nba_viz_utils import fetch_nba_stats, scatter_nba_stats

# Pull advanced stats for a season
df = fetch_nba_stats("Advanced", ["2024-25"], season_type="Regular Season")

# Plot two stats against each other
scatter_nba_stats(df, x_stat="USG_PCT", y_stat="TS_PCT", season="2024-25", min_gp=40)
```

## Status

- [x] Multi-source feature pipeline (~1,000 engineered features per player-season)
- [x] Caching, selective refresh, and incremental updates
- [x] Backfill to 2015-16
- [x] Salary data ingestion (BBR scrape) + player-identity join
- [x] Salary model + residual analysis (R² 0.845, out-of-time)
- [x] Interactive app (per-player SHAP waterfall + league ladder)
- [x] Results write-up / [article](https://medium.com/@siddharthravindran/i-built-a-model-that-prices-nba-players-the-way-the-market-actually-does-and-the-interesting-part-1b71dbc77e6f)
- [x] LEBRON impact-comparison model — cohort SHAP analysis of where plus-minus impact reveals market mispricing
- [ ] LEBRON impact tab in app (disagreement leaderboard + per-player impact view)
- [ ] Follow-up write-up on the LEBRON findings (drafted)
- [ ] v2: forward model on already-signed 2026-27 contracts

## Roadmap (v2)

- **Forward model:** validate against real future contracts (already scraped, 2026-31) — "is the salary a
  player is *already locked into* a bargain, given current production?"
- **Rookie-scale flag** to separate "underpaid by rule" from "underpaid by market."

## Notes

- The NBA Stats API rate-limits aggressively; the pipeline retries failed calls and caches successes, so
  re-runs only fetch the gaps.
- Tracking and Synergy availability limit usable history to ~2015-16 onward.

## License

MIT
