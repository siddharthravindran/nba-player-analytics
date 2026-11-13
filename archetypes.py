"""Offensive role archetypes from Synergy play-type shares.

Clusters players by *what* they do on offense, not how well they do it. The ten
Synergy play types carry a share-of-possessions figure each (POSS_PCT), so a
player is a composition vector — 32% pick-and-roll handler, 22% spot-up, and so
on — and the clustering groups players whose offense is distributed alike.

Efficiency (PPP) is deliberately left out of the features. Including it would mix
"what is this player's role" with "how good is he at it" and the clusters would
come out as quality tiers. Kept separate, role is the grouping and efficiency
becomes something you can measure *across* archetypes.

Run it: python3 archetypes.py [SEASON]
"""
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from nba_viz_utils import DATA_DIR, FIG_DIR

PLAY_TYPES = ['CUT', 'DHO', 'ISO', 'OFFSCREEN', 'PNRBH',
              'POST', 'PUTBACK', 'RM', 'SPOT', 'TRANS']

PRETTY = {
    'CUT': 'Cut', 'DHO': 'Handoff', 'ISO': 'Isolation', 'OFFSCREEN': 'Off screen',
    'PNRBH': 'P&R handler', 'POST': 'Post up', 'PUTBACK': 'Putback',
    'RM': 'P&R roll man', 'SPOT': 'Spot up', 'TRANS': 'Transition',
}

# Synergy suppresses a player below its per-play-type possession minimum, so a
# thin season reads as a handful of concentrated shares that aren't really a
# profile. These cuts keep players whose composition covers most of their offense.
MIN_MINUTES = 10.0      # rotation players only
MIN_COVERAGE = 0.60     # the 10 shares must account for >=60% of his offense
MIN_POSS_PG = 4.0       # tracked possessions per game


def load_play_type_matrix(season, season_type="Regular Season"):
    """Return (frame, composition) for players with a usable play-type profile."""
    master = pd.read_parquet(DATA_DIR / "df_master_current.parquet")
    master['SEASON'] = master['SEASON'].astype(str)

    df = master[
        (master['SEASON'] == season)
        & (master['SEASON_TYPE'].astype(str).str.strip() == season_type)
    ].copy()

    if df.empty:
        raise ValueError(f"no rows for {season} {season_type}")

    pct_cols = [f'{p}_POSS_PCT' for p in PLAY_TYPES]
    poss_cols = [f'{p}_POSS' for p in PLAY_TYPES]

    df['_coverage'] = df[pct_cols].fillna(0).sum(axis=1)
    df['_poss_pg'] = df[poss_cols].fillna(0).sum(axis=1)

    keep = df[
        (df['MIN'] >= MIN_MINUTES)
        & (df['_coverage'] >= MIN_COVERAGE)
        & (df['_poss_pg'] >= MIN_POSS_PG)
    ].copy()

    # A missing play type means "negligible share", so 0 is the right fill. Shares
    # cover ~90% of a player's offense (the rest is unclassified), so renormalize
    # to make the vectors comparable rather than penalizing whoever has more
    # unclassified possessions.
    comp = keep[pct_cols].fillna(0)
    comp.columns = PLAY_TYPES
    comp = comp.div(comp.sum(axis=1), axis=0)

    print(f"📊 {season} {season_type}: {len(df)} players -> {len(keep)} with a usable profile")
    print(f"   (MIN>={MIN_MINUTES}, coverage>={MIN_COVERAGE}, poss/g>={MIN_POSS_PG})")

    return keep, comp


def choose_k(X, k_range=range(3, 11), random_state=0):
    """Pick k by silhouette, printing the whole curve so the choice is visible."""
    print("\n🔍 k selection:")
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=25, random_state=random_state).fit_predict(X)
        scores[k] = silhouette_score(X, labels)
        sizes = np.bincount(labels)
        print(f"   k={k:2d}  silhouette={scores[k]:.3f}  smallest={sizes.min():3d}")
    return scores


def stability(X, k, n_seeds=20, random_state=0):
    """Mean ARI of re-runs against a reference fit.

    There's no ground truth for a player's role, so the honest question isn't
    "are these clusters correct" but "are they a stable feature of the data or an
    artifact of one random initialization". ARI near 1 means the structure
    survives re-seeding.
    """
    reference = KMeans(n_clusters=k, n_init=25, random_state=random_state).fit_predict(X)
    aris = [
        adjusted_rand_score(
            reference,
            KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X),
        )
        for seed in range(1, n_seeds + 1)
    ]
    return float(np.mean(aris)), float(np.min(aris))


def name_archetype(mean_share):
    """Label a cluster from its composition.

    Keyed off the shares rather than the cluster id, because KMeans ids are
    arbitrary and reshuffle between runs — a hardcoded id->name map silently
    mislabels everything the first time a seed changes.
    """
    s = mean_share
    big_man = s['CUT'] + s['PUTBACK'] + s['RM']
    on_ball = s['PNRBH'] + s['ISO'] + s['DHO']

    if big_man >= 0.45:
        return "Rim runner"
    if s['PNRBH'] >= 0.25 or on_ball >= 0.40:
        return "On-ball creator"
    if s.max() < 0.30 and s['POST'] >= 0.08:
        return "Frontcourt hub"
    if s['SPOT'] >= 0.30 and big_man > on_ball:
        return "Spacing big"
    if s['SPOT'] >= 0.30:
        return "Off-ball wing"
    return "Secondary creator"


def fit_archetypes(season="2025-26", season_type="Regular Season", k=None):
    keep, comp = load_play_type_matrix(season, season_type)

    # Hellinger transform: these are proportions, and sqrt before Euclidean
    # distance is the standard handling for compositional data. It also tolerates
    # the structural zeros that a log-ratio transform would choke on.
    X = np.sqrt(comp.to_numpy())

    scores = choose_k(X)
    if k is None:
        k = max(scores, key=scores.get)
        print(f"\n✅ k={k} (silhouette {scores[k]:.3f})")

    km = KMeans(n_clusters=k, n_init=50, random_state=0).fit(X)
    mean_ari, min_ari = stability(X, k)
    print(f"🔁 stability over 20 re-seeds: mean ARI {mean_ari:.3f}, worst {min_ari:.3f}")

    keep = keep.assign(cluster=km.labels_)

    # Distance to the nearest two centroids -> how archetypal a player is. A
    # player sitting between two centroids is a genuine hybrid, not noise, and
    # that's worth surfacing rather than hiding behind a hard label.
    d = km.transform(X)
    nearest = np.sort(d, axis=1)[:, :2]
    with np.errstate(divide='ignore', invalid='ignore'):
        keep['archetype_fit'] = 1 - (nearest[:, 0] / nearest[:, 1])

    names, profiles = {}, {}
    for c in range(k):
        mean_share = comp.loc[keep.index[keep.cluster == c]].mean()
        profiles[c] = mean_share
        names[c] = name_archetype(mean_share)

    # two clusters can land on the same label; keep them distinguishable
    seen = {}
    for c, nm in list(names.items()):
        seen[nm] = seen.get(nm, 0) + 1
        if seen[nm] > 1:
            names[c] = f"{nm} {seen[nm]}"

    keep['archetype'] = keep['cluster'].map(names)

    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X)
    keep['pc1'], keep['pc2'] = coords[:, 0], coords[:, 1]

    profile_df = pd.DataFrame(profiles).T
    profile_df.index = [names[c] for c in profile_df.index]

    return keep, comp, profile_df, pca, scores


def report(keep, profile_df):
    print("\n🏀 archetypes:")
    for name, row in profile_df.iterrows():
        members = keep[keep['archetype'] == name]
        top = ", ".join(
            f"{PRETTY[p]} {row[p]*100:.0f}%" for p in row.sort_values(ascending=False).head(4).index
        )
        # Rank exemplars by volume, not by archetype_fit. Fit rewards a
        # concentrated composition, which picks out low-usage specialists whose
        # few tracked possessions happen to be all one type — accurate, but the
        # names mean nothing to a reader trying to check the cluster by eye.
        exemplars = members.nlargest(5, '_poss_pg')['PLAYER_NAME'].tolist()
        print(f"\n   {name}  (n={len(members)})")
        print(f"     {top}")
        print(f"     highest volume: {', '.join(exemplars)}")

    hybrids = keep.nsmallest(6, 'archetype_fit')[['PLAYER_NAME', 'archetype', 'archetype_fit']]
    print("\n🔀 least archetypal (between two roles):")
    for _, r in hybrids.iterrows():
        print(f"     {r['PLAYER_NAME']:26s} {r['archetype']:20s} fit={r['archetype_fit']:.3f}")


def _axis_label(pca, axis):
    """Describe a principal component by the play types that load on it.

    "PC1" tells a reader nothing. Reading the loadings off the fitted PCA keeps
    the label honest if the data shifts — nothing here is hardcoded to the shape
    of one particular season.
    """
    load = pd.Series(pca.components_[axis], index=PLAY_TYPES)
    lo = ", ".join(PRETTY[p] for p in load.nsmallest(2).index)
    hi = ", ".join(PRETTY[p] for p in load.nlargest(2).index)
    return f"PC{axis + 1}:   ←  {lo}      |      {hi}  →"


def plot(keep, profile_df, season, out_path, pca=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(17, 7.5), gridspec_kw={'width_ratios': [1.25, 1]}
    )

    order = profile_df.index.tolist()
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(order)))

    for color, name in zip(colors, order):
        sub = keep[keep['archetype'] == name]
        ax1.scatter(sub['pc1'], sub['pc2'], s=26 + 90 * sub['archetype_fit'],
                    color=color, alpha=0.72, edgecolor='white', linewidth=0.5,
                    label=f"{name} ({len(sub)})")

    # Label the highest-volume few per cluster — recognizable names are what let
    # someone check the clustering against what they've actually watched. Skip any
    # that would land on top of a label already placed.
    placed = []
    span = max(np.ptp(keep['pc1']), np.ptp(keep['pc2']))
    for name in order:
        for _, r in keep[keep['archetype'] == name].nlargest(4, '_poss_pg').iterrows():
            if any(abs(r['pc1'] - x) < span * 0.045 and abs(r['pc2'] - y) < span * 0.028
                   for x, y in placed):
                continue
            ax1.annotate(r['PLAYER_NAME'], (r['pc1'], r['pc2']), fontsize=7.5,
                         xytext=(5, 3), textcoords='offset points', color='#333')
            placed.append((r['pc1'], r['pc2']))

    ax1.set_title(f"Offensive role archetypes — {season}\n"
                  "play-type composition, PCA projection", fontsize=12, loc='left')
    if pca is not None:
        ax1.set_xlabel(_axis_label(pca, 0), fontsize=9)
        ax1.set_ylabel(_axis_label(pca, 1), fontsize=9)
    else:
        ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
    ax1.legend(fontsize=8, frameon=False, loc='best')
    ax1.spines[['top', 'right']].set_visible(False)

    heat = profile_df[PLAY_TYPES] * 100
    im = ax2.imshow(heat.to_numpy(), cmap='YlGnBu', aspect='auto')
    ax2.set_xticks(range(len(PLAY_TYPES)))
    ax2.set_xticklabels([PRETTY[p] for p in PLAY_TYPES], rotation=45, ha='right', fontsize=9)
    ax2.set_yticks(range(len(heat)))
    ax2.set_yticklabels(heat.index, fontsize=9)
    ax2.set_title("What defines each archetype\n% of player's offense", fontsize=12, loc='left')
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.iat[i, j]
            if v >= 4:
                ax2.text(j, i, f"{v:.0f}", ha='center', va='center', fontsize=8,
                         color='white' if v > heat.to_numpy().max() * 0.6 else '#222')
    fig.colorbar(im, ax=ax2, fraction=0.035, pad=0.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n🖼️  saved {out_path}")


if __name__ == "__main__":
    season = sys.argv[1] if len(sys.argv) > 1 else "2025-26"

    keep, comp, profile_df, pca, scores = fit_archetypes(season)
    report(keep, profile_df)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot(keep, profile_df, season,
         FIG_DIR / f"archetypes_{season.replace('-', '_')}.png", pca=pca)

    out_cols = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'SEASON', 'MIN',
                '_poss_pg', 'archetype', 'archetype_fit', 'pc1', 'pc2']
    export = keep[out_cols].rename(columns={'_poss_pg': 'POSS_PG'})
    export = export.join(comp.add_suffix('_SHARE'))
    path = DATA_DIR / f"archetypes_{season.replace('-', '_')}.parquet"
    export.to_parquet(path, index=False)
    print(f"💾 saved {path}  ({export.shape[0]} players, {export.shape[1]} cols)")
    print(f"   PCA explains {pca.explained_variance_ratio_.sum()*100:.0f}% of variance in 2D")
