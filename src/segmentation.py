"""
RESPONSIBILITY OF THIS FILE
    Cluster leads into engagement tiers with K-Means, then translate
    numeric cluster ids into human-readable tier names (hot, warm, cold,
    unqualified) a sales rep can act on.

WHAT CONCEPT THIS FILE TEACHES
    Third application of the same K-Means + centroid-naming pattern from
    Pulse and Beacon — with the RANK-BASED naming fix (from Beacon's
    concepts.md section 4) applied from the start this time instead of
    discovered as a bug. The lesson: a pattern learned from a mistake on
    one project should be applied proactively on the next, not
    rediscovered.

CONNECTS TO PULSE AND BEACON
    Structurally identical file to both. "unqualified" here is a
    BUSINESS OVERRIDE on top of the unsupervised clusters (low budget +
    small company, regardless of engagement) — same pattern as Pulse's
    vip_monetary_floor override on top of K-Means.

WHERE ELSE THIS PATTERN APPLIES
    Any prioritization tiering: support ticket triage, sales pipeline
    stages, content-moderation review queues.

WHEN NOT TO USE THIS PATTERN
    When qualification is a hard compliance/eligibility rule (e.g. "must
    be in a licensed territory") — that's a filter applied before
    clustering, not a segment name assigned after it.

PUBLIC API
    segment_leads(features: pandas.DataFrame, config: dict) -> pandas.DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def segment_leads(features: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Cluster leads with K-Means and assign human-readable tier names.

    WHAT GOES IN / WHAT COMES OUT
        in:  features DataFrame (output of
             feature_engineering.build_lead_features), must contain
             every column in config["segmentation"]["clustering_features"]
        out: same DataFrame with three new columns:
             cluster_id (int), segment_name (str), silhouette_avg (float)
             Example: cluster_id=2, segment_name="hot", silhouette_avg=0.33

    Parameters
    ----------
    features : pandas.DataFrame
        Lead-level feature table.
    config : dict
        Full project config. Reads config["segmentation"] and
        config["recommendations"] (for the unqualified override).

    Returns
    -------
    pandas.DataFrame
        Input features with cluster_id, segment_name, silhouette_avg
        columns added.

    Example
    -------
    >>> segmented = segment_leads(features, cfg)
    >>> segmented["segment_name"].value_counts()
    cold           2100
    warm           1400
    hot             900
    unqualified     600
    """
    seg_cfg = config["segmentation"]
    feature_cols = seg_cfg["clustering_features"]

    X = features[feature_cols].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=seg_cfg["n_clusters"],
        random_state=seg_cfg["random_state"],
        n_init=seg_cfg["n_init"],
    )
    cluster_ids = kmeans.fit_predict(X_scaled)

    sample_size = min(2000, len(X_scaled))
    sample_idx = np.random.default_rng(seg_cfg["random_state"]).choice(
        len(X_scaled), size=sample_size, replace=False
    )
    sil_score = silhouette_score(X_scaled[sample_idx], cluster_ids[sample_idx])

    out = features.copy()
    out["cluster_id"] = cluster_ids
    out["silhouette_avg"] = round(float(sil_score), 3)
    out["segment_name"] = _name_segments(out, config)

    return out


def _name_segments(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    Map raw cluster_id values to human-readable tier names.

    WHAT GOES IN / WHAT COMES OUT
        in:  df with cluster_id, total_events, demo_requests,
             engagement_velocity, budget_tier, company_size
        out: Series of str, one of {"hot", "warm", "cold", "unqualified"}

    Parameters
    ----------
    df : pandas.DataFrame
        Feature table with cluster_id already assigned.
    config : dict
        Full project config. Reads config["recommendations"] for the
        unqualified override thresholds.

    Returns
    -------
    pandas.Series
        Segment name per row.

    Example
    -------
    >>> _name_segments(df, cfg).unique()
    array(['hot', 'warm', 'cold', 'unqualified'], dtype=object)
    """
    centroid_stats = df.groupby("cluster_id").agg(
        mean_events=("total_events", "mean"),
        mean_demo=("demo_requests", "mean"),
        mean_velocity=("engagement_velocity", "mean"),
    )

    # WHAT: rank clusters relative to each other, not against fixed
    #       thresholds — the fix carried over from Beacon's segmentation
    #       bug, applied proactively this time
    # WHY: guarantees every tier name is used exactly once, regardless
    #      of how spread out this run's centroids happen to be
    remaining = list(centroid_stats.index)
    label_by_cluster = {}

    hot_id = centroid_stats.loc[remaining, "mean_demo"].idxmax()
    label_by_cluster[hot_id] = "hot"
    remaining.remove(hot_id)

    warm_id = centroid_stats.loc[remaining, "mean_velocity"].idxmax()
    label_by_cluster[warm_id] = "warm"
    remaining.remove(warm_id)

    cold_id = centroid_stats.loc[remaining, "mean_events"].idxmin()
    label_by_cluster[cold_id] = "cold"
    remaining.remove(cold_id)

    for cid in remaining:
        label_by_cluster[cid] = "warm"

    names = df["cluster_id"].map(label_by_cluster)

    # WHAT: business override — low budget AND small company gets
    #       "unqualified" regardless of which engagement cluster they
    #       landed in
    # WHY: a small, low-budget company that clicks every email is still
    #      not a good sales target — engagement predicts INTEREST, not
    #      DEAL VIABILITY, and conflating the two wastes a rep's time.
    #      Same override pattern as Pulse's vip_monetary_floor.
    rec_cfg = config["recommendations"]
    is_unqualified = (
        (df["budget_tier"] == rec_cfg["unqualified_budget_tier"])
        & (df["company_size"] < rec_cfg["unqualified_company_size_floor"])
    )
    names = names.where(~is_unqualified, "unqualified")

    return names


if __name__ == "__main__":
    # Standalone smoke test — run with: python src/segmentation.py
    from config_loader import load_config
    from synthetic_data import generate_synthetic_leads
    from cleaning import clean_leads
    from feature_engineering import build_lead_features

    cfg = load_config()
    raw = generate_synthetic_leads(cfg)
    cleaned = clean_leads(raw["leads"], raw["events"], cfg)
    features = build_lead_features(cleaned["leads"], cleaned["events"], cfg)
    segmented = segment_leads(features, cfg)
    print(segmented["segment_name"].value_counts())
    print(f"\nSilhouette score: {segmented['silhouette_avg'].iloc[0]}")
