

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def segment_leads(features: pd.DataFrame, config: dict) -> pd.DataFrame:
 
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

    centroid_stats = df.groupby("cluster_id").agg(
        mean_events=("total_events", "mean"),
        mean_demo=("demo_requests", "mean"),
        mean_velocity=("engagement_velocity", "mean"),
    )

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

  
    rec_cfg = config["recommendations"]
    is_unqualified = (
        (df["budget_tier"] == rec_cfg["unqualified_budget_tier"])
        & (df["company_size"] < rec_cfg["unqualified_company_size_floor"])
    )
    names = names.where(~is_unqualified, "unqualified")

    return names


if __name__ == "__main__":
   
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
