
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

MODEL_FEATURE_COLS = [
    "days_since_created",
    "total_events",
    "email_opens",
    "email_clicks",
    "website_visits",
    "form_fills",
    "demo_requests",
    "days_since_last_engagement",
    "engagement_velocity",
    "company_size",
    "budget_tier_score",
    "source_organic",
    "source_referral",
    "source_event",
    "source_paid_search",
    "source_cold_outreach",
]


def train_lead_model(features: pd.DataFrame, config: dict) -> dict:
  
    scoring_cfg = config["scoring"]

    feature_cols = [c for c in MODEL_FEATURE_COLS if c in features.columns]
    X = features[feature_cols].to_numpy()
    y = features["converted"].to_numpy().astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=scoring_cfg["test_size"],
        random_state=scoring_cfg["random_state"],
        stratify=y, 
                    
    )

    model = RandomForestClassifier(
        n_estimators=scoring_cfg["n_estimators"],
        max_depth=scoring_cfg["max_depth"],
        random_state=scoring_cfg["random_state"],
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "test_auc": round(float(test_auc), 3),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def predict_lead_score(features: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:

    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]

    X = features[feature_cols].to_numpy()
    probs = model.predict_proba(X)[:, 1]

    out = features.copy()
    out["lead_score"] = np.round(probs * 100, 1)
    out["primary_signal"] = _determine_primary_signal(out)
    return out


def _determine_primary_signal(df: pd.DataFrame) -> pd.Series:
  
    signals = []
    for _, row in df.iterrows():
      
        if row["demo_requests"] >= 1:
            signals.append("Requested a demo — highest-intent action available")
        elif row["form_fills"] >= 1:
            signals.append("Filled out a form")
        elif row["budget_tier_score"] >= 2.0 and row["total_events"] >= 3:
            signals.append("Enterprise budget tier with active engagement")
        elif row["engagement_velocity"] > 0 and row["days_since_last_engagement"] <= 7:
            signals.append("Actively engaging in the last week")
        elif row["total_events"] == 0:
            signals.append("No engagement yet")
        else:
            signals.append("Some engagement, no strong single signal")

    return pd.Series(signals, index=df.index)


if __name__ == "__main__":
  
    from config_loader import load_config
    from synthetic_data import generate_synthetic_leads
    from cleaning import clean_leads
    from feature_engineering import build_lead_features
    from segmentation import segment_leads

    cfg = load_config()
    raw = generate_synthetic_leads(cfg)
    cleaned = clean_leads(raw["leads"], raw["events"], cfg)
    features = build_lead_features(cleaned["leads"], cleaned["events"], cfg)
    segmented = segment_leads(features, cfg)

    bundle = train_lead_model(segmented, cfg)
    print(f"Trained on {bundle['n_train']} leads, "
          f"tested on {bundle['n_test']}, AUC = {bundle['test_auc']}")

    scored = predict_lead_score(segmented, bundle)
    print(scored[["lead_id", "segment_name", "lead_score",
                   "primary_signal"]].sort_values(
        "lead_score", ascending=False
    ).head())
