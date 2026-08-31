"""
RESPONSIBILITY OF THIS FILE
    Train a supervised classifier predicting conversion probability
    (0-100 lead score) from engagement + firmographic features, then
    score every lead and flag the single most explanatory reason.

WHAT CONCEPT THIS FILE TEACHES
    Third application of the RandomForestClassifier + predict_proba
    pattern from Beacon — now trained on a MIX of behavioral features
    (engagement counts, velocity) and categorical/ordinal firmographic
    features (budget_tier_score, company_size) together. Mixing feature
    types in one model is the new piece: Beacon's classifier only saw
    behavioral trend numbers.

CONNECTS TO BEACON
    Same classifier family, same stratified-split + class_weight
    discipline for an imbalanced label (conversion is rare, same as
    churn was). Compare MODEL_FEATURE_COLS here to Beacon's — notice
    firmographic features (budget_tier_score, company_size) sit
    alongside behavioral ones, which Beacon's feature set didn't need.

WHERE ELSE THIS PATTERN APPLIES
    Any propensity model combining "who they are" (firmographics/
    demographics) with "what they did" (behavior): loan approval,
    insurance underwriting, university admissions scoring.

WHEN NOT TO USE THIS PATTERN
    When firmographic attributes are protected-class-adjacent (age,
    gender, zip code as a race proxy, etc.) — mixing behavioral and
    demographic features into one score requires a fairness review
    before deployment, flagged explicitly in about_the_project.md.

PUBLIC API
    train_lead_model(features: pandas.DataFrame, config: dict) -> dict
    predict_lead_score(features: pandas.DataFrame, model_bundle: dict) -> pandas.DataFrame
"""

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
    """
    Train a RandomForestClassifier to predict converted.

    WHAT GOES IN / WHAT COMES OUT
        in:  features DataFrame with converted and every column in
             MODEL_FEATURE_COLS
        out: dict "model bundle":
             {"model": fitted RandomForestClassifier,
              "feature_cols": list[str], "test_auc": float,
              "n_train": int, "n_test": int}

    Parameters
    ----------
    features : pandas.DataFrame
        Segmented lead feature table.
    config : dict
        Full project config. Reads config["scoring"].

    Returns
    -------
    dict
        Model bundle.

    Example
    -------
    >>> bundle = train_lead_model(segmented, cfg)
    >>> bundle["test_auc"] > 0.7
    True
    """
    scoring_cfg = config["scoring"]

    feature_cols = [c for c in MODEL_FEATURE_COLS if c in features.columns]
    X = features[feature_cols].to_numpy()
    y = features["converted"].to_numpy().astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=scoring_cfg["test_size"],
        random_state=scoring_cfg["random_state"],
        stratify=y,  # conversion is a minority class — keep the same
                     # conversion rate in both splits (same reasoning as Beacon)
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
    """
    Score every lead with lead_score (0-100) and a primary reason.

    WHAT GOES IN / WHAT COMES OUT
        in:  features DataFrame, model_bundle from train_lead_model()
        out: same DataFrame with two new columns:
             lead_score (float, 0-100), primary_signal (str)
             Example: lead_score=82.3, primary_signal="Requested a demo"

    Parameters
    ----------
    features : pandas.DataFrame
        Lead-level feature table to score.
    model_bundle : dict
        Output of train_lead_model().

    Returns
    -------
    pandas.DataFrame
        Input features with lead_score, primary_signal appended.

    Example
    -------
    >>> scored = predict_lead_score(segmented, bundle)
    >>> scored["lead_score"].between(0, 100).all()
    True
    """
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]

    X = features[feature_cols].to_numpy()
    probs = model.predict_proba(X)[:, 1]

    out = features.copy()
    out["lead_score"] = np.round(probs * 100, 1)
    out["primary_signal"] = _determine_primary_signal(out)
    return out


def _determine_primary_signal(df: pd.DataFrame) -> pd.Series:
    """
    Pick the single most explanatory signal per lead, in plain English.

    WHAT GOES IN / WHAT COMES OUT
        in:  df with demo_requests, form_fills, engagement_velocity,
             days_since_last_engagement, budget_tier_score
        out: Series of str, one sentence per row

    Parameters
    ----------
    df : pandas.DataFrame
        Scored lead table.

    Returns
    -------
    pandas.Series
        Plain-English reason per row.

    Example
    -------
    >>> _determine_primary_signal(df).iloc[0]
    'Requested a demo'
    """
    signals = []
    for _, row in df.iterrows():
        # WHAT: same honest, rule-based ranking approach as Beacon's
        #       _determine_primary_reason — highest-intent signal wins
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
    # Standalone smoke test — run with: python src/scoring.py
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
