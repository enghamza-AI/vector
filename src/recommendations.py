"""
RESPONSIBILITY OF THIS FILE
    Translate lead_score + segment_name + primary_signal into a plain-
    English recommended action and urgency level a sales rep can act on
    the same day.

WHAT CONCEPT THIS FILE TEACHES
    Same ML-to-business translation boundary as Pulse and Beacon. The
    new wrinkle: "unqualified" leads get a recommendation that
    deliberately IGNORES a high lead_score — a lead can be highly
    engaged and still not worth a rep's time, which is exactly why the
    business-rule override exists as its own layer instead of trusting
    the score alone.

CONNECTS TO PULSE AND BEACON
    Same file role. Compare the "unqualified overrides a high score"
    rule here to Beacon's "new accounts ignore their own risk score" —
    same underlying lesson: a model's number is an input to a decision,
    not the decision itself.

PUBLIC API
    generate_recommendations(scored: pandas.DataFrame, config: dict) -> pandas.DataFrame
"""

from __future__ import annotations

import pandas as pd


def generate_recommendations(scored: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Attach recommended_action and urgency columns to scored leads.

    WHAT GOES IN / WHAT COMES OUT
        in:  scored DataFrame with segment_name, lead_score,
             primary_signal, is_too_new
        out: same DataFrame with two new columns:
             recommended_action (str), urgency (str: "high"/"medium"/"low")
             Example: recommended_action="Hot lead (84/100) — requested
             a demo. Call today.", urgency="high"

    Parameters
    ----------
    scored : pandas.DataFrame
        Output of scoring.predict_lead_score.
    config : dict
        Full project config. Reads config["recommendations"].

    Returns
    -------
    pandas.DataFrame
        Input with recommended_action, urgency columns appended.

    Example
    -------
    >>> recs = generate_recommendations(scored, cfg)
    >>> recs["urgency"].value_counts()
    low       3400
    medium     900
    high       700
    """
    rec_cfg = config["recommendations"]
    hot_threshold = rec_cfg["hot_score_threshold"]
    warm_threshold = rec_cfg["warm_score_threshold"]

    out = scored.copy()

    actions = []
    urgencies = []

    for _, row in out.iterrows():
        score = row["lead_score"]
        signal = row["primary_signal"]
        segment = row["segment_name"]
        too_new = row["is_too_new"]

        if segment == "unqualified":
            action = (
                f"Engaged but doesn't fit the ideal customer profile "
                f"(budget/company size too small) — deprioritize even "
                f"though the score is {score:.0f}/100. Route to "
                f"self-serve/nurture, not a rep's time."
            )
            urgency = "low"

        elif too_new:
            action = (
                "Too new to score reliably — let the first nurture "
                "email or two land before acting on this lead's score."
            )
            urgency = "low"

        elif score >= hot_threshold:
            action = (
                f"Hot lead ({score:.0f}/100) — {signal.lower()}. "
                f"Call today, don't let this go cold."
            )
            urgency = "high"

        elif score >= warm_threshold:
            action = (
                f"Warm lead ({score:.0f}/100) — {signal.lower()}. "
                f"Add to this week's outreach list."
            )
            urgency = "medium"

        else:
            action = (
                f"Cold lead ({score:.0f}/100) — {signal.lower()}. "
                f"Keep in a low-touch nurture sequence, not active outreach."
            )
            urgency = "low"

        actions.append(action)
        urgencies.append(urgency)

    out["recommended_action"] = actions
    out["urgency"] = urgencies
    return out


if __name__ == "__main__":
    # Standalone smoke test — run with: python src/recommendations.py
    from config_loader import load_config
    from synthetic_data import generate_synthetic_leads
    from cleaning import clean_leads
    from feature_engineering import build_lead_features
    from segmentation import segment_leads
    from scoring import train_lead_model, predict_lead_score

    cfg = load_config()
    raw = generate_synthetic_leads(cfg)
    cleaned = clean_leads(raw["leads"], raw["events"], cfg)
    features = build_lead_features(cleaned["leads"], cleaned["events"], cfg)
    segmented = segment_leads(features, cfg)
    bundle = train_lead_model(segmented, cfg)
    scored = predict_lead_score(segmented, bundle)
    recs = generate_recommendations(scored, cfg)
    print(recs["urgency"].value_counts())
    print(recs[["lead_id", "segment_name", "urgency",
                 "recommended_action"]].head(3).to_string())
