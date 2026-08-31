
from __future__ import annotations

import pandas as pd


def generate_recommendations(scored: pd.DataFrame, config: dict) -> pd.DataFrame:
 
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
