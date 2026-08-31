
from __future__ import annotations

import numpy as np
import pandas as pd


def build_lead_features(leads: pd.DataFrame, events: pd.DataFrame, config: dict) -> pd.DataFrame:
 
    fe_cfg = config["feature_engineering"]
    too_new_days = fe_cfg["too_new_days_threshold"]

    anchor_candidates = [leads["created_date"].max()]
    if len(events) > 0:
        anchor_candidates.append(events["event_date"].max())
    anchor_date = max(anchor_candidates) + pd.Timedelta(days=1)

    event_type_cols = ["email_open", "email_click", "website_visit",
                        "form_fill", "demo_request"]

    if len(events) > 0:
      
        event_counts = (
            events.groupby(["lead_id", "event_type"]).size().unstack(fill_value=0)
        )
        for col in event_type_cols:
            if col not in event_counts.columns:
                event_counts[col] = 0
        event_counts = event_counts[event_type_cols]
        last_engagement = events.groupby("lead_id")["event_date"].max()
    else:
        event_counts = pd.DataFrame(columns=event_type_cols)
        last_engagement = pd.Series(dtype="datetime64[ns]")

    rows = []
    for _, lead in leads.iterrows():
        lead_id = lead["lead_id"]
        days_since_created = (anchor_date - lead["created_date"]).days

        if lead_id in event_counts.index:
            counts = event_counts.loc[lead_id]
        else:
            counts = pd.Series({c: 0 for c in event_type_cols})

        total_events = int(counts.sum())

        if lead_id in last_engagement.index:
            days_since_last = (anchor_date - last_engagement.loc[lead_id]).days
        else:
           
            days_since_last = days_since_created

      
        engagement_velocity = total_events / max(1, days_since_created)

        is_too_new = days_since_created <= too_new_days

        rows.append(
            {
                "lead_id": lead_id,
                "days_since_created": days_since_created,
                "total_events": total_events,
                "email_opens": int(counts["email_open"]),
                "email_clicks": int(counts["email_click"]),
                "website_visits": int(counts["website_visit"]),
                "form_fills": int(counts["form_fill"]),
                "demo_requests": int(counts["demo_request"]),
                "days_since_last_engagement": days_since_last,
                "engagement_velocity": round(float(engagement_velocity), 4),
                "company_size": lead["company_size"],
                "budget_tier": lead["budget_tier"],
                "source": lead["source"],
                "is_too_new": is_too_new,
                "converted": bool(lead["converted"]),
            }
        )

    features = pd.DataFrame(rows)

    budget_order = {"low": 0.0, "mid": 1.0, "enterprise": 2.0}
    features["budget_tier_score"] = features["budget_tier"].map(budget_order)

  
    source_dummies = pd.get_dummies(features["source"], prefix="source")
    features = pd.concat([features, source_dummies], axis=1)

    return features


if __name__ == "__main__":
  
    from config_loader import load_config
    from synthetic_data import generate_synthetic_leads
    from cleaning import clean_leads

    cfg = load_config()
    raw = generate_synthetic_leads(cfg)
    cleaned = clean_leads(raw["leads"], raw["events"], cfg)
    features = build_lead_features(cleaned["leads"], cleaned["events"], cfg)
    print(f"Built features for {len(features):,} leads")
    print(features.describe(include="all"))
    print(f"\nConverted: {features['converted'].mean():.1%}")
    print(f"Too new to score: {features['is_too_new'].mean():.1%}")
