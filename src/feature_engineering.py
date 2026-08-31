"""
RESPONSIBILITY OF THIS FILE
    Collapse event-level rows into ONE row per lead, combining
    engagement features (from events) with firmographic features (from
    the leads table) and the observed conversion label.

WHAT CONCEPT THIS FILE TEACHES
    Feature engineering from TWO SOURCES at once — a fact table
    (events) and a dimension table (leads) — merged into one feature
    row per entity. Also introduces WEIGHTED event scoring: not every
    event counts equally (a demo_request means more than an email_open),
    which neither Pulse nor Beacon needed, since their "events" were
    already homogeneous (orders, usage weeks).

CONNECTS TO PULSE AND BEACON
    Same groupby-to-entity-level discipline as both previous projects.
    New skill: joining an aggregated fact table back onto a dimension
    table, and encoding categorical firmographic columns (source,
    budget_tier) into model-ready numeric form — neither Pulse's nor
    Beacon's entities had categorical attributes to encode.

WHERE ELSE THIS PATTERN APPLIES
    Any lead-scoring, propensity-modeling, or account-scoring system
    combining behavioral events with firmographic/demographic
    attributes — which is most real B2B and B2C scoring systems.

WHEN NOT TO USE THIS PATTERN
    When behavioral and attribute data live in genuinely unrelated
    entities that shouldn't be merged (would create a meaningless join).

PUBLIC API
    build_lead_features(leads: pandas.DataFrame, events: pandas.DataFrame, config: dict) -> pandas.DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_lead_features(leads: pd.DataFrame, events: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Build one feature row per lead from firmographics + engagement events.

    WHAT GOES IN / WHAT COMES OUT
        in:  leads DataFrame (lead_id, created_date, source, budget_tier,
             company_size, converted), events DataFrame (lead_id,
             event_id, event_date, event_type)
        out: DataFrame, one row per lead_id, columns:
             lead_id, days_since_created, total_events, email_opens,
             email_clicks, website_visits, form_fills, demo_requests,
             days_since_last_engagement, engagement_velocity,
             company_size, budget_tier, budget_tier_score, source,
             is_too_new (bool), converted (bool, label)
             Example row: lead_id=1042, days_since_created=45,
             total_events=12, email_opens=6, email_clicks=3,
             website_visits=2, form_fills=1, demo_requests=0,
             days_since_last_engagement=4, engagement_velocity=0.27,
             company_size=34, budget_tier="mid", budget_tier_score=1.0,
             source="referral", is_too_new=False, converted=True

    Parameters
    ----------
    leads : pandas.DataFrame
        Cleaned lead-level data.
    events : pandas.DataFrame
        Cleaned event-level data.
    config : dict
        Full project config. Reads config["feature_engineering"].

    Returns
    -------
    pandas.DataFrame
        Lead-level feature table, one row per lead_id.

    Example
    -------
    >>> features = build_lead_features(leads, events, cfg)
    >>> features["converted"].mean() < 0.5
    True
    """
    fe_cfg = config["feature_engineering"]
    too_new_days = fe_cfg["too_new_days_threshold"]

    anchor_candidates = [leads["created_date"].max()]
    if len(events) > 0:
        anchor_candidates.append(events["event_date"].max())
    anchor_date = max(anchor_candidates) + pd.Timedelta(days=1)

    event_type_cols = ["email_open", "email_click", "website_visit",
                        "form_fill", "demo_request"]

    if len(events) > 0:
        # WHAT: pivot event_type into count columns per lead
        # WHY: turns "a list of typed events" into per-type counts a
        #      model can actually use — the pandas equivalent of
        #      one-hot-and-sum
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
            # WHAT: no engagement at all -> fall back to days_since_created
            # WHY: a literal "days since last engagement" is undefined
            #      with zero events; treating it as "as long as they've
            #      existed" (maximally stale) is the honest default
            days_since_last = days_since_created

        # WHAT: events per day since creation
        # WHY: 10 events over 5 days is far hotter than 10 events over
        #      150 days — raw count alone can't distinguish those, but
        #      velocity can
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

    # WHAT: encode budget_tier as an ORDINAL score (low < mid < enterprise)
    # WHY: budget tier has a genuine order (more budget is strictly
    #      "better" for conversion likelihood) — ordinal encoding
    #      preserves that order for the model, unlike one-hot encoding,
    #      which would treat the three tiers as unrelated categories
    budget_order = {"low": 0.0, "mid": 1.0, "enterprise": 2.0}
    features["budget_tier_score"] = features["budget_tier"].map(budget_order)

    # WHAT: one-hot encode source into source_<channel> columns
    # WHY: unlike budget_tier, source channels have NO natural order
    #      (referral isn't "more" than paid_search the way enterprise is
    #      "more" than low budget) — one-hot is the correct encoding for
    #      unordered categories, versus the ordinal encoding used above
    #      for budget_tier. Source is a real driver of conversion
    #      probability in this dataset's generator, so leaving it out
    #      (as an earlier draft of this file did) measurably weakens the
    #      classifier — see concepts.md for the before/after AUC.
    source_dummies = pd.get_dummies(features["source"], prefix="source")
    features = pd.concat([features, source_dummies], axis=1)

    return features


if __name__ == "__main__":
    # Standalone smoke test — run with: python src/feature_engineering.py
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
