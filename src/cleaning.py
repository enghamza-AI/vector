"""
RESPONSIBILITY OF THIS FILE
    Clean raw lead and event data (synthetic or, later, a real CRM/
    marketing-automation export) into a trustworthy pair of tables.

WHAT CONCEPT THIS FILE TEACHES
    Same defensive cleaning discipline as Pulse and Beacon, applied to
    TWO related tables at once (leads and events) instead of one — the
    new judgment call is that a lead with ZERO events is valid and kept
    (silence is real signal here), unlike Pulse where an order below a
    value floor was dropped as noise.

CONNECTS TO PULSE AND BEACON
    Same shape of file, same role. Compare the "what counts as noise
    vs. signal" decision here (zero engagement = signal, keep it) to
    Pulse's (near-zero order value = noise, drop it) — same discipline,
    opposite conclusion, because the business meaning is opposite.

WHERE ELSE THIS PATTERN APPLIES
    Any two-table (dimension + fact) cleaning job: customers + orders,
    accounts + usage events, patients + visits.

WHEN NOT TO USE THIS PATTERN
    Single-table datasets — nothing to reconcile between two sources.

PUBLIC API
    clean_leads(leads, events, config) -> dict {"leads": df, "events": df}
"""

from __future__ import annotations

import pandas as pd


def clean_leads(leads: pd.DataFrame, events: pd.DataFrame, config: dict) -> dict:
    """
    Clean raw lead and event tables.

    WHAT GOES IN / WHAT COMES OUT
        in:  leads DataFrame (lead_id, created_date, source, budget_tier,
             company_size, converted), events DataFrame (lead_id,
             event_id, event_date, event_type)
        out: dict {"leads": cleaned leads DataFrame,
                    "events": cleaned events DataFrame}
             Leads with zero events are KEPT — see file header.

    Parameters
    ----------
    leads : pandas.DataFrame
        Raw lead-level data.
    events : pandas.DataFrame
        Raw event-level data.
    config : dict
        Full project config. Only config["cleaning"] is read.

    Returns
    -------
    dict
        {"leads": pandas.DataFrame, "events": pandas.DataFrame}

    Example
    -------
    >>> cleaned = clean_leads(raw_leads, raw_events, cfg)
    >>> cleaned["leads"]["company_size"].min() >= 1
    True
    """
    leads_df = leads.copy()
    events_df = events.copy()

    # WHAT: drop leads missing required fields
    # WHY: a lead with no created_date or source can't be feature-
    #      engineered or attributed to a channel
    required_lead_cols = ["lead_id", "created_date", "source", "budget_tier",
                           "company_size", "converted"]
    leads_df = leads_df.dropna(subset=required_lead_cols)
    leads_df["created_date"] = pd.to_datetime(leads_df["created_date"], errors="coerce")
    leads_df = leads_df.dropna(subset=["created_date"])

    # WHAT: company_size must be a positive integer
    # WHY: a real CRM export can have a 0 or negative "employees" field
    #      from a bad data entry — nonsensical for a firmographic feature
    leads_df = leads_df[leads_df["company_size"] >= 1]

    leads_df = leads_df.drop_duplicates(subset=["lead_id"], keep="first")

    # WHAT: clean the events table, but do NOT drop leads with zero events
    # WHY: see file header — silence is real signal here, not noise
    if len(events_df) > 0:
        required_event_cols = ["lead_id", "event_id", "event_date", "event_type"]
        events_df = events_df.dropna(subset=required_event_cols)
        events_df["event_date"] = pd.to_datetime(events_df["event_date"], errors="coerce")
        events_df = events_df.dropna(subset=["event_date"])
        events_df = events_df.drop_duplicates(subset=["event_id"], keep="first")

        # WHAT: only keep events belonging to a lead that survived cleaning
        # WHY: an event referencing a dropped/invalid lead_id is an
        #      orphaned row that would otherwise silently disappear when
        #      joined later — better to drop it explicitly here
        valid_lead_ids = set(leads_df["lead_id"])
        events_df = events_df[events_df["lead_id"].isin(valid_lead_ids)]

    leads_df = leads_df.sort_values("lead_id").reset_index(drop=True)
    if len(events_df) > 0:
        events_df = events_df.sort_values(["lead_id", "event_date"]).reset_index(drop=True)

    return {"leads": leads_df, "events": events_df}


if __name__ == "__main__":
    # Standalone smoke test — run with: python src/cleaning.py
    from config_loader import load_config
    from synthetic_data import generate_synthetic_leads

    cfg = load_config()
    raw = generate_synthetic_leads(cfg)
    cleaned = clean_leads(raw["leads"], raw["events"], cfg)
    print(f"Leads: {len(raw['leads']):,} -> {len(cleaned['leads']):,}")
    print(f"Events: {len(raw['events']):,} -> {len(cleaned['events']):,}")
