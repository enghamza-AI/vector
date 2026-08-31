
from __future__ import annotations

import pandas as pd


def clean_leads(leads: pd.DataFrame, events: pd.DataFrame, config: dict) -> dict:

    leads_df = leads.copy()
    events_df = events.copy()


    required_lead_cols = ["lead_id", "created_date", "source", "budget_tier",
                           "company_size", "converted"]
    leads_df = leads_df.dropna(subset=required_lead_cols)
    leads_df["created_date"] = pd.to_datetime(leads_df["created_date"], errors="coerce")
    leads_df = leads_df.dropna(subset=["created_date"])

   
    leads_df = leads_df[leads_df["company_size"] >= 1]

    leads_df = leads_df.drop_duplicates(subset=["lead_id"], keep="first")

   
    if len(events_df) > 0:
        required_event_cols = ["lead_id", "event_id", "event_date", "event_type"]
        events_df = events_df.dropna(subset=required_event_cols)
        events_df["event_date"] = pd.to_datetime(events_df["event_date"], errors="coerce")
        events_df = events_df.dropna(subset=["event_date"])
        events_df = events_df.drop_duplicates(subset=["event_id"], keep="first")

       
        valid_lead_ids = set(leads_df["lead_id"])
        events_df = events_df[events_df["lead_id"].isin(valid_lead_ids)]

    leads_df = leads_df.sort_values("lead_id").reset_index(drop=True)
    if len(events_df) > 0:
        events_df = events_df.sort_values(["lead_id", "event_date"]).reset_index(drop=True)

    return {"leads": leads_df, "events": events_df}


if __name__ == "__main__":
    
    from config_loader import load_config
    from synthetic_data import generate_synthetic_leads

    cfg = load_config()
    raw = generate_synthetic_leads(cfg)
    cleaned = clean_leads(raw["leads"], raw["events"], cfg)
    print(f"Leads: {len(raw['leads']):,} -> {len(cleaned['leads']):,}")
    print(f"Events: {len(raw['events']):,} -> {len(cleaned['events']):,}")
