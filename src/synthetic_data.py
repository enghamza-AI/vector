
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_leads(config: dict) -> dict:

    sd = config["synthetic_data"]
    rng = np.random.default_rng(sd["random_seed"])

    n_leads = sd["n_leads"]
    sim_days = sd["simulation_days"]
    sources = list(sd["sources"].keys())
    source_mults = sd["sources"]
    budget_tiers = list(sd["budget_tiers"].keys())
    budget_mults = sd["budget_tiers"]
    event_types = list(sd["event_type_weights"].keys())
    event_weights = np.array(list(sd["event_type_weights"].values()))
    event_probs = event_weights / event_weights.sum()

    sim_end_date = pd.Timestamp.today().normalize()
    sim_start_date = sim_end_date - pd.Timedelta(days=sim_days)

    lead_rows = []
    event_rows = []
    event_counter = 0

    for lead_id in range(1, n_leads + 1):
    
        true_intent = rng.lognormal(
            mean=sd["true_intent_lognormal_mean"],
            sigma=sd["true_intent_lognormal_sigma"],
        )

        source = sources[rng.integers(0, len(sources))]
        budget_tier = budget_tiers[rng.integers(0, len(budget_tiers))]
        company_size = max(
            1,
            int(round(rng.lognormal(
                mean=sd["company_size_lognormal_mean"],
                sigma=sd["company_size_lognormal_sigma"],
            ))),
        )

        created_offset = rng.integers(0, sim_days)
        created_date = sim_start_date + pd.Timedelta(days=int(created_offset))
        days_available = sim_days - created_offset

       
        expected_events = sd["baseline_total_events"] * true_intent
        n_events = rng.poisson(max(0.1, expected_events))

        for _ in range(n_events):
            event_counter += 1
            event_type = event_types[
                rng.choice(len(event_types), p=event_probs)
            ]
            event_offset = rng.integers(0, max(1, days_available))
            event_date = created_date + pd.Timedelta(days=int(event_offset))
            event_rows.append(
                {
                    "lead_id": lead_id,
                    "event_id": f"evt_{event_counter:08d}",
                    "event_date": event_date,
                    "event_type": event_type,
                }
            )


        combined_score = (
            np.log(true_intent + 1e-6)
            + np.log(source_mults[source])
            + np.log(budget_mults[budget_tier])
        )
        z = (
            sd["conversion_logistic_intercept"]
            + sd["conversion_logistic_steepness"] * combined_score
        )
        conversion_prob = 1.0 / (1.0 + np.exp(-z))
        converted = bool(rng.random() < conversion_prob)

        lead_rows.append(
            {
                "lead_id": lead_id,
                "created_date": created_date,
                "source": source,
                "budget_tier": budget_tier,
                "company_size": company_size,
                "converted": converted,
            }
        )

    leads = pd.DataFrame(lead_rows)
    events = pd.DataFrame(event_rows)
    if len(events) > 0:
        events = events.sort_values(["lead_id", "event_date"]).reset_index(drop=True)

    return {"leads": leads, "events": events}


if __name__ == "__main__":
  
    from config_loader import load_config

    cfg = load_config()
    data = generate_synthetic_leads(cfg)
    leads, events = data["leads"], data["events"]
    print(f"Generated {len(leads):,} leads, {len(events):,} engagement events")
    print(f"Conversion rate: {leads['converted'].mean():.1%}")
    print(leads.head())
    print(events.head())
