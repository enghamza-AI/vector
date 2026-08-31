"""
RESPONSIBILITY OF THIS FILE
    Generate a realistic, fully synthetic lead-engagement dataset — one
    row per engagement EVENT (email open, click, website visit, form
    fill, demo request), plus a firmographic row per lead — with a
    hidden "true intent" per lead that drives both event frequency and
    eventual conversion, never exposed as a feature.

WHAT CONCEPT THIS FILE TEACHES
    A THIRD variation of the same hidden-ground-truth simulation
    pattern used in Pulse (hidden repurchase cadence) and Beacon (hidden
    decay trajectory). Here the hidden variable is "true intent," and it
    drives TWO separate observable outcomes — event frequency AND
    conversion probability — which is closer to how real latent
    variables usually work (one underlying cause, multiple downstream
    effects) than either previous project's single-effect design.

CONNECTS TO PULSE AND BEACON
    Structurally closest to Pulse: event-level rows collapsed to one
    row per entity (lead, here) via feature_engineering.py. The
    conversion label is like Beacon's — genuinely observed within a
    fixed window, not censored. The event-type MIX (not just count) is
    new — a lead with 10 email opens and a lead with 2 demo requests
    have very different intent despite similar raw event counts, which
    is a distinction neither prior project needed to make.

WHERE ELSE THIS PATTERN APPLIES
    Marketing attribution, product-led-growth activation scoring, sales
    pipeline prioritization — anywhere discrete user actions of
    different "weight" need to be combined into one intent signal.

WHEN NOT TO USE THIS PATTERN
    Once real CRM/marketing-automation event data exists.

PUBLIC API
    generate_synthetic_leads(config: dict) -> dict
        returns {"leads": DataFrame, "events": DataFrame}
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_leads(config: dict) -> dict:
    """
    Generate synthetic lead firmographics and engagement events.

    WHAT GOES IN / WHAT COMES OUT
        in:  config["synthetic_data"] dict
        out: dict with two DataFrames:
             "leads": one row per lead — lead_id, created_date, source,
                       budget_tier, company_size, converted (bool)
             "events": one row per engagement event — lead_id, event_id,
                        event_date, event_type
             Example leads row: lead_id=1042, created_date=2025-01-14,
             source="referral", budget_tier="mid", company_size=34,
             converted=True
             Example events row: lead_id=1042, event_id="evt_00004821",
             event_date=2025-01-20, event_type="demo_request"

    Parameters
    ----------
    config : dict
        Full project config. Only config["synthetic_data"] is read.

    Returns
    -------
    dict
        {"leads": pandas.DataFrame, "events": pandas.DataFrame}

    Example
    -------
    >>> from src.config_loader import load_config
    >>> cfg = load_config()
    >>> data = generate_synthetic_leads(cfg)
    >>> sorted(data.keys())
    ['events', 'leads']
    """
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
        # WHAT: each lead's hidden true intent — drives BOTH event
        #       frequency and conversion probability
        # WHY: this dual effect is the new concept this file teaches —
        #      one latent cause, two observable downstream effects
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

        # WHAT: expected total events for this lead, scaled by intent
        # WHY: a Poisson draw around this expectation gives a realistic,
        #      variable event count per lead rather than a fixed number
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

        # WHAT: combine hidden intent, source quality, and budget
        #       quality into a conversion probability via a logistic
        #       function
        # WHY: a logistic (sigmoid) function is the standard way to turn
        #      an unbounded combined "score" into a valid 0-1
        #      probability — the same function a real logistic
        #      regression model would fit, used here to GENERATE the
        #      ground truth the classifier later has to recover
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
    # Standalone smoke test — run with: python src/synthetic_data.py
    from config_loader import load_config

    cfg = load_config()
    data = generate_synthetic_leads(cfg)
    leads, events = data["leads"], data["events"]
    print(f"Generated {len(leads):,} leads, {len(events):,} engagement events")
    print(f"Conversion rate: {leads['converted'].mean():.1%}")
    print(leads.head())
    print(events.head())
