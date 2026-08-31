"""
RESPONSIBILITY OF THIS FILE
    Wire together every other src/ module into one callable pipeline:
    load/generate -> clean -> feature-engineer -> segment -> score ->
    recommend. The only file app.py and api/main.py import from directly.

WHAT CONCEPT THIS FILE TEACHES
    Third application of the composition-root pattern — by now, notice
    this file is almost boilerplate to write, because the DESIGN
    decision (single orchestrator, single entry point, no logic of its
    own) was made once and just gets reapplied. That's what "the
    pattern paid for itself" looks like.

CONNECTS TO PULSE AND BEACON
    Structurally identical to both. Only difference: Vector's raw data
    is TWO tables (leads + events) instead of one, so data_mode="demo"
    loads two CSVs instead of one.

PUBLIC API
    run_pipeline(config: dict, data_mode: str = "demo") -> dict
"""

from __future__ import annotations

import pandas as pd

from src.config_loader import load_config
from src.synthetic_data import generate_synthetic_leads
from src.cleaning import clean_leads
from src.feature_engineering import build_lead_features
from src.segmentation import segment_leads
from src.scoring import train_lead_model, predict_lead_score
from src.recommendations import generate_recommendations


def run_pipeline(config: dict, data_mode: str = "demo") -> dict:
    """
    Run the full Vector pipeline end to end.

    WHAT GOES IN / WHAT COMES OUT
        in:  config dict, data_mode: "demo" or "local"
        out: dict with keys:
             "result" (pandas.DataFrame, one row per lead with every
             feature + segment_name + lead_score + primary_signal +
             recommended_action + urgency),
             "model_auc" (float), "silhouette_avg" (float),
             "n_leads" (int)

    Parameters
    ----------
    config : dict
        Full project config.
    data_mode : str
        "demo" (pre-baked small CSVs) or "local" (full synthetic
        generation, or swap in a real CRM export — see
        about_the_project.md).

    Returns
    -------
    dict
        Pipeline result bundle.

    Example
    -------
    >>> cfg = load_config()
    >>> out = run_pipeline(cfg, data_mode="demo")
    >>> out["result"].columns.tolist()[:3]
    ['lead_id', 'days_since_created', 'total_events']
    """
    if data_mode == "demo":
        leads_path = config["app"]["demo_csv_path"]
        events_path = leads_path.replace("demo_sample.csv", "demo_sample_events.csv")
        raw_leads = pd.read_csv(leads_path, parse_dates=["created_date"])
        raw_events = pd.read_csv(events_path, parse_dates=["event_date"])
    elif data_mode == "local":
        generated = generate_synthetic_leads(config)
        raw_leads = generated["leads"]
        raw_events = generated["events"]
    else:
        raise ValueError(
            f"Unknown data_mode '{data_mode}'. Expected 'demo' or 'local'."
        )

    cleaned = clean_leads(raw_leads, raw_events, config)
    features = build_lead_features(cleaned["leads"], cleaned["events"], config)
    segmented = segment_leads(features, config)
    model_bundle = train_lead_model(segmented, config)
    scored = predict_lead_score(segmented, model_bundle)
    result = generate_recommendations(scored, config)

    return {
        "result": result,
        "model_auc": model_bundle["test_auc"],
        "silhouette_avg": float(result["silhouette_avg"].iloc[0]),
        "n_leads": len(result),
    }


if __name__ == "__main__":
    # Standalone smoke test — run with: python -m src.pipeline
    cfg = load_config()
    output = run_pipeline(cfg, data_mode="local")
    print(f"Scored {output['n_leads']:,} leads")
    print(f"Model AUC: {output['model_auc']}")
    print(f"Silhouette score: {output['silhouette_avg']}")
    print(output["result"]["segment_name"].value_counts())
