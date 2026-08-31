
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
 
    cfg = load_config()
    output = run_pipeline(cfg, data_mode="local")
    print(f"Scored {output['n_leads']:,} leads")
    print(f"Model AUC: {output['model_auc']}")
    print(f"Silhouette score: {output['silhouette_avg']}")
    print(output["result"]["segment_name"].value_counts())
