
from __future__ import annotations

from src.config_loader import load_config
from src.synthetic_data import generate_synthetic_leads


def main() -> None:
    config = load_config()
    demo_max_rows = config["app"]["demo_max_rows"]
    leads_path = config["app"]["demo_csv_path"]
    events_path = leads_path.replace("demo_sample.csv", "demo_sample_events.csv")

    demo_config = {
        **config,
        "synthetic_data": {
            **config["synthetic_data"],
            "n_leads": demo_max_rows,
        },
    }

    generated = generate_synthetic_leads(demo_config)
    generated["leads"].to_csv(leads_path, index=False)
    generated["events"].to_csv(events_path, index=False)

    print(f"Wrote {len(generated['leads']):,} leads to {leads_path}")
    print(f"Wrote {len(generated['events']):,} events to {events_path}")


if __name__ == "__main__":
    main()
