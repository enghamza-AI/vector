
from __future__ import annotations

from pathlib import Path
import sys

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.pipeline import run_pipeline


OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "demo"
    / "vector_results.pkl"
)


def main() -> None:
    config = load_config()

    print("Running Vector pipeline...")
    output = run_pipeline(config, data_mode="demo")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, OUTPUT_PATH)

    print()
    print("Vector demo results saved successfully.")
    print(f"File: {OUTPUT_PATH}")
    print(f"Leads: {output['n_leads']:,}")
    print(f"Model AUC: {output['model_auc']}")
    print(f"Silhouette score: {output['silhouette_avg']}")


if __name__ == "__main__":
    main()

