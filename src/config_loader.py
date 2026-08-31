
from __future__ import annotations

import os
from pathlib import Path

import yaml


def load_config(path: str = "config/config.yaml") -> dict:


    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / path

    if not config_path.exists():
       
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            f"Make sure you're running from the project root, or pass an "
            f"explicit path to load_config()."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_data_mode(config: dict) -> str:

    default = config.get("app", {}).get("data_mode_default", "demo")
    return os.getenv("DATA_MODE", default)


if __name__ == "__main__":

    cfg = load_config()
    print("Loaded config sections:", list(cfg.keys()))
    print("n_clusters:", cfg["segmentation"]["n_clusters"])
    print("resolved DATA_MODE:", get_data_mode(cfg))
