"""
RESPONSIBILITY OF THIS FILE
    Load config/config.yaml into a plain Python dict and hand it to every
    other file in src/. Nothing else in the project should open a YAML file
    directly — this is the one place that knows the config's file path.

WHAT CONCEPT THIS FILE TEACHES
    Config-driven design. You already do this informally (your Stage 2
    projects used YAML config for cost-sensitive learning thresholds) —
    this file formalizes the pattern: one loader function, imported
    everywhere, so there is exactly one place that knows "where is the
    config file" and "what if it's missing."

CONNECTS TO STAGE 1/2/3
    Stage 2: you already wrote YAML config for fairness constraints and
    Pareto thresholds. This is the same idea, generalized into its own
    module instead of being loaded ad-hoc inside a script.

WHERE ELSE THIS PATTERN APPLIES
    Any project with more than ~5 tunable numbers: API rate limits, model
    hyperparameters, feature flags, environment-specific paths (dev vs
    prod database URLs). Anywhere you'd otherwise be tempted to hardcode
    a number and forget where you hardcoded it.

WHEN NOT TO USE THIS PATTERN
    Single-script, one-off analyses where the "config" is three numbers
    you're actively iterating on in a notebook — a YAML round-trip just
    adds friction. Also not useful for secrets (API keys, passwords):
    those belong in environment variables or a secrets manager, never in
    a committed YAML file.

PUBLIC API
    load_config(path: str = "config/config.yaml") -> dict
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    """
    Load the project's YAML config file into a nested dict.

    Parameters
    ----------
    path : str
        Path to the config file. Resolved relative to the project root
        (the directory containing this src/ folder), not relative to
        whatever directory the caller happens to be running from — this
        is what lets both `python src/pipeline.py` (run from project root)
        and Streamlit (which sometimes changes the working directory)
        find the same file reliably.

    Returns
    -------
    dict
        Nested dict mirroring config.yaml's structure, e.g.
        config["segmentation"]["n_clusters"] -> 4

    Example
    -------
    >>> cfg = load_config()
    >>> cfg["segmentation"]["n_clusters"]
    4
    """
    # WHAT: build an absolute path anchored to this file's location, not cwd
    # WHY: Streamlit and pytest both sometimes run from a different working
    #      directory than the project root, which silently breaks a plain
    #      relative path like "config/config.yaml"
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / path

    if not config_path.exists():
        # Helpful error: say what went wrong AND what to do about it
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            f"Make sure you're running from the project root, or pass an "
            f"explicit path to load_config()."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_data_mode(config: dict) -> str:
    """
    Resolve DATA_MODE: environment variable wins over config.yaml default.

    Parameters
    ----------
    config : dict
        Loaded config dict (from load_config()).

    Returns
    -------
    str
        Either "demo" or "local".

    Example
    -------
    >>> cfg = load_config()
    >>> get_data_mode(cfg)
    'demo'
    """
    # WHAT: environment variable takes priority over the YAML default
    # WHY: this is exactly the mechanism Hugging Face Spaces uses to flip
    #      a deployed app into demo mode without touching committed code —
    #      you set DATA_MODE=demo as a Space secret/variable instead
    default = config.get("app", {}).get("data_mode_default", "demo")
    return os.getenv("DATA_MODE", default)


if __name__ == "__main__":
    # Standalone smoke test — run with: python src/config_loader.py
    cfg = load_config()
    print("Loaded config sections:", list(cfg.keys()))
    print("n_clusters:", cfg["segmentation"]["n_clusters"])
    print("resolved DATA_MODE:", get_data_mode(cfg))
