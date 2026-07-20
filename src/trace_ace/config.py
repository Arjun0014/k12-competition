from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(project_root: Path) -> dict[str, Any]:
    path = project_root / "configs" / "project.json"
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    required = {
        "cache_version",
        "closing_fraction",
        "fold_seed",
        "n_splits",
        "objective_smoothing",
        "probability_clip",
        "text_baseline",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing project configuration keys: {sorted(missing)}")
    if not 0.0 < float(config["closing_fraction"]) <= 1.0:
        raise ValueError("closing_fraction must be in (0, 1].")
    if int(config["n_splits"]) < 2:
        raise ValueError("n_splits must be at least 2.")
    return config
