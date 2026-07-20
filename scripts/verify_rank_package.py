from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    package = root / "submission_builds" / "final_ensemble_v04_rank.zip"
    smoke_data = root / "tmp" / "final_ensemble_v04_rank_submission" / "data"
    replay = root / "tmp" / "final_ensemble_v04_rank_replay"
    if not package.exists() or not smoke_data.is_dir():
        raise FileNotFoundError("Rank package or smoke data is missing.")
    if replay.exists():
        shutil.rmtree(replay)
    replay.mkdir(parents=True)

    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        if "main.py" not in names or any(name.startswith("data/") for name in names):
            raise RuntimeError("Rank ZIP has an invalid root or contains prohibited data.")
        required = {
            "assets/final_ensemble_v04_rank.joblib",
            "assets/final_ensemble_v04_rank.metadata.json",
            "assets/supervised_delta.safetensors",
        }
        if not required.issubset(names):
            raise RuntimeError(f"Rank ZIP is missing assets: {sorted(required - set(names))}")
        if not any(name.startswith("assets/bge-base-en-v1.5/") for name in names):
            raise RuntimeError("Rank ZIP is missing BGE-base assets.")
        if not any(name.startswith("assets/bge-small-en-v1.5/") for name in names):
            raise RuntimeError("Rank ZIP is missing BGE-small assets.")
        archive.extractall(replay)
    shutil.copytree(smoke_data, replay / "data")

    expected = pd.read_csv(
        replay / "data" / "submission_format.csv", dtype={"response_id": "string"}
    )
    outputs: list[bytes] = []
    probabilities: list[np.ndarray] = []
    elapsed: list[float] = []
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, "main.py"],
            cwd=replay,
            check=True,
            capture_output=True,
            text=True,
        )
        if completed.stdout or completed.stderr:
            raise RuntimeError("Rank package emitted forbidden runtime output.")
        output_path = replay / "submission.csv"
        raw = output_path.read_bytes()
        frame = pd.read_csv(output_path, dtype={"response_id": "string"})
        if list(frame.columns) != ["response_id", "probability"]:
            raise RuntimeError("Rank package output schema is invalid.")
        if not frame["response_id"].equals(expected["response_id"]):
            raise RuntimeError("Rank package output order does not match submission format.")
        values = frame["probability"].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not ((values >= 0.0) & (values <= 1.0)).all():
            raise RuntimeError("Rank package emitted invalid probabilities.")
        outputs.append(raw)
        probabilities.append(values)
        elapsed.append(float(completed.returncode))

    max_probability_difference = float(
        np.max(np.abs(probabilities[0] - probabilities[1]))
    )
    if outputs[0] != outputs[1] or max_probability_difference != 0.0:
        raise RuntimeError("Rank package failed exact deterministic replay.")

    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "package": package.name,
        "package_bytes": package.stat().st_size,
        "package_sha256": sha256(package),
        "archive_members": len(names),
        "smoke_rows": len(expected),
        "replay_count": 2,
        "stdout_stderr_empty": True,
        "output_bytes_identical": True,
        "max_probability_difference": max_probability_difference,
        "probability_min": float(probabilities[0].min()),
        "probability_max": float(probabilities[0].max()),
        "probability_mean": float(probabilities[0].mean()),
    }
    report_path = package.with_suffix(".verification.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
