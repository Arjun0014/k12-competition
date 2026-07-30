from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from trace_ace.bge_base_multiview_screen import (
    _encode,
    _load_encoder,
    assert_runtime,
    verify_sources,
)
from trace_ace.metrics import binary_metrics
from trace_ace.source_robust_validation import _current_rss_bytes
from trace_ace.timing_dynamics_validation import (
    _macro_cluster_bootstrap,
    load_component_oof,
)


PROTOCOL_ID = "E880_alter_math_success_transfer_v1"
SEED = 20260730
SOURCE_REVISION = "efaaa64e2dd08c67d9ebef3ab3141d7de28c5d40"
SOURCE_SHA256 = "50370bde7e0bb4ed691ca3a1bcf7533f7494966b63f3d6086224c66136d7d63b"
SOURCE_BYTES = 23_995_917
RAW_ROWS = 24_116
SESSION_ROWS = 2_318
SOURCE_COLUMNS = (
    "id",
    "id2",
    "content",
    "Success",
    "confirmatory feedback",
    "negative feedback",
    "correcting",
    "giving instruction",
    "giving explanation",
    "providing further references",
    "questioning",
    "asking for elaboration",
    "praising and encouraging",
    "managing frustration",
    "managing discussions",
    "giving answers",
    "encouraging peer tutoring",
    "guiding peer tutoring",
    "acknowledging tutor issue",
    "other",
    "irrelevant statement",
    "computational skill",
    "linguistic knowledge",
    "conceptual knowledge",
    "strategic knowledge",
    "affective control",
)
EXPECTED_FOLD_COUNTS = {
    0: (262, 166),
    1: (305, 191),
    2: (299, 167),
    3: (283, 192),
    4: (264, 189),
}
TASK_LINE = (
    "Task: represent whether the student's mathematics problem was successfully "
    "resolved during the discussion."
)
REGULARIZATION_C = 0.1
MAX_ITERATIONS = 1_000
BOOTSTRAP_REPLICATES = 5_000
GAMMAS = (0.10, 0.20, 0.30)
SELECTION_ENVIRONMENTS = ("V_seen", "V_objective", "V_style")
CONFIRMATION_ENVIRONMENT = "V_joint"
PROBABILITY_CLIP = 1e-6
V05_WEIGHTS = {
    "pred_full": 0.25,
    "pred_role": 0.25,
    "pred_bge_base": 0.50,
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root).resolve()
    return {
        "root": root,
        "raw": (
            root
            / "Datasets"
            / "ALTER_Math"
            / "LEVI_tutoring_dataset_round1.csv"
        ),
        "source_manifest": root / "E880_SOURCE_MANIFEST_2026-07-30.json",
        "sessions": root / "data_cache" / "alter_math_e880_sessions.parquet",
        "source_audit": root / "data_cache" / "alter_math_e880_source_audit.json",
        "embeddings": root / "data_cache" / "alter_math_e880_bge_base.npy",
        "embedding_meta": (
            root / "data_cache" / "alter_math_e880_bge_base_metadata.json"
        ),
        "probe": root / "models" / "alter_math_e880_success_probe.joblib",
        "evidence": (
            root / "data_cache" / "alter_math_e880_competition_evidence.parquet"
        ),
        "runs": root / "experiments" / "runs",
    }


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _runtime() -> dict[str, str]:
    observed = assert_runtime()
    observed["pandas"] = pd.__version__
    if observed["pandas"] != "2.3.3":
        raise RuntimeError(f"E880 pandas runtime changed: {observed['pandas']}")
    return observed


def source_fold(session_id: str | int) -> int:
    value = hashlib.sha256(f"E880|{session_id}".encode()).hexdigest()
    return int(value[:16], 16) % 5


def _normalize_document(value: object) -> str:
    text = str(value)
    if "*document*" not in text:
        raise ValueError("E880 source row lacks the *document* delimiter.")
    document = text.split("*document*", maxsplit=1)[1]
    return re.sub(r"\s+", " ", document).strip()


def _role_marker(raw_role: str) -> str:
    role = raw_role.strip().lower()
    if role.startswith("u"):
        return "[STUDENT]"
    if role.startswith("e"):
        return "[EXPERT_TUTOR]"
    if role.startswith("p"):
        return "[PEER_TUTOR]"
    return "[OTHER]"


def _document_turns(document: str) -> list[str]:
    raw_turns = re.split(r"\s+\*\s+", str(document).strip())
    turns: list[str] = []
    for raw in raw_turns:
        text = re.sub(r"\s+", " ", raw).strip()
        if not text:
            continue
        role, separator, utterance = text.partition(":")
        if not separator:
            marker = "[OTHER]"
            utterance = text
        else:
            marker = _role_marker(role)
        words = utterance.strip().split()[:16]
        if words:
            turns.append(f"{marker} {' '.join(words)}")
    return turns


def compact_document(document: str) -> str:
    turns = _document_turns(document)
    if not turns:
        return TASK_LINE
    ending_start = max(0, len(turns) - 8)
    ending_indices = list(range(ending_start, len(turns)))
    earlier_indices = [
        index for index in range(min(4, len(turns))) if index not in ending_indices
    ]
    parts = [TASK_LINE]
    if ending_indices:
        parts.append(
            "End of mathematics discussion: "
            + " ".join(turns[index] for index in ending_indices)
        )
    if earlier_indices:
        parts.append(
            "Earlier mathematics discussion: "
            + " ".join(turns[index] for index in earlier_indices)
        )
    return "\n".join(parts)


def build_source_sessions(raw: pd.DataFrame) -> pd.DataFrame:
    if tuple(raw.columns) != SOURCE_COLUMNS:
        raise ValueError("E880 source columns changed.")
    if len(raw) != RAW_ROWS:
        raise ValueError(f"E880 expected {RAW_ROWS} raw rows; found {len(raw)}.")
    if raw[["id", "id2"]].duplicated().any():
        raise ValueError("E880 source contains duplicate (id, id2) rows.")
    if set(raw["Success"].unique()) != {0, 1}:
        raise ValueError("E880 Success must contain exactly binary labels.")

    working = raw[["id", "id2", "content", "Success"]].copy()
    working["document"] = working["content"].map(_normalize_document)
    rows: list[dict[str, object]] = []
    for session_id, group in working.groupby("id", sort=True):
        ordered = group.sort_values("id2", kind="mergesort")
        expected_ids = np.arange(len(ordered), dtype=np.int64)
        observed_ids = ordered["id2"].to_numpy(dtype=np.int64)
        if not np.array_equal(observed_ids, expected_ids):
            raise ValueError(f"E880 id2 sequence changed for session {session_id}.")
        labels = ordered["Success"].unique()
        documents = ordered["document"].unique()
        if len(labels) != 1 or len(documents) != 1:
            raise ValueError(
                f"E880 label/document is not constant for session {session_id}."
            )
        document = str(documents[0])
        turns = _document_turns(document)
        rows.append(
            {
                "session_id": str(session_id),
                "target": int(labels[0]),
                "fold": source_fold(session_id),
                "turns": len(turns),
                "words": len(document.split()),
                "text": compact_document(document),
                "document_sha256": hashlib.sha256(
                    document.encode("utf-8")
                ).hexdigest(),
            }
        )
    sessions = pd.DataFrame(rows).sort_values(
        "session_id", kind="mergesort"
    ).reset_index(drop=True)
    if len(sessions) != SESSION_ROWS or sessions["session_id"].duplicated().any():
        raise ValueError("E880 session reconstruction row count changed.")
    if sessions["document_sha256"].duplicated().any():
        raise ValueError("E880 contains duplicate normalized session documents.")
    observed_counts = {
        int(fold): (
            int((group["target"] == 0).sum()),
            int((group["target"] == 1).sum()),
        )
        for fold, group in sessions.groupby("fold", sort=True)
    }
    if observed_counts != EXPECTED_FOLD_COUNTS:
        raise ValueError(f"E880 source fold counts changed: {observed_counts}")
    return sessions


def prepare_source(project_root: str | Path) -> dict[str, object]:
    paths = _paths(project_root)
    if not paths["raw"].is_file():
        raise FileNotFoundError(f"Missing E880 raw source: {paths['raw']}")
    if paths["raw"].stat().st_size != SOURCE_BYTES:
        raise ValueError("E880 raw source byte count changed.")
    if _sha256(paths["raw"]) != SOURCE_SHA256:
        raise ValueError("E880 raw source SHA-256 changed.")
    manifest = json.loads(paths["source_manifest"].read_text(encoding="utf-8"))
    if (
        manifest.get("file_sha256") != SOURCE_SHA256
        or manifest.get("hugging_face_revision") != SOURCE_REVISION
        or manifest.get("license") != "Apache-2.0"
    ):
        raise ValueError("E880 tracked source manifest changed.")
    raw = pd.read_csv(paths["raw"])
    sessions = build_source_sessions(raw)
    paths["sessions"].parent.mkdir(parents=True, exist_ok=True)
    sessions.to_parquet(paths["sessions"], index=False)
    audit = {
        "protocol_id": PROTOCOL_ID,
        "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "raw_rows": int(len(raw)),
        "raw_columns": int(len(raw.columns)),
        "sessions": int(len(sessions)),
        "negative_sessions": int((sessions["target"] == 0).sum()),
        "positive_sessions": int((sessions["target"] == 1).sum()),
        "fold_counts_negative_positive": {
            str(key): list(value) for key, value in EXPECTED_FOLD_COUNTS.items()
        },
        "turns": {
            "minimum": int(sessions["turns"].min()),
            "median": float(sessions["turns"].median()),
            "maximum": int(sessions["turns"].max()),
        },
        "words": {
            "minimum": int(sessions["words"].min()),
            "median": float(sessions["words"].median()),
            "mean": float(sessions["words"].mean()),
            "maximum": int(sessions["words"].max()),
        },
        "session_cache": str(paths["sessions"].relative_to(paths["root"])),
        "session_cache_sha256": _sha256(paths["sessions"]),
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    _json_write(paths["source_audit"], audit)
    audit["source_audit_sha256"] = _sha256(paths["source_audit"])
    return audit


def synthetic_benchmark(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    runtime = _runtime()
    verify_sources(project_root)
    documents = []
    for index in range(32):
        outcome = (
            "I understand why the denominator stays the same."
            if index % 2
            else "I am still unsure which operation to use."
        )
        documents.append(
            "u1: How do I solve this fraction problem? * "
            "e1: Start by finding a common denominator. * "
            f"u1: {outcome} * "
            "e1: Explain the next step in your own words."
        )
    texts = [compact_document(document) for document in documents]
    model = _load_encoder(project_root)
    values = _encode(model, texts)
    runtime_seconds = float(time.perf_counter() - started)
    projected_seconds = runtime_seconds * SESSION_ROWS / len(texts)
    rss_bytes = int(_current_rss_bytes())
    clauses = {
        "shape_32_by_768": values.shape == (32, 768),
        "finite": bool(np.isfinite(values).all()),
        "unit_normalized": bool(
            np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=2e-5, rtol=0)
        ),
        "projected_under_30_minutes": projected_seconds < 1_800,
        "rss_under_8_gib": rss_bytes < 8 * 1024**3,
    }
    result = {
        "protocol_id": PROTOCOL_ID,
        "benchmark_rows": len(texts),
        "runtime_seconds": runtime_seconds,
        "projected_source_seconds": projected_seconds,
        "rss_bytes": rss_bytes,
        "runtime": runtime,
        "clauses": clauses,
        "passes_benchmark": bool(all(clauses.values())),
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    if not result["passes_benchmark"]:
        raise RuntimeError(f"E880 synthetic benchmark failed: {result}")
    return result


def build_embedding_cache(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    paths = _paths(project_root)
    runtime = _runtime()
    verify_sources(project_root)
    sessions = pd.read_parquet(paths["sessions"])
    if (
        len(sessions) != SESSION_ROWS
        or sessions["session_id"].duplicated().any()
        or list(sessions.columns)
        != [
            "session_id",
            "target",
            "fold",
            "turns",
            "words",
            "text",
            "document_sha256",
        ]
    ):
        raise ValueError("E880 prepared session cache changed.")
    model = _load_encoder(project_root)
    embeddings = _encode(model, sessions["text"].astype(str).tolist())
    paths["embeddings"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["embeddings"], embeddings)
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "rows": int(len(embeddings)),
        "columns": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "runtime_seconds": float(time.perf_counter() - started),
        "runtime": runtime,
        "source_session_cache_sha256": _sha256(paths["sessions"]),
        "embedding_cache_sha256": _sha256(paths["embeddings"]),
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    _json_write(paths["embedding_meta"], metadata)
    metadata["metadata_sha256"] = _sha256(paths["embedding_meta"])
    return metadata


def _loss_rows(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    y = np.asarray(target, dtype=np.float64)
    p = np.clip(
        np.asarray(probability, dtype=np.float64),
        PROBABILITY_CLIP,
        1.0 - PROBABILITY_CLIP,
    )
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def _external_bootstrap(
    target: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
) -> dict[str, float | int]:
    gain = _loss_rows(target, baseline) - _loss_rows(target, candidate)
    rng = np.random.default_rng(seed)
    sampled = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 128):
        stop = min(start + 128, replicates)
        indices = rng.integers(0, len(gain), size=(stop - start, len(gain)))
        sampled[start:stop] = gain[indices].mean(axis=1)
    return {
        "replicates": int(replicates),
        "observed_gain": float(gain.mean()),
        "bootstrap_mean_gain": float(sampled.mean()),
        "ci_lower": float(np.quantile(sampled, 0.025)),
        "ci_upper": float(np.quantile(sampled, 0.975)),
        "support_positive_gain": float(np.mean(sampled > 0.0)),
    }


def _new_probe() -> LogisticRegression:
    return LogisticRegression(
        C=REGULARIZATION_C,
        solver="lbfgs",
        max_iter=MAX_ITERATIONS,
        random_state=SEED,
    )


def external_oof(
    sessions: pd.DataFrame,
    embeddings: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if embeddings.shape != (SESSION_ROWS, 768):
        raise ValueError(f"E880 external embedding shape changed: {embeddings.shape}")
    target = sessions["target"].to_numpy(dtype=np.int8)
    folds = sessions["fold"].to_numpy(dtype=np.int8)
    candidate = np.full(SESSION_ROWS, np.nan, dtype=np.float64)
    baseline = np.full(SESSION_ROWS, np.nan, dtype=np.float64)
    fit_rows: list[dict[str, object]] = []
    for fold in range(5):
        validation = folds == fold
        training = ~validation
        model = _new_probe()
        model.fit(embeddings[training], target[training])
        candidate[validation] = model.predict_proba(embeddings[validation])[:, 1]
        baseline[validation] = float(target[training].mean())
        candidate_metrics = binary_metrics(target[validation], candidate[validation])
        baseline_metrics = binary_metrics(target[validation], baseline[validation])
        fit_rows.append(
            {
                "fold": fold,
                "training_sessions": int(training.sum()),
                "validation_sessions": int(validation.sum()),
                "training_prevalence": float(target[training].mean()),
                "candidate_log_loss": candidate_metrics["log_loss"],
                "baseline_log_loss": baseline_metrics["log_loss"],
                "log_loss_gain": (
                    baseline_metrics["log_loss"] - candidate_metrics["log_loss"]
                ),
                "candidate_roc_auc": candidate_metrics["roc_auc"],
                "candidate_brier_score": candidate_metrics["brier_score"],
                "candidate_ece_10": candidate_metrics["ece_10"],
                "iterations": int(model.n_iter_[0]),
            }
        )
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise RuntimeError("E880 external OOF predictions are incomplete.")
    scored = sessions[
        ["session_id", "target", "fold", "turns", "words"]
    ].copy()
    scored["baseline_prediction"] = baseline
    scored["candidate_prediction"] = candidate
    candidate_metrics = binary_metrics(target, candidate)
    baseline_metrics = binary_metrics(target, baseline)
    bootstrap = _external_bootstrap(target, baseline, candidate)
    macro_f1 = float(f1_score(target, candidate >= 0.5, average="macro"))
    worst_fold_gain = min(row["log_loss_gain"] for row in fit_rows)
    clauses = {
        "candidate_log_loss_at_most_0_59": (
            candidate_metrics["log_loss"] <= 0.59
        ),
        "log_loss_gain_at_least_0_07": (
            baseline_metrics["log_loss"] - candidate_metrics["log_loss"] >= 0.07
        ),
        "roc_auc_at_least_0_75": candidate_metrics["roc_auc"] >= 0.75,
        "brier_gain_at_least_0_02": (
            baseline_metrics["brier_score"]
            - candidate_metrics["brier_score"]
            >= 0.02
        ),
        "macro_f1_at_least_0_65": macro_f1 >= 0.65,
        "ece_10_at_most_0_08": candidate_metrics["ece_10"] <= 0.08,
        "all_five_folds_improve_log_loss": all(
            row["log_loss_gain"] > 0.0 for row in fit_rows
        ),
        "worst_fold_gain_at_least_0_01": worst_fold_gain >= 0.01,
        "bootstrap_support_at_least_0_99": (
            bootstrap["support_positive_gain"] >= 0.99
        ),
        "bootstrap_lower_bound_at_least_0_03": bootstrap["ci_lower"] >= 0.03,
    }
    summary: dict[str, object] = {
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "log_loss_gain": (
            baseline_metrics["log_loss"] - candidate_metrics["log_loss"]
        ),
        "brier_gain": (
            baseline_metrics["brier_score"] - candidate_metrics["brier_score"]
        ),
        "macro_f1": macro_f1,
        "folds": fit_rows,
        "worst_fold_log_loss_gain": worst_fold_gain,
        "session_bootstrap": bootstrap,
        "gate_clauses": clauses,
        "passes_external_gate": bool(all(clauses.values())),
    }
    return scored, summary


def _evidence_from_margin(
    margin: np.ndarray,
    *,
    median: float,
    iqr: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(iqr) or iqr <= 1e-8:
        raise ValueError("E880 external margin IQR is invalid.")
    z = np.clip((np.asarray(margin, dtype=np.float64) - median) / iqr, -4.0, 4.0)
    return z, np.tanh(z / 2.0)


def _applicability(evidence: np.ndarray) -> dict[str, object]:
    values = np.asarray(evidence, dtype=np.float64)
    median = float(np.median(values))
    iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    standard_deviation = float(values.std(ddof=0))
    saturation = float(np.mean(np.abs(values) > 0.95))
    clauses = {
        "all_35072_rows_finite": (
            values.shape == (35_072,) and bool(np.isfinite(values).all())
        ),
        "standard_deviation_at_least_0_10": standard_deviation >= 0.10,
        "iqr_at_least_0_20": iqr >= 0.20,
        "absolute_median_at_most_0_75": abs(median) <= 0.75,
        "saturation_at_most_0_20": saturation <= 0.20,
    }
    return {
        "rows": int(len(values)),
        "median": median,
        "iqr": iqr,
        "standard_deviation": standard_deviation,
        "saturation_fraction": saturation,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "clauses": clauses,
        "passes_applicability_gate": bool(all(clauses.values())),
    }


def validate_external(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    paths = _paths(project_root)
    runtime = _runtime()
    verify_sources(project_root)
    sessions = pd.read_parquet(paths["sessions"])
    embeddings = np.load(paths["embeddings"])
    scored, summary = external_oof(sessions, embeddings)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_e880_alter_math")
    run_dir = paths["runs"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    oof_path = run_dir / "external_oof_predictions.parquet"
    report_path = run_dir / "report.json"
    scored.to_parquet(oof_path, index=False)

    artifacts: dict[str, dict[str, object]] = {
        "external_oof_predictions": {
            "path": str(oof_path.relative_to(paths["root"])),
            "sha256": _sha256(oof_path),
        },
        "external_embeddings": {
            "path": str(paths["embeddings"].relative_to(paths["root"])),
            "sha256": _sha256(paths["embeddings"]),
        },
        "prepared_sessions": {
            "path": str(paths["sessions"].relative_to(paths["root"])),
            "sha256": _sha256(paths["sessions"]),
        },
    }
    applicability: dict[str, object] | None = None
    if summary["passes_external_gate"]:
        target = sessions["target"].to_numpy(dtype=np.int8)
        model = _new_probe()
        model.fit(embeddings, target)
        external_margin = model.decision_function(embeddings)
        margin_median = float(np.median(external_margin))
        margin_iqr = float(
            np.quantile(external_margin, 0.75)
            - np.quantile(external_margin, 0.25)
        )
        checkpoint = {
            "protocol_id": PROTOCOL_ID,
            "source_revision": SOURCE_REVISION,
            "source_sha256": SOURCE_SHA256,
            "regularization_c": REGULARIZATION_C,
            "solver": "lbfgs",
            "max_iterations": MAX_ITERATIONS,
            "seed": SEED,
            "coef": model.coef_[0].astype(np.float64),
            "intercept": float(model.intercept_[0]),
            "margin_median": margin_median,
            "margin_iqr": margin_iqr,
            "embedding_dimension": 768,
        }
        paths["probe"].parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(checkpoint, paths["probe"], compress=3)

        modeling = pd.read_parquet(
            paths["root"] / "data_cache" / "modeling_base.parquet",
            columns=["response_id"],
        )
        competition_embeddings = np.load(
            paths["root"] / "data_cache" / "bge_base_context_256.npy",
            mmap_mode="r",
        )
        if competition_embeddings.shape != (35_072, 768):
            raise ValueError("E880 competition BGE context cache shape changed.")
        competition_margin = (
            np.asarray(competition_embeddings @ checkpoint["coef"])
            + checkpoint["intercept"]
        )
        z, evidence = _evidence_from_margin(
            competition_margin,
            median=margin_median,
            iqr=margin_iqr,
        )
        applicability = _applicability(evidence)
        evidence_frame = modeling.copy()
        evidence_frame["external_margin"] = competition_margin
        evidence_frame["external_z"] = z
        evidence_frame["external_evidence"] = evidence
        evidence_frame.to_parquet(paths["evidence"], index=False)
        artifacts["probe_checkpoint"] = {
            "path": str(paths["probe"].relative_to(paths["root"])),
            "sha256": _sha256(paths["probe"]),
        }
        artifacts["competition_evidence"] = {
            "path": str(paths["evidence"].relative_to(paths["root"])),
            "sha256": _sha256(paths["evidence"]),
        }

    passes_transfer = bool(
        summary["passes_external_gate"]
        and applicability is not None
        and applicability["passes_applicability_gate"]
    )
    report: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": "ALTER-Math/annotated-math-tutoring-dataset",
            "revision": SOURCE_REVISION,
            "license": "Apache-2.0",
            "raw_sha256": SOURCE_SHA256,
            "sessions": SESSION_ROWS,
        },
        "configuration": {
            "seed": SEED,
            "fold": "sha256('E880|' + session_id) modulo 5",
            "regularization_c": REGULARIZATION_C,
            "solver": "lbfgs",
            "max_iterations": MAX_ITERATIONS,
            "encoder": "BAAI/bge-base-en-v1.5",
            "embedding_dimension": 768,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "external_validation": summary,
        "competition_target_free_applicability": applicability,
        "passes_external_and_applicability_gate": passes_transfer,
        "runtime_seconds": float(time.perf_counter() - started),
        "environment": runtime,
        "artifacts": artifacts,
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    _json_write(report_path, report)
    report["report_path"] = str(report_path.relative_to(paths["root"]))
    report["report_sha256"] = _sha256(report_path)
    return report


def build_candidate(
    baseline: np.ndarray,
    evidence: np.ndarray,
    gamma: float,
) -> np.ndarray:
    raw = np.clip(
        np.asarray(baseline, dtype=np.float64),
        PROBABILITY_CLIP,
        1.0 - PROBABILITY_CLIP,
    )
    transfer = np.asarray(evidence, dtype=np.float64)
    if raw.shape != transfer.shape or not np.isfinite(transfer).all():
        raise ValueError("E880 baseline/evidence values are invalid.")
    return np.clip(
        expit(logit(raw) + float(gamma) * transfer),
        PROBABILITY_CLIP,
        1.0 - PROBABILITY_CLIP,
    )


def _baseline(component: pd.DataFrame) -> np.ndarray:
    return sum(
        weight * component[name].to_numpy(dtype=np.float64)
        for name, weight in V05_WEIGHTS.items()
    )


def _load_bound_evidence(
    project_root: str | Path,
    binding_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    root = Path(project_root).resolve()
    binding = json.loads(Path(binding_path).resolve().read_text(encoding="utf-8"))
    required = {
        "protocol_id": PROTOCOL_ID,
        "external_gate_passed": True,
        "applicability_gate_passed": True,
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    observed = {key: binding.get(key) for key in required}
    if observed != required:
        raise ValueError(f"E880 competition binding contract changed: {observed}")
    for name in ("external_report", "probe_checkpoint", "competition_evidence"):
        artifact = binding.get("artifacts", {}).get(name)
        if not isinstance(artifact, dict):
            raise ValueError(f"E880 binding lacks artifact {name}.")
        path = root / str(artifact["path"])
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"E880 bound artifact changed: {name}.")
    report_path = root / binding["artifacts"]["external_report"]["path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("passes_external_and_applicability_gate"):
        raise ValueError("E880 bound external report does not pass.")
    evidence_path = root / binding["artifacts"]["competition_evidence"]["path"]
    evidence = pd.read_parquet(evidence_path)
    if (
        list(evidence.columns)
        != [
            "response_id",
            "external_margin",
            "external_z",
            "external_evidence",
        ]
        or len(evidence) != 35_072
        or evidence["response_id"].duplicated().any()
    ):
        raise ValueError("E880 bound evidence schema changed.")
    return evidence, binding


def _environment_predictions(
    component: pd.DataFrame,
    evidence: pd.DataFrame,
    gammas: Sequence[float],
) -> pd.DataFrame:
    aligned = component[["response_id"]].merge(
        evidence[["response_id", "external_evidence"]],
        on="response_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if aligned["external_evidence"].isna().any():
        raise ValueError("E880 evidence does not align with component OOF.")
    base = _baseline(component)
    metadata = component[
        [
            "environment",
            "response_id",
            "session_id",
            "learning_objective_id",
            "semantic_family",
            "fold",
            "evaluation_eligible",
            "target",
        ]
    ].copy()
    rows: list[pd.DataFrame] = []
    for gamma in gammas:
        candidate = metadata.copy()
        candidate["gamma"] = float(gamma)
        candidate["pred_v05_raw"] = base
        candidate["external_evidence"] = aligned[
            "external_evidence"
        ].to_numpy(dtype=np.float64)
        candidate["prediction"] = build_candidate(
            base,
            candidate["external_evidence"].to_numpy(),
            gamma,
        )
        rows.append(candidate)
    return pd.concat(rows, ignore_index=True)


def _metric_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    for (environment, gamma, fold), group in predictions.groupby(
        ["environment", "gamma", "fold"],
        sort=True,
    ):
        scored = group.loc[group["evaluation_eligible"].astype(bool)]
        target = scored["target"].to_numpy(dtype=np.int8)
        candidate = binary_metrics(target, scored["prediction"])
        baseline = binary_metrics(target, scored["pred_v05_raw"])
        row: dict[str, object] = {
            "environment": str(environment),
            "gamma": float(gamma),
            "fold": int(fold),
            "rows": int(len(scored)),
        }
        for name, value in candidate.items():
            row[name] = value
            row[f"baseline_{name}"] = baseline[name]
            row[f"delta_{name}_vs_v05"] = value - baseline[name]
        fold_rows.append(row)
    folds = pd.DataFrame(fold_rows)
    metric_columns = [
        "log_loss",
        "roc_auc",
        "brier_score",
        "ece_10",
        "prediction_mean",
    ]
    aggregate = [
        *metric_columns,
        *[f"baseline_{name}" for name in metric_columns],
        *[f"delta_{name}_vs_v05" for name in metric_columns],
    ]
    environments = (
        folds.groupby(["environment", "gamma"], as_index=False)[aggregate]
        .mean()
        .sort_values(["environment", "gamma"], kind="mergesort")
        .reset_index(drop=True)
    )
    environments["aggregation"] = "equal_fold_macro"
    return folds, environments


def _development_selection(
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
    environments: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object], float | None]:
    rows: list[dict[str, object]] = []
    bootstraps: dict[str, object] = {}
    for index, gamma in enumerate(GAMMAS):
        gamma_predictions = predictions.loc[predictions["gamma"].eq(gamma)]
        gamma_environments = environments.loc[environments["gamma"].eq(gamma)]
        gamma_folds = folds.loc[folds["gamma"].eq(gamma)]
        session_bootstrap = _macro_cluster_bootstrap(
            gamma_predictions,
            candidate_column="prediction",
            baseline_column="pred_v05_raw",
            group_column="session_id",
            n_replicates=BOOTSTRAP_REPLICATES,
            seed=SEED + index,
        )
        family_bootstrap = _macro_cluster_bootstrap(
            gamma_predictions,
            candidate_column="prediction",
            baseline_column="pred_v05_raw",
            group_column="semantic_family",
            n_replicates=BOOTSTRAP_REPLICATES,
            seed=SEED + 100 + index,
        )
        bootstraps[f"{gamma:.2f}"] = {
            "session": session_bootstrap,
            "semantic_family": family_bootstrap,
        }
        loss_delta = gamma_environments[
            "delta_log_loss_vs_v05"
        ].to_numpy(dtype=np.float64)
        clauses = {
            "mean_log_loss_gain_at_least_0_0016": (
                -float(loss_delta.mean()) >= 0.0016
            ),
            "all_three_environments_improve": bool(np.all(loss_delta < 0.0)),
            "worst_environment_no_regression": float(loss_delta.max()) <= 0.0,
            "mean_auroc_non_regression": (
                float(
                    gamma_environments["delta_roc_auc_vs_v05"].mean()
                )
                >= 0.0
            ),
            "mean_brier_non_regression": (
                float(
                    gamma_environments["delta_brier_score_vs_v05"].mean()
                )
                <= 0.0
            ),
            "mean_ece_non_regression": (
                float(gamma_environments["delta_ece_10_vs_v05"].mean()) <= 0.0
            ),
            "worst_fold_loss_regression_at_most_0_0005": (
                float(gamma_folds["delta_log_loss_vs_v05"].max()) <= 0.0005
            ),
            "session_bootstrap_support_at_least_0_95": (
                session_bootstrap["support_positive_gain"] >= 0.95
            ),
            "session_bootstrap_lower_bound_positive": (
                session_bootstrap["ci_lower"] > 0.0
            ),
            "family_bootstrap_support_at_least_0_95": (
                family_bootstrap["support_positive_gain"] >= 0.95
            ),
            "family_bootstrap_lower_bound_positive": (
                family_bootstrap["ci_lower"] > 0.0
            ),
        }
        rows.append(
            {
                "gamma": gamma,
                "mean_log_loss_gain_vs_v05": -float(loss_delta.mean()),
                "improved_environments": int(np.sum(loss_delta < 0.0)),
                "worst_environment_delta_log_loss": float(loss_delta.max()),
                "mean_delta_roc_auc": float(
                    gamma_environments["delta_roc_auc_vs_v05"].mean()
                ),
                "mean_delta_brier_score": float(
                    gamma_environments["delta_brier_score_vs_v05"].mean()
                ),
                "mean_delta_ece_10": float(
                    gamma_environments["delta_ece_10_vs_v05"].mean()
                ),
                "worst_fold_delta_log_loss": float(
                    gamma_folds["delta_log_loss_vs_v05"].max()
                ),
                "session_bootstrap_support": session_bootstrap[
                    "support_positive_gain"
                ],
                "semantic_family_bootstrap_support": family_bootstrap[
                    "support_positive_gain"
                ],
                "passes_gate": bool(all(clauses.values())),
                "gate_clauses": clauses,
            }
        )
    selection = pd.DataFrame(rows).sort_values(
        ["gamma"], kind="mergesort"
    ).reset_index(drop=True)
    passing = selection.loc[selection["passes_gate"]].sort_values(
        ["mean_log_loss_gain_vs_v05", "gamma"],
        ascending=[False, True],
        kind="mergesort",
    )
    selected = None if passing.empty else float(passing.iloc[0]["gamma"])
    return selection, bootstraps, selected


def _confirmation_gate(
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
    environments: pd.DataFrame,
    gamma: float,
) -> dict[str, object]:
    environment = environments.iloc[0]
    session_bootstrap = _macro_cluster_bootstrap(
        predictions,
        candidate_column="prediction",
        baseline_column="pred_v05_raw",
        group_column="session_id",
        n_replicates=BOOTSTRAP_REPLICATES,
        seed=SEED + 1_000,
    )
    family_bootstrap = _macro_cluster_bootstrap(
        predictions,
        candidate_column="prediction",
        baseline_column="pred_v05_raw",
        group_column="semantic_family",
        n_replicates=BOOTSTRAP_REPLICATES,
        seed=SEED + 1_100,
    )
    clauses = {
        "log_loss_non_regression": (
            float(environment["delta_log_loss_vs_v05"]) <= 0.0
        ),
        "auroc_non_regression": (
            float(environment["delta_roc_auc_vs_v05"]) >= 0.0
        ),
        "brier_non_regression": (
            float(environment["delta_brier_score_vs_v05"]) <= 0.0
        ),
        "ece_non_regression": (
            float(environment["delta_ece_10_vs_v05"]) <= 0.0
        ),
        "worst_fold_loss_regression_at_most_0_0005": (
            float(folds["delta_log_loss_vs_v05"].max()) <= 0.0005
        ),
        "session_bootstrap_support_at_least_0_90": (
            session_bootstrap["support_positive_gain"] >= 0.90
        ),
        "session_bootstrap_lower_bound_positive": (
            session_bootstrap["ci_lower"] > 0.0
        ),
        "family_bootstrap_support_at_least_0_90": (
            family_bootstrap["support_positive_gain"] >= 0.90
        ),
        "family_bootstrap_lower_bound_positive": (
            family_bootstrap["ci_lower"] > 0.0
        ),
    }
    return {
        "gamma": gamma,
        "environment_metrics": environment.to_dict(),
        "worst_fold_delta_log_loss": float(
            folds["delta_log_loss_vs_v05"].max()
        ),
        "session_bootstrap": session_bootstrap,
        "semantic_family_bootstrap": family_bootstrap,
        "gate_clauses": clauses,
        "passes_confirmation_gate": bool(all(clauses.values())),
    }


def validate_competition(
    project_root: str | Path,
    binding_path: str | Path,
) -> dict[str, object]:
    started = time.perf_counter()
    paths = _paths(project_root)
    runtime = _runtime()
    evidence, binding = _load_bound_evidence(project_root, binding_path)

    development_parts: list[pd.DataFrame] = []
    for environment in SELECTION_ENVIRONMENTS:
        component = load_component_oof(project_root, environment)
        development_parts.append(
            _environment_predictions(component, evidence, GAMMAS)
        )
    development_predictions = pd.concat(development_parts, ignore_index=True)
    development_folds, development_environments = _metric_tables(
        development_predictions
    )
    selection, bootstraps, selected_gamma = _development_selection(
        development_predictions,
        development_folds,
        development_environments,
    )

    confirmation: dict[str, object] | None = None
    confirmation_predictions: pd.DataFrame | None = None
    confirmation_folds: pd.DataFrame | None = None
    confirmation_environments: pd.DataFrame | None = None
    if selected_gamma is not None:
        joint = load_component_oof(project_root, CONFIRMATION_ENVIRONMENT)
        confirmation_predictions = _environment_predictions(
            joint, evidence, (selected_gamma,)
        )
        confirmation_folds, confirmation_environments = _metric_tables(
            confirmation_predictions
        )
        confirmation = _confirmation_gate(
            confirmation_predictions,
            confirmation_folds,
            confirmation_environments,
            selected_gamma,
        )

    passes = bool(
        selected_gamma is not None
        and confirmation is not None
        and confirmation["passes_confirmation_gate"]
    )
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ_e880_competition_validation"
    )
    run_dir = paths["runs"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    prediction_path = run_dir / "development_predictions.parquet"
    folds_path = run_dir / "development_fold_metrics.csv"
    environments_path = run_dir / "development_environment_metrics.csv"
    selection_path = run_dir / "development_selection.json"
    bootstrap_path = run_dir / "development_bootstraps.json"
    report_path = run_dir / "report.json"
    development_predictions.to_parquet(prediction_path, index=False)
    development_folds.to_csv(folds_path, index=False, lineterminator="\n")
    development_environments.to_csv(
        environments_path, index=False, lineterminator="\n"
    )
    _json_write(
        selection_path,
        {
            "rows": selection.to_dict(orient="records"),
            "selected_gamma": selected_gamma,
        },
    )
    _json_write(bootstrap_path, bootstraps)
    artifacts: dict[str, dict[str, str]] = {}
    for name, artifact_path in (
        ("development_predictions", prediction_path),
        ("development_fold_metrics", folds_path),
        ("development_environment_metrics", environments_path),
        ("development_selection", selection_path),
        ("development_bootstraps", bootstrap_path),
    ):
        artifacts[name] = {
            "path": str(artifact_path.relative_to(paths["root"])),
            "sha256": _sha256(artifact_path),
        }
    if confirmation_predictions is not None:
        confirmation_prediction_path = run_dir / "confirmation_predictions.parquet"
        confirmation_folds_path = run_dir / "confirmation_fold_metrics.csv"
        confirmation_environments_path = (
            run_dir / "confirmation_environment_metrics.csv"
        )
        confirmation_predictions.to_parquet(
            confirmation_prediction_path, index=False
        )
        assert confirmation_folds is not None
        assert confirmation_environments is not None
        confirmation_folds.to_csv(
            confirmation_folds_path, index=False, lineterminator="\n"
        )
        confirmation_environments.to_csv(
            confirmation_environments_path,
            index=False,
            lineterminator="\n",
        )
        for name, artifact_path in (
            ("confirmation_predictions", confirmation_prediction_path),
            ("confirmation_fold_metrics", confirmation_folds_path),
            ("confirmation_environment_metrics", confirmation_environments_path),
        ):
            artifacts[name] = {
                "path": str(artifact_path.relative_to(paths["root"])),
                "sha256": _sha256(artifact_path),
            }

    report: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "binding": binding,
        "configuration": {
            "gammas": list(GAMMAS),
            "formula": "sigmoid(logit(raw_v05) + gamma * external_evidence)",
            "selection_environments": list(SELECTION_ENVIRONMENTS),
            "confirmation_environment": CONFIRMATION_ENVIRONMENT,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "seed": SEED,
        },
        "development_environment_metrics": development_environments.to_dict(
            orient="records"
        ),
        "development_selection": selection.to_dict(orient="records"),
        "development_bootstraps": bootstraps,
        "selected_gamma": selected_gamma,
        "confirmation": confirmation,
        "passes_development_and_confirmation_gates": passes,
        "projected_public_loss_from_0_6054": (
            None
            if selected_gamma is None
            else 0.6054
            - float(
                selection.loc[
                    selection["gamma"].eq(selected_gamma),
                    "mean_log_loss_gain_vs_v05",
                ].iloc[0]
            )
        ),
        "runtime_seconds": float(time.perf_counter() - started),
        "environment": runtime,
        "artifacts": artifacts,
        "competition_outcomes_accessed": True,
        "V_joint_accessed": selected_gamma is not None,
        "V_final_accessed": False,
        "decision": "accept_for_packaging" if passes else "reject_without_rescue",
    }
    _json_write(report_path, report)
    report["report_path"] = str(report_path.relative_to(paths["root"]))
    report["report_sha256"] = _sha256(report_path)
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen E880 ALTER-Math success transfer stages."
    )
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "benchmark",
            "build-cache",
            "validate-external",
            "validate-competition",
        ),
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--binding")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.stage == "prepare":
        result = prepare_source(args.project_root)
    elif args.stage == "benchmark":
        result = synthetic_benchmark(args.project_root)
    elif args.stage == "build-cache":
        result = build_embedding_cache(args.project_root)
    elif args.stage == "validate-external":
        result = validate_external(args.project_root)
    else:
        if not args.binding:
            parser.error("--binding is required for validate-competition")
        result = validate_competition(args.project_root, args.binding)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
