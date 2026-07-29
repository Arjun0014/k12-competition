from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss


PROTOCOL_ID = "E850_lexical_speaker_role_denoising_target_free_v1"
SEED = 20260730
EXPECTED_UTTERANCE_SHA256 = (
    "80a231d18dbb0641989ebb972f08988f0ddd3cb7c4fb769ad2fe90b5a98ba5a4"
)
EXPECTED_STYLE_SHA256 = (
    "600664b9ce6c3809ff4b17206397285cfdd905a203a5c5f4d6969b403dfc9b40"
)
SAMPLE_SESSION_MODULUS = 24
N_FEATURES = 2**18
ALPHA = 1e-5
MAX_ITERATIONS = 20
CONFIDENCE_THRESHOLD = 0.98
INFERENCE_BATCH_ROWS = 100_000
ROLE_NAMES = ("student", "tutor")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_integer(value: str, namespace: str) -> int:
    payload = f"E850|{namespace}|{value}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _session_selected(session_id: str) -> bool:
    return _stable_integer(session_id, "sample") % SAMPLE_SESSION_MODULUS == 0


def _session_fold(session_id: str) -> int:
    return _stable_integer(session_id, "fold") % 5


def enrich_session_text(group: pd.DataFrame) -> list[str]:
    ordered = group.sort_values("utterance_id", kind="mergesort")
    roles = ordered["role"].astype(str).str.casefold().tolist()
    contents = ordered["content"].fillna("").astype(str).tolist()
    output: list[str] = []
    last_index = len(roles) - 1
    for index, content in enumerate(contents):
        previous = roles[index - 1] if index > 0 else "boundary"
        following = roles[index + 1] if index < last_index else "boundary"
        position = index / max(last_index, 1)
        position_token = (
            "opening"
            if position < 0.2
            else "closing"
            if position >= 0.8
            else "middle"
        )
        word_count = len(content.split())
        length_token = (
            "very_short"
            if word_count <= 2
            else "short"
            if word_count <= 8
            else "medium"
            if word_count <= 30
            else "long"
        )
        question_token = "has_question" if "?" in content else "no_question"
        output.append(
            " ".join(
                [
                    f"prev_role_{previous}",
                    f"next_role_{following}",
                    f"position_{position_token}",
                    f"length_{length_token}",
                    question_token,
                    content,
                ]
            )
        )
    return output


def _vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        n_features=N_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
    )


def _classifier(seed: int) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=ALPHA,
        max_iter=MAX_ITERATIONS,
        tol=1e-4,
        shuffle=True,
        random_state=seed,
        average=True,
    )


def _top_label_ece(target: np.ndarray, probability: np.ndarray) -> float:
    predicted = (probability >= 0.5).astype(np.int8)
    confidence = np.maximum(probability, 1.0 - probability)
    correct = (predicted == target).astype(np.float64)
    edges = np.linspace(0.5, 1.0, 11)
    total = len(target)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence >= lower) & (
            confidence <= upper if upper == 1.0 else confidence < upper
        )
        if mask.any():
            value += (mask.sum() / total) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return float(value)


def _prepare_sample(utterances: pd.DataFrame) -> pd.DataFrame:
    eligible = utterances.loc[
        utterances["role"].astype(str).str.casefold().isin(ROLE_NAMES)
        & utterances["session_id"].astype(str).map(_session_selected)
    ].copy()
    eligible["role"] = eligible["role"].astype(str).str.casefold()
    eligible["fold"] = eligible["session_id"].astype(str).map(_session_fold)
    enriched: list[pd.DataFrame] = []
    for _, group in eligible.groupby("session_id", sort=True):
        chunk = group.sort_values("utterance_id", kind="mergesort").copy()
        chunk["model_text"] = enrich_session_text(chunk)
        enriched.append(chunk)
    sample = pd.concat(enriched, ignore_index=True)
    if set(sample["fold"]) != set(range(5)):
        raise ValueError("E850 sample does not cover all five folds.")
    return sample


def cross_validated_role_screen(
    sample: pd.DataFrame,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    vectorizer = _vectorizer()
    matrix = vectorizer.transform(sample["model_text"].astype(str))
    target = sample["role"].eq("student").to_numpy(dtype=np.int8)
    folds = sample["fold"].to_numpy(dtype=np.int8)
    probability = np.full(len(sample), np.nan, dtype=np.float64)
    summaries: list[dict[str, object]] = []
    for fold in range(5):
        validation = folds == fold
        training = ~validation
        training_sessions = set(sample.loc[training, "session_id"].astype(str))
        validation_sessions = set(sample.loc[validation, "session_id"].astype(str))
        if training_sessions & validation_sessions:
            raise RuntimeError(f"E850 session leakage in fold {fold}.")
        model = _classifier(SEED + fold)
        model.fit(matrix[training], target[training])
        probability[validation] = model.predict_proba(matrix[validation])[:, 1]
        summaries.append(
            {
                "fold": fold,
                "training_rows": int(training.sum()),
                "validation_rows": int(validation.sum()),
                "training_sessions": len(training_sessions),
                "validation_sessions": len(validation_sessions),
                "iterations": int(model.n_iter_),
            }
        )
    if not np.isfinite(probability).all():
        raise RuntimeError("E850 OOF role probabilities are incomplete.")
    return probability, summaries


def _sample_metrics(
    sample: pd.DataFrame,
    probability: np.ndarray,
) -> dict[str, float]:
    target = sample["role"].eq("student").to_numpy(dtype=np.int8)
    prediction = (probability >= 0.5).astype(np.int8)
    row_key = (
        sample["session_id"].astype(str)
        + "|"
        + sample["utterance_id"].astype(str)
    )
    corrupted = row_key.map(
        lambda value: _stable_integer(value, "corrupt") % 20 == 0
    ).to_numpy()
    confident = np.maximum(probability, 1.0 - probability) >= CONFIDENCE_THRESHOLD
    confident_corrupted = corrupted & confident
    return {
        "rows": int(len(sample)),
        "sessions": int(sample["session_id"].nunique()),
        "student_rate": float(target.mean()),
        "accuracy": float(accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro")),
        "log_loss": float(log_loss(target, probability)),
        "top_label_ece_10": _top_label_ece(target, probability),
        "high_confidence_coverage": float(confident.mean()),
        "high_confidence_agreement": float(
            np.mean(prediction[confident] == target[confident])
        ),
        "synthetic_corruption_rows": int(corrupted.sum()),
        "synthetic_high_confidence_rows": int(confident_corrupted.sum()),
        "synthetic_corruption_recovery": float(
            np.mean(
                prediction[confident_corrupted] == target[confident_corrupted]
            )
        ),
    }


def _session_style_map(project_root: str | Path) -> dict[str, int]:
    frame = pd.read_parquet(
        Path(project_root).resolve()
        / "data_cache"
        / "validation_environments_selection.parquet",
        columns=["session_id", "style_cell"],
    ).drop_duplicates()
    counts = frame.groupby("session_id")["style_cell"].nunique()
    if int(counts.max()) != 1:
        raise ValueError("E850 style cell varies within session.")
    return {
        str(row.session_id): int(row.style_cell)
        for row in frame.drop_duplicates("session_id").itertuples(index=False)
    }


def fit_final_role_model(sample: pd.DataFrame) -> tuple[HashingVectorizer, SGDClassifier]:
    vectorizer = _vectorizer()
    matrix = vectorizer.transform(sample["model_text"].astype(str))
    target = sample["role"].eq("student").to_numpy(dtype=np.int8)
    model = _classifier(SEED + 100)
    model.fit(matrix, target)
    return vectorizer, model


def audit_full_corpus(
    project_root: str | Path,
    utterances: pd.DataFrame,
    vectorizer: HashingVectorizer,
    model: SGDClassifier,
) -> tuple[pd.DataFrame, dict[str, object]]:
    style_lookup = _session_style_map(project_root)
    correction_rows: list[dict[str, object]] = []
    eligible_rows = 0
    eligible_sessions = 0
    affected_sessions: set[str] = set()
    style_eligible_sessions: dict[int, set[str]] = {
        style: set() for style in range(20)
    }
    style_affected_sessions: dict[int, set[str]] = {
        style: set() for style in range(20)
    }
    style_corrections = {style: 0 for style in range(20)}
    direction = {"student_to_tutor": 0, "tutor_to_student": 0}

    batch_text: list[str] = []
    batch_metadata: list[tuple[str, int, str, int]] = []

    def flush() -> None:
        nonlocal batch_text, batch_metadata
        if not batch_text:
            return
        matrix = vectorizer.transform(batch_text)
        probability = model.predict_proba(matrix)[:, 1]
        for (session_id, utterance_id, observed_role, style), value in zip(
            batch_metadata, probability, strict=True
        ):
            proposed_role = "student" if value >= 0.5 else "tutor"
            confidence = max(float(value), 1.0 - float(value))
            if (
                confidence >= CONFIDENCE_THRESHOLD
                and proposed_role != observed_role
            ):
                affected_sessions.add(session_id)
                style_affected_sessions[style].add(session_id)
                style_corrections[style] += 1
                key = f"{observed_role}_to_{proposed_role}"
                direction[key] += 1
                correction_rows.append(
                    {
                        "session_id": session_id,
                        "utterance_id": utterance_id,
                        "observed_role": observed_role,
                        "proposed_role": proposed_role,
                        "student_probability": float(value),
                        "confidence": confidence,
                        "style_cell": style,
                    }
                )
        batch_text = []
        batch_metadata = []

    for session_id, group in utterances.groupby("session_id", sort=True):
        session_id = str(session_id)
        ordered = group.sort_values("utterance_id", kind="mergesort")
        role_mask = ordered["role"].astype(str).str.casefold().isin(ROLE_NAMES)
        eligible = ordered.loc[role_mask].copy()
        if eligible.empty:
            continue
        style = style_lookup.get(session_id)
        if style is None:
            raise ValueError(f"E850 session lacks target-free style cell: {session_id}")
        eligible_sessions += 1
        style_eligible_sessions[style].add(session_id)
        texts = enrich_session_text(eligible)
        observed = eligible["role"].astype(str).str.casefold().tolist()
        utterance_ids = eligible["utterance_id"].astype(int).tolist()
        eligible_rows += len(eligible)
        batch_text.extend(texts)
        batch_metadata.extend(
            [
                (session_id, utterance_id, role, style)
                for utterance_id, role in zip(
                    utterance_ids, observed, strict=True
                )
            ]
        )
        if len(batch_text) >= INFERENCE_BATCH_ROWS:
            flush()
    flush()

    corrections = pd.DataFrame(
        correction_rows,
        columns=[
            "session_id",
            "utterance_id",
            "observed_role",
            "proposed_role",
            "student_probability",
            "confidence",
            "style_cell",
        ],
    ).sort_values(
        ["session_id", "utterance_id"], kind="mergesort"
    )
    correction_count = len(corrections)
    style_rows = {
        str(style): {
            "eligible_sessions": len(style_eligible_sessions[style]),
            "affected_sessions": len(style_affected_sessions[style]),
            "affected_session_fraction": (
                len(style_affected_sessions[style])
                / max(len(style_eligible_sessions[style]), 1)
            ),
            "corrections": style_corrections[style],
        }
        for style in range(20)
    }
    summary: dict[str, object] = {
        "eligible_utterances": eligible_rows,
        "eligible_sessions": eligible_sessions,
        "corrections": correction_count,
        "correction_rate": correction_count / eligible_rows,
        "affected_sessions": len(affected_sessions),
        "affected_session_fraction": len(affected_sessions) / eligible_sessions,
        "direction_counts": direction,
        "direction_ratio_student_to_tutor_over_tutor_to_student": (
            direction["student_to_tutor"]
            / max(direction["tutor_to_student"], 1)
        ),
        "style_cells": style_rows,
        "minimum_style_corrections": min(style_corrections.values()),
        "minimum_style_affected_session_fraction": min(
            row["affected_session_fraction"] for row in style_rows.values()
        ),
    }
    return corrections, summary


def _gate(
    sample: dict[str, float],
    corpus: dict[str, object],
) -> dict[str, bool]:
    direction_ratio = float(
        corpus["direction_ratio_student_to_tutor_over_tutor_to_student"]
    )
    return {
        "sample_macro_f1_at_least_0_95": sample["macro_f1"] >= 0.95,
        "sample_top_label_ece_at_most_0_05": (
            sample["top_label_ece_10"] <= 0.05
        ),
        "sample_high_confidence_coverage_at_least_0_50": (
            sample["high_confidence_coverage"] >= 0.50
        ),
        "sample_high_confidence_agreement_at_least_0_98": (
            sample["high_confidence_agreement"] >= 0.98
        ),
        "synthetic_corruption_recovery_at_least_0_95": (
            sample["synthetic_corruption_recovery"] >= 0.95
        ),
        "full_corpus_correction_rate_at_least_0_0025": (
            float(corpus["correction_rate"]) >= 0.0025
        ),
        "full_corpus_correction_rate_at_most_0_03": (
            float(corpus["correction_rate"]) <= 0.03
        ),
        "affected_session_fraction_at_least_0_05": (
            float(corpus["affected_session_fraction"]) >= 0.05
        ),
        "affected_session_fraction_at_most_0_60": (
            float(corpus["affected_session_fraction"]) <= 0.60
        ),
        "direction_ratio_between_0_25_and_4": 0.25 <= direction_ratio <= 4.0,
        "at_least_20_corrections_in_every_style_cell": (
            int(corpus["minimum_style_corrections"]) >= 20
        ),
        "at_least_0_02_affected_sessions_in_every_style_cell": (
            float(corpus["minimum_style_affected_session_fraction"]) >= 0.02
        ),
    }


def benchmark(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    root = Path(project_root).resolve()
    source = root / "data_cache" / "utterances.parquet"
    frame = pd.read_parquet(
        source,
        columns=["session_id", "utterance_id", "role", "content"],
    )
    loaded = time.perf_counter()
    sample = _prepare_sample(frame)
    sampled = time.perf_counter()
    subset = sample.iloc[: min(50_000, len(sample))]
    vectorizer = _vectorizer()
    matrix = vectorizer.transform(subset["model_text"].astype(str))
    transformed = time.perf_counter()
    model = _classifier(SEED)
    target = subset["role"].eq("student").to_numpy(dtype=np.int8)
    model.fit(matrix, target)
    fitted = time.perf_counter()
    model.predict_proba(matrix)
    inferred = time.perf_counter()

    load_seconds = loaded - started
    sample_preparation_seconds = sampled - loaded
    transform_seconds = transformed - sampled
    fit_seconds = fitted - transformed
    inference_seconds = inferred - fitted
    elapsed = inferred - started
    benchmark_rows = max(len(subset), 1)
    eligible_rows = int(
        frame["role"].astype(str).str.casefold().isin(ROLE_NAMES).sum()
    )
    sample_scale = len(sample) / benchmark_rows
    corpus_scale = eligible_rows / benchmark_rows
    projected_components = {
        "source_load_seconds": load_seconds,
        "sample_preparation_seconds": sample_preparation_seconds,
        "full_corpus_enrichment_seconds": (
            sample_preparation_seconds * eligible_rows / max(len(sample), 1)
        ),
        "two_sample_vectorizations_seconds": (
            transform_seconds * 2.0 * sample_scale
        ),
        "five_sample_equivalent_fits_seconds": (
            fit_seconds * 5.0 * sample_scale
        ),
        "sample_plus_corpus_inference_seconds": (
            inference_seconds * (sample_scale + corpus_scale)
        ),
        "full_corpus_vectorization_seconds": transform_seconds * corpus_scale,
    }
    projected = 1.25 * sum(projected_components.values())
    return {
        "protocol_id": PROTOCOL_ID,
        "benchmark_rows": int(len(subset)),
        "source_rows": int(len(frame)),
        "eligible_source_rows": eligible_rows,
        "sample_rows": int(len(sample)),
        "elapsed_seconds": elapsed,
        "measured_components_seconds": {
            "source_load": load_seconds,
            "sample_preparation": sample_preparation_seconds,
            "vectorize_benchmark_rows": transform_seconds,
            "fit_benchmark_rows": fit_seconds,
            "infer_benchmark_rows": inference_seconds,
        },
        "projected_components_seconds_before_25pct_margin": projected_components,
        "projected_full_seconds_upper_bound": projected,
        "projected_full_hours_upper_bound": projected / 3600.0,
        "projected_below_one_hour": projected < 3600.0,
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }


def run(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    root = Path(project_root).resolve()
    utterance_path = root / "data_cache" / "utterances.parquet"
    style_path = root / "data_cache" / "validation_environments_selection.parquet"
    if _sha256(utterance_path) != EXPECTED_UTTERANCE_SHA256:
        raise ValueError("E850 utterance source SHA-256 changed.")
    if _sha256(style_path) != EXPECTED_STYLE_SHA256:
        raise ValueError("E850 style source SHA-256 changed.")
    utterances = pd.read_parquet(
        utterance_path,
        columns=["session_id", "utterance_id", "role", "content"],
    )
    sample = _prepare_sample(utterances)
    oof_probability, fits = cross_validated_role_screen(sample)
    sample_metrics = _sample_metrics(sample, oof_probability)
    vectorizer, model = fit_final_role_model(sample)
    corrections, corpus = audit_full_corpus(
        project_root, utterances, vectorizer, model
    )
    clauses = _gate(sample_metrics, corpus)

    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ_speaker_role_denoising"
    )
    run_dir = root / "experiments" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    corrections_path = run_dir / "proposed_role_corrections.parquet"
    sample_path = run_dir / "sample_oof_predictions.parquet"
    report_path = run_dir / "report.json"
    corrections.to_parquet(corrections_path, index=False)
    sample_output = sample[
        ["session_id", "utterance_id", "role", "fold"]
    ].copy()
    sample_output["student_probability"] = oof_probability
    sample_output.to_parquet(sample_path, index=False)
    report: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "seed": SEED,
            "sample_session_modulus": SAMPLE_SESSION_MODULUS,
            "hash_features": N_FEATURES,
            "word_ngrams": [1, 2],
            "alternate_sign": False,
            "sgd_loss": "log_loss",
            "sgd_alpha": ALPHA,
            "sgd_max_iterations": MAX_ITERATIONS,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "inference_batch_rows": INFERENCE_BATCH_ROWS,
        },
        "source": {
            "utterances_path": str(utterance_path.relative_to(root)),
            "utterances_sha256": _sha256(utterance_path),
            "style_path": str(style_path.relative_to(root)),
            "style_sha256": _sha256(style_path),
            "rows": int(len(utterances)),
            "sessions": int(utterances["session_id"].nunique()),
            "role_counts": {
                str(key): int(value)
                for key, value in utterances["role"].value_counts().items()
            },
        },
        "sample_oof_metrics": sample_metrics,
        "fit_summaries": fits,
        "full_corpus_audit": corpus,
        "gate_clauses": clauses,
        "passes_target_free_gate": bool(all(clauses.values())),
        "runtime_seconds": float(time.perf_counter() - started),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "artifacts": {
            "proposed_role_corrections": {
                "path": str(corrections_path.relative_to(root)),
                "sha256": _sha256(corrections_path),
                "rows": int(len(corrections)),
            },
            "sample_oof_predictions": {
                "path": str(sample_path.relative_to(root)),
                "sha256": _sha256(sample_path),
                "rows": int(len(sample_output)),
            },
        },
        "competition_outcomes_accessed": False,
        "V_seen_accessed": False,
        "V_objective_accessed": False,
        "V_style_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
        "submission_built": False,
        "competition_upload_or_submission": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "passes_target_free_gate": report["passes_target_free_gate"],
        "sample_oof_metrics": sample_metrics,
        "full_corpus_audit": corpus,
        "gate_clauses": clauses,
        "artifact_sha256": {
            name: value["sha256"] for name, value in report["artifacts"].items()
        },
        "report_sha256": _sha256(report_path),
        "runtime_seconds": report["runtime_seconds"],
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run E850 target-free lexical speaker-role denoising."
    )
    parser.add_argument("stage", choices=("benchmark", "run"))
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    result = (
        benchmark(args.project_root)
        if args.stage == "benchmark"
        else run(args.project_root)
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
