from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import platform
import random
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from trace_ace.external_sra_transfer import (
    ENCODER_LEARNING_RATE,
    EPOCHS,
    EVAL_BATCH_SIZE,
    HEAD_LEARNING_RATE,
    MODEL_DIRECTORY,
    MODEL_NAME,
    TRAIN_BATCH_SIZE,
    UNFROZEN_LAYERS,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
    _freeze_deberta,
    _sha256,
    _trainable_state_dict,
)
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.source_robust_validation import _current_rss_bytes


SEED = 20260727
MAX_LENGTH = 384
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260727
EXPECTED_RUNTIME = {
    "python": "3.12.8",
    "numpy": "2.5.1",
    "scikit_learn": "1.8.0",
    "torch": "2.13.0+cpu",
    "transformers": "5.14.1",
}
SOURCE_REVISION = "c61f335f89005161b6ef439872cc1735bee26745"
SOURCE_HASHES = {
    "mathdial_train_atc.csv": (
        "bd3906355e98d357cbfa5e0e67e3b778e976f6f1a340cef35928a33325b7e6ad"
    ),
    "mathdial_test_atc.csv": (
        "f9d583489b6fe57f62b2a5239194a013f6b0662bcbcfe5b5218cf97b0a1efe63"
    ),
    "comta_atc.csv": (
        "fdfebc0108bec3f6eab0839a5ccc8b58f24ba4c0e55e2511e67c7116845de9e1"
    ),
    "COMTA_LICENSE.txt": (
        "d047c993977d0017314176b237da232de7525bc6b2faad8341bbbb242debc761"
    ),
}
EXPECTED_DIALOGUE_COUNTS = {
    "mathdial_train_raw": 2253,
    "mathdial_train_legal": 1679,
    "mathdial_test": 595,
    "comta_eval_only": 153,
}
EXPECTED_TURN_COUNTS = {
    "mathdial_train": 7980,
    "mathdial_test": 2573,
    "comta_eval_only": 623,
}
EXPECTED_INVALID_DIALOGUES = {
    "mathdial_train": 15,
    "mathdial_test": 7,
    "comta_eval_only": 0,
}
EXPECTED_CANONICAL_CONTENT_SHA256 = (
    "f6d5cf38716bf5b6a8fabade6346a17d9d3a4cbd7966a600907b4bb1ffa45458"
)
EXPECTED_BINARY_HEAD_WEIGHT_SHA256 = (
    "c06504ffa78c64e79d3a6ecbad025b542a50bc7188463f7f4b1fd07ca267abb1"
)
EXPECTED_BINARY_HEAD_BIAS_SHA256 = (
    "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"
)


def assert_e630_runtime() -> dict[str, str]:
    import sklearn
    import torch
    import transformers

    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    if observed != EXPECTED_RUNTIME:
        raise RuntimeError(
            f"E630 runtime differs from the frozen .venv contract: {observed}"
        )
    return observed


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def _parse_literal(value: object) -> object:
    if not isinstance(value, str):
        return value
    return ast.literal_eval(value)


def _ordered_content_sha256(frame: pd.DataFrame) -> str:
    columns = [
        "split",
        "dialogue_id",
        "group_id",
        "turn_number",
        "history",
        "objective",
        "label",
    ]
    digest = hashlib.sha256()
    for row in frame[columns].itertuples(index=False, name=None):
        digest.update(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _objective_text(kcs: list[object]) -> str:
    cleaned = [_clean_text(value) for value in kcs]
    unique = list(dict.fromkeys(value for value in cleaned if value))
    return " | ".join(value[:180] for value in unique[:3])


def _history_through_tutor(dialogue: list[dict[str, object]], index: int) -> str:
    lines = ["Dialogue history:"]
    for prior in dialogue[:index]:
        teacher = _clean_text(prior.get("teacher"))
        student = _clean_text(prior.get("student"))
        if teacher:
            lines.append(f"Tutor: {teacher}")
        if student:
            lines.append(f"Student: {student}")
    current_teacher = _clean_text(dialogue[index].get("teacher"))
    if current_teacher:
        lines.append(f"Tutor: {current_teacher}")
    return "\n".join(lines)


def _correct_final_turn(
    annotation: dict[str, object],
    dialogue: list[dict[str, object]],
    row: pd.Series,
    *,
    source: str,
) -> dict[str, object]:
    corrected = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in annotation.items()
    }
    final_key = f"turn {dialogue[-1]['turn']}"
    final = corrected.get(final_key)
    if not isinstance(final, dict):
        return corrected
    if not final.get("kcs") or final.get("correct") is None:
        return corrected
    if source == "mathdial":
        outcome = str(row["self-correctness"])
        final["correct"] = (
            True if outcome == "Yes" else False if outcome == "No" else None
        )
    elif source == "comta":
        metadata = row["meta_data"]
        final["correct"] = metadata.get("expected_result") == "Answer Accepted"
    else:
        raise ValueError(f"Unknown E630 source: {source}")
    return corrected


def canonical_turn_rows(
    frame: pd.DataFrame,
    *,
    split: str,
    source: str,
) -> tuple[list[dict[str, object]], int]:
    records: list[dict[str, object]] = []
    invalid_dialogues = 0
    for _, row in frame.iterrows():
        dialogue = _parse_literal(row["dialogue"])
        annotation = _parse_literal(row["annotation"])
        metadata = _parse_literal(row["meta_data"])
        if (
            not isinstance(dialogue, list)
            or not dialogue
            or not isinstance(annotation, dict)
            or "error" in annotation
            or not isinstance(metadata, dict)
        ):
            invalid_dialogues += 1
            continue
        parsed_row = row.copy()
        parsed_row["meta_data"] = metadata
        annotation = _correct_final_turn(
            annotation, dialogue, parsed_row, source=source
        )
        if source == "mathdial":
            dialogue_id = str(row["index"])
            group_id = str(row["qid"])
        else:
            dialogue_id = str(row["index"])
            group_id = dialogue_id
        for turn_index, turn in enumerate(dialogue):
            turn_number = int(turn["turn"])
            turn_annotation = annotation.get(f"turn {turn_number}")
            if not isinstance(turn_annotation, dict):
                continue
            label = turn_annotation.get("correct")
            kcs = turn_annotation.get("kcs")
            teacher = _clean_text(turn.get("teacher"))
            if label is None or not isinstance(kcs, list) or not kcs or not teacher:
                continue
            history = _history_through_tutor(dialogue, turn_index)
            objective = _objective_text(kcs)
            if not objective:
                continue
            records.append(
                {
                    "split": split,
                    "source": source,
                    "dialogue_id": dialogue_id,
                    "group_id": group_id,
                    "turn_number": turn_number,
                    "history": history,
                    "objective": objective,
                    "label": int(bool(label)),
                }
            )
    return records, invalid_dialogues


def prepare_dialogue_kt_cache(project_root: str | Path) -> dict[str, object]:
    runtime = assert_e630_runtime()
    paths = discover_project_paths(project_root)
    source_root = paths.root / "Datasets" / "dialogue-kt" / "data" / "annotated"
    for filename, expected_hash in SOURCE_HASHES.items():
        path = source_root / filename
        if not path.exists() or _sha256(path) != expected_hash:
            raise ValueError(f"E630 source hash mismatch: {path}")

    math_train = pd.read_csv(source_root / "mathdial_train_atc.csv")
    math_test = pd.read_csv(source_root / "mathdial_test_atc.csv")
    comta = pd.read_csv(source_root / "comta_atc.csv")
    if {
        "mathdial_train_raw": len(math_train),
        "mathdial_test": len(math_test),
        "comta_eval_only": len(comta),
    } != {
        key: EXPECTED_DIALOGUE_COUNTS[key]
        for key in ("mathdial_train_raw", "mathdial_test", "comta_eval_only")
    }:
        raise ValueError("E630 raw dialogue counts changed.")
    test_qids = set(math_test["qid"].astype(str))
    legal_math_train = math_train.loc[
        ~math_train["qid"].astype(str).isin(test_qids)
    ].reset_index(drop=True)
    if len(legal_math_train) != EXPECTED_DIALOGUE_COUNTS["mathdial_train_legal"]:
        raise ValueError("E630 leakage-safe MathDial training count changed.")
    overlap = set(legal_math_train["qid"].astype(str)) & test_qids
    if overlap:
        raise ValueError("E630 legal MathDial training still overlaps test qids.")

    record_sets: list[list[dict[str, object]]] = []
    invalid_counts: dict[str, int] = {}
    for frame, split, source in (
        (legal_math_train, "mathdial_train", "mathdial"),
        (math_test, "mathdial_test", "mathdial"),
        (comta, "comta_eval_only", "comta"),
    ):
        rows, invalid = canonical_turn_rows(frame, split=split, source=source)
        record_sets.append(rows)
        invalid_counts[split] = invalid
    canonical = pd.DataFrame([row for rows in record_sets for row in rows])
    observed_turns = canonical.groupby("split", sort=False).size().to_dict()
    if observed_turns != EXPECTED_TURN_COUNTS:
        raise ValueError(f"E630 canonical turn counts changed: {observed_turns}")
    if invalid_counts != EXPECTED_INVALID_DIALOGUES:
        raise ValueError(f"E630 invalid-dialogue counts changed: {invalid_counts}")
    train_groups = set(
        canonical.loc[canonical["split"].eq("mathdial_train"), "group_id"]
    )
    test_groups = set(
        canonical.loc[canonical["split"].eq("mathdial_test"), "group_id"]
    )
    if train_groups & test_groups:
        raise ValueError("E630 canonical MathDial qid leakage detected.")

    ordered_hash = _ordered_content_sha256(canonical)
    cache_path = paths.cache_dir / "dialogue_kt_e630_canonical.parquet"
    metadata_path = paths.cache_dir / "dialogue_kt_e630_canonical.metadata.json"
    canonical.to_parquet(cache_path, index=False)
    metadata = {
        "candidate": "E630_dialogue_kt_next_response",
        "source_revision": SOURCE_REVISION,
        "source_hashes": SOURCE_HASHES,
        "dialogue_counts": EXPECTED_DIALOGUE_COUNTS,
        "turn_counts": observed_turns,
        "invalid_dialogues": invalid_counts,
        "mathdial_train_qids": len(train_groups),
        "mathdial_test_qids": len(test_groups),
        "qid_overlap": 0,
        "ordered_content_sha256": ordered_hash,
        "parquet_sha256": _sha256(cache_path),
        "runtime_contract": runtime,
        "comta_training_rows": 0,
        "current_student_response_in_input": False,
        "V_final_accessed": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "cache_path": str(cache_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }


def _tensor_sha256(tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def validate_binary_head_initialization(model) -> dict[str, object]:
    import torch

    audit = {
        "seed": SEED,
        "weight_shape": list(model.classifier.weight.shape),
        "bias_shape": list(model.classifier.bias.shape),
        "weight_sha256": _tensor_sha256(model.classifier.weight),
        "bias_sha256": _tensor_sha256(model.classifier.bias),
        "bias_values": model.classifier.bias.detach().cpu().tolist(),
    }
    failures: list[str] = []
    if audit["weight_shape"] != [2, 768] or audit["bias_shape"] != [2]:
        failures.append("binary classifier shape changed")
    if not bool(
        torch.isfinite(model.classifier.weight).all()
        and torch.isfinite(model.classifier.bias).all()
    ):
        failures.append("binary classifier contains non-finite values")
    if (
        EXPECTED_BINARY_HEAD_WEIGHT_SHA256
        and audit["weight_sha256"] != EXPECTED_BINARY_HEAD_WEIGHT_SHA256
    ):
        failures.append("binary classifier weight hash changed")
    if (
        EXPECTED_BINARY_HEAD_BIAS_SHA256
        and audit["bias_sha256"] != EXPECTED_BINARY_HEAD_BIAS_SHA256
    ):
        failures.append("binary classifier bias hash changed")
    if failures:
        raise RuntimeError("E630 head initialization drift: " + "; ".join(failures))
    return audit


def _load_frame(paths) -> tuple[pd.DataFrame, dict[str, object]]:
    cache_path = paths.cache_dir / "dialogue_kt_e630_canonical.parquet"
    metadata_path = paths.cache_dir / "dialogue_kt_e630_canonical.metadata.json"
    if not cache_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("E630 requires its canonical cache and metadata.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    content_hash = metadata.get("ordered_content_sha256")
    if (
        EXPECTED_CANONICAL_CONTENT_SHA256
        and content_hash != EXPECTED_CANONICAL_CONTENT_SHA256
    ):
        raise ValueError("E630 canonical content hash changed.")
    if metadata.get("parquet_sha256") != _sha256(cache_path):
        raise ValueError("E630 canonical Parquet hash changed.")
    frame = pd.read_parquet(cache_path)
    counts = frame.groupby("split", sort=False).size().to_dict()
    if counts != EXPECTED_TURN_COUNTS:
        raise ValueError(f"E630 cached turn counts changed: {counts}")
    return frame, metadata


def _tokenize(tokenizer, frame: pd.DataFrame):
    import torch

    tokenizer.truncation_side = "left"
    encoded = tokenizer(
        frame["history"].astype(str).tolist(),
        frame["objective"].astype(str).tolist(),
        padding="max_length",
        truncation="only_first",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    keys = [
        key
        for key in ("input_ids", "attention_mask", "token_type_ids")
        if key in encoded
    ]
    return keys, [encoded[key].to(dtype=torch.long) for key in keys]


def _new_model(paths):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model_path = paths.root / "assets" / "pretrained" / MODEL_DIRECTORY
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    head_audit = validate_binary_head_initialization(model)
    freeze_summary = _freeze_deberta(model, UNFROZEN_LAYERS)
    return tokenizer, model, head_audit, freeze_summary


def _optimizer(model):
    import torch

    classifier_ids = {id(parameter) for parameter in model.classifier.parameters()}
    head_parameters = list(model.classifier.parameters())
    encoder_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in classifier_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": ENCODER_LEARNING_RATE},
            {"params": head_parameters, "lr": HEAD_LEARNING_RATE},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def benchmark_dialogue_kt_transfer(project_root: str | Path) -> dict[str, object]:
    import torch

    runtime = assert_e630_runtime()
    paths = discover_project_paths(project_root)
    frame, metadata = _load_frame(paths)
    torch.set_num_threads(6)
    tokenizer, model, head_audit, freeze_summary = _new_model(paths)
    train = frame.loc[frame["split"].eq("mathdial_train")].iloc[:32]
    evaluation = pd.concat(
        [
            frame.loc[frame["split"].eq("mathdial_test")].iloc[:64],
            frame.loc[frame["split"].eq("comta_eval_only")].iloc[:64],
        ],
        ignore_index=True,
    )
    train_keys, train_tensors = _tokenize(tokenizer, train)
    eval_keys, eval_tensors = _tokenize(tokenizer, evaluation)
    synthetic_labels = torch.arange(len(train), dtype=torch.long) % 2
    optimizer = _optimizer(model)
    train_seconds = 0.0
    model.train()
    for start in (0, 16):
        inputs = {
            key: value[start : start + 16]
            for key, value in zip(train_keys, train_tensors, strict=True)
        }
        optimizer.zero_grad(set_to_none=True)
        began = time.perf_counter()
        output = model(**inputs, labels=synthetic_labels[start : start + 16])
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()
        train_seconds += time.perf_counter() - began
    eval_seconds = 0.0
    model.eval()
    with torch.inference_mode():
        for start in (0, 64):
            inputs = {
                key: value[start : start + 64]
                for key, value in zip(eval_keys, eval_tensors, strict=True)
            }
            began = time.perf_counter()
            model(**inputs)
            eval_seconds += time.perf_counter() - began
    train_batches = math.ceil(EXPECTED_TURN_COUNTS["mathdial_train"] / TRAIN_BATCH_SIZE)
    eval_batches = math.ceil(
        EXPECTED_TURN_COUNTS["mathdial_test"] / EVAL_BATCH_SIZE
    ) + math.ceil(EXPECTED_TURN_COUNTS["comta_eval_only"] / EVAL_BATCH_SIZE)
    projected_seconds = (
        train_seconds / 2.0 * train_batches + eval_seconds / 2.0 * eval_batches
    )
    rss_bytes = _current_rss_bytes()
    report = {
        "candidate": "E630_dialogue_kt_next_response",
        "canonical_content_sha256": metadata["ordered_content_sha256"],
        "two_train_batch_seconds": train_seconds,
        "two_eval_batch_seconds": eval_seconds,
        "projected_train_batches": train_batches,
        "projected_eval_batches": eval_batches,
        "projected_seconds": projected_seconds,
        "projected_hours": projected_seconds / 3600.0,
        "rss_bytes": rss_bytes,
        "passes_time_gate": projected_seconds < 3 * 3600,
        "passes_memory_gate": rss_bytes < 8 * 1024**3,
        "head_initialization": head_audit,
        "freeze_summary": freeze_summary,
        "runtime_contract": runtime,
        "synthetic_labels_only": True,
        "prediction_metric_computed": False,
        "V_final_accessed": False,
    }
    output_path = paths.cache_dir / "dialogue_kt_e630_benchmark.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report["benchmark_sha256"] = _sha256(output_path)
    return {"benchmark_path": str(output_path), "report": report}


def _evaluate(model, keys, tensors, labels: np.ndarray):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            inputs = {key: value for key, value in zip(keys, batch, strict=True)}
            probability = torch.softmax(model(**inputs).logits, dim=1)[:, 1]
            probabilities.append(probability.cpu().numpy().astype(np.float64))
            if batch_number % 20 == 0 or batch_number == len(loader):
                print(
                    f"e630_eval_batch={batch_number}/{len(loader)}",
                    flush=True,
                )
    probability = np.concatenate(probabilities)
    metrics = {
        "rows": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "accuracy": float(accuracy_score(labels, probability >= 0.5)),
        "macro_f1": float(f1_score(labels, probability >= 0.5, average="macro")),
        **binary_metrics(labels, probability),
    }
    return probability, metrics


def group_bootstrap(
    frame: pd.DataFrame,
    labels: np.ndarray,
    prior_probability: np.ndarray,
    candidate_probability: np.ndarray,
) -> dict[str, object]:
    target = labels.astype(np.float64)
    prior = np.clip(prior_probability, 1e-6, 1.0 - 1e-6)
    candidate = np.clip(candidate_probability, 1e-6, 1.0 - 1e-6)
    gains = pd.DataFrame(
        {
            "group_id": frame["group_id"].astype(str),
            "gain": -(
                target * np.log(prior) + (1.0 - target) * np.log1p(-prior)
            )
            + (
                target * np.log(candidate)
                + (1.0 - target) * np.log1p(-candidate)
            ),
        }
    )
    grouped = gains.groupby("group_id", sort=True)["gain"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=np.float64)
    counts = grouped["count"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        values[replicate] = sums[sampled].sum() / counts[sampled].sum()
    return {
        "resampler": "qid" if frame["split"].iloc[0] == "mathdial_test" else "dialogue",
        "unique_groups": int(len(grouped)),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "support_positive_log_loss_gain": float((values > 0.0).mean()),
        "mean_log_loss_gain": float(values.mean()),
        "ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def external_gate(
    split: str,
    metrics: dict[str, float | int],
    prior_metrics: dict[str, float],
    bootstrap: dict[str, object],
) -> dict[str, bool]:
    loss_gain = float(prior_metrics["log_loss"]) - float(metrics["log_loss"])
    brier_gain = float(prior_metrics["brier_score"]) - float(
        metrics["brier_score"]
    )
    if split == "mathdial_test":
        return {
            "auroc_at_least_0_70": float(metrics["roc_auc"]) >= 0.70,
            "macro_f1_at_least_0_60": float(metrics["macro_f1"]) >= 0.60,
            "log_loss_at_most_0_66": float(metrics["log_loss"]) <= 0.66,
            "ece_10_at_most_0_10": float(metrics["ece_10"]) <= 0.10,
            "log_loss_gain_at_least_0_02": loss_gain >= 0.02,
            "brier_gain_at_least_0_005": brier_gain >= 0.005,
            "qid_bootstrap_support_at_least_0_95": float(
                bootstrap["support_positive_log_loss_gain"]
            )
            >= 0.95,
        }
    if split == "comta_eval_only":
        return {
            "auroc_at_least_0_60": float(metrics["roc_auc"]) >= 0.60,
            "macro_f1_at_least_0_55": float(metrics["macro_f1"]) >= 0.55,
            "log_loss_at_most_0_68": float(metrics["log_loss"]) <= 0.68,
            "ece_10_at_most_0_15": float(metrics["ece_10"]) <= 0.15,
            "log_loss_non_regression": loss_gain >= 0.0,
            "brier_non_regression": brier_gain >= 0.0,
            "dialogue_bootstrap_support_at_least_0_90": float(
                bootstrap["support_positive_log_loss_gain"]
            )
            >= 0.90,
        }
    raise ValueError(f"Unknown E630 external split: {split}")


def train_dialogue_kt_transfer(project_root: str | Path) -> dict[str, object]:
    import torch
    from safetensors.torch import load_file, save_file
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import get_linear_schedule_with_warmup

    started = time.perf_counter()
    runtime = assert_e630_runtime()
    paths = discover_project_paths(project_root)
    frame, metadata = _load_frame(paths)
    benchmark_path = paths.cache_dir / "dialogue_kt_e630_benchmark.json"
    if not benchmark_path.exists():
        raise FileNotFoundError("E630 benchmark report is missing.")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if not (
        benchmark.get("passes_time_gate") and benchmark.get("passes_memory_gate")
    ):
        raise RuntimeError("E630 resource gate did not pass.")

    torch.set_num_threads(6)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    tokenizer, model, head_audit, freeze_summary = _new_model(paths)
    splits = {
        split: frame.loc[frame["split"].eq(split)].reset_index(drop=True)
        for split in EXPECTED_TURN_COUNTS
    }
    tokenized = {split: _tokenize(tokenizer, value) for split, value in splits.items()}
    keys = tokenized["mathdial_train"][0]
    if any(value[0] != keys for value in tokenized.values()):
        raise RuntimeError("E630 tokenizer fields differ across splits.")
    train_labels_np = splits["mathdial_train"]["label"].to_numpy(dtype=np.int64)
    train_labels = torch.tensor(train_labels_np, dtype=torch.long)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        TensorDataset(*tokenized["mathdial_train"][1], train_labels),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    optimizer = _optimizer(model)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(round(total_steps * WARMUP_FRACTION)),
        num_training_steps=total_steps,
    )
    training_rows: list[dict[str, float | int]] = []
    global_step = 0
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        seen = 0
        for batch_number, batch in enumerate(train_loader, start=1):
            labels = batch[-1]
            inputs = {key: value for key, value in zip(keys, batch[:-1], strict=True)}
            optimizer.zero_grad(set_to_none=True)
            output = model(**inputs, labels=labels)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            seen += len(labels)
            running_loss += float(output.loss.item()) * len(labels)
            if batch_number % 25 == 0 or batch_number == len(train_loader):
                print(
                    f"e630_epoch={epoch + 1} step={batch_number}/{len(train_loader)} "
                    f"train_loss={running_loss / seen:.6f}",
                    flush=True,
                )
        training_rows.append(
            {
                "epoch": epoch + 1,
                "steps": global_step,
                "train_loss": running_loss / seen,
            }
        )

    train_prior = float(train_labels_np.mean())
    evaluation: dict[str, dict[str, object]] = {}
    clauses: dict[str, dict[str, bool]] = {}
    prediction_frames: list[pd.DataFrame] = []
    for split in ("mathdial_test", "comta_eval_only"):
        split_frame = splits[split]
        labels = split_frame["label"].to_numpy(dtype=np.int64)
        probability, metrics = _evaluate(
            model, tokenized[split][0], tokenized[split][1], labels
        )
        prior_probability = np.full(len(labels), train_prior, dtype=np.float64)
        prior_metrics = binary_metrics(labels, prior_probability)
        bootstrap = group_bootstrap(
            split_frame, labels, prior_probability, probability
        )
        split_clauses = external_gate(split, metrics, prior_metrics, bootstrap)
        evaluation[split] = {
            "metrics": metrics,
            "train_prior_metrics": prior_metrics,
            "paired_group_bootstrap": bootstrap,
        }
        clauses[split] = split_clauses
        prediction_frames.append(
            pd.DataFrame(
                {
                    "split": split,
                    "dialogue_id": split_frame["dialogue_id"],
                    "group_id": split_frame["group_id"],
                    "turn_number": split_frame["turn_number"],
                    "target": labels,
                    "probability": probability,
                    "train_prior_probability": train_prior,
                }
            )
        )
    passes_gate = all(all(values.values()) for values in clauses.values())

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_dialogue_kt_transfer")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    delta_path = run_dir / "dialogue_kt_deberta_delta.safetensors"
    delta = _trainable_state_dict(model)
    save_file(delta, str(delta_path))
    loaded = load_file(str(delta_path), device="cpu")
    max_difference = max(
        float(torch.max(torch.abs(delta[name] - loaded[name])).item()) for name in delta
    )
    if set(loaded) != set(delta) or max_difference != 0.0:
        raise RuntimeError("E630 delta serialization was not exact.")
    pd.DataFrame(training_rows).to_csv(
        run_dir / "training_metrics.csv", index=False, lineterminator="\n"
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_parquet(run_dir / "external_predictions.parquet", index=False)
    pd.DataFrame(
        [{"split": split, **result["metrics"]} for split, result in evaluation.items()]
    ).to_csv(run_dir / "external_metrics.csv", index=False, lineterminator="\n")
    report = {
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "candidate": "E630_dialogue_kt_next_response",
        "model": MODEL_NAME,
        "model_license": "Apache-2.0",
        "training_source": "MathDial Dialogue-KT annotations",
        "training_source_license": "CC BY-SA 4.0",
        "evaluation_only_source": "CoMTA Dialogue-KT annotations",
        "evaluation_only_license_sha256": SOURCE_HASHES["COMTA_LICENSE.txt"],
        "source_revision": SOURCE_REVISION,
        "source_hashes": SOURCE_HASHES,
        "canonical_content_sha256": metadata["ordered_content_sha256"],
        "canonical_parquet_sha256": metadata["parquet_sha256"],
        "training_rows": len(splits["mathdial_train"]),
        "train_positive_rate": train_prior,
        "comta_training_rows": 0,
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "batch_size": TRAIN_BATCH_SIZE,
        "encoder_learning_rate": ENCODER_LEARNING_RATE,
        "head_learning_rate": HEAD_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_fraction": WARMUP_FRACTION,
        "seed": SEED,
        "binary_head_initialization": head_audit,
        "freeze_summary": freeze_summary,
        "external_evaluation": evaluation,
        "external_gate_clauses": clauses,
        "passes_external_gate": passes_gate,
        "delta_file": delta_path.name,
        "delta_bytes": delta_path.stat().st_size,
        "delta_sha256": _sha256(delta_path),
        "prediction_sha256": _sha256(
            run_dir / "external_predictions.parquet"
        ),
        "trainable_tensor_count": len(delta),
        "max_serialization_difference": max_difference,
        "elapsed_seconds": time.perf_counter() - started,
        "runtime_contract": runtime,
        "rejected_checkpoint_loaded": False,
        "current_student_response_in_input": False,
        "competition_cache_built": False,
        "V_final_accessed": False,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "report": report,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare, benchmark, or train frozen E630 Dialogue-KT transfer."
    )
    parser.add_argument("stage", choices=("prepare", "benchmark", "train"))
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.stage == "prepare":
        result = prepare_dialogue_kt_cache(args.project_root)
    elif args.stage == "benchmark":
        result = benchmark_dialogue_kt_transfer(args.project_root)
    else:
        result = train_dialogue_kt_transfer(args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
