from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from trace_ace.external_sra_transfer import _freeze_deberta, _sha256
from trace_ace.io import discover_project_paths


SOURCE_COMMIT = "b06c020a0a1f57a87577fec33e657b63e7eb476e"
EXPECTED_SOURCE_SHA256 = {
    "train": "997d11eaf11dfb6c883b8d415f20969f705ff2df1c9cda49cecae71251806216",
    "test": "8659dccbd891335a8a84c466ce661ad4d1fde9967117128eae36d9e3e9d3934b",
}
MODEL_DIRECTORY = "nli-deberta-v3-small"
MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
SEED = 20260723
MAX_LENGTH = 256
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
EPOCHS = 1
ENCODER_LEARNING_RATE = 2e-5
HEAD_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.10
UNFROZEN_LAYERS = 2
EXPECTED_BINARY_HEAD_WEIGHT_SHA256 = (
    "ee1cf8a8d8a045fa48245c8cbe1fbbdb78d83672eacefe2d72f0be1bb5dc608f"
)
EXPECTED_BINARY_HEAD_BIAS_SHA256 = (
    "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"
)
LABEL_MAP = {
    "Yes": 1,
    "Yes, but I had to reveal the answer": 0,
    "No": 0,
}


def _tensor_sha256(tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _validate_binary_head_initialization(model) -> dict[str, object]:
    import torch

    weight = model.classifier.weight
    bias = model.classifier.bias
    audit = {
        "seed": SEED,
        "weight_shape": list(weight.shape),
        "bias_shape": list(bias.shape),
        "weight_sha256": _tensor_sha256(weight),
        "bias_sha256": _tensor_sha256(bias),
        "weight_mean": float(weight.detach().mean()),
        "weight_std": float(weight.detach().std(unbiased=False)),
        "bias_values": bias.detach().cpu().tolist(),
        "initializer_range": float(model.config.initializer_range),
        "hidden_size": int(model.config.hidden_size),
        "num_labels": int(model.config.num_labels),
        "source_head_labels": 3,
        "replacement_head_labels": 2,
    }
    failures: list[str] = []
    if audit["weight_shape"] != [2, 768]:
        failures.append(f"weight shape {audit['weight_shape']} != [2, 768]")
    if audit["bias_shape"] != [2]:
        failures.append(f"bias shape {audit['bias_shape']} != [2]")
    if not bool(torch.isfinite(weight).all() and torch.isfinite(bias).all()):
        failures.append("classifier head contains non-finite values")
    if audit["weight_sha256"] != EXPECTED_BINARY_HEAD_WEIGHT_SHA256:
        failures.append(
            "weight SHA-256 "
            f"{audit['weight_sha256']} != {EXPECTED_BINARY_HEAD_WEIGHT_SHA256}"
        )
    if audit["bias_sha256"] != EXPECTED_BINARY_HEAD_BIAS_SHA256:
        failures.append(
            f"bias SHA-256 {audit['bias_sha256']} != {EXPECTED_BINARY_HEAD_BIAS_SHA256}"
        )
    if failures:
        raise RuntimeError(
            "MathDial binary classifier-head initialization drifted: "
            + "; ".join(failures)
        )
    return audit


def parse_mathdial(project_root: str | Path) -> pd.DataFrame:
    paths = discover_project_paths(project_root)
    source_root = paths.root / "Datasets" / "MathDial" / "data"
    frames: list[pd.DataFrame] = []
    for split in ("train", "test"):
        source_path = source_root / f"{split}.csv"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing MathDial source split: {source_path}")
        observed_sha = _sha256(source_path)
        if observed_sha != EXPECTED_SOURCE_SHA256[split]:
            raise ValueError(f"MathDial {split} source SHA-256 changed: {observed_sha}")
        source = pd.read_csv(source_path)
        required = {
            "qid",
            "scenario",
            "question",
            "student_incorrect_solution",
            "self-correctness",
            "conversation",
        }
        missing = sorted(required.difference(source.columns))
        if missing:
            raise ValueError(f"MathDial {split} is missing columns: {missing}")
        source["source_row"] = np.arange(len(source), dtype=np.int32)
        source = source.loc[source["self-correctness"].notna()].copy()
        unknown = sorted(set(source["self-correctness"].astype(str)).difference(LABEL_MAP))
        if unknown:
            raise ValueError(f"Unknown MathDial correctness labels: {unknown}")
        source["split"] = split
        source["outcome_label"] = source["self-correctness"].map(LABEL_MAP).astype(np.int8)
        source["example_id"] = (
            split
            + "-"
            + source["source_row"].map(lambda value: f"{int(value):06d}")
            + "-"
            + source["qid"].astype(str)
            + "-"
            + source["scenario"].astype(str)
        )
        for column in ("question", "student_incorrect_solution", "conversation"):
            source[column] = source[column].astype(str).map(
                lambda value: " ".join(value.split())
            )
            if source[column].eq("").any():
                raise ValueError(f"MathDial {split} has empty {column} values.")
        frames.append(
            source[
                [
                    "example_id",
                    "qid",
                    "scenario",
                    "split",
                    "question",
                    "student_incorrect_solution",
                    "conversation",
                    "self-correctness",
                    "outcome_label",
                ]
            ]
        )
    frame = pd.concat(frames, ignore_index=True)
    if frame["example_id"].duplicated().any():
        raise ValueError("MathDial example IDs are not unique across official splits.")
    train_qids = set(frame.loc[frame["split"].eq("train"), "qid"].astype(str))
    test_qids = set(frame.loc[frame["split"].eq("test"), "qid"].astype(str))
    if not train_qids or not test_qids:
        raise ValueError("MathDial contains no train or test question groups.")
    overlap = train_qids.intersection(test_qids)
    frame["external_train_eligible"] = frame["split"].eq("train") & ~frame[
        "qid"
    ].astype(str).isin(overlap)
    frame["external_test_eligible"] = frame["split"].eq("test")
    return frame.sort_values(["split", "example_id"], kind="mergesort").reset_index(
        drop=True
    )


def prepare_mathdial_cache(project_root: str | Path) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    frame = parse_mathdial(project_root)
    cache_path = paths.cache_dir / "mathdial_outcome_canonical.parquet"
    metadata_path = paths.cache_dir / "mathdial_outcome_canonical.metadata.json"
    frame.to_parquet(cache_path, index=False)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "MathDial",
        "source_commit": SOURCE_COMMIT,
        "license": "CC BY-SA 4.0",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "rows": int(len(frame)),
        "split_rows": {
            key: int(value) for key, value in frame["split"].value_counts().items()
        },
        "label_rows": {
            f"{split}:{label}": int(len(group))
            for (split, label), group in frame.groupby(
                ["split", "self-correctness"], sort=True
            )
        },
        "label_mapping": LABEL_MAP,
        "overlapping_official_question_ids_purged_from_train": int(
            frame.loc[
                frame["split"].eq("train") & ~frame["external_train_eligible"], "qid"
            ].nunique()
        ),
        "external_train_rows_after_question_purge": int(
            frame["external_train_eligible"].sum()
        ),
        "parquet_sha256": _sha256(cache_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "cache_path": str(cache_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }


def _texts(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    premise = [
        f"[QUESTION] {question} [INITIAL INCORRECT ANSWER] {answer}"
        for question, answer in zip(frame["question"], frame["student_incorrect_solution"])
    ]
    tutoring = [f"[TUTORING] {value}" for value in frame["conversation"]]
    return premise, tutoring


def _tokenize(tokenizer, frame: pd.DataFrame):
    import torch

    premise, tutoring = _texts(frame)
    encoded = tokenizer(
        premise,
        tutoring,
        padding="max_length",
        truncation="longest_first",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    keys = [
        key for key in ("input_ids", "attention_mask", "token_type_ids") if key in encoded
    ]
    return keys, [encoded[key].to(dtype=torch.long) for key in keys]


def _evaluate_binary(model, keys, tensors, labels: np.ndarray) -> tuple[np.ndarray, dict]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    batches: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            inputs = {key: value for key, value in zip(keys, batch)}
            batches.append(model(**inputs).logits.cpu().numpy().astype(np.float64))
            if batch_number % 10 == 0 or batch_number == len(loader):
                print(f"mathdial_eval_batch={batch_number}/{len(loader)}", flush=True)
    logits = np.concatenate(batches)
    logits -= logits.max(axis=1, keepdims=True)
    probability_matrix = np.exp(logits)
    probability_matrix /= probability_matrix.sum(axis=1, keepdims=True)
    probability = probability_matrix[:, 1]
    prediction = (probability >= 0.5).astype(np.int8)
    metrics = {
        "rows": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "prediction_mean": float(probability.mean()),
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(labels, probability)),
    }
    return probability, metrics


def train_mathdial_outcome_transfer(project_root: str | Path) -> dict[str, object]:
    import torch
    from safetensors.torch import load_file, save_file
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    paths = discover_project_paths(project_root)
    cache_path = paths.cache_dir / "mathdial_outcome_canonical.parquet"
    metadata_path = paths.cache_dir / "mathdial_outcome_canonical.metadata.json"
    if not cache_path.exists() or not metadata_path.exists():
        prepare_mathdial_cache(project_root)
    frame = pd.read_parquet(cache_path)
    train = frame.loc[frame["external_train_eligible"].astype(bool)].reset_index(drop=True)
    test = frame.loc[frame["external_test_eligible"].astype(bool)].reset_index(drop=True)
    if set(train["qid"].astype(str)).intersection(test["qid"].astype(str)):
        raise RuntimeError("MathDial outcome transfer has question leakage after purge.")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(6, max(1, torch.get_num_threads())))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    model_path = paths.root / "assets" / "pretrained" / MODEL_DIRECTORY
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    binary_head_initialization = _validate_binary_head_initialization(model)
    model.config.id2label = {0: "not_independent", 1: "independent"}
    model.config.label2id = {"not_independent": 0, "independent": 1}
    freeze_summary = _freeze_deberta(model, UNFROZEN_LAYERS)
    train_keys, train_tensors = _tokenize(tokenizer, train)
    test_keys, test_tensors = _tokenize(tokenizer, test)
    if train_keys != test_keys:
        raise RuntimeError("MathDial tokenizer fields differ across official splits.")
    train_labels = torch.tensor(train["outcome_label"].to_numpy(), dtype=torch.long)
    test_labels = test["outcome_label"].to_numpy(dtype=np.int64)
    counts = np.bincount(train_labels.numpy(), minlength=2).astype(np.float64)
    class_weights = torch.tensor(len(train) / (2.0 * counts), dtype=torch.float32)
    prior = float(train_labels.to(dtype=torch.float32).mean())
    prior_probability = np.full(len(test), prior, dtype=np.float64)
    prior_log_loss = float(log_loss(test_labels, prior_probability, labels=[0, 1]))

    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(*train_tensors, train_labels),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    classifier_ids = {id(parameter) for parameter in model.classifier.parameters()}
    head_parameters = list(model.classifier.parameters())
    encoder_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in classifier_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": ENCODER_LEARNING_RATE},
            {"params": head_parameters, "lr": HEAD_LEARNING_RATE},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = len(loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(round(total_steps * WARMUP_FRACTION)),
        num_training_steps=total_steps,
    )
    training_rows: list[dict[str, float | int]] = []
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        seen = 0
        for batch_number, batch in enumerate(loader, start=1):
            labels = batch[-1]
            inputs = {key: value for key, value in zip(train_keys, batch[:-1])}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            batch_rows = int(len(labels))
            seen += batch_rows
            running_loss += float(loss.item()) * batch_rows
            if batch_number % 20 == 0 or batch_number == len(loader):
                print(
                    f"mathdial_epoch={epoch + 1} step={batch_number}/{len(loader)} "
                    f"train_loss={running_loss / seen:.6f}",
                    flush=True,
                )
        training_rows.append(
            {"epoch": epoch + 1, "steps": len(loader), "train_loss": running_loss / seen}
        )

    probability, test_metrics = _evaluate_binary(
        model, test_keys, test_tensors, test_labels
    )
    clauses = {
        "test_roc_auc_at_least_0_65": test_metrics["roc_auc"] >= 0.65,
        "test_macro_f1_at_least_0_55": test_metrics["macro_f1"] >= 0.55,
        "test_log_loss_beats_train_prior": test_metrics["log_loss"] < prior_log_loss,
    }
    passes_gate = bool(all(clauses.values()))
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_mathdial_outcome_transfer")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    state = model.state_dict()
    delta = {
        name: state[name].detach().cpu().contiguous() for name in sorted(trainable)
    }
    delta_path = run_dir / "mathdial_deberta_delta.safetensors"
    save_file(delta, str(delta_path))
    loaded = load_file(str(delta_path), device="cpu")
    if set(loaded) != set(delta):
        raise RuntimeError("MathDial serialized delta key mismatch.")
    pd.DataFrame(training_rows).to_csv(run_dir / "training_metrics.csv", index=False)
    pd.DataFrame(
        {
            "example_id": test["example_id"],
            "target": test_labels,
            "probability": probability,
        }
    ).to_parquet(run_dir / "test_predictions.parquet", index=False)
    report = {
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "candidate": "E430_mathdial_outcome_deberta",
        "model": MODEL_NAME,
        "model_license": "Apache-2.0",
        "dataset": "MathDial",
        "dataset_license": "CC BY-SA 4.0",
        "source_commit": SOURCE_COMMIT,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "label_mapping": LABEL_MAP,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_positive_rate": prior,
        "test_metrics": test_metrics,
        "prior_log_loss": prior_log_loss,
        "external_gate_clauses": clauses,
        "passes_external_gate": passes_gate,
        "freeze_summary": freeze_summary,
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "batch_size": TRAIN_BATCH_SIZE,
        "class_weights": class_weights.tolist(),
        "binary_head_initialization": binary_head_initialization,
        "delta_file": delta_path.name,
        "delta_sha256": _sha256(delta_path),
        "delta_bytes": delta_path.stat().st_size,
        "V_final_accessed": False,
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"run_id": run_id, "run_dir": str(run_dir), "report": report}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare or train the preregistered MathDial outcome transfer."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--stage", choices=("prepare", "train", "pipeline"), default="pipeline")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.stage == "prepare":
        result = prepare_mathdial_cache(args.project_root)
    else:
        prepare_mathdial_cache(args.project_root)
        result = train_mathdial_outcome_transfer(args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
