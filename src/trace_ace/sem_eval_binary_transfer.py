from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
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
    MAX_LENGTH,
    MODEL_DIRECTORY,
    MODEL_NAME,
    TRAIN_BATCH_SIZE,
    UNFROZEN_LAYERS,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
    _freeze_deberta,
    _sha256,
    _tokenize,
    _trainable_state_dict,
)
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics


SEED = 20260724
EXPECTED_DATASET_CONTENT_SHA256 = (
    "7a51ebe3b8c6d1537648c511836eb54360831006053eb65980f7736bd6351dfb"
)
EXPECTED_BINARY_HEAD_WEIGHT_SHA256 = (
    "696ad4a9b5d7c57ccf4906b3aca4a9e9d85921429b92ca7f096aacd6c4e6c55c"
)
EXPECTED_BINARY_HEAD_BIAS_SHA256 = (
    "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"
)
EXPECTED_SPLIT_ROWS = {
    "train": 8910,
    "test-unseen-questions": 1552,
    "test-unseen-domains": 4562,
}
EXPECTED_RUNTIME = {
    "python": "3.12.8",
    "numpy": "2.5.1",
    "scikit_learn": "1.8.0",
    "torch": "2.13.0+cpu",
    "transformers": "5.14.1",
}
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260724


def assert_e460_runtime() -> dict[str, str]:
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
            f"E460 runtime differs from the frozen .venv contract: {observed}"
        )
    return observed


def binary_mastery_labels(frame: pd.DataFrame) -> np.ndarray:
    labels = frame["sra_label"].astype(str).eq("correct").to_numpy(dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("E460 split must contain both binary mastery labels.")
    return labels


def _tensor_sha256(tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def validate_binary_head_initialization(model) -> dict[str, object]:
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
        failures.append(f"weight SHA-256 drifted to {audit['weight_sha256']}")
    if audit["bias_sha256"] != EXPECTED_BINARY_HEAD_BIAS_SHA256:
        failures.append(f"bias SHA-256 drifted to {audit['bias_sha256']}")
    if failures:
        raise RuntimeError(
            "E460 binary classifier-head initialization drifted: "
            + "; ".join(failures)
        )
    return audit


def _evaluate_binary(
    model, keys, tensors, labels: np.ndarray
) -> tuple[np.ndarray, dict[str, float | int]]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            inputs = {key: value for key, value in zip(keys, batch, strict=True)}
            probability = torch.softmax(model(**inputs).logits, dim=1)[:, 1]
            batches.append(probability.cpu().numpy().astype(np.float64))
            if batch_number % 25 == 0 or batch_number == len(loader):
                print(f"binary_eval_batch={batch_number}/{len(loader)}", flush=True)
    probability = np.concatenate(batches)
    if len(probability) != len(labels) or not np.isfinite(probability).all():
        raise RuntimeError("E460 produced invalid external probabilities.")
    metrics: dict[str, float | int] = {
        "rows": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "accuracy": float(accuracy_score(labels, probability >= 0.5)),
        "macro_f1": float(f1_score(labels, probability >= 0.5, average="macro")),
        **binary_metrics(labels, probability),
    }
    return probability, metrics


def question_bootstrap(
    frame: pd.DataFrame,
    labels: np.ndarray,
    prior_probability: np.ndarray,
    candidate_probability: np.ndarray,
) -> dict[str, object]:
    target = labels.astype(np.float64)
    prior = np.clip(prior_probability, 1e-6, 1.0 - 1e-6)
    candidate = np.clip(candidate_probability, 1e-6, 1.0 - 1e-6)
    prior_loss = -(target * np.log(prior) + (1 - target) * np.log1p(-prior))
    candidate_loss = -(
        target * np.log(candidate) + (1 - target) * np.log1p(-candidate)
    )
    gains = pd.DataFrame(
        {"question_id": frame["question_id"].astype(str), "gain": prior_loss - candidate_loss}
    )
    grouped = gains.groupby("question_id", sort=True)["gain"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=np.float64)
    counts = grouped["count"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        values[replicate] = sums[sampled].sum() / counts[sampled].sum()
    return {
        "resampler": "question_id",
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


def _gate_for_split(
    split: str,
    metrics: dict[str, float | int],
    prior_metrics: dict[str, float],
    bootstrap: dict[str, object],
) -> dict[str, bool]:
    auc_floor = 0.72 if split == "test-unseen-questions" else 0.76
    return {
        "auroc_at_least_floor": float(metrics["roc_auc"]) >= auc_floor,
        "macro_f1_at_least_0_65": float(metrics["macro_f1"]) >= 0.65,
        "ece_10_at_most_0_10": float(metrics["ece_10"]) <= 0.10,
        "log_loss_gain_at_least_0_02": (
            float(prior_metrics["log_loss"]) - float(metrics["log_loss"]) >= 0.02
        ),
        "brier_gain_at_least_0_01": (
            float(prior_metrics["brier_score"]) - float(metrics["brier_score"])
            >= 0.01
        ),
        "question_bootstrap_support_at_least_0_95": (
            float(bootstrap["support_positive_log_loss_gain"]) >= 0.95
        ),
    }


def train_sem_eval_binary_transfer(project_root: str | Path) -> dict[str, object]:
    import torch
    from safetensors.torch import load_file, save_file
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    runtime = assert_e460_runtime()
    paths = discover_project_paths(project_root)
    cache_path = paths.cache_dir / "sem_eval_2013_task7_canonical.parquet"
    metadata_path = paths.cache_dir / "sem_eval_2013_task7_canonical.metadata.json"
    if not cache_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("E460 requires the audited canonical SemEval cache.")
    dataset_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if dataset_metadata.get("ordered_content_sha256") != EXPECTED_DATASET_CONTENT_SHA256:
        raise ValueError("E460 canonical SemEval content SHA-256 changed.")
    frame = pd.read_parquet(cache_path)
    splits = {
        split: frame.loc[frame["split"].eq(split)].reset_index(drop=True)
        for split in EXPECTED_SPLIT_ROWS
    }
    observed_rows = {split: len(value) for split, value in splits.items()}
    if observed_rows != EXPECTED_SPLIT_ROWS:
        raise ValueError(f"E460 external split counts changed: {observed_rows}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(6)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

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
    tokenized = {split: _tokenize(tokenizer, value) for split, value in splits.items()}
    keys = tokenized["train"][0]
    if any(value[0] != keys for value in tokenized.values()):
        raise RuntimeError("E460 tokenizer fields differ across external splits.")

    train_labels_np = binary_mastery_labels(splits["train"])
    train_labels = torch.tensor(train_labels_np, dtype=torch.long)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        TensorDataset(*tokenized["train"][1], train_labels),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
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
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(round(total_steps * WARMUP_FRACTION)),
        num_training_steps=total_steps,
    )
    training_rows: list[dict[str, float | int]] = []
    model.train()
    global_step = 0
    for epoch in range(EPOCHS):
        running_loss = 0.0
        seen = 0
        for batch_number, batch in enumerate(train_loader, start=1):
            batch_labels = batch[-1]
            inputs = {key: value for key, value in zip(keys, batch[:-1], strict=True)}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**inputs, labels=batch_labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            batch_rows = int(len(batch_labels))
            seen += batch_rows
            running_loss += float(outputs.loss.item()) * batch_rows
            if batch_number % 25 == 0 or batch_number == len(train_loader):
                print(
                    f"binary_epoch={epoch + 1} step={batch_number}/{len(train_loader)} "
                    f"train_loss={running_loss / seen:.6f}",
                    flush=True,
                )
        training_rows.append(
            {"epoch": epoch + 1, "steps": global_step, "train_loss": running_loss / seen}
        )

    train_prior = float(train_labels_np.mean())
    evaluation: dict[str, dict[str, object]] = {}
    prediction_frames: list[pd.DataFrame] = []
    all_clauses: dict[str, dict[str, bool]] = {}
    for split in ("test-unseen-questions", "test-unseen-domains"):
        split_frame = splits[split]
        labels = binary_mastery_labels(split_frame)
        probability, metrics = _evaluate_binary(
            model, tokenized[split][0], tokenized[split][1], labels
        )
        prior_probability = np.full(len(labels), train_prior, dtype=np.float64)
        prior_metrics = binary_metrics(labels, prior_probability)
        bootstrap = question_bootstrap(
            split_frame, labels, prior_probability, probability
        )
        clauses = _gate_for_split(split, metrics, prior_metrics, bootstrap)
        evaluation[split] = {
            "metrics": metrics,
            "train_prior_metrics": prior_metrics,
            "paired_question_bootstrap": bootstrap,
        }
        all_clauses[split] = clauses
        prediction_frames.append(
            pd.DataFrame(
                {
                    "student_answer_id": split_frame["student_answer_id"],
                    "question_id": split_frame["question_id"],
                    "split": split,
                    "target": labels,
                    "probability": probability,
                    "train_prior_probability": train_prior,
                }
            )
        )
    passes_gate = bool(
        all(all(clauses.values()) for clauses in all_clauses.values())
    )

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_sem_eval_binary_transfer")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    delta_path = run_dir / "sem_eval_binary_deberta_delta.safetensors"
    delta = _trainable_state_dict(model)
    save_file(delta, str(delta_path))
    loaded = load_file(str(delta_path), device="cpu")
    if set(loaded) != set(delta):
        raise RuntimeError("E460 serialized delta key mismatch.")
    max_difference = max(
        float(torch.max(torch.abs(delta[name] - loaded[name])).item()) for name in delta
    )
    if max_difference != 0.0:
        raise RuntimeError("E460 delta serialization is not exact.")
    pd.DataFrame(training_rows).to_csv(
        run_dir / "training_metrics.csv", index=False, lineterminator="\n"
    )
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        run_dir / "external_predictions.parquet", index=False
    )
    metric_rows = []
    for split, result in evaluation.items():
        metric_rows.append({"split": split, **result["metrics"]})
    pd.DataFrame(metric_rows).to_csv(
        run_dir / "external_metrics.csv", index=False, lineterminator="\n"
    )
    report = {
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "candidate": "E460_sem_eval_binary_mastery_transfer",
        "model": MODEL_NAME,
        "model_license": "Apache-2.0",
        "external_dataset": "SemEval-2013 Task 7 Student Response Analysis",
        "external_dataset_license": "CC BY-SA 3.0",
        "dataset_content_sha256": EXPECTED_DATASET_CONTENT_SHA256,
        "label_contract": "correct=1; every other five-way label=0",
        "training_rows": int(len(splits["train"])),
        "train_positive_rate": train_prior,
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
        "external_gate_clauses": all_clauses,
        "passes_external_gate": passes_gate,
        "delta_file": delta_path.name,
        "delta_bytes": delta_path.stat().st_size,
        "delta_sha256": _sha256(delta_path),
        "trainable_tensor_count": len(delta),
        "max_serialization_difference": max_difference,
        "runtime_contract": runtime,
        "E400_delta_loaded": False,
        "competition_cache_built": False,
        "V_final_accessed": False,
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"run_id": run_id, "run_dir": str(run_dir), "report": report}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train the frozen E460 binary SemEval mastery transfer."
    )
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = train_sem_eval_binary_transfer(args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
