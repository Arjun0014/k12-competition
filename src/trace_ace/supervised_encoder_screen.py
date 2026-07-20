from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from trace_ace.hard_validation import _fold_masks
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.multiview_validation import BASELINE_RUN_ID, PILOT_PROTOCOL


SEED = 20260717
TRAIN_ROWS = 4096
VALIDATION_FOLD = 0
MAX_LENGTH = 192
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
EPOCHS = 1
ENCODER_LEARNING_RATE = 3e-5
HEAD_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.10
UNFROZEN_LAYERS = 4
BLEND_WEIGHTS = (0.10, 0.20, 0.30, 0.40, 0.50)
HYPOTHESIS_PREFIX = "The student demonstrates mastery of this learning objective: "


def mastery_hypothesis(objective: str) -> str:
    return f"{HYPOTHESIS_PREFIX}{str(objective).strip()}."


def stratified_screen_indices(
    candidate_indices: np.ndarray,
    target: np.ndarray,
    train_rows: int = TRAIN_ROWS,
    seed: int = SEED,
) -> np.ndarray:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if len(candidate_indices) <= train_rows:
        return np.sort(candidate_indices)
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=int(train_rows), random_state=int(seed)
    )
    positions, _ = next(
        splitter.split(np.zeros(len(candidate_indices)), target[candidate_indices])
    )
    return np.sort(candidate_indices[positions])


def _tokenize_pairs(tokenizer, contexts: list[str], objectives: list[str], max_length: int):
    import torch

    encoded = tokenizer(
        contexts,
        [mastery_hypothesis(value) for value in objectives],
        padding="max_length",
        truncation="only_first",
        max_length=int(max_length),
        return_tensors="pt",
    )
    keys = [key for key in ("input_ids", "attention_mask", "token_type_ids") if key in encoded]
    return keys, [encoded[key].to(dtype=torch.long) for key in keys]


def _freeze_lower_bert_layers(model, unfrozen_layers: int) -> dict[str, int]:
    if not hasattr(model, "bert"):
        raise TypeError("The supervised screen expects a BERT-family encoder.")
    layers = model.bert.encoder.layer
    if not 1 <= unfrozen_layers <= len(layers):
        raise ValueError("unfrozen_layers must retain at least one trainable encoder layer.")
    for parameter in model.bert.embeddings.parameters():
        parameter.requires_grad = False
    frozen_count = len(layers) - int(unfrozen_layers)
    for layer in layers[:frozen_count]:
        for parameter in layer.parameters():
            parameter.requires_grad = False
    return {"total_encoder_layers": len(layers), "frozen_encoder_layers": frozen_count}


def run_supervised_encoder_screen(
    project_root: str | Path,
    train_rows: int = TRAIN_ROWS,
    validation_fold: int = VALIDATION_FOLD,
    max_length: int = MAX_LENGTH,
    epochs: int = EPOCHS,
) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    paths = discover_project_paths(project_root)
    model_path = paths.root / "assets" / "pretrained" / "bge-small-en-v1.5"
    if not model_path.is_dir():
        raise FileNotFoundError(f"Packaged encoder not found: {model_path}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(6, max(1, torch.get_num_threads())))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    contexts = pd.read_parquet(
        paths.cache_dir / "response_objective_context.parquet",
        columns=["response_id", "objective_context"],
    )
    if list(contexts["response_id"]) != list(frame["response_id"]):
        raise ValueError("Objective-context rows do not align with the modeling frame.")
    baseline = pd.read_parquet(
        paths.experiments_dir / "runs" / BASELINE_RUN_ID / "oof_predictions.parquet"
    )
    baseline = baseline.loc[baseline["protocol"].eq(PILOT_PROTOCOL)].reset_index(drop=True)
    if list(baseline["response_id"]) != list(frame["response_id"]):
        raise ValueError("Pilot baseline rows do not align with the modeling frame.")

    folds = baseline["fold"].to_numpy(dtype=np.int8)
    train_mask, validation_mask = _fold_masks(frame, folds, int(validation_fold))
    candidate_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    target = frame["target"].to_numpy(dtype=np.int8)
    train_indices = stratified_screen_indices(candidate_indices, target, train_rows)
    if set(frame.loc[train_indices, "session_id"]).intersection(
        set(frame.loc[validation_indices, "session_id"])
    ):
        raise RuntimeError("Session leakage detected in the supervised screen.")
    if set(frame.loc[train_indices, "learning_objective_id"]).intersection(
        set(frame.loc[validation_indices, "learning_objective_id"])
    ):
        raise RuntimeError("Objective leakage detected in the supervised screen.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    train_keys, train_tensors = _tokenize_pairs(
        tokenizer,
        contexts.loc[train_indices, "objective_context"].astype(str).tolist(),
        frame.loc[train_indices, "learning_objective"].astype(str).tolist(),
        max_length,
    )
    validation_keys, validation_tensors = _tokenize_pairs(
        tokenizer,
        contexts.loc[validation_indices, "objective_context"].astype(str).tolist(),
        frame.loc[validation_indices, "learning_objective"].astype(str).tolist(),
        max_length,
    )
    if train_keys != validation_keys:
        raise RuntimeError("Train and validation tokenizer fields differ.")
    train_labels = torch.tensor(target[train_indices], dtype=torch.float32)
    validation_labels = target[validation_indices].astype(np.float64)
    train_dataset = TensorDataset(*train_tensors, train_labels)
    validation_dataset = TensorDataset(*validation_tensors)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    freeze_summary = _freeze_lower_bert_layers(model, UNFROZEN_LAYERS)
    prior = float(np.mean(target[train_indices]))
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.fill_(math.log(prior / (1.0 - prior)))

    head_ids = {id(parameter) for parameter in model.classifier.parameters()}
    encoder_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": ENCODER_LEARNING_RATE},
            {"params": list(model.classifier.parameters()), "lr": HEAD_LEARNING_RATE},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * int(epochs)
    warmup_steps = int(round(total_steps * WARMUP_FRACTION))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    loss_function = torch.nn.BCEWithLogitsLoss()

    training_rows: list[dict[str, float | int]] = []
    model.train()
    global_step = 0
    for epoch in range(int(epochs)):
        running_loss = 0.0
        seen = 0
        for batch_number, batch in enumerate(train_loader, start=1):
            labels = batch[-1]
            inputs = {key: value for key, value in zip(train_keys, batch[:-1])}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits.squeeze(-1)
            loss = loss_function(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            batch_rows = int(len(labels))
            running_loss += float(loss.item()) * batch_rows
            seen += batch_rows
            if batch_number % 25 == 0 or batch_number == len(train_loader):
                average_loss = running_loss / max(1, seen)
                print(
                    f"epoch={epoch + 1} step={batch_number}/{len(train_loader)} "
                    f"train_loss={average_loss:.6f}",
                    flush=True,
                )
        training_rows.append(
            {"epoch": epoch + 1, "steps": global_step, "train_loss": running_loss / seen}
        )

    model.eval()
    prediction_batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch_number, batch in enumerate(validation_loader, start=1):
            inputs = {key: value for key, value in zip(validation_keys, batch)}
            probability = torch.sigmoid(model(**inputs).logits.squeeze(-1))
            prediction_batches.append(probability.cpu().numpy().astype(np.float64))
            if batch_number % 50 == 0 or batch_number == len(validation_loader):
                print(
                    f"validation_batch={batch_number}/{len(validation_loader)}",
                    flush=True,
                )
    prediction = np.concatenate(prediction_batches)
    if len(prediction) != len(validation_indices) or not np.isfinite(prediction).all():
        raise RuntimeError("Supervised screen produced invalid validation predictions.")

    baseline_prediction = baseline.loc[validation_indices, "pred_v02"].to_numpy(
        dtype=np.float64
    )
    baseline_metrics = binary_metrics(validation_labels, baseline_prediction)
    metric_rows = [
        {"model": "v02_baseline", "model_weight": 0.0, **baseline_metrics}
    ]
    standalone_metrics = binary_metrics(validation_labels, prediction)
    metric_rows.append(
        {"model": "supervised_bge_small", "model_weight": 1.0, **standalone_metrics}
    )
    for weight in BLEND_WEIGHTS:
        blended = (1.0 - weight) * baseline_prediction + weight * prediction
        metrics = binary_metrics(validation_labels, blended)
        metric_rows.append(
            {
                "model": f"v02_blend_supervised_{weight:.2f}",
                "model_weight": weight,
                **metrics,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics["delta_log_loss_vs_v02"] = metrics["log_loss"] - baseline_metrics["log_loss"]
    metrics["delta_roc_auc_vs_v02"] = metrics["roc_auc"] - baseline_metrics["roc_auc"]
    metrics = metrics.sort_values(["log_loss", "roc_auc"], ascending=[True, False])

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_supervised_encoder_screen")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")
    pd.DataFrame(training_rows).to_csv(
        run_dir / "training_metrics.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(
        {
            "response_id": frame.loc[validation_indices, "response_id"].to_numpy(),
            "target": validation_labels,
            "fold": int(validation_fold),
            "pred_v02": baseline_prediction,
            "pred_supervised": prediction,
        }
    ).to_parquet(run_dir / "validation_predictions.parquet", index=False)
    best = metrics.iloc[0].to_dict()
    report = {
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "model": "BAAI/bge-small-en-v1.5",
        "license": "MIT",
        "protocol": PILOT_PROTOCOL,
        "validation_fold": int(validation_fold),
        "training_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "max_length": int(max_length),
        "epochs": int(epochs),
        "train_batch_size": TRAIN_BATCH_SIZE,
        "encoder_learning_rate": ENCODER_LEARNING_RATE,
        "head_learning_rate": HEAD_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_fraction": WARMUP_FRACTION,
        "unfrozen_layers": UNFROZEN_LAYERS,
        "seed": SEED,
        "training_positive_rate": prior,
        "freeze_summary": freeze_summary,
        "best_validation_result": best,
        "continuation_rule": (
            "Continue only if the fixed final checkpoint adds at least 0.001 log-loss "
            "improvement without AUROC loss, or 0.005 AUROC without log-loss regression."
        ),
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"run_id": run_id, "run_dir": str(run_dir), "metrics": metrics}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered supervised BGE-small outcome-adaptation screen."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--train-rows", type=int, default=TRAIN_ROWS)
    parser.add_argument("--validation-fold", type=int, default=VALIDATION_FOLD)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_supervised_encoder_screen(
        args.project_root,
        train_rows=args.train_rows,
        validation_fold=args.validation_fold,
        max_length=args.max_length,
        epochs=args.epochs,
    )
    print(f"Supervised encoder screen complete: {result['run_id']}")
    print(result["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
