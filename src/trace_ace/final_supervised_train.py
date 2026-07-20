from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from trace_ace.io import discover_project_paths
from trace_ace.supervised_encoder_screen import (
    ENCODER_LEARNING_RATE,
    EPOCHS,
    HEAD_LEARNING_RATE,
    MAX_LENGTH,
    SEED,
    TRAIN_BATCH_SIZE,
    UNFROZEN_LAYERS,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
    _freeze_lower_bert_layers,
    _tokenize_pairs,
)


def trainable_state_dict(model) -> dict[str, object]:
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = model.state_dict()
    result = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in state.items()
        if name in trainable
    }
    if set(result) != trainable:
        missing = sorted(trainable - set(result))
        raise RuntimeError(f"Trainable parameters missing from state dict: {missing}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_final_supervised_train(
    project_root: str | Path,
    max_length: int = MAX_LENGTH,
    epochs: int = EPOCHS,
) -> dict[str, object]:
    import torch
    from safetensors.torch import load_file, save_file
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
    target = frame["target"].to_numpy(dtype=np.int8)
    if len(target) != 35072:
        raise ValueError(f"Unexpected final training-row count: {len(target)}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    train_keys, train_tensors = _tokenize_pairs(
        tokenizer,
        contexts["objective_context"].astype(str).tolist(),
        frame["learning_objective"].astype(str).tolist(),
        int(max_length),
    )
    labels = torch.tensor(target, dtype=torch.float32)
    dataset = TensorDataset(*train_tensors, labels)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    freeze_summary = _freeze_lower_bert_layers(model, UNFROZEN_LAYERS)
    prior = float(np.mean(target))
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
    total_steps = len(loader) * int(epochs)
    warmup_steps = int(round(total_steps * WARMUP_FRACTION))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    loss_function = torch.nn.BCEWithLogitsLoss()

    training_rows: list[dict[str, float | int]] = []
    model.train()
    global_step = 0
    for epoch in range(int(epochs)):
        running_loss = 0.0
        seen = 0
        for batch_number, batch in enumerate(loader, start=1):
            batch_labels = batch[-1]
            inputs = {key: value for key, value in zip(train_keys, batch[:-1])}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits.squeeze(-1)
            loss = loss_function(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            batch_rows = int(len(batch_labels))
            running_loss += float(loss.item()) * batch_rows
            seen += batch_rows
            if batch_number % 50 == 0 or batch_number == len(loader):
                print(
                    f"epoch={epoch + 1} step={batch_number}/{len(loader)} "
                    f"train_loss={running_loss / max(1, seen):.6f}",
                    flush=True,
                )
        training_rows.append(
            {"epoch": epoch + 1, "steps": global_step, "train_loss": running_loss / seen}
        )

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_final_supervised_train")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    delta_path = run_dir / "supervised_delta.safetensors"
    delta = trainable_state_dict(model)
    save_file(delta, str(delta_path))

    # Verify the serialized delta exactly reconstructs every trainable tensor.
    loaded = load_file(str(delta_path), device="cpu")
    if set(loaded) != set(delta):
        raise RuntimeError("Serialized supervised delta key mismatch.")
    max_serialization_difference = max(
        float(torch.max(torch.abs(delta[name] - loaded[name])).item()) for name in delta
    )
    if max_serialization_difference != 0.0:
        raise RuntimeError("Serialized supervised delta failed exact tensor parity.")

    pd.DataFrame(training_rows).to_csv(
        run_dir / "training_metrics.csv", index=False, lineterminator="\n"
    )
    report = {
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "model": "BAAI/bge-small-en-v1.5",
        "license": "MIT",
        "training_rows": int(len(frame)),
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
        "total_steps": total_steps,
        "trainable_tensor_count": len(delta),
        "delta_file": delta_path.name,
        "delta_bytes": delta_path.stat().st_size,
        "delta_sha256": _sha256(delta_path),
        "max_serialization_difference": max_serialization_difference,
        "locked_submission_weight": 0.30,
        "confirmation_run_ids": [
            "20260717T063544Z_supervised_encoder_screen",
            "20260717T082126Z_supervised_encoder_screen",
        ],
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"run_id": run_id, "run_dir": str(run_dir), "report": report}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train the locked supervised BGE-small component on all labeled rows."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_final_supervised_train(
        args.project_root,
        max_length=args.max_length,
        epochs=args.epochs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
