from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from trace_ace.external_sra_transfer import (
    ENCODER_LEARNING_RATE,
    EPOCHS,
    EVAL_BATCH_SIZE,
    EXPECTED_LABELS,
    EXPECTED_SPLIT_ROWS,
    HEAD_LEARNING_RATE,
    MAX_LENGTH,
    MODEL_DIRECTORY,
    MODEL_NAME,
    TRAIN_BATCH_SIZE,
    UNFROZEN_LAYERS,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
    _evaluate_external,
    _freeze_deberta,
    build_competition_session_cache as build_adapted_session_cache,
    _premises,
    _tokenize,
    _trainable_state_dict,
)
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.source_robust_validation import _current_rss_bytes


PROTOCOL_ID = "E580_semeval_gsm8k_correctness_transfer_v1"
SEED = 20260725
GSM_TRAIN_QUESTIONS = 4_455
TOTAL_TRAIN_ROWS = 17_820
MAX_PROJECTED_HOURS = 6.0
MAX_RSS_BYTES = 8 * 1024**3
EXPECTED_RUNTIME = {
    "python": "3.12.8",
    "numpy": "2.5.1",
    "scikit_learn": "1.8.0",
    "torch": "2.13.0+cpu",
    "transformers": "5.14.1",
}
SOURCE_HASHES = {
    "semeval": "527256be4e5f2e509f725b302ab8b163cc57760560a58621b2c02880ebe5dbb1",
    "gsm_train": "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d",
    "gsm_test": "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
    "model": "ebc79588dd73ccfb6a3f6078519cfbf512c5305384c5ea1845bc71cd32216e86",
    "config": "885d0dceae8fa5c136da9209121ec9eb11160488e840de3bc1f29353674e5712",
    "tokenizer": "5124ef2ead1a10a717703bc436de7f353da76d6340e4587719b42b1693707964",
    "spm": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
}
SELECTED_QUESTION_SHA256 = (
    "2bbcc18022ccf879c60a2cbaf46d8487e89e74c22c723fe4130a9f8ea1bb4092"
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_frame_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    digest = hashlib.sha256()
    for row in frame[columns].itertuples(index=False, name=None):
        normalized = tuple(
            None
            if value is pd.NA or pd.isna(value)
            else value.item()
            if isinstance(value, np.generic)
            else value
            for value in row
        )
        digest.update(
            json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _paths(project_root: str | Path) -> dict[str, Path]:
    paths = discover_project_paths(project_root)
    model = paths.root / "assets" / "pretrained" / MODEL_DIRECTORY
    gsm = paths.root / "Datasets" / "Grade School Math (GSM8k)" / "main"
    return {
        "semeval": paths.cache_dir / "sem_eval_2013_task7_canonical.parquet",
        "gsm_train": gsm / "train.parquet",
        "gsm_test": gsm / "test.parquet",
        "model": model / "model.safetensors",
        "config": model / "config.json",
        "tokenizer": model / "tokenizer.json",
        "spm": model / "spm.model",
        "model_dir": model,
        "cache": paths.cache_dir / "multicorpus_correctness_e580.parquet",
        "metadata": paths.cache_dir / "multicorpus_correctness_e580.metadata.json",
        "benchmark": paths.cache_dir / "multicorpus_correctness_e580_benchmark.json",
    }


def verify_sources(project_root: str | Path) -> dict[str, str]:
    paths = _paths(project_root)
    for name, expected in SOURCE_HASHES.items():
        path = paths[name]
        if not path.exists():
            raise FileNotFoundError(f"Missing E580 {name}: {path}")
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"E580 {name} SHA-256 changed: {observed}")
    return dict(SOURCE_HASHES)


def assert_runtime() -> dict[str, str]:
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
        raise RuntimeError(f"E580 runtime differs from frozen .venv: {observed}")
    return observed


def final_integer(answer: str) -> int:
    value = str(answer).rsplit("####", 1)
    if len(value) != 2:
        raise ValueError("E580 GSM answer lacks a final #### marker.")
    normalized = value[1].strip().replace(",", "")
    if not re.fullmatch(r"[-+]?\d+", normalized):
        raise ValueError(f"E580 GSM final answer is not an integer: {normalized!r}")
    return int(normalized)


def corrupt_final_integer(answer: str) -> tuple[str, int, int]:
    original = final_integer(answer)
    corrupted = original + 1 if original >= 0 else original - 1
    prefix, _ = str(answer).rsplit("####", 1)
    return f"{prefix}#### {corrupted}", original, corrupted


def _question_hash_order(frame: pd.DataFrame) -> np.ndarray:
    rows = list(range(len(frame)))
    rows.sort(
        key=lambda index: (
            hashlib.sha256(str(frame.iloc[index]["question"]).encode()).hexdigest(),
            index,
        )
    )
    return np.asarray(rows, dtype=np.int64)


def build_gsm_rows(
    frame: pd.DataFrame,
    *,
    split: str,
    selected_rows: int | None,
) -> pd.DataFrame:
    required = {"question", "answer"}
    if not required.issubset(frame.columns):
        raise ValueError("E580 GSM source schema changed.")
    if frame["question"].duplicated().any():
        raise ValueError("E580 GSM questions are not unique.")
    order = _question_hash_order(frame)
    if selected_rows is not None:
        if not 0 < selected_rows <= len(order):
            raise ValueError("E580 selected GSM row count is invalid.")
        order = order[:selected_rows]
    records: list[dict[str, object]] = []
    for source_row in order:
        question = " ".join(str(frame.iloc[source_row]["question"]).split())
        answer = str(frame.iloc[source_row]["answer"])
        corrupted_answer, correct_integer, corrupted_integer = corrupt_final_integer(
            answer
        )
        hypothesis = f"The correct final answer is {correct_integer}."
        for variant, candidate, label in (
            ("correct", answer, 1),
            ("corrupted", corrupted_answer, 0),
        ):
            records.append(
                {
                    "example_id": f"gsm_{split}_{source_row}_{variant}",
                    "corpus": "GSM8K",
                    "split": split,
                    "premise": f"[QUESTION] {question} [STUDENT ANSWER] {candidate}",
                    "hypothesis": hypothesis,
                    "nli_label": label,
                    "source_row": int(source_row),
                    "variant": variant,
                    "correct_integer": correct_integer,
                    "candidate_integer": (
                        correct_integer if variant == "correct" else corrupted_integer
                    ),
                }
            )
    result = pd.DataFrame(records)
    expected = 2 * (selected_rows if selected_rows is not None else len(frame))
    if len(result) != expected or result["example_id"].duplicated().any():
        raise RuntimeError("E580 GSM construction failed.")
    return result


def prepare_multicorpus_cache(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    source_hashes = verify_sources(project_root)
    paths = _paths(project_root)
    semeval = pd.read_parquet(paths["semeval"])
    semeval_train = semeval.loc[semeval["split"].eq("train")].reset_index(drop=True)
    if len(semeval_train) != EXPECTED_SPLIT_ROWS["train"]:
        raise ValueError("E580 SemEval train count changed.")
    gsm_train_source = pd.read_parquet(paths["gsm_train"])
    gsm_test_source = pd.read_parquet(paths["gsm_test"])
    if len(gsm_train_source) != 7_473 or len(gsm_test_source) != 1_319:
        raise ValueError("E580 GSM source row counts changed.")
    if any(final_integer(value) is None for value in gsm_train_source["answer"]):
        raise RuntimeError("E580 unreachable GSM parse audit failed.")
    selected_order = _question_hash_order(gsm_train_source)[:GSM_TRAIN_QUESTIONS]
    selected_question_sha = hashlib.sha256(
        "\n".join(
            gsm_train_source.iloc[selected_order]["question"].astype(str)
        ).encode()
    ).hexdigest()
    if selected_question_sha != SELECTED_QUESTION_SHA256:
        raise ValueError("E580 selected GSM question hash changed.")

    gsm_train = build_gsm_rows(
        gsm_train_source, split="train", selected_rows=GSM_TRAIN_QUESTIONS
    )
    gsm_test = build_gsm_rows(gsm_test_source, split="test", selected_rows=None)
    semeval_rows = pd.DataFrame(
        {
            "example_id": (
                "semeval_" + semeval_train["student_answer_id"].astype(str)
            ),
            "corpus": "SemEval",
            "split": "train",
            "premise": _premises(semeval_train),
            "hypothesis": semeval_train["reference_answer"].astype(str),
            "nli_label": semeval_train["nli_label"].astype(np.int64),
            "source_row": np.arange(len(semeval_train), dtype=np.int64),
            "variant": semeval_train["sra_label"].astype(str),
            "correct_integer": pd.array([pd.NA] * len(semeval_train), dtype="Int64"),
            "candidate_integer": pd.array([pd.NA] * len(semeval_train), dtype="Int64"),
        }
    )
    combined_train = pd.concat([semeval_rows, gsm_train], ignore_index=True)
    if len(combined_train) != TOTAL_TRAIN_ROWS:
        raise RuntimeError("E580 combined training count changed.")
    rng = np.random.default_rng(SEED)
    combined_train = combined_train.iloc[
        rng.permutation(len(combined_train))
    ].reset_index(drop=True)
    combined_train["split"] = "train"
    cache = pd.concat([combined_train, gsm_test], ignore_index=True)
    columns = [
        "example_id",
        "corpus",
        "split",
        "premise",
        "hypothesis",
        "nli_label",
        "source_row",
        "variant",
        "correct_integer",
        "candidate_integer",
    ]
    cache.to_parquet(paths["cache"], index=False, compression="zstd")
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(cache),
        "training_rows": len(combined_train),
        "training_rows_by_corpus": {
            key: int(value)
            for key, value in combined_train["corpus"].value_counts().items()
        },
        "gsm_test_rows": len(gsm_test),
        "selected_gsm_train_questions": GSM_TRAIN_QUESTIONS,
        "selected_question_sha256": selected_question_sha,
        "ordered_content_sha256": _ordered_frame_sha256(cache, columns),
        "parquet_sha256": _sha256(paths["cache"]),
        "source_hashes": source_hashes,
        "runtime": runtime,
        "V_final_accessed": False,
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"cache_path": str(paths["cache"]), "metadata": metadata}


def _load_cache(project_root: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    paths = _paths(project_root)
    if not paths["cache"].exists() or not paths["metadata"].exists():
        prepare_multicorpus_cache(project_root)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if (
        metadata.get("protocol_id") != PROTOCOL_ID
        or metadata.get("source_hashes") != verify_sources(project_root)
        or metadata.get("parquet_sha256") != _sha256(paths["cache"])
    ):
        raise ValueError("E580 cache binding changed.")
    return pd.read_parquet(paths["cache"]), metadata


def _tokenize_pairs(tokenizer, frame: pd.DataFrame):
    import torch

    encoded = tokenizer(
        frame["premise"].astype(str).tolist(),
        frame["hypothesis"].astype(str).tolist(),
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


def _gsm_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    if probability.shape != (len(labels), 3):
        raise ValueError("E580 GSM probability shape changed.")
    two_class = probability[:, :2]
    denominator = two_class.sum(axis=1)
    if np.any(denominator <= 0):
        raise ValueError("E580 GSM binary normalization failed.")
    correct_probability = two_class[:, 1] / denominator
    metrics = binary_metrics(labels, correct_probability)
    return {
        "rows": len(labels),
        "accuracy": float(accuracy_score(labels, correct_probability >= 0.5)),
        "roc_auc": float(roc_auc_score(labels, correct_probability)),
        "log_loss": float(log_loss(labels, correct_probability, labels=[0, 1])),
        "brier_score": metrics["brier_score"],
        "ece_10": metrics["ece_10"],
        "prediction_mean": metrics["prediction_mean"],
        "mean_neutral_probability": float(probability[:, 2].mean()),
    }


def _evaluate_gsm(model, keys, tensors, labels: np.ndarray):
    probability, _ = _evaluate_external(model, keys, tensors, labels)
    return probability, _gsm_metrics(labels, probability)


def _configure_runtime() -> None:
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(6)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _load_model_and_tokenizer(project_root: str | Path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    paths = _paths(project_root)
    tokenizer = AutoTokenizer.from_pretrained(
        paths["model_dir"], local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        paths["model_dir"], local_files_only=True
    )
    actual_labels = {
        int(key): str(value).lower() for key, value in model.config.id2label.items()
    }
    if actual_labels != EXPECTED_LABELS:
        raise ValueError(f"E580 base NLI labels changed: {actual_labels}")
    return model, tokenizer


def benchmark(project_root: str | Path) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    runtime = assert_runtime()
    verify_sources(project_root)
    _configure_runtime()
    cache, metadata = _load_cache(project_root)
    train = cache.loc[cache["split"].eq("train")].reset_index(drop=True)
    gsm_test = cache.loc[cache["split"].eq("test")].reset_index(drop=True)
    model, tokenizer = _load_model_and_tokenizer(project_root)
    freeze_summary = _freeze_deberta(model, UNFROZEN_LAYERS)
    train_sample = train.iloc[: 2 * TRAIN_BATCH_SIZE].reset_index(drop=True)
    eval_sample = gsm_test.iloc[: 2 * EVAL_BATCH_SIZE].reset_index(drop=True)
    train_keys, train_tensors = _tokenize_pairs(tokenizer, train_sample)
    eval_keys, eval_tensors = _tokenize_pairs(tokenizer, eval_sample)
    labels = torch.tensor(train_sample["nli_label"].to_numpy(), dtype=torch.long)
    loader = DataLoader(
        TensorDataset(*train_tensors, labels),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    trainable = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=ENCODER_LEARNING_RATE)
    model.train()
    started = time.perf_counter()
    for batch in loader:
        batch_labels = batch[-1]
        inputs = {key: value for key, value in zip(train_keys, batch[:-1])}
        optimizer.zero_grad(set_to_none=True)
        loss = model(**inputs, labels=batch_labels).loss
        loss.backward()
        optimizer.step()
    train_elapsed = time.perf_counter() - started
    model.eval()
    eval_loader = DataLoader(
        TensorDataset(*eval_tensors),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in eval_loader:
            model(**{key: value for key, value in zip(eval_keys, batch)})
    eval_elapsed = time.perf_counter() - started
    total_steps = int(np.ceil(TOTAL_TRAIN_ROWS / TRAIN_BATCH_SIZE))
    total_eval_rows = (
        EXPECTED_SPLIT_ROWS["test-unseen-questions"]
        + EXPECTED_SPLIT_ROWS["test-unseen-domains"]
        + 2 * 1_319
    )
    total_eval_batches = int(np.ceil(total_eval_rows / EVAL_BATCH_SIZE))
    projected_seconds = (
        train_elapsed / len(loader) * total_steps
        + 2.0 * eval_elapsed / len(eval_loader) * total_eval_batches
    )
    peak_rss = _current_rss_bytes()
    result = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_benchmark_batches": len(loader),
        "train_benchmark_seconds": train_elapsed,
        "eval_benchmark_batches": len(eval_loader),
        "eval_benchmark_seconds": eval_elapsed,
        "projected_training_steps": total_steps,
        "projected_external_eval_rows_per_pass": total_eval_rows,
        "projected_seconds": projected_seconds,
        "projected_hours": projected_seconds / 3600.0,
        "peak_rss_bytes": peak_rss,
        "proceed": bool(
            projected_seconds <= MAX_PROJECTED_HOURS * 3600
            and peak_rss < MAX_RSS_BYTES
        ),
        "freeze_summary": freeze_summary,
        "cache_metadata": metadata,
        "source_hashes": verify_sources(project_root),
        "runtime": runtime,
        "V_final_accessed": False,
    }
    _paths(project_root)["benchmark"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _load_benchmark(project_root: str | Path) -> dict[str, object]:
    paths = _paths(project_root)
    if not paths["benchmark"].exists():
        raise FileNotFoundError("Run the E580 benchmark before training.")
    result = json.loads(paths["benchmark"].read_text(encoding="utf-8"))
    if (
        result.get("protocol_id") != PROTOCOL_ID
        or result.get("source_hashes") != verify_sources(project_root)
        or result.get("proceed") is not True
    ):
        raise ValueError("E580 benchmark binding or gate is invalid.")
    return result


def train(project_root: str | Path) -> dict[str, object]:
    import torch
    from safetensors.torch import load_file, save_file
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import get_linear_schedule_with_warmup

    runtime = assert_runtime()
    paths = discover_project_paths(project_root)
    local_paths = _paths(project_root)
    source_hashes = verify_sources(project_root)
    benchmark_result = _load_benchmark(project_root)
    _configure_runtime()
    cache, cache_metadata = _load_cache(project_root)
    train_frame = cache.loc[cache["split"].eq("train")].reset_index(drop=True)
    gsm_test = cache.loc[cache["split"].eq("test")].reset_index(drop=True)
    semeval = pd.read_parquet(local_paths["semeval"])
    unseen_questions = semeval.loc[
        semeval["split"].eq("test-unseen-questions")
    ].reset_index(drop=True)
    unseen_domains = semeval.loc[
        semeval["split"].eq("test-unseen-domains")
    ].reset_index(drop=True)
    if len(train_frame) != TOTAL_TRAIN_ROWS or len(gsm_test) != 2_638:
        raise ValueError("E580 material row counts changed.")
    model, tokenizer = _load_model_and_tokenizer(project_root)
    freeze_summary = _freeze_deberta(model, UNFROZEN_LAYERS)
    train_keys, train_tensors = _tokenize_pairs(tokenizer, train_frame)
    gsm_keys, gsm_tensors = _tokenize_pairs(tokenizer, gsm_test)
    question_keys, question_tensors = _tokenize(tokenizer, unseen_questions)
    domain_keys, domain_tensors = _tokenize(tokenizer, unseen_domains)
    if not (train_keys == gsm_keys == question_keys == domain_keys):
        raise RuntimeError("E580 tokenizer fields differ across splits.")
    question_labels = unseen_questions["nli_label"].to_numpy(dtype=np.int64)
    domain_labels = unseen_domains["nli_label"].to_numpy(dtype=np.int64)
    gsm_labels = gsm_test["nli_label"].to_numpy(dtype=np.int64)
    base_question_probability, base_question_metrics = _evaluate_external(
        model, question_keys, question_tensors, question_labels
    )
    base_domain_probability, base_domain_metrics = _evaluate_external(
        model, domain_keys, domain_tensors, domain_labels
    )
    base_gsm_probability, base_gsm_metrics = _evaluate_gsm(
        model, gsm_keys, gsm_tensors, gsm_labels
    )

    labels = torch.tensor(train_frame["nli_label"].to_numpy(), dtype=torch.long)
    loader = DataLoader(
        TensorDataset(*train_tensors, labels),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    classifier_ids = {id(value) for value in model.classifier.parameters()}
    head_parameters = list(model.classifier.parameters())
    encoder_parameters = [
        value
        for value in model.parameters()
        if value.requires_grad and id(value) not in classifier_ids
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
    global_step = 0
    for epoch in range(EPOCHS):
        running_loss = 0.0
        seen = 0
        for batch_number, batch in enumerate(loader, start=1):
            batch_labels = batch[-1]
            inputs = {key: value for key, value in zip(train_keys, batch[:-1])}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**inputs, labels=batch_labels)
            if not torch.isfinite(outputs.loss):
                raise RuntimeError("E580 encountered non-finite training loss.")
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [value for value in model.parameters() if value.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            batch_rows = len(batch_labels)
            seen += batch_rows
            running_loss += float(outputs.loss.detach()) * batch_rows
            if batch_number % 25 == 0 or batch_number == len(loader):
                print(
                    f"e580_epoch={epoch + 1} step={batch_number}/{len(loader)} "
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

    question_probability, question_metrics = _evaluate_external(
        model, question_keys, question_tensors, question_labels
    )
    domain_probability, domain_metrics = _evaluate_external(
        model, domain_keys, domain_tensors, domain_labels
    )
    gsm_probability, gsm_metrics = _evaluate_gsm(
        model, gsm_keys, gsm_tensors, gsm_labels
    )
    clauses = {
        "unseen_questions_correct_auroc_at_least_0_70": (
            question_metrics["correct_vs_rest_auroc"] >= 0.70
        ),
        "unseen_domains_correct_auroc_at_least_0_68": (
            domain_metrics["correct_vs_rest_auroc"] >= 0.68
        ),
        "unseen_questions_macro_f1_at_least_0_45": (
            question_metrics["macro_f1"] >= 0.45
        ),
        "unseen_questions_correct_auroc_not_regressed": (
            question_metrics["correct_vs_rest_auroc"]
            >= base_question_metrics["correct_vs_rest_auroc"]
        ),
        "unseen_domains_correct_auroc_not_regressed": (
            domain_metrics["correct_vs_rest_auroc"]
            >= base_domain_metrics["correct_vs_rest_auroc"]
        ),
        "gsm_correct_vs_corrupt_auroc_at_least_0_95": (
            gsm_metrics["roc_auc"] >= 0.95
        ),
        "gsm_binary_log_loss_at_most_0_35": gsm_metrics["log_loss"] <= 0.35,
        "gsm_auroc_not_regressed": (
            gsm_metrics["roc_auc"] >= base_gsm_metrics["roc_auc"]
        ),
        "gsm_log_loss_not_regressed": (
            gsm_metrics["log_loss"] <= base_gsm_metrics["log_loss"]
        ),
    }
    passes_gate = bool(all(clauses.values()))
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_multicorpus_correctness_transfer")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    delta_path = run_dir / "multicorpus_deberta_delta.safetensors"
    delta = _trainable_state_dict(model)
    save_file(delta, str(delta_path))
    loaded = load_file(str(delta_path), device="cpu")
    if set(loaded) != set(delta):
        raise RuntimeError("E580 serialized delta key mismatch.")
    max_difference = max(
        float(torch.max(torch.abs(delta[name] - loaded[name])).detach().item())
        for name in delta
    )
    if max_difference != 0.0:
        raise RuntimeError("E580 delta serialization is not exact.")
    training_path = run_dir / "training_metrics.csv"
    metrics_path = run_dir / "external_metrics.csv"
    predictions_path = run_dir / "external_predictions.parquet"
    pd.DataFrame(training_rows).to_csv(
        training_path, index=False, lineterminator="\n"
    )
    pd.DataFrame(
        [
            {"split": "test-unseen-questions", **question_metrics},
            {"split": "test-unseen-domains", **domain_metrics},
            {"split": "gsm-test-paired", **gsm_metrics},
        ]
    ).to_csv(metrics_path, index=False, lineterminator="\n")
    prediction_frames: list[pd.DataFrame] = []
    for name, labels_array, probability in (
        ("test-unseen-questions", question_labels, question_probability),
        ("test-unseen-domains", domain_labels, domain_probability),
        ("gsm-test-paired", gsm_labels, gsm_probability),
        ("test-unseen-questions-base", question_labels, base_question_probability),
        ("test-unseen-domains-base", domain_labels, base_domain_probability),
        ("gsm-test-paired-base", gsm_labels, base_gsm_probability),
    ):
        prediction_frames.append(
            pd.DataFrame(
                {
                    "split": name,
                    "row": np.arange(len(labels_array), dtype=np.int64),
                    "nli_label": labels_array,
                    "prob_contradiction": probability[:, 0],
                    "prob_entailment": probability[:, 1],
                    "prob_neutral": probability[:, 2],
                }
            )
        )
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        predictions_path, index=False
    )
    report = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "candidate": "E580_semeval_gsm8k_correctness_transfer",
        "model": MODEL_NAME,
        "model_license": "Apache-2.0",
        "datasets": {
            "SemEval-2013 Task 7": "CC BY-SA 3.0; organizer ruling supplied",
            "GSM8K": "MIT",
        },
        "training_rows": len(train_frame),
        "training_rows_by_corpus": {
            key: int(value)
            for key, value in train_frame["corpus"].value_counts().items()
        },
        "epochs": EPOCHS,
        "batch_size": TRAIN_BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "encoder_learning_rate": ENCODER_LEARNING_RATE,
        "head_learning_rate": HEAD_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_fraction": WARMUP_FRACTION,
        "seed": SEED,
        "freeze_summary": freeze_summary,
        "external_metrics": {
            "test-unseen-questions": question_metrics,
            "test-unseen-domains": domain_metrics,
            "gsm-test-paired": gsm_metrics,
        },
        "base_nli_external_metrics": {
            "test-unseen-questions": base_question_metrics,
            "test-unseen-domains": base_domain_metrics,
            "gsm-test-paired": base_gsm_metrics,
        },
        "external_gate_clauses": clauses,
        "passes_external_gate": passes_gate,
        "delta_file": delta_path.name,
        "delta_bytes": delta_path.stat().st_size,
        "delta_sha256": _sha256(delta_path),
        "trainable_tensor_count": len(delta),
        "max_serialization_difference": max_difference,
        "cache_metadata": cache_metadata,
        "benchmark": benchmark_result,
        "source_hashes": source_hashes,
        "runtime": runtime,
        "artifact_sha256": {
            "training_metrics": _sha256(training_path),
            "external_metrics": _sha256(metrics_path),
            "external_predictions": _sha256(predictions_path),
        },
        "competition_cache_built": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "report": report,
        "report_sha256": _sha256(report_path),
    }


def build_competition_session_cache(
    project_root: str | Path,
    transfer_run_id: str,
    *,
    batch_size: int = EVAL_BATCH_SIZE,
) -> dict[str, object]:
    assert_runtime()
    return build_adapted_session_cache(
        project_root,
        transfer_run_id,
        batch_size=batch_size,
        candidate_name="E580_semeval_gsm8k_correctness_session",
        experiment_label="E580",
    )


def validate_competition_features(
    project_root: str | Path,
    transfer_run_id: str,
) -> dict[str, object]:
    from trace_ace.external_sra_validation import run_external_sra_validation

    assert_runtime()
    return run_external_sra_validation(
        project_root,
        transfer_run_id,
        candidate_code="e580_correctness",
        candidate_family="E580_semeval_gsm8k_correctness_blend",
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen E580 SemEval plus GSM8K correctness transfer."
    )
    parser.add_argument(
        "stage",
        choices=("prepare", "benchmark", "train", "cache", "validate", "pipeline"),
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--transfer-run-id")
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.stage == "prepare":
        result = prepare_multicorpus_cache(args.project_root)
    elif args.stage == "benchmark":
        result = benchmark(args.project_root)
    elif args.stage == "train":
        result = train(args.project_root)
    elif args.stage == "cache":
        if not args.transfer_run_id:
            raise ValueError("--transfer-run-id is required for the cache stage.")
        result = build_competition_session_cache(
            args.project_root,
            args.transfer_run_id,
            batch_size=args.batch_size,
        )
    elif args.stage == "validate":
        if not args.transfer_run_id:
            raise ValueError("--transfer-run-id is required for the validate stage.")
        result = validate_competition_features(
            args.project_root,
            args.transfer_run_id,
        )
    else:
        cache_result = prepare_multicorpus_cache(args.project_root)
        benchmark_result = benchmark(args.project_root)
        if not benchmark_result["proceed"]:
            raise RuntimeError("E580 benchmark failed; training is prohibited.")
        trained = train(args.project_root)
        if trained["report"]["passes_external_gate"]:
            session_cache = build_competition_session_cache(
                args.project_root,
                trained["run_id"],
                batch_size=args.batch_size,
            )
            validation = validate_competition_features(
                args.project_root,
                trained["run_id"],
            )
        else:
            session_cache = None
            validation = None
        result = {
            "cache": cache_result,
            "benchmark": benchmark_result,
            "train": trained,
            "session_cache": session_cache,
            "validation": validation,
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
