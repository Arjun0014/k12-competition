from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import random
import time
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from trace_ace.external_sra_transfer import (
    ENCODER_LEARNING_RATE,
    EPOCHS,
    EVAL_BATCH_SIZE,
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
    _premises,
    _tokenize,
    _trainable_state_dict,
)
from trace_ace.io import discover_project_paths
from trace_ace.multicorpus_correctness_transfer import (
    MAX_PROJECTED_HOURS,
    MAX_RSS_BYTES,
    _load_model_and_tokenizer,
    _ordered_frame_sha256,
    _sha256,
    _tokenize_pairs,
    assert_runtime,
)
from trace_ace.source_robust_validation import _current_rss_bytes


PROTOCOL_ID = "E590_semeval_prm800k_correctness_transfer_v1"
SEED = 20260727
PRM_ROWS_PER_CLASS = 2_970
PRM_TRAIN_ROWS = 8_910
TOTAL_TRAIN_ROWS = 17_820
PRM_TEST_ROWS = 25_530
PRM_BOOTSTRAP_REPLICATES = 2_000
SOURCE_HASHES = {
    "semeval": "527256be4e5f2e509f725b302ab8b163cc57760560a58621b2c02880ebe5dbb1",
    "prm_train": "1110237feeb51d1bc200cb37b8f965cfdc1036eac7d506094049366fe7dc1089",
    "prm_test": "6b172efa884ac8341a946dd82e06947c135b7254109fb3f7aa907c715d98aaad",
    "math_train": "90d96daeac3fe343ebb1e22ce93dd99690f75983e957f88de42f87cffe1e8076",
    "math_test": "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132",
    "license": "f213be7e9bf1040b5407cdc7b55c24a053c8fe7bc9b9f755541d822ff8af814b",
    "model": "ebc79588dd73ccfb6a3f6078519cfbf512c5305384c5ea1845bc71cd32216e86",
    "config": "885d0dceae8fa5c136da9209121ec9eb11160488e840de3bc1f29353674e5712",
    "tokenizer": "5124ef2ead1a10a717703bc436de7f353da76d6340e4587719b42b1693707964",
    "spm": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
}
SELECTED_CONTENT_SHA256 = (
    "73eed21b2a1fe1d26a5ab93b9751dd616b8892687294d4d1796147f4d3bddc3d"
)
TEST_CONTENT_SHA256 = (
    "456ecb3150c941b2d457bfb37476e687b47a8ad5847f100d17a87c72a9d74774"
)
HYPOTHESIS = (
    "The candidate reasoning step is mathematically correct and advances the solution."
)


def _paths(project_root: str | Path) -> dict[str, Path]:
    paths = discover_project_paths(project_root)
    prm = paths.root / "Datasets" / "PRM800K"
    model = paths.root / "assets" / "pretrained" / MODEL_DIRECTORY
    return {
        "semeval": paths.cache_dir / "sem_eval_2013_task7_canonical.parquet",
        "prm_train": prm / "prm800k" / "data" / "phase2_train.jsonl",
        "prm_test": prm / "prm800k" / "data" / "phase2_test.jsonl",
        "math_train": prm / "prm800k" / "math_splits" / "train.jsonl",
        "math_test": prm / "prm800k" / "math_splits" / "test.jsonl",
        "license": prm / "LICENSE",
        "model": model / "model.safetensors",
        "config": model / "config.json",
        "tokenizer": model / "tokenizer.json",
        "spm": model / "spm.model",
        "cache": paths.cache_dir / "multicorpus_prm_correctness_e590.parquet",
        "metadata": (
            paths.cache_dir / "multicorpus_prm_correctness_e590.metadata.json"
        ),
        "benchmark": (
            paths.cache_dir / "multicorpus_prm_correctness_e590_benchmark.json"
        ),
    }


def verify_sources(project_root: str | Path) -> dict[str, str]:
    paths = _paths(project_root)
    for name, expected in SOURCE_HASHES.items():
        path = paths[name]
        if not path.exists():
            raise FileNotFoundError(f"Missing E590 source {name}: {path}")
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"E590 {name} SHA-256 changed: {observed}")
    return dict(SOURCE_HASHES)


def _math_problem_set(path: Path) -> set[str]:
    problems: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            problems.add(str(row["problem"]))
    return problems


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").split())


def _canonical_record(
    problem: str,
    ground_truth: str,
    prior_steps: list[str],
    candidate: str,
    rating: int,
) -> tuple[bytes, bytes, dict[str, object]]:
    record: dict[str, object] = {
        "problem": problem,
        "ground_truth_solution": ground_truth,
        "prior_steps": list(prior_steps),
        "candidate": candidate,
        "rating": rating,
    }
    raw = json.dumps(
        record, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    key = hashlib.sha256(raw.rsplit(b',\"rating\":', 1)[0]).digest()
    return key, raw, record


def _iter_prm_examples(
    path: Path,
    allowed_problems: set[str],
    *,
    exclude_qc_screen: bool,
) -> Iterator[tuple[bytes, int, bytes, dict[str, object]]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            problem = str(row["question"]["problem"])
            if problem not in allowed_problems:
                continue
            if exclude_qc_screen and (
                row.get("is_quality_control_question")
                or row.get("is_initial_screening_question")
            ):
                continue
            ground_truth = str(row["question"]["ground_truth_solution"])
            prior_steps: list[str] = []
            for step in (row.get("label") or {}).get("steps") or []:
                completions = step.get("completions") or []
                for completion in completions:
                    rating = completion.get("rating")
                    candidate = _normalize_text(completion.get("text"))
                    if (
                        rating not in (-1, 0, 1)
                        or bool(completion.get("flagged"))
                        or not candidate
                    ):
                        continue
                    key, raw, record = _canonical_record(
                        problem,
                        ground_truth,
                        prior_steps,
                        candidate,
                        int(rating),
                    )
                    yield key, int(rating), raw, record
                chosen = step.get("chosen_completion")
                if chosen is not None and 0 <= int(chosen) < len(completions):
                    chosen_text = _normalize_text(completions[int(chosen)].get("text"))
                    if chosen_text:
                        prior_steps.append(chosen_text)
                elif step.get("human_completion"):
                    human = _normalize_text(step["human_completion"])
                    if human:
                        prior_steps.append(human)


def _rating_map(
    path: Path,
    allowed_problems: set[str],
    *,
    exclude_qc_screen: bool,
) -> dict[bytes, int]:
    ratings: dict[bytes, int] = {}
    for key, rating, _, _ in _iter_prm_examples(
        path,
        allowed_problems,
        exclude_qc_screen=exclude_qc_screen,
    ):
        previous = ratings.get(key)
        if previous is None:
            ratings[key] = rating
        elif previous != rating:
            ratings[key] = 2
    return ratings


def _compact_words(value: str, count: int, *, tail: bool = False) -> str:
    words = value.split()
    selected = words[-count:] if tail else words[:count]
    return " ".join(selected)


def _premise(record: dict[str, object]) -> str:
    prior = " ".join(str(value) for value in record["prior_steps"])
    return (
        f"[CANDIDATE STEP] {_compact_words(str(record['candidate']), 64)} "
        f"[PROBLEM] {_compact_words(str(record['problem']), 64)} "
        f"[GROUND TRUTH SOLUTION] "
        f"{_compact_words(str(record['ground_truth_solution']), 80)} "
        f"[PRIOR REASONING] {_compact_words(prior, 48, tail=True)}"
    ).strip()


def _record_to_row(
    key: bytes,
    record: dict[str, object],
    *,
    split: str,
) -> dict[str, object]:
    rating = int(record["rating"])
    return {
        "example_id": f"prm_{split}_{key.hex()}",
        "corpus": "PRM800K",
        "split": split,
        "premise": _premise(record),
        "hypothesis": HYPOTHESIS,
        "nli_label": rating + 1,
        "group_id": hashlib.sha256(
            str(record["problem"]).encode("utf-8")
        ).hexdigest(),
        "rating": rating,
    }


def _digest_rows(rows: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for raw in rows:
        digest.update(raw)
        digest.update(b"\n")
    return digest.hexdigest()


def build_prm_rows(
    path: Path,
    allowed_problems: set[str],
    *,
    split: str,
    selected_per_class: int | None,
    exclude_qc_screen: bool,
) -> tuple[pd.DataFrame, str]:
    ratings = _rating_map(
        path,
        allowed_problems,
        exclude_qc_screen=exclude_qc_screen,
    )
    emitted: set[bytes] = set()
    if selected_per_class is not None:
        heaps: dict[int, list[tuple[int, bytes, bytes, dict[str, object]]]] = {
            -1: [],
            0: [],
            1: [],
        }
        for key, rating, raw, record in _iter_prm_examples(
            path,
            allowed_problems,
            exclude_qc_screen=exclude_qc_screen,
        ):
            if ratings.get(key) != rating or key in emitted:
                continue
            emitted.add(key)
            heap = heaps[rating]
            item = (-int.from_bytes(key), key, raw, record)
            if len(heap) < selected_per_class:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
        ordered: list[tuple[bytes, bytes, dict[str, object]]] = []
        for rating in (-1, 0, 1):
            if len(heaps[rating]) != selected_per_class:
                raise ValueError(f"E590 PRM class {rating} is undersized.")
            ordered.extend(
                (key, raw, record)
                for _, key, raw, record in sorted(
                    heaps[rating], key=lambda item: item[1]
                )
            )
    else:
        ordered = []
        for key, rating, raw, record in _iter_prm_examples(
            path,
            allowed_problems,
            exclude_qc_screen=exclude_qc_screen,
        ):
            if ratings.get(key) != rating or key in emitted:
                continue
            emitted.add(key)
            ordered.append((key, raw, record))
        ordered.sort(key=lambda item: item[0])
    content_hash = _digest_rows(raw for _, raw, _ in ordered)
    frame = pd.DataFrame(
        [_record_to_row(key, record, split=split) for key, _, record in ordered]
    )
    return frame, content_hash


def prepare_multicorpus_cache(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    source_hashes = verify_sources(project_root)
    paths = _paths(project_root)
    math_train = _math_problem_set(paths["math_train"])
    math_test = _math_problem_set(paths["math_test"])
    if len(math_train) != 11_999 or len(math_test) != 500 or math_train & math_test:
        raise ValueError("E590 official MATH split contract changed.")
    prm_train, selected_hash = build_prm_rows(
        paths["prm_train"],
        math_train,
        split="train",
        selected_per_class=PRM_ROWS_PER_CLASS,
        exclude_qc_screen=True,
    )
    prm_test, test_hash = build_prm_rows(
        paths["prm_test"],
        math_test,
        split="test",
        selected_per_class=None,
        exclude_qc_screen=False,
    )
    if (
        len(prm_train) != PRM_TRAIN_ROWS
        or selected_hash != SELECTED_CONTENT_SHA256
        or len(prm_test) != PRM_TEST_ROWS
        or test_hash != TEST_CONTENT_SHA256
    ):
        raise ValueError("E590 deterministic PRM selection changed.")
    expected_test = {-1: 5_810, 0: 1_938, 1: 17_782}
    if prm_test["rating"].value_counts().to_dict() != expected_test:
        raise ValueError("E590 PRM test label counts changed.")

    semeval = pd.read_parquet(paths["semeval"])
    semeval_train = semeval.loc[semeval["split"].eq("train")].reset_index(drop=True)
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
            "group_id": (
                "semeval_"
                + semeval_train["question_id"].astype(str)
                if "question_id" in semeval_train
                else "semeval_" + semeval_train["student_answer_id"].astype(str)
            ),
            "rating": semeval_train["nli_label"].astype(np.int64) - 1,
        }
    )
    combined = pd.concat([semeval_rows, prm_train], ignore_index=True)
    if len(combined) != TOTAL_TRAIN_ROWS:
        raise ValueError("E590 combined train row count changed.")
    rng = np.random.default_rng(SEED)
    combined = combined.iloc[rng.permutation(len(combined))].reset_index(drop=True)
    cache = pd.concat([combined, prm_test], ignore_index=True)
    columns = [
        "example_id",
        "corpus",
        "split",
        "premise",
        "hypothesis",
        "nli_label",
        "group_id",
        "rating",
    ]
    cache.to_parquet(paths["cache"], index=False, compression="zstd")
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(cache),
        "training_rows": len(combined),
        "training_rows_by_corpus": {
            key: int(value) for key, value in combined["corpus"].value_counts().items()
        },
        "prm_test_rows": len(prm_test),
        "prm_test_by_rating": expected_test,
        "selected_content_sha256": selected_hash,
        "test_content_sha256": test_hash,
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
        raise ValueError("E590 cache binding changed.")
    return pd.read_parquet(paths["cache"]), metadata


def _prm_metrics(
    labels: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float | int]:
    prediction = probability.argmax(axis=1)
    return {
        "rows": len(labels),
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "multiclass_log_loss": float(
            log_loss(labels, probability, labels=[0, 1, 2])
        ),
        "contradiction_vs_rest_auroc": float(
            roc_auc_score(labels == 0, probability[:, 0])
        ),
    }


def _evaluate_prm(model, keys, tensors, labels: np.ndarray):
    probability, _ = _evaluate_external(model, keys, tensors, labels)
    return probability, _prm_metrics(labels, probability)


def _question_bootstrap(
    labels: np.ndarray,
    base_probability: np.ndarray,
    candidate_probability: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float | int]:
    row = np.arange(len(labels))
    base_loss = -np.log(np.clip(base_probability[row, labels], 1e-12, 1.0))
    candidate_loss = -np.log(
        np.clip(candidate_probability[row, labels], 1e-12, 1.0)
    )
    unique, inverse = np.unique(groups.astype(str), return_inverse=True)
    counts = np.bincount(inverse)
    gain_sums = np.bincount(inverse, weights=base_loss - candidate_loss)
    rng = np.random.default_rng(SEED)
    gains = np.empty(PRM_BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(PRM_BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, len(unique), size=len(unique))
        gains[replicate] = gain_sums[sampled].sum() / counts[sampled].sum()
    return {
        "groups": len(unique),
        "replicates": PRM_BOOTSTRAP_REPLICATES,
        "mean_log_loss_gain": float(gains.mean()),
        "ci_lower_95": float(np.quantile(gains, 0.025)),
        "ci_upper_95": float(np.quantile(gains, 0.975)),
        "probability_gain_positive": float(np.mean(gains > 0.0)),
    }


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


def benchmark(project_root: str | Path) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    runtime = assert_runtime()
    source_hashes = verify_sources(project_root)
    _configure_runtime()
    cache, metadata = _load_cache(project_root)
    train_frame = cache.loc[cache["split"].eq("train")].reset_index(drop=True)
    test_frame = cache.loc[cache["split"].eq("test")].reset_index(drop=True)
    model, tokenizer = _load_model_and_tokenizer(project_root)
    freeze_summary = _freeze_deberta(model, UNFROZEN_LAYERS)
    train_sample = train_frame.iloc[: 2 * TRAIN_BATCH_SIZE].reset_index(drop=True)
    eval_sample = test_frame.iloc[: 2 * EVAL_BATCH_SIZE].reset_index(drop=True)
    train_keys, train_tensors = _tokenize_pairs(tokenizer, train_sample)
    eval_keys, eval_tensors = _tokenize_pairs(tokenizer, eval_sample)
    labels = torch.tensor(train_sample["nli_label"].to_numpy(), dtype=torch.long)
    loader = DataLoader(
        TensorDataset(*train_tensors, labels),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad],
        lr=ENCODER_LEARNING_RATE,
    )
    started = time.perf_counter()
    model.train()
    for batch in loader:
        inputs = {key: value for key, value in zip(train_keys, batch[:-1])}
        optimizer.zero_grad(set_to_none=True)
        loss = model(**inputs, labels=batch[-1]).loss
        loss.backward()
        optimizer.step()
    train_elapsed = time.perf_counter() - started
    eval_loader = DataLoader(
        TensorDataset(*eval_tensors),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for batch in eval_loader:
            model(**{key: value for key, value in zip(eval_keys, batch)})
    eval_elapsed = time.perf_counter() - started
    total_steps = int(np.ceil(TOTAL_TRAIN_ROWS / TRAIN_BATCH_SIZE))
    total_eval_rows = (
        EXPECTED_SPLIT_ROWS["test-unseen-questions"]
        + EXPECTED_SPLIT_ROWS["test-unseen-domains"]
        + PRM_TEST_ROWS
    )
    total_eval_batches = int(np.ceil(total_eval_rows / EVAL_BATCH_SIZE))
    projected_seconds = (
        train_elapsed / len(loader) * total_steps
        + 2.0 * eval_elapsed / len(eval_loader) * total_eval_batches
    )
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
        "peak_rss_bytes": _current_rss_bytes(),
        "proceed": bool(
            projected_seconds <= MAX_PROJECTED_HOURS * 3600
            and _current_rss_bytes() < MAX_RSS_BYTES
        ),
        "freeze_summary": freeze_summary,
        "cache_metadata": metadata,
        "source_hashes": source_hashes,
        "runtime": runtime,
        "V_final_accessed": False,
    }
    _paths(project_root)["benchmark"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _load_benchmark(project_root: str | Path) -> dict[str, object]:
    path = _paths(project_root)["benchmark"]
    if not path.exists():
        raise FileNotFoundError("Run the E590 benchmark before training.")
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("protocol_id") != PROTOCOL_ID
        or result.get("source_hashes") != verify_sources(project_root)
        or result.get("proceed") is not True
    ):
        raise ValueError("E590 benchmark binding or gate is invalid.")
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
    prm_test = cache.loc[cache["split"].eq("test")].reset_index(drop=True)
    semeval = pd.read_parquet(local_paths["semeval"])
    questions = semeval.loc[
        semeval["split"].eq("test-unseen-questions")
    ].reset_index(drop=True)
    domains = semeval.loc[
        semeval["split"].eq("test-unseen-domains")
    ].reset_index(drop=True)
    model, tokenizer = _load_model_and_tokenizer(project_root)
    freeze_summary = _freeze_deberta(model, UNFROZEN_LAYERS)
    train_keys, train_tensors = _tokenize_pairs(tokenizer, train_frame)
    prm_keys, prm_tensors = _tokenize_pairs(tokenizer, prm_test)
    question_keys, question_tensors = _tokenize(tokenizer, questions)
    domain_keys, domain_tensors = _tokenize(tokenizer, domains)
    if not (train_keys == prm_keys == question_keys == domain_keys):
        raise RuntimeError("E590 tokenizer fields differ across splits.")
    question_labels = questions["nli_label"].to_numpy(dtype=np.int64)
    domain_labels = domains["nli_label"].to_numpy(dtype=np.int64)
    prm_labels = prm_test["nli_label"].to_numpy(dtype=np.int64)
    base_question_probability, base_question_metrics = _evaluate_external(
        model, question_keys, question_tensors, question_labels
    )
    base_domain_probability, base_domain_metrics = _evaluate_external(
        model, domain_keys, domain_tensors, domain_labels
    )
    base_prm_probability, base_prm_metrics = _evaluate_prm(
        model, prm_keys, prm_tensors, prm_labels
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
            inputs = {key: value for key, value in zip(train_keys, batch[:-1])}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**inputs, labels=batch[-1])
            if not torch.isfinite(outputs.loss):
                raise RuntimeError("E590 encountered non-finite training loss.")
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [value for value in model.parameters() if value.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            batch_rows = len(batch[-1])
            seen += batch_rows
            running_loss += float(outputs.loss.detach()) * batch_rows
            if batch_number % 25 == 0 or batch_number == len(loader):
                print(
                    f"e590_epoch={epoch + 1} step={batch_number}/{len(loader)} "
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
    prm_probability, prm_metrics = _evaluate_prm(
        model, prm_keys, prm_tensors, prm_labels
    )
    bootstrap = _question_bootstrap(
        prm_labels,
        base_prm_probability,
        prm_probability,
        prm_test["group_id"].to_numpy(),
    )
    prm_loss_gain = (
        base_prm_metrics["multiclass_log_loss"]
        - prm_metrics["multiclass_log_loss"]
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
        "prm_macro_f1_at_least_0_50": prm_metrics["macro_f1"] >= 0.50,
        "prm_contradiction_auroc_at_least_0_75": (
            prm_metrics["contradiction_vs_rest_auroc"] >= 0.75
        ),
        "prm_multiclass_log_loss_at_most_0_90": (
            prm_metrics["multiclass_log_loss"] <= 0.90
        ),
        "prm_log_loss_gain_at_least_0_10": prm_loss_gain >= 0.10,
        "prm_contradiction_auroc_not_regressed": (
            prm_metrics["contradiction_vs_rest_auroc"]
            >= base_prm_metrics["contradiction_vs_rest_auroc"]
        ),
        "prm_question_bootstrap_support_at_least_0_95": (
            bootstrap["probability_gain_positive"] >= 0.95
        ),
    }
    passes_gate = bool(all(clauses.values()))
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_prm_correctness_transfer")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    delta_path = run_dir / "prm_correctness_deberta_delta.safetensors"
    delta = _trainable_state_dict(model)
    save_file(delta, str(delta_path))
    loaded = load_file(str(delta_path), device="cpu")
    max_difference = max(
        float(torch.max(torch.abs(delta[name] - loaded[name])).detach().item())
        for name in delta
    )
    if set(loaded) != set(delta) or max_difference != 0.0:
        raise RuntimeError("E590 delta serialization is not exact.")
    training_path = run_dir / "training_metrics.csv"
    metrics_path = run_dir / "external_metrics.csv"
    predictions_path = run_dir / "external_predictions.parquet"
    bootstrap_path = run_dir / "prm_question_bootstrap.json"
    pd.DataFrame(training_rows).to_csv(training_path, index=False, lineterminator="\n")
    pd.DataFrame(
        [
            {"split": "test-unseen-questions", **question_metrics},
            {"split": "test-unseen-domains", **domain_metrics},
            {"split": "prm-phase2-test", **prm_metrics},
        ]
    ).to_csv(metrics_path, index=False, lineterminator="\n")
    frames: list[pd.DataFrame] = []
    for name, labels_array, probability in (
        ("test-unseen-questions", question_labels, question_probability),
        ("test-unseen-domains", domain_labels, domain_probability),
        ("prm-phase2-test", prm_labels, prm_probability),
        ("test-unseen-questions-base", question_labels, base_question_probability),
        ("test-unseen-domains-base", domain_labels, base_domain_probability),
        ("prm-phase2-test-base", prm_labels, base_prm_probability),
    ):
        frames.append(
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
    pd.concat(frames, ignore_index=True).to_parquet(predictions_path, index=False)
    bootstrap_path.write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "candidate": "E590_semeval_prm800k_correctness_transfer",
        "model": MODEL_NAME,
        "model_license": "Apache-2.0",
        "datasets": {
            "SemEval-2013 Task 7": "CC BY-SA 3.0; organizer ruling supplied",
            "PRM800K": "MIT",
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
            "prm-phase2-test": prm_metrics,
        },
        "base_nli_external_metrics": {
            "test-unseen-questions": base_question_metrics,
            "test-unseen-domains": base_domain_metrics,
            "prm-phase2-test": base_prm_metrics,
        },
        "prm_log_loss_gain": prm_loss_gain,
        "prm_question_bootstrap": bootstrap,
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
            "prm_question_bootstrap": _sha256(bootstrap_path),
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


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen E590 SemEval plus PRM800K correctness transfer."
    )
    parser.add_argument("stage", choices=("prepare", "benchmark", "train", "pipeline"))
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.stage == "prepare":
        result = prepare_multicorpus_cache(args.project_root)
    elif args.stage == "benchmark":
        result = benchmark(args.project_root)
    elif args.stage == "train":
        result = train(args.project_root)
    else:
        cache_result = prepare_multicorpus_cache(args.project_root)
        benchmark_result = benchmark(args.project_root)
        if not benchmark_result["proceed"]:
            raise RuntimeError("E590 benchmark failed; training is prohibited.")
        result = {
            "cache": cache_result,
            "benchmark": benchmark_result,
            "train": train(args.project_root),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
