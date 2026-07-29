from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trace_ace.bge_base_multiview_screen import _current_rss_bytes, assert_runtime
from trace_ace.io import discover_project_paths


PROTOCOL_ID = "E820_mathdial_contrastive_objective_alignment_v1"
SEED = 20260729
MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
MODEL_SHA256 = "c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7"
MATHDIAL_REVISION = "b06c020a0a1f57a87577fec33e657b63e7eb476e"
SOURCE_HASHES = {
    "train.jsonl": "96980babee081a3da48ed0f1fb3068ab52f23ddc67e63bfd5ec99ad701fd29cc",
    "test.jsonl": "28d1e537d65a6e6ff7b8bde602a2c7e2493b93d6bc785d1848506a650997f122",
}
EXPECTED_COUNTS = {
    "train_rows": 2262,
    "test_rows": 599,
    "train_qids": 1035,
    "test_qids": 394,
    "overlap_qids": 318,
    "legal_train_rows": 1677,
    "legal_train_qids": 717,
}
QUERY_PREFIX = (
    "Represent this K-12 math learning target for retrieving its tutoring dialogue: "
)
QUERY_MAX_LENGTH = 96
DOCUMENT_MAX_LENGTH = 256
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 32
EPOCHS = 2
UNFROZEN_LAYERS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.10
TEMPERATURE = 0.05
GRADIENT_CLIP = 1.0
BOOTSTRAP_REPLICATES = 2_000
MAX_PROJECTED_SECONDS = 4 * 3600
MAX_RSS_BYTES = 12 * 1024**3


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def objective_query(text: str) -> str:
    return f"{QUERY_PREFIX}{str(text).strip()}"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{line_number} is not an object.")
            rows.append(row)
    return rows


def load_mathdial_alignment_source(
    project_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    root = (
        Path(project_root).resolve()
        / "Datasets"
        / "MathDial"
        / "data"
    )
    for name, expected in SOURCE_HASHES.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned MathDial source: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"MathDial {name} SHA-256 changed: expected {expected}, got {actual}."
            )

    raw_train = _read_jsonl(root / "train.jsonl")
    raw_test = _read_jsonl(root / "test.jsonl")
    test_qids = {str(row["qid"]) for row in raw_test}
    train_qids = {str(row["qid"]) for row in raw_train}
    legal_train = [
        row for row in raw_train if str(row["qid"]) not in test_qids
    ]

    observed = {
        "train_rows": len(raw_train),
        "test_rows": len(raw_test),
        "train_qids": len(train_qids),
        "test_qids": len(test_qids),
        "overlap_qids": len(train_qids.intersection(test_qids)),
        "legal_train_rows": len(legal_train),
        "legal_train_qids": len({str(row["qid"]) for row in legal_train}),
    }
    if observed != EXPECTED_COUNTS:
        raise ValueError(
            f"MathDial source cardinalities changed: {observed!r}."
        )

    def canonicalize(
        rows: list[dict[str, object]], split: str
    ) -> pd.DataFrame:
        result: list[dict[str, str]] = []
        for row in rows:
            qid = str(row["qid"])
            question = str(row["question"]).strip()
            conversation = str(row["conversation"]).strip()
            confusion = str(row["teacher_described_confusion"]).strip()
            if not question or not conversation or not confusion:
                raise ValueError(f"MathDial {split} has an empty required field.")
            result.append(
                {
                    "split": split,
                    "qid_hash": _text_hash(qid),
                    "row_hash": _text_hash(
                        "\n".join((qid, question, confusion, conversation))
                    ),
                    "question": question,
                    "confusion": confusion,
                    "conversation": conversation,
                }
            )
        frame = pd.DataFrame(result)
        return frame.sort_values(
            ["qid_hash", "row_hash"], kind="stable"
        ).reset_index(drop=True)

    train = canonicalize(legal_train, "train")
    test = canonicalize(raw_test, "test")
    if set(train["qid_hash"]).intersection(set(test["qid_hash"])):
        raise RuntimeError("MathDial legal train/test qid leakage remains.")

    content = "\n".join(
        f"{row.split}\t{row.qid_hash}\t{row.row_hash}"
        for row in pd.concat([train, test], ignore_index=True).itertuples(
            index=False
        )
    )
    audit: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "source": "official eth-nlped/mathdial",
        "source_revision": MATHDIAL_REVISION,
        "license": "CC BY-SA 4.0",
        "source_hashes": dict(SOURCE_HASHES),
        "counts": observed,
        "canonical_identity_sha256": _text_hash(content),
        "consumed_fields": [
            "qid",
            "question",
            "teacher_described_confusion",
            "conversation",
        ],
        "prohibited_fields_consumed": False,
        "outcome_fields_consumed": [],
    }
    return train, test, audit


def epoch_training_rows(
    frame: pd.DataFrame, epoch: int, seed: int = SEED
) -> pd.DataFrame:
    if epoch < 0:
        raise ValueError("epoch must be nonnegative.")
    selected: list[pd.Series] = []
    for _, group in frame.groupby("qid_hash", sort=True):
        ordered = group.sort_values("row_hash", kind="stable")
        selected.append(ordered.iloc[epoch % len(ordered)])
    result = pd.DataFrame(selected).reset_index(drop=True)
    rng = np.random.default_rng(int(seed) + int(epoch))
    return result.iloc[rng.permutation(len(result))].reset_index(drop=True)


def _freeze_encoder(model, unfrozen_layers: int) -> dict[str, int]:
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layer"):
        raise TypeError("E820 requires a BERT encoder with encoder.layer.")
    layers = model.encoder.layer
    if not 1 <= int(unfrozen_layers) < len(layers):
        raise ValueError("Invalid E820 unfrozen layer count.")
    for parameter in model.parameters():
        parameter.requires_grad = False
    frozen = len(layers) - int(unfrozen_layers)
    for layer in layers[frozen:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    return {
        "total_encoder_layers": len(layers),
        "frozen_encoder_layers": frozen,
        "unfrozen_encoder_layers": int(unfrozen_layers),
        "trainable_parameters": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
    }


def _tokenize(
    tokenizer,
    texts: list[str],
    max_length: int,
):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )


def _normalized_cls(model, encoded):
    import torch.nn.functional as functional

    output = model(**encoded).last_hidden_state[:, 0]
    return functional.normalize(output, p=2, dim=1)


def _contrastive_loss(query_embedding, document_embedding):
    import torch
    import torch.nn.functional as functional

    similarity = query_embedding @ document_embedding.T / TEMPERATURE
    labels = torch.arange(len(query_embedding), device=similarity.device)
    return 0.5 * (
        functional.cross_entropy(similarity, labels)
        + functional.cross_entropy(similarity.T, labels)
    )


def _encode_texts(
    model,
    tokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int = EVAL_BATCH_SIZE,
) -> np.ndarray:
    import torch

    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), int(batch_size)):
            encoded = _tokenize(
                tokenizer, texts[start : start + int(batch_size)], max_length
            )
            embedding = _normalized_cls(model, encoded)
            batches.append(embedding.cpu().numpy().astype(np.float32))
    result = np.concatenate(batches, axis=0)
    if result.shape != (len(texts), 768) or not np.isfinite(result).all():
        raise RuntimeError("E820 encoder produced invalid embeddings.")
    return result


def retrieval_rows(
    query_embedding: np.ndarray,
    document_embedding: np.ndarray,
    query_qids: Iterable[str],
    document_qids: Iterable[str],
) -> pd.DataFrame:
    query_qids_array = np.asarray(list(query_qids), dtype=object)
    document_qids_array = np.asarray(list(document_qids), dtype=object)
    similarity = np.asarray(query_embedding) @ np.asarray(
        document_embedding
    ).T
    if similarity.shape != (
        len(query_qids_array),
        len(document_qids_array),
    ):
        raise ValueError("E820 retrieval matrix shape mismatch.")
    result: list[dict[str, object]] = []
    for index, qid in enumerate(query_qids_array):
        positive = document_qids_array == qid
        if not positive.any():
            raise ValueError("E820 query has no positive document.")
        order = np.argsort(-similarity[index], kind="stable")
        ranks = np.flatnonzero(positive[order]) + 1
        best_rank = int(ranks.min())
        result.append(
            {
                "qid_hash": str(qid),
                "reciprocal_rank": 1.0 / best_rank,
                "hit_at_1": int(best_rank <= 1),
                "hit_at_5": int(best_rank <= 5),
                "hit_at_10": int(best_rank <= 10),
            }
        )
    return pd.DataFrame(result)


def summarize_retrieval(rows: pd.DataFrame) -> dict[str, float]:
    return {
        "mrr": float(rows["reciprocal_rank"].mean()),
        "recall_at_1": float(rows["hit_at_1"].mean()),
        "recall_at_5": float(rows["hit_at_5"].mean()),
        "recall_at_10": float(rows["hit_at_10"].mean()),
    }


def qid_bootstrap(
    base_rows: pd.DataFrame,
    adapted_rows: pd.DataFrame,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
) -> dict[str, float | int | list[float]]:
    merged = base_rows[["qid_hash", "reciprocal_rank"]].merge(
        adapted_rows[["qid_hash", "reciprocal_rank"]],
        on="qid_hash",
        suffixes=("_base", "_adapted"),
        validate="many_to_many",
    )
    by_qid = (
        merged.assign(
            gain=(
                merged["reciprocal_rank_adapted"]
                - merged["reciprocal_rank_base"]
            )
        )
        .groupby("qid_hash", sort=True)["gain"]
        .mean()
        .to_numpy(dtype=np.float64)
    )
    if len(by_qid) < 2:
        raise ValueError("E820 bootstrap needs at least two qids.")
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(replicates), dtype=np.float64)
    for start in range(0, int(replicates), 250):
        count = min(250, int(replicates) - start)
        indices = rng.integers(
            0, len(by_qid), size=(count, len(by_qid))
        )
        draws[start : start + count] = by_qid[indices].mean(axis=1)
    return {
        "replicates": int(replicates),
        "qids": int(len(by_qid)),
        "observed_gain": float(by_qid.mean()),
        "mean_gain": float(draws.mean()),
        "interval_95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "positive_gain_support": float(np.mean(draws > 0.0)),
    }


def _configure_torch() -> None:
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _load_model(project_root: str | Path):
    from transformers import AutoModel, AutoTokenizer

    model_path = (
        Path(project_root).resolve()
        / "assets"
        / "pretrained"
        / "bge-base-en-v1.5"
    )
    model_file = model_path / "model.safetensors"
    if not model_file.is_file() or _sha256(model_file) != MODEL_SHA256:
        raise ValueError("Pinned E820 BGE-base model is missing or changed.")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True
    )
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    return model, tokenizer


def audit_source(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    train, test, audit = load_mathdial_alignment_source(project_root)
    audit.update(
        {
            "runtime": runtime,
            "legal_train_rows": len(train),
            "evaluation_rows": len(test),
            "competition_outcomes_accessed": False,
            "v_joint_accessed": False,
            "v_final_accessed": False,
        }
    )
    return audit


def benchmark(project_root: str | Path) -> dict[str, object]:
    import torch

    runtime = assert_runtime()
    train, test, audit = load_mathdial_alignment_source(project_root)
    _configure_torch()
    model, tokenizer = _load_model(project_root)
    freeze = _freeze_encoder(model, UNFROZEN_LAYERS)
    sample = epoch_training_rows(train, epoch=0).iloc[:TRAIN_BATCH_SIZE]
    query = _tokenize(
        tokenizer,
        [objective_query(value) for value in sample["question"]],
        QUERY_MAX_LENGTH,
    )
    document = _tokenize(
        tokenizer,
        sample["conversation"].astype(str).tolist(),
        DOCUMENT_MAX_LENGTH,
    )

    started = time.perf_counter()
    model.train()
    query_embedding = _normalized_cls(model, query)
    document_embedding = _normalized_cls(model, document)
    loss = _contrastive_loss(query_embedding, document_embedding)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        max_norm=GRADIENT_CLIP,
    )
    training_step_seconds = time.perf_counter() - started

    eval_sample = test.iloc[: min(64, len(test))]
    started = time.perf_counter()
    _encode_texts(
        model,
        tokenizer,
        [
            objective_query(value)
            for value in eval_sample["confusion"].astype(str)
        ],
        QUERY_MAX_LENGTH,
    )
    evaluation_seconds = time.perf_counter() - started

    training_steps = (
        math.ceil(EXPECTED_COUNTS["legal_train_qids"] / TRAIN_BATCH_SIZE)
        * EPOCHS
    )
    evaluation_texts = len(test) * 3 * 2
    projected_evaluation = (
        evaluation_seconds / len(eval_sample) * evaluation_texts
    )
    projected_seconds = (
        training_step_seconds * training_steps + projected_evaluation
    )
    peak_rss = _current_rss_bytes()
    result: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "runtime": runtime,
        "source_audit": audit,
        "freeze_summary": freeze,
        "benchmark_batch_size": TRAIN_BATCH_SIZE,
        "training_step_seconds": training_step_seconds,
        "training_steps": training_steps,
        "evaluation_sample_rows": len(eval_sample),
        "evaluation_seconds": evaluation_seconds,
        "projected_evaluation_seconds": projected_evaluation,
        "projected_total_seconds": projected_seconds,
        "peak_rss_bytes": peak_rss,
        "resource_gate": {
            "max_projected_seconds": MAX_PROJECTED_SECONDS,
            "max_rss_bytes": MAX_RSS_BYTES,
            "passes_duration": projected_seconds <= MAX_PROJECTED_SECONDS,
            "passes_memory": peak_rss < MAX_RSS_BYTES,
        },
        "proceed": bool(
            projected_seconds <= MAX_PROJECTED_SECONDS
            and peak_rss < MAX_RSS_BYTES
        ),
        "competition_outcomes_accessed": False,
        "v_joint_accessed": False,
        "v_final_accessed": False,
    }
    paths = discover_project_paths(project_root)
    output = paths.cache_dir / "e820_resource_benchmark.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["benchmark_path"] = str(output)
    result["benchmark_sha256"] = _sha256(output)
    return result


def _parameter_delta(
    model, initial: dict[str, object]
) -> dict[str, object]:
    import torch

    state = model.state_dict()
    result: dict[str, torch.Tensor] = {}
    for name, base in initial.items():
        delta = state[name].detach().cpu() - base
        if torch.count_nonzero(delta).item():
            result[name] = delta.contiguous()
    if not result:
        raise RuntimeError("E820 training produced an empty parameter delta.")
    return result


def train_and_evaluate(project_root: str | Path) -> dict[str, object]:
    import torch
    from safetensors.torch import save_file
    from transformers import get_linear_schedule_with_warmup

    runtime = assert_runtime()
    paths = discover_project_paths(project_root)
    benchmark_path = paths.cache_dir / "e820_resource_benchmark.json"
    if not benchmark_path.is_file():
        raise FileNotFoundError("E820 resource benchmark is required.")
    benchmark_report = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if (
        benchmark_report.get("protocol_id") != PROTOCOL_ID
        or not benchmark_report.get("proceed")
    ):
        raise RuntimeError("E820 resource gate is not satisfied.")

    train, test, source_audit = load_mathdial_alignment_source(project_root)
    _configure_torch()
    model, tokenizer = _load_model(project_root)
    freeze = _freeze_encoder(model, UNFROZEN_LAYERS)
    initial = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.state_dict().items()
        if name in {
            parameter_name
            for parameter_name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
    }

    test_problem_queries = [
        objective_query(value) for value in test["question"].astype(str)
    ]
    test_confusion_queries = [
        objective_query(value) for value in test["confusion"].astype(str)
    ]
    test_documents = test["conversation"].astype(str).tolist()
    base_problem_embedding = _encode_texts(
        model, tokenizer, test_problem_queries, QUERY_MAX_LENGTH
    )
    base_confusion_embedding = _encode_texts(
        model, tokenizer, test_confusion_queries, QUERY_MAX_LENGTH
    )
    base_document_embedding = _encode_texts(
        model, tokenizer, test_documents, DOCUMENT_MAX_LENGTH
    )

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    steps_per_epoch = math.ceil(
        EXPECTED_COUNTS["legal_train_qids"] / TRAIN_BATCH_SIZE
    )
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(round(total_steps * WARMUP_FRACTION))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    training_metrics: list[dict[str, float | int]] = []
    global_step = 0
    training_started = time.perf_counter()
    for epoch in range(EPOCHS):
        rows = epoch_training_rows(train, epoch=epoch)
        model.train()
        running_loss = 0.0
        seen = 0
        for start in range(0, len(rows), TRAIN_BATCH_SIZE):
            batch = rows.iloc[start : start + TRAIN_BATCH_SIZE]
            if batch["qid_hash"].duplicated().any():
                raise RuntimeError("E820 in-batch false negative detected.")
            query = _tokenize(
                tokenizer,
                [
                    objective_query(value)
                    for value in batch["question"].astype(str)
                ],
                QUERY_MAX_LENGTH,
            )
            document = _tokenize(
                tokenizer,
                batch["conversation"].astype(str).tolist(),
                DOCUMENT_MAX_LENGTH,
            )
            optimizer.zero_grad(set_to_none=True)
            query_embedding = _normalized_cls(model, query)
            document_embedding = _normalized_cls(model, document)
            loss = _contrastive_loss(query_embedding, document_embedding)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters, max_norm=GRADIENT_CLIP
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            batch_rows = len(batch)
            running_loss += float(loss.item()) * batch_rows
            seen += batch_rows
            if (
                global_step % 15 == 0
                or global_step == total_steps
            ):
                print(
                    f"E820 epoch={epoch + 1}/{EPOCHS} "
                    f"step={global_step}/{total_steps} "
                    f"loss={running_loss / seen:.6f}",
                    flush=True,
                )
        training_metrics.append(
            {
                "epoch": epoch + 1,
                "steps": global_step,
                "rows": seen,
                "mean_loss": running_loss / seen,
            }
        )
    training_seconds = time.perf_counter() - training_started

    adapted_problem_embedding = _encode_texts(
        model, tokenizer, test_problem_queries, QUERY_MAX_LENGTH
    )
    adapted_confusion_embedding = _encode_texts(
        model, tokenizer, test_confusion_queries, QUERY_MAX_LENGTH
    )
    adapted_document_embedding = _encode_texts(
        model, tokenizer, test_documents, DOCUMENT_MAX_LENGTH
    )

    query_qids = test["qid_hash"].astype(str).tolist()
    document_qids = test["qid_hash"].astype(str).tolist()
    views = {
        "problem": (base_problem_embedding, adapted_problem_embedding),
        "confusion": (
            base_confusion_embedding,
            adapted_confusion_embedding,
        ),
    }
    score_frames: list[pd.DataFrame] = []
    metrics: dict[str, object] = {}
    bootstraps: dict[str, object] = {}
    for view, (base_query, adapted_query) in views.items():
        base_rows = retrieval_rows(
            base_query,
            base_document_embedding,
            query_qids,
            document_qids,
        )
        adapted_rows = retrieval_rows(
            adapted_query,
            adapted_document_embedding,
            query_qids,
            document_qids,
        )
        base_metrics = summarize_retrieval(base_rows)
        adapted_metrics = summarize_retrieval(adapted_rows)
        metrics[view] = {
            "base": base_metrics,
            "adapted": adapted_metrics,
            "gain": {
                key: adapted_metrics[key] - base_metrics[key]
                for key in base_metrics
            },
        }
        bootstraps[view] = qid_bootstrap(base_rows, adapted_rows)
        combined = base_rows.rename(
            columns={
                column: f"base_{column}"
                for column in base_rows.columns
                if column != "qid_hash"
            }
        )
        renamed_adapted = adapted_rows.rename(
            columns={
                column: f"adapted_{column}"
                for column in adapted_rows.columns
                if column != "qid_hash"
            }
        )
        combined = combined.join(
            renamed_adapted.drop(columns=["qid_hash"])
        )
        combined.insert(0, "view", view)
        score_frames.append(combined)

    rng = np.random.default_rng(SEED)
    sample_size = min(10_000, len(adapted_document_embedding) ** 2)
    left = rng.integers(
        0, len(adapted_document_embedding), size=sample_size
    )
    right = rng.integers(
        0, len(adapted_document_embedding), size=sample_size
    )
    distinct = left != right
    off_diagonal = np.sum(
        adapted_document_embedding[left[distinct]]
        * adapted_document_embedding[right[distinct]],
        axis=1,
    )
    geometry = {
        "sample_pairs": int(len(off_diagonal)),
        "off_diagonal_cosine_mean": float(off_diagonal.mean()),
        "off_diagonal_cosine_std": float(off_diagonal.std()),
    }

    confusion_gain = metrics["confusion"]["gain"]
    problem_gain = metrics["problem"]["gain"]
    clauses = {
        "confusion_mrr_gain_at_least_0_04": (
            confusion_gain["mrr"] >= 0.04
        ),
        "confusion_recall5_gain_at_least_0_04": (
            confusion_gain["recall_at_5"] >= 0.04
        ),
        "confusion_bootstrap_support_at_least_0_95": (
            bootstraps["confusion"]["positive_gain_support"] >= 0.95
        ),
        "problem_mrr_regression_at_most_0_005": (
            problem_gain["mrr"] >= -0.005
        ),
        "problem_recall5_regression_at_most_0_005": (
            problem_gain["recall_at_5"] >= -0.005
        ),
        "no_embedding_collapse_mean_below_0_90": (
            geometry["off_diagonal_cosine_mean"] < 0.90
        ),
        "no_embedding_collapse_std_above_0_02": (
            geometry["off_diagonal_cosine_std"] > 0.02
        ),
        "training_loss_decreases": (
            training_metrics[-1]["mean_loss"]
            < training_metrics[0]["mean_loss"]
        ),
    }
    passes = all(clauses.values())

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime(
        "%Y%m%dT%H%M%SZ_mathdial_contrastive_alignment"
    )
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    delta_path = run_dir / "e820_bge_base_delta.safetensors"
    save_file(_parameter_delta(model, initial), str(delta_path))
    pd.DataFrame(training_metrics).to_csv(
        run_dir / "training_metrics.csv",
        index=False,
        lineterminator="\n",
    )
    pd.concat(score_frames, ignore_index=True).to_parquet(
        run_dir / "retrieval_scores.parquet", index=False
    )

    report: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "runtime": runtime,
        "source_audit": source_audit,
        "model": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "model_sha256": MODEL_SHA256,
            "freeze_summary": freeze,
        },
        "configuration": {
            "query_prefix": QUERY_PREFIX,
            "query_max_length": QUERY_MAX_LENGTH,
            "document_max_length": DOCUMENT_MAX_LENGTH,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "epochs": EPOCHS,
            "unfrozen_layers": UNFROZEN_LAYERS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_fraction": WARMUP_FRACTION,
            "temperature": TEMPERATURE,
            "gradient_clip": GRADIENT_CLIP,
            "seed": SEED,
        },
        "training_seconds": training_seconds,
        "training_metrics": training_metrics,
        "retrieval_metrics": metrics,
        "qid_bootstraps": bootstraps,
        "embedding_geometry": geometry,
        "gate_clauses": clauses,
        "passes_target_free_gate": passes,
        "decision": (
            "eligible_for_frozen_competition_preregistration"
            if passes
            else "reject_E820_before_competition_outcomes"
        ),
        "competition_outcomes_accessed": False,
        "v_seen_accessed": False,
        "v_objective_accessed": False,
        "v_style_accessed": False,
        "v_joint_accessed": False,
        "v_final_accessed": False,
        "peak_rss_bytes": _current_rss_bytes(),
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hashes = {
        path.name: _sha256(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file()
    }
    (run_dir / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "passes_target_free_gate": passes,
        "metrics": metrics,
        "bootstraps": bootstraps,
        "report_sha256": _sha256(report_path),
        "delta_sha256": _sha256(delta_path),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the E820 target-free MathDial alignment protocol."
    )
    parser.add_argument(
        "mode", choices=("audit", "benchmark", "train")
    )
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.mode == "audit":
        result = audit_source(args.project_root)
    elif args.mode == "benchmark":
        result = benchmark(args.project_root)
    else:
        result = train_and_evaluate(args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
