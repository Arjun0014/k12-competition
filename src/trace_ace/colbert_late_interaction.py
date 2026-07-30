from __future__ import annotations

import argparse
import hashlib
import json
import platform
import string
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trace_ace.bge_base_multiview_screen import _current_rss_bytes
from trace_ace.io import discover_project_paths
from trace_ace.mathdial_contrastive_alignment import (
    _encode_texts as encode_bge_texts,
)
from trace_ace.mathdial_contrastive_alignment import (
    _load_model as load_bge_model,
)
from trace_ace.mathdial_contrastive_alignment import (
    load_mathdial_alignment_source,
    objective_query,
    qid_bootstrap,
    summarize_retrieval,
)


PROTOCOL_ID = "E900_colbert_late_interaction_external_v1"
MODEL_NAME = "colbert-ir/colbertv2.0"
MODEL_REVISION = "c1e84128e85ef755c096a95bdb06b47793b13acf"
MODEL_LICENSE = "MIT"
MODEL_FILES = {
    "model.safetensors": (
        "3f58890b1dfdfec066ef12ba431fa9d56992da9e30a53489242c4156e37e9017"
    ),
    "config.json": (
        "cbdcc01dc7772fd0e5e1d85d5de695faf8c4935ae3c7ea3fdb6c24397b284f3c"
    ),
    "artifact.metadata": (
        "0ddc5a54234cff6d13bc9411250a5479d9d96f3ffbace76d8a1884144377e434"
    ),
    "README.md": (
        "44e68292037d4fac7db91b9b24ae36535ae5807eef2b323adcc83a615b91751d"
    ),
    "special_tokens_map.json": (
        "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3"
    ),
    "tokenizer.json": (
        "5fd1c882abbd30517dced455a2c9768945ec726b96727927e4959348d9de550b"
    ),
    "tokenizer_config.json": (
        "a16a31c8d474339d7a46afe7eb8583b24f8b74806af6e25294506523f9d0acfb"
    ),
    "vocab.txt": (
        "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
    ),
}
OFFICIAL_CODE_REVISION = "cc4f3dc91c0b45d2d08c251d9d95178285c65f1c"
OFFICIAL_CODE_LICENSE_SHA256 = (
    "4e19aae8ed1a14290a67c3cb32f1da34df5ffcd713841f8f508270aa0fadeefe"
)
QUERY_MAX_LENGTH = 32
DOCUMENT_MAX_LENGTH = 180
PROJECTION_DIMENSION = 128
ENCODE_BATCH_SIZE = 8
SCORE_QUERY_BATCH_SIZE = 8
CPU_THREADS = 6
BENCHMARK_ROWS = 32
MAX_PROJECTED_SECONDS = 3_600.0
MAX_RSS_BYTES = 8 * 1024**3
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20260730
EXPECTED_RUNTIME = {
    "python": "3.12.8",
    "numpy": "2.5.1",
    "scikit_learn": "1.8.0",
    "torch": "2.13.0+cpu",
    "transformers": "5.14.1",
}
GATES = {
    "confusion_mrr_gain": 0.08,
    "confusion_recall5_gain": 0.08,
    "confusion_bootstrap_support": 0.95,
    "problem_mrr_min_gain": -0.005,
    "problem_recall5_min_gain": -0.005,
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
        raise RuntimeError(f"E900 runtime contract changed: {observed}")
    return observed


def _model_directory(project_root: str | Path) -> Path:
    return (
        Path(project_root).resolve()
        / "assets"
        / "pretrained"
        / "colbertv2.0"
    )


def verify_model_files(project_root: str | Path) -> dict[str, str]:
    directory = _model_directory(project_root)
    for name, expected in MODEL_FILES.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing E900 model file: {path}")
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(
                f"E900 {name} SHA-256 changed: expected {expected}, got {observed}."
            )
    return dict(MODEL_FILES)


def configure_torch() -> None:
    import torch

    torch.manual_seed(20260730)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def insert_marker(tensor, marker_id: int):
    import torch

    if tensor.ndim != 2 or tensor.shape[1] < 2:
        raise ValueError("E900 marker insertion expects a two-dimensional tensor.")
    marker = torch.full(
        (tensor.shape[0], 1),
        int(marker_id),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor[:, :1], marker, tensor[:, 1:]], dim=1)


def _load_colbert(project_root: str | Path):
    import torch
    from safetensors import safe_open
    from transformers import BertModel, BertTokenizerFast

    verify_model_files(project_root)
    directory = _model_directory(project_root)
    tokenizer = BertTokenizerFast.from_pretrained(
        directory,
        local_files_only=True,
    )
    model = BertModel.from_pretrained(
        directory,
        local_files_only=True,
        add_pooling_layer=False,
        use_safetensors=True,
    )
    with safe_open(
        directory / "model.safetensors",
        framework="pt",
        device="cpu",
    ) as handle:
        projection = handle.get_tensor("linear.weight").contiguous()
    if tuple(projection.shape) != (PROJECTION_DIMENSION, 768):
        raise ValueError(f"E900 projection shape changed: {projection.shape}")
    if model.config.num_hidden_layers != 12 or model.config.hidden_size != 768:
        raise ValueError("E900 expected the released 12-layer BERT encoder.")
    if tokenizer.convert_tokens_to_ids("[unused0]") != 1:
        raise ValueError("E900 query marker token changed.")
    if tokenizer.convert_tokens_to_ids("[unused1]") != 2:
        raise ValueError("E900 document marker token changed.")
    model.eval()
    return model, tokenizer, projection.to(dtype=torch.float32)


def _punctuation_ids(tokenizer) -> set[int]:
    values: set[int] = set()
    for symbol in string.punctuation:
        encoded = tokenizer.encode(symbol, add_special_tokens=False)
        if encoded:
            values.add(int(encoded[0]))
    return values


def encode_colbert(
    model,
    tokenizer,
    projection,
    texts: list[str],
    *,
    is_query: bool,
    batch_size: int = ENCODE_BATCH_SIZE,
):
    import torch
    import torch.nn.functional as functional

    if not texts or any(not str(value).strip() for value in texts):
        raise ValueError("E900 requires nonempty input texts.")
    if batch_size <= 0:
        raise ValueError("E900 batch size must be positive.")
    max_length = QUERY_MAX_LENGTH if is_query else DOCUMENT_MAX_LENGTH
    marker_id = 1 if is_query else 2
    punctuation = _punctuation_ids(tokenizer)
    encoded_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), int(batch_size)):
            batch = tokenizer(
                texts[start : start + int(batch_size)],
                padding="max_length",
                truncation=True,
                max_length=max_length - 1,
                return_tensors="pt",
            )
            input_ids = insert_marker(batch["input_ids"], marker_id)
            attention_mask = insert_marker(batch["attention_mask"], 1)
            if is_query:
                input_ids[input_ids == tokenizer.pad_token_id] = (
                    tokenizer.mask_token_id
                )
            hidden = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
            embedding = functional.linear(hidden, projection)
            keep = input_ids.ne(tokenizer.pad_token_id)
            if not is_query:
                for token_id in punctuation:
                    keep &= input_ids.ne(token_id)
            embedding = functional.normalize(embedding, p=2, dim=2)
            embedding = embedding * keep.unsqueeze(-1)
            encoded_parts.append(embedding.cpu())
            mask_parts.append(keep.cpu())
    result = torch.cat(encoded_parts)
    mask = torch.cat(mask_parts)
    expected = (len(texts), max_length, PROJECTION_DIMENSION)
    if tuple(result.shape) != expected or not bool(torch.isfinite(result).all()):
        raise RuntimeError(f"E900 produced invalid embeddings: {result.shape}")
    return result, mask


def colbert_score_matrix(
    query_embedding,
    document_embedding,
    document_mask,
    *,
    query_batch_size: int = SCORE_QUERY_BATCH_SIZE,
) -> np.ndarray:
    import torch

    if query_embedding.ndim != 3 or document_embedding.ndim != 3:
        raise ValueError("E900 score inputs must be three-dimensional.")
    if document_mask.shape != document_embedding.shape[:2]:
        raise ValueError("E900 document mask shape mismatch.")
    if query_embedding.shape[2] != document_embedding.shape[2]:
        raise ValueError("E900 query/document dimensions differ.")
    if query_batch_size <= 0:
        raise ValueError("E900 score batch size must be positive.")
    output = np.empty(
        (query_embedding.shape[0], document_embedding.shape[0]),
        dtype=np.float32,
    )
    with torch.inference_mode():
        for start in range(0, query_embedding.shape[0], query_batch_size):
            end = min(query_embedding.shape[0], start + query_batch_size)
            token_scores = torch.einsum(
                "qle,dke->qdkl",
                query_embedding[start:end],
                document_embedding,
            )
            token_scores = token_scores.masked_fill(
                ~document_mask[None, :, :, None],
                -9_999.0,
            )
            output[start:end] = (
                token_scores.max(dim=2).values.sum(dim=2).cpu().numpy()
            )
    if not np.isfinite(output).all():
        raise RuntimeError("E900 score matrix contains non-finite values.")
    return output


def retrieval_rows_from_scores(
    scores: np.ndarray,
    query_qids: list[str],
    document_qids: list[str],
) -> pd.DataFrame:
    matrix = np.asarray(scores, dtype=np.float64)
    query_ids = np.asarray(query_qids, dtype=object)
    document_ids = np.asarray(document_qids, dtype=object)
    if matrix.shape != (len(query_ids), len(document_ids)):
        raise ValueError("E900 retrieval score shape mismatch.")
    rows: list[dict[str, object]] = []
    for index, qid in enumerate(query_ids):
        positive = document_ids == qid
        if not positive.any():
            raise ValueError("E900 query has no positive document.")
        order = np.argsort(-matrix[index], kind="stable")
        best_rank = int((np.flatnonzero(positive[order]) + 1).min())
        rows.append(
            {
                "qid_hash": str(qid),
                "reciprocal_rank": 1.0 / best_rank,
                "hit_at_1": int(best_rank <= 1),
                "hit_at_5": int(best_rank <= 5),
                "hit_at_10": int(best_rank <= 10),
            }
        )
    return pd.DataFrame(rows)


def select_benchmark_rows(frame: pd.DataFrame) -> np.ndarray:
    if len(frame) != 599:
        raise ValueError("E900 expected all 599 official MathDial test rows.")
    lengths = frame["conversation"].astype(str).str.len().to_numpy()
    order = np.argsort(lengths, kind="stable")
    positions = np.rint(
        np.linspace(0, len(frame) - 1, BENCHMARK_ROWS)
    ).astype(np.int64)
    selected = order[positions]
    if len(np.unique(selected)) != BENCHMARK_ROWS:
        raise RuntimeError("E900 benchmark selection is not unique.")
    return selected


def audit(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    _, test, source_audit = load_mathdial_alignment_source(project_root)
    return {
        "protocol_id": PROTOCOL_ID,
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "model_license": MODEL_LICENSE,
        "model_files": verify_model_files(project_root),
        "official_code_revision": OFFICIAL_CODE_REVISION,
        "official_code_license_sha256": OFFICIAL_CODE_LICENSE_SHA256,
        "mathdial_source": source_audit,
        "evaluation_rows": len(test),
        "runtime": runtime,
        "competition_text_accessed": False,
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    paths = discover_project_paths(project_root)
    output_path = paths.cache_dir / "e900_colbert_resource_benchmark.json"
    _, test, source_audit = load_mathdial_alignment_source(project_root)
    selected = select_benchmark_rows(test)
    sample = test.iloc[selected].reset_index(drop=True)
    configure_torch()
    peak_rss = _current_rss_bytes()

    colbert, tokenizer, projection = _load_colbert(project_root)
    started = time.perf_counter()
    queries, _ = encode_colbert(
        colbert,
        tokenizer,
        projection,
        sample["confusion"].astype(str).tolist(),
        is_query=True,
    )
    query_seconds = time.perf_counter() - started
    peak_rss = max(peak_rss, _current_rss_bytes())

    started = time.perf_counter()
    documents, document_mask = encode_colbert(
        colbert,
        tokenizer,
        projection,
        sample["conversation"].astype(str).tolist(),
        is_query=False,
    )
    document_seconds = time.perf_counter() - started
    peak_rss = max(peak_rss, _current_rss_bytes())

    started = time.perf_counter()
    colbert_score_matrix(queries, documents, document_mask)
    score_seconds = time.perf_counter() - started
    peak_rss = max(peak_rss, _current_rss_bytes())

    bge_model, bge_tokenizer = load_bge_model(project_root)
    started = time.perf_counter()
    encode_bge_texts(
        bge_model,
        bge_tokenizer,
        [
            objective_query(value)
            for value in sample["confusion"].astype(str)
        ],
        96,
    )
    encode_bge_texts(
        bge_model,
        bge_tokenizer,
        sample["conversation"].astype(str).tolist(),
        256,
    )
    bge_seconds = time.perf_counter() - started
    peak_rss = max(peak_rss, _current_rss_bytes())

    projection_scale = len(test) / BENCHMARK_ROWS
    projected_seconds = (
        (2.0 * query_seconds)
        + document_seconds
        + bge_seconds
    ) * projection_scale + score_seconds * projection_scale**2 * 2.0
    proceed = bool(
        projected_seconds <= MAX_PROJECTED_SECONDS
        and peak_rss < MAX_RSS_BYTES
    )
    result: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime,
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "model_files": verify_model_files(project_root),
        "mathdial_identity_sha256": source_audit[
            "canonical_identity_sha256"
        ],
        "benchmark_rows": BENCHMARK_ROWS,
        "selected_rows": selected.tolist(),
        "query_max_length": QUERY_MAX_LENGTH,
        "document_max_length": DOCUMENT_MAX_LENGTH,
        "encode_batch_size": ENCODE_BATCH_SIZE,
        "score_query_batch_size": SCORE_QUERY_BATCH_SIZE,
        "query_seconds": query_seconds,
        "document_seconds": document_seconds,
        "score_seconds": score_seconds,
        "bge_seconds": bge_seconds,
        "projected_total_seconds": projected_seconds,
        "max_projected_seconds": MAX_PROJECTED_SECONDS,
        "peak_rss_bytes": peak_rss,
        "max_rss_bytes": MAX_RSS_BYTES,
        "proceed": proceed,
        "competition_text_accessed": False,
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    _write_json(output_path, result)
    result["benchmark_path"] = str(output_path)
    result["benchmark_sha256"] = _sha256(output_path)
    return result


def _view_result(
    bge_scores: np.ndarray,
    colbert_scores: np.ndarray,
    qids: list[str],
) -> tuple[dict[str, object], pd.DataFrame]:
    base_rows = retrieval_rows_from_scores(bge_scores, qids, qids)
    candidate_rows = retrieval_rows_from_scores(colbert_scores, qids, qids)
    base_metrics = summarize_retrieval(base_rows)
    candidate_metrics = summarize_retrieval(candidate_rows)
    bootstrap = qid_bootstrap(
        base_rows,
        candidate_rows,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    combined = base_rows.rename(
        columns={
            column: f"bge_{column}"
            for column in base_rows.columns
            if column != "qid_hash"
        }
    )
    candidate = candidate_rows.rename(
        columns={
            column: f"colbert_{column}"
            for column in candidate_rows.columns
            if column != "qid_hash"
        }
    )
    combined = combined.join(candidate.drop(columns=["qid_hash"]))
    return (
        {
            "bge": base_metrics,
            "colbert": candidate_metrics,
            "gain": {
                key: candidate_metrics[key] - base_metrics[key]
                for key in base_metrics
            },
            "bootstrap": bootstrap,
        },
        combined,
    )


def run_external_screen(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    paths = discover_project_paths(project_root)
    benchmark_path = paths.cache_dir / "e900_colbert_resource_benchmark.json"
    if not benchmark_path.is_file():
        raise FileNotFoundError("E900 resource benchmark is required.")
    benchmark_report = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if (
        benchmark_report.get("protocol_id") != PROTOCOL_ID
        or not benchmark_report.get("proceed")
    ):
        raise RuntimeError("E900 resource benchmark did not pass.")

    _, test, source_audit = load_mathdial_alignment_source(project_root)
    configure_torch()
    peak_rss = _current_rss_bytes()
    started = time.perf_counter()

    colbert, tokenizer, projection = _load_colbert(project_root)
    problem_query, _ = encode_colbert(
        colbert,
        tokenizer,
        projection,
        test["question"].astype(str).tolist(),
        is_query=True,
    )
    confusion_query, _ = encode_colbert(
        colbert,
        tokenizer,
        projection,
        test["confusion"].astype(str).tolist(),
        is_query=True,
    )
    documents, document_mask = encode_colbert(
        colbert,
        tokenizer,
        projection,
        test["conversation"].astype(str).tolist(),
        is_query=False,
    )
    colbert_problem_scores = colbert_score_matrix(
        problem_query,
        documents,
        document_mask,
    )
    colbert_confusion_scores = colbert_score_matrix(
        confusion_query,
        documents,
        document_mask,
    )
    peak_rss = max(peak_rss, _current_rss_bytes())

    bge_model, bge_tokenizer = load_bge_model(project_root)
    bge_problem_query = encode_bge_texts(
        bge_model,
        bge_tokenizer,
        [objective_query(value) for value in test["question"].astype(str)],
        96,
    )
    bge_confusion_query = encode_bge_texts(
        bge_model,
        bge_tokenizer,
        [objective_query(value) for value in test["confusion"].astype(str)],
        96,
    )
    bge_documents = encode_bge_texts(
        bge_model,
        bge_tokenizer,
        test["conversation"].astype(str).tolist(),
        256,
    )
    bge_problem_scores = bge_problem_query @ bge_documents.T
    bge_confusion_scores = bge_confusion_query @ bge_documents.T
    peak_rss = max(peak_rss, _current_rss_bytes())

    qids = test["qid_hash"].astype(str).tolist()
    problem, problem_rows = _view_result(
        bge_problem_scores,
        colbert_problem_scores,
        qids,
    )
    confusion, confusion_rows = _view_result(
        bge_confusion_scores,
        colbert_confusion_scores,
        qids,
    )
    clauses = {
        "confusion_mrr_gain_at_least_0_08": (
            confusion["gain"]["mrr"] >= GATES["confusion_mrr_gain"]
        ),
        "confusion_recall5_gain_at_least_0_08": (
            confusion["gain"]["recall_at_5"]
            >= GATES["confusion_recall5_gain"]
        ),
        "confusion_bootstrap_support_at_least_0_95": (
            confusion["bootstrap"]["positive_gain_support"]
            >= GATES["confusion_bootstrap_support"]
        ),
        "problem_mrr_regression_at_most_0_005": (
            problem["gain"]["mrr"] >= GATES["problem_mrr_min_gain"]
        ),
        "problem_recall5_regression_at_most_0_005": (
            problem["gain"]["recall_at_5"]
            >= GATES["problem_recall5_min_gain"]
        ),
    }
    passes = all(clauses.values())
    elapsed = time.perf_counter() - started
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_colbert_late_interaction")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    score_path = run_dir / "retrieval_scores.parquet"
    pd.concat(
        [
            problem_rows.assign(view="problem"),
            confusion_rows.assign(view="confusion"),
        ],
        ignore_index=True,
    ).to_parquet(score_path, index=False)
    report_path = run_dir / "report.json"
    report: dict[str, object] = {
        "run_id": run_id,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": timestamp.isoformat(),
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "model_license": MODEL_LICENSE,
        "model_files": verify_model_files(project_root),
        "source_audit": source_audit,
        "runtime": runtime,
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": peak_rss,
        "evaluation_rows": len(test),
        "query_max_length": QUERY_MAX_LENGTH,
        "document_max_length": DOCUMENT_MAX_LENGTH,
        "pooling": "128-dimensional normalized token matrices",
        "interaction": "sum over query-token maximum document-token cosine",
        "problem": problem,
        "confusion": confusion,
        "clauses": clauses,
        "passes_external_gate": passes,
        "retrieval_scores_sha256": _sha256(score_path),
        "competition_text_accessed": False,
        "competition_outcomes_accessed": False,
        "V_seen_accessed": False,
        "V_objective_accessed": False,
        "V_style_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    _write_json(report_path, report)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "retrieval_scores_path": str(score_path),
        "retrieval_scores_sha256": _sha256(score_path),
        "passes_external_gate": passes,
        "clauses": clauses,
        "problem": problem,
        "confusion": confusion,
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": peak_rss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and run frozen E900 ColBERT late interaction."
    )
    parser.add_argument(
        "stage",
        choices=("audit", "benchmark", "screen"),
    )
    parser.add_argument("--project-root", default=".")
    arguments = parser.parse_args()
    if arguments.stage == "audit":
        result = audit(arguments.project_root)
    elif arguments.stage == "benchmark":
        result = benchmark(arguments.project_root)
    else:
        result = run_external_screen(arguments.project_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
