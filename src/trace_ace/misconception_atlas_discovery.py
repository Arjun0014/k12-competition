from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sentence_transformers import SentenceTransformer


PROTOCOL_ID = "E870_mae_misconception_atlas_external_v1"
SEED = 20260730
BOOTSTRAP_REPLICATES = 5_000
MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 16
THREADS = 6
SOURCE_REVISION = "12bee142d49dfb3c874149035cfbad28547a818b"
EXPECTED_HASHES = {
    "data/data.json": (
        "8223b3a6222c02f4efe8519c4e6abaa2e8e35d96a380bda3882b4f1cdd1fe4e1"
    ),
    "LICENSE": (
        "cca1a8c2bc40c9e58cddec5c88da0330384dafd6600059a063f2ea2cba1d35e4"
    ),
    "README.md": (
        "c91fc759503a5d4c9117e8fe8595026f8528b16b6a8bee15e82f2ffe5dcd8087"
    ),
}
EXPECTED_MODEL_SHA256 = (
    "c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7"
)
SPACE_RE = re.compile(r"\s+")
NONWORD_RE = re.compile(r"[^a-z0-9]+")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_root(project_root: str | Path) -> Path:
    return Path(project_root).resolve() / "Datasets" / "MaE"


def _model_root(project_root: str | Path) -> Path:
    return (
        Path(project_root).resolve()
        / "submission_src"
        / "assets"
        / "bge-base-en-v1.5"
    )


def _normalized_question(value: object) -> str:
    return NONWORD_RE.sub(" ", _text(value).casefold()).strip()


def _text(value: object) -> str:
    if value is None or bool(pd.isna(value)):
        return ""
    return str(value).strip()


def query_text(row: pd.Series | object) -> str:
    return "\n".join(
        [
            "Task: identify the student's mathematical misconception.",
            f"Question: {_text(getattr(row, 'Question'))}",
            f"Student answer: {_text(getattr(row, 'Incorrect_Answer'))}",
        ]
    )


def prototype_text(
    misconception: str,
    examples: pd.DataFrame,
    excluded_question: str | None,
) -> tuple[str, int]:
    eligible = examples
    if excluded_question is not None:
        eligible = examples.loc[
            examples["normalized_question"] != excluded_question
        ]
    parts = [
        "Task: represent a general mathematical misconception.",
        f"Misconception: {SPACE_RE.sub(' ', misconception).strip()}",
    ]
    for index, row in enumerate(eligible.itertuples(index=False), start=1):
        parts.extend(
            [
                f"Example {index} question: {_text(row.Question)}",
                (
                    f"Example {index} student answer: "
                    f"{_text(row.Incorrect_Answer)}"
                ),
            ]
        )
        explanation = _text(row.Explanation)
        if explanation:
            parts.append(f"Example {index} explanation: {explanation}")
    return "\n".join(parts), int(len(eligible))


def verify_source(project_root: str | Path) -> dict[str, object]:
    root = _source_root(project_root)
    observed = {name: _sha256(root / name) for name in EXPECTED_HASHES}
    if observed != EXPECTED_HASHES:
        raise ValueError(f"E870 MaE source hashes changed: {observed}")
    model_hash = _sha256(_model_root(project_root) / "model.safetensors")
    if model_hash != EXPECTED_MODEL_SHA256:
        raise ValueError("E870 BGE-base model hash changed.")
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"E870 MaE source revision changed: {revision}")
    return {
        "revision": revision,
        "license": "MIT",
        "hashes": observed,
        "model_sha256": model_hash,
    }


def load_examples(project_root: str | Path) -> pd.DataFrame:
    raw = json.loads(
        (
            _source_root(project_root) / "data" / "data.json"
        ).read_text(encoding="utf-8")
    )
    frame = pd.DataFrame(raw).rename(
        columns={
            "Misconception ID": "misconception_id",
            "Incorrect Answer": "Incorrect_Answer",
            "Correct Answer": "Correct_Answer",
            "Example Number": "example_number",
        }
    )
    required = (
        "Misconception",
        "misconception_id",
        "Topic",
        "example_number",
        "Question",
        "Incorrect_Answer",
        "Correct_Answer",
        "Explanation",
    )
    if not set(required).issubset(frame.columns):
        raise ValueError("E870 MaE source schema changed.")
    frame = frame[list(required)].copy()
    frame["normalized_question"] = frame["Question"].map(_normalized_question)
    counts = frame.groupby("misconception_id", sort=True).size()
    descriptions = frame.groupby("misconception_id", sort=True)[
        "Misconception"
    ].nunique()
    topics = frame.groupby("misconception_id", sort=True)["Topic"].nunique()
    if (
        len(frame) != 220
        or len(counts) != 55
        or not counts.eq(4).all()
        or not descriptions.eq(1).all()
        or not topics.eq(1).all()
    ):
        raise ValueError("E870 expects 55 misconceptions with four examples.")
    return frame.sort_values(
        ["misconception_id", "example_number"], kind="mergesort"
    ).reset_index(drop=True)


def _load_model(project_root: str | Path) -> SentenceTransformer:
    import torch

    torch.set_num_threads(THREADS)
    model = SentenceTransformer(str(_model_root(project_root)), device="cpu")
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    return model


def _encode(model: SentenceTransformer, values: list[str]) -> np.ndarray:
    return np.asarray(
        model.encode(
            values,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )


def build_external_scores(
    project_root: str | Path,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    model = _load_model(project_root)
    labels = sorted(frame["misconception_id"].unique())
    label_index = {label: index for index, label in enumerate(labels)}
    grouped = {
        label: group.copy()
        for label, group in frame.groupby("misconception_id", sort=True)
    }
    full_prototypes: list[str] = []
    for label in labels:
        group = grouped[label]
        text, _ = prototype_text(
            str(group["Misconception"].iloc[0]), group, None
        )
        full_prototypes.append(text)
    full_vectors = _encode(model, full_prototypes)

    queries = [query_text(row) for row in frame.itertuples(index=False)]
    query_vectors = _encode(model, queries)
    leave_one_texts: list[str] = []
    prototype_example_counts: list[int] = []
    for row in frame.itertuples(index=False):
        group = grouped[str(row.misconception_id)]
        text, count = prototype_text(
            str(row.Misconception),
            group,
            str(row.normalized_question),
        )
        leave_one_texts.append(text)
        prototype_example_counts.append(count)
    leave_one_vectors = _encode(model, leave_one_texts)

    similarity = np.asarray(query_vectors @ full_vectors.T, dtype=np.float64)
    true_indices = frame["misconception_id"].map(label_index).to_numpy()
    similarity[np.arange(len(frame)), true_indices] = np.sum(
        query_vectors * leave_one_vectors, axis=1, dtype=np.float64
    )
    order = np.argsort(-similarity, axis=1, kind="stable")
    ranks = (
        np.argmax(order == true_indices[:, np.newaxis], axis=1) + 1
    ).astype(np.int16)

    topic_ranks: list[int] = []
    for row_index, row in enumerate(frame.itertuples(index=False)):
        topic_labels = sorted(
            frame.loc[frame["Topic"].eq(row.Topic), "misconception_id"].unique()
        )
        indices = np.asarray([label_index[label] for label in topic_labels])
        topic_order = indices[
            np.argsort(-similarity[row_index, indices], kind="stable")
        ]
        topic_ranks.append(
            int(np.flatnonzero(topic_order == true_indices[row_index])[0] + 1)
        )
    topic_ranks_array = np.asarray(topic_ranks, dtype=np.int16)
    true_similarity = similarity[np.arange(len(frame)), true_indices]
    false_similarity = similarity.copy()
    false_similarity[np.arange(len(frame)), true_indices] = -np.inf
    hardest_false = np.max(false_similarity, axis=1)
    scored = frame[
        [
            "misconception_id",
            "Topic",
            "example_number",
            "normalized_question",
        ]
    ].copy()
    scored["prototype_examples"] = prototype_example_counts
    scored["true_similarity"] = true_similarity
    scored["hardest_false_similarity"] = hardest_false
    scored["similarity_margin"] = true_similarity - hardest_false
    scored["rank_all_55"] = ranks
    scored["rank_within_topic"] = topic_ranks_array

    topic_rows: list[dict[str, object]] = []
    for topic, group in scored.groupby("Topic", sort=True):
        topic_rows.append(
            {
                "topic": str(topic),
                "rows": int(len(group)),
                "labels": int(group["misconception_id"].nunique()),
                "top1_all_55": float(np.mean(group["rank_all_55"] <= 1)),
                "top5_all_55": float(np.mean(group["rank_all_55"] <= 5)),
                "top1_within_topic": float(
                    np.mean(group["rank_within_topic"] <= 1)
                ),
                "top3_within_topic": float(
                    np.mean(group["rank_within_topic"] <= 3)
                ),
            }
        )
    topic_frame = pd.DataFrame(topic_rows)
    summary: dict[str, object] = {
        "rows": int(len(scored)),
        "misconceptions": int(scored["misconception_id"].nunique()),
        "topics": int(scored["Topic"].nunique()),
        "minimum_leave_one_prototype_examples": int(
            scored["prototype_examples"].min()
        ),
        "top1_all_55": float(np.mean(ranks <= 1)),
        "top3_all_55": float(np.mean(ranks <= 3)),
        "top5_all_55": float(np.mean(ranks <= 5)),
        "mrr_all_55": float(np.mean(1.0 / ranks)),
        "top1_within_topic": float(np.mean(topic_ranks_array <= 1)),
        "top3_within_topic": float(np.mean(topic_ranks_array <= 3)),
        "mrr_within_topic": float(np.mean(1.0 / topic_ranks_array)),
        "median_similarity_margin": float(
            scored["similarity_margin"].median()
        ),
        "minimum_topic_top5_all_55": float(topic_frame["top5_all_55"].min()),
        "minimum_topic_top3_within_topic": float(
            topic_frame["top3_within_topic"].min()
        ),
        "topic_metrics": topic_rows,
    }
    return scored, summary


def _misconception_bootstrap(scored: pd.DataFrame) -> dict[str, float]:
    grouped = {
        label: group.index.to_numpy()
        for label, group in scored.groupby("misconception_id", sort=True)
    }
    labels = np.asarray(sorted(grouped))
    ranks = scored["rank_all_55"].to_numpy()
    rng = np.random.default_rng(SEED)
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected = rng.choice(labels, size=len(labels), replace=True)
        indices = np.concatenate([grouped[label] for label in selected])
        values[replicate] = np.mean(ranks[indices] <= 5)
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "group_column": "misconception_id",
        "observed_top5": float(np.mean(ranks <= 5)),
        "bootstrap_mean_top5": float(values.mean()),
        "ci_lower": float(np.quantile(values, 0.025)),
        "ci_upper": float(np.quantile(values, 0.975)),
        "support_top5_at_least_0_55": float(np.mean(values >= 0.55)),
    }


def _gate(
    summary: dict[str, object],
    bootstrap: dict[str, float],
) -> dict[str, bool]:
    return {
        "all_220_rows_and_55_misconceptions": (
            int(summary["rows"]) == 220
            and int(summary["misconceptions"]) == 55
        ),
        "at_least_2_nonduplicate_prototype_examples": (
            int(summary["minimum_leave_one_prototype_examples"]) >= 2
        ),
        "top1_all_55_at_least_0_25": (
            float(summary["top1_all_55"]) >= 0.25
        ),
        "top5_all_55_at_least_0_65": (
            float(summary["top5_all_55"]) >= 0.65
        ),
        "mrr_all_55_at_least_0_40": (
            float(summary["mrr_all_55"]) >= 0.40
        ),
        "top1_within_topic_at_least_0_50": (
            float(summary["top1_within_topic"]) >= 0.50
        ),
        "top3_within_topic_at_least_0_75": (
            float(summary["top3_within_topic"]) >= 0.75
        ),
        "mrr_within_topic_at_least_0_65": (
            float(summary["mrr_within_topic"]) >= 0.65
        ),
        "minimum_topic_top5_all_55_at_least_0_40": (
            float(summary["minimum_topic_top5_all_55"]) >= 0.40
        ),
        "minimum_topic_top3_within_topic_at_least_0_50": (
            float(summary["minimum_topic_top3_within_topic"]) >= 0.50
        ),
        "bootstrap_lower_top5_at_least_0_55": (
            float(bootstrap["ci_lower"]) >= 0.55
        ),
        "bootstrap_support_at_least_0_95": (
            float(bootstrap["support_top5_at_least_0_55"]) >= 0.95
        ),
    }


def run(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    root = Path(project_root).resolve()
    source = verify_source(root)
    frame = load_examples(root)
    scored, summary = build_external_scores(root, frame)
    bootstrap = _misconception_bootstrap(scored)
    clauses = _gate(summary, bootstrap)

    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ_misconception_atlas"
    )
    run_dir = root / "experiments" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    scores_path = run_dir / "external_leave_one_example_scores.parquet"
    report_path = run_dir / "report.json"
    scored.drop(columns=["normalized_question"]).to_parquet(
        scores_path, index=False
    )
    report: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "configuration": {
            "seed": SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "batch_size": BATCH_SIZE,
            "threads": THREADS,
            "query_uses_correct_answer": False,
            "query_uses_educator_explanation": False,
            "exclude_duplicate_normalized_questions_from_true_prototype": True,
        },
        "external_metrics": summary,
        "misconception_bootstrap": bootstrap,
        "gate_clauses": clauses,
        "passes_external_gate": bool(all(clauses.values())),
        "runtime_seconds": float(time.perf_counter() - started),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "artifacts": {
            "external_scores": {
                "path": str(scores_path.relative_to(root)),
                "sha256": _sha256(scores_path),
                "rows": int(len(scored)),
            }
        },
        "competition_text_encoded": False,
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
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "external_metrics": summary,
        "misconception_bootstrap": bootstrap,
        "gate_clauses": clauses,
        "passes_external_gate": report["passes_external_gate"],
        "external_scores_sha256": _sha256(scores_path),
        "report_sha256": _sha256(report_path),
        "runtime_seconds": report["runtime_seconds"],
        "competition_text_encoded": False,
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run E870 MaE misconception-atlas external gate."
    )
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    print(
        json.dumps(run(args.project_root), indent=2, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
