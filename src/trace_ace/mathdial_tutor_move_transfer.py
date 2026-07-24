from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from trace_ace.io import discover_project_paths


SOURCE_COMMIT = "b06c020a0a1f57a87577fec33e657b63e7eb476e"
SOURCE_SHA256 = {
    "train": "96980babee081a3da48ed0f1fb3068ab52f23ddc67e63bfd5ec99ad701fd29cc",
    "test": "28d1e537d65a6e6ff7b8bde602a2c7e2493b93d6bc785d1848506a650997f122",
}
MOVE_TAXONOMY = ("generic", "probing", "focus", "telling")
CLASS_ORDER = tuple(sorted(MOVE_TAXONOMY))
SEED = 20260724
BOOTSTRAP_REPLICATES = 5_000
WORD_MAX_FEATURES = 50_000
CHAR_MAX_FEATURES = 75_000
EXPECTED_DIALOGUES = {"train": 1_679, "test": 595}
EXPECTED_MOVE_ROWS = {"train": 11_106, "test": 3_664}
EXPECTED_CLASS_COUNTS = {
    "train": {"focus": 4_102, "generic": 2_611, "probing": 2_567, "telling": 1_826},
    "test": {"focus": 1_241, "generic": 884, "probing": 946, "telling": 593},
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(value: object) -> str:
    return " ".join(str(value).split())


def _load_source(project_root: str | Path, split: str) -> list[dict[str, Any]]:
    paths = discover_project_paths(project_root)
    source_path = paths.root / "Datasets" / "MathDial" / "data" / f"{split}.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing MathDial source split: {source_path}")
    observed_sha = _sha256(source_path)
    if observed_sha != SOURCE_SHA256[split]:
        raise ValueError(f"MathDial {split} JSONL SHA-256 changed: {observed_sha}")
    rows: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for source_row, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "qid",
                "scenario",
                "self-correctness",
                "student_incorrect_solution",
                "conversation",
            }
            missing = sorted(required.difference(row))
            if missing:
                raise ValueError(
                    f"MathDial {split} row {source_row} is missing columns: {missing}"
                )
            row["_source_row"] = source_row
            rows.append(row)
    return rows


def _is_labeled_dialogue(row: dict[str, Any]) -> bool:
    value = row.get("self-correctness")
    return value is not None and _normalize(value) != ""


def _parse_dialogue_moves(
    row: dict[str, Any],
    *,
    split: str,
    audit: Counter[str],
) -> list[dict[str, object]]:
    source_row = int(row["_source_row"])
    qid = str(row["qid"])
    scenario = str(row["scenario"])
    dialogue_id = f"{split}-{source_row:06d}-{qid}-{scenario}"
    previous_non_teacher = _normalize(row["student_incorrect_solution"])
    if not previous_non_teacher:
        raise ValueError(f"MathDial {dialogue_id} has an empty initial student solution.")

    parsed: list[dict[str, object]] = []
    for turn_index, raw_turn in enumerate(str(row["conversation"]).split("|EOM|")):
        turn = _normalize(raw_turn)
        if not turn:
            audit["empty_turns_excluded"] += 1
            continue
        if ":" not in turn:
            previous_non_teacher = turn
            audit["unmarked_non_teacher_turns"] += 1
            continue
        speaker, body = turn.split(":", 1)
        speaker = _normalize(speaker)
        body = _normalize(body)
        if speaker.casefold() != "teacher":
            previous_non_teacher = body if body else turn
            audit["non_teacher_turns_seen"] += 1
            continue

        audit["teacher_turns_seen"] += 1
        if not body.startswith("(") or ")" not in body:
            audit["invalid_move_marker_excluded"] += 1
            continue
        marker, teacher_text = body[1:].split(")", 1)
        move = _normalize(marker).casefold()
        teacher_text = _normalize(teacher_text)
        if move not in MOVE_TAXONOMY:
            audit["unknown_move_excluded"] += 1
            continue
        if not teacher_text:
            audit["empty_teacher_text_excluded"] += 1
            continue
        if not previous_non_teacher:
            audit["missing_previous_non_teacher_excluded"] += 1
            continue
        parsed.append(
            {
                "dialogue_id": dialogue_id,
                "qid": qid,
                "scenario": scenario,
                "split": split,
                "source_row": source_row,
                "turn_index": turn_index,
                "previous_student": previous_non_teacher,
                "teacher_text": teacher_text,
                "move": move,
                "model_text": (
                    f"[STUDENT] {previous_non_teacher}\n[TUTOR] {teacher_text}"
                ),
            }
        )
    return parsed


def parse_mathdial_tutor_moves(
    project_root: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    sources = {split: _load_source(project_root, split) for split in ("train", "test")}
    labeled = {
        split: [row for row in rows if _is_labeled_dialogue(row)]
        for split, rows in sources.items()
    }
    test_qids = {str(row["qid"]) for row in labeled["test"]}
    train_qids = {str(row["qid"]) for row in labeled["train"]}
    overlap = train_qids.intersection(test_qids)
    eligible = {
        "train": [
            row for row in labeled["train"] if str(row["qid"]) not in overlap
        ],
        "test": labeled["test"],
    }
    if len(eligible["train"]) != EXPECTED_DIALOGUES["train"]:
        raise ValueError(
            "MathDial leakage-safe train dialogue count changed: "
            f"{len(eligible['train'])}"
        )
    if len(eligible["test"]) != EXPECTED_DIALOGUES["test"]:
        raise ValueError(
            f"MathDial official test dialogue count changed: {len(eligible['test'])}"
        )
    if {str(row["qid"]) for row in eligible["train"]}.intersection(test_qids):
        raise RuntimeError("MathDial tutor-move transfer has question leakage.")

    records: list[dict[str, object]] = []
    split_audits: dict[str, dict[str, object]] = {}
    for split in ("train", "test"):
        counter: Counter[str] = Counter()
        for row in eligible[split]:
            records.extend(_parse_dialogue_moves(row, split=split, audit=counter))
        split_frame = pd.DataFrame(
            record for record in records if record["split"] == split
        )
        class_counts = {
            move: int((split_frame["move"] == move).sum())
            for move in CLASS_ORDER
        }
        if len(split_frame) != EXPECTED_MOVE_ROWS[split]:
            raise ValueError(
                f"MathDial {split} move-row count changed: {len(split_frame)}"
            )
        if class_counts != EXPECTED_CLASS_COUNTS[split]:
            raise ValueError(
                f"MathDial {split} move class counts changed: {class_counts}"
            )
        split_audits[split] = {
            "raw_dialogues": len(sources[split]),
            "missing_outcome_dialogues_excluded": (
                len(sources[split]) - len(labeled[split])
            ),
            "overlap_question_ids": len(overlap),
            "overlap_dialogues_purged": (
                len(labeled["train"]) - len(eligible["train"])
                if split == "train"
                else 0
            ),
            "eligible_dialogues": len(eligible[split]),
            "eligible_question_ids": len(
                {str(row["qid"]) for row in eligible[split]}
            ),
            "move_rows": len(split_frame),
            "class_counts": class_counts,
            **{key: int(value) for key, value in sorted(counter.items())},
        }

    frame = pd.DataFrame.from_records(records)
    frame = frame.sort_values(
        ["split", "source_row", "turn_index"], kind="mergesort"
    ).reset_index(drop=True)
    if frame.empty or frame["model_text"].eq("").any():
        raise ValueError("MathDial tutor-move cache contains empty model text.")
    if not set(frame["move"]).issubset(MOVE_TAXONOMY):
        raise ValueError("MathDial tutor-move cache contains an unknown class.")
    audit = {
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "taxonomy": list(MOVE_TAXONOMY),
        "class_order": list(CLASS_ORDER),
        "qid_overlap_count": len(overlap),
        "splits": split_audits,
    }
    return frame, audit


def prepare_mathdial_tutor_move_cache(
    project_root: str | Path,
) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    frame, audit = parse_mathdial_tutor_moves(project_root)
    cache_path = paths.cache_dir / "mathdial_tutor_moves_e530.parquet"
    metadata_path = paths.cache_dir / "mathdial_tutor_moves_e530.metadata.json"
    frame.to_parquet(cache_path, index=False)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "E530_mathdial_tutor_move_transfer",
        "dataset": "MathDial",
        "dataset_license": "CC BY-SA 4.0",
        "audit": audit,
        "cache_rows": len(frame),
        "cache_sha256": _sha256(cache_path),
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


def build_vectorizers() -> tuple[TfidfVectorizer, TfidfVectorizer]:
    shared = {
        "lowercase": True,
        "strip_accents": "unicode",
        "min_df": 2,
        "sublinear_tf": True,
        "norm": "l2",
        "dtype": np.float64,
    }
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=WORD_MAX_FEATURES,
        **shared,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=CHAR_MAX_FEATURES,
        **shared,
    )
    return word, char


def _combine_blocks(
    word_block: sparse.spmatrix, char_block: sparse.spmatrix
) -> sparse.csr_matrix:
    scale = 1.0 / np.sqrt(2.0)
    return sparse.hstack(
        [word_block * scale, char_block * scale], format="csr"
    )


def _top_label_ece(
    labels: np.ndarray, probability: np.ndarray, *, n_bins: int = 10
) -> float:
    confidence = probability.max(axis=1)
    prediction = probability.argmax(axis=1)
    correctness = prediction == labels
    bin_index = np.minimum((confidence * n_bins).astype(np.int64), n_bins - 1)
    value = 0.0
    for index in range(n_bins):
        mask = bin_index == index
        if mask.any():
            value += float(mask.mean()) * abs(
                float(correctness[mask].mean()) - float(confidence[mask].mean())
            )
    return float(value)


def _multiclass_metrics(
    labels: Sequence[str],
    probability: np.ndarray,
    classes: Sequence[str],
) -> dict[str, object]:
    class_array = np.asarray(classes, dtype=object)
    class_to_index = {str(label): index for index, label in enumerate(class_array)}
    encoded = np.asarray([class_to_index[str(label)] for label in labels], dtype=np.int64)
    prediction_index = probability.argmax(axis=1)
    prediction = class_array[prediction_index]
    clipped = np.clip(probability, 1e-15, 1.0)
    row_loss = -np.log(clipped[np.arange(len(encoded)), encoded])
    one_hot = np.eye(len(class_array), dtype=np.float64)[encoded]
    class_f1 = f1_score(
        np.asarray(labels, dtype=object),
        prediction,
        labels=class_array.tolist(),
        average=None,
        zero_division=0,
    )
    return {
        "rows": len(encoded),
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(class_f1.mean()),
        "class_f1": {
            str(label): float(score)
            for label, score in zip(class_array, class_f1)
        },
        "top_label_ece_10": _top_label_ece(encoded, probability, n_bins=10),
        "log_loss": float(row_loss.mean()),
        "multiclass_brier": float(np.square(probability - one_hot).sum(axis=1).mean()),
        "prediction_class_counts": {
            str(label): int((prediction == label).sum()) for label in class_array
        },
        "prediction_confidence_mean": float(probability.max(axis=1).mean()),
    }


def _qid_bootstrap(
    qids: Sequence[str],
    labels: Sequence[str],
    candidate_probability: np.ndarray,
    prior_probability: np.ndarray,
    classes: Sequence[str],
    *,
    n_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
) -> dict[str, object]:
    if n_replicates < BOOTSTRAP_REPLICATES:
        raise ValueError("E530 requires at least 5,000 qid bootstrap replicates.")
    class_to_index = {str(label): index for index, label in enumerate(classes)}
    encoded = np.asarray([class_to_index[str(label)] for label in labels], dtype=np.int64)
    candidate_loss = -np.log(
        np.clip(candidate_probability, 1e-15, 1.0)[
            np.arange(len(encoded)), encoded
        ]
    )
    prior_loss = -np.log(
        np.clip(prior_probability, 1e-15, 1.0)[np.arange(len(encoded)), encoded]
    )
    qid_array = np.asarray(qids, dtype=str)
    unique_qids = np.unique(qid_array)
    if len(unique_qids) < 2:
        raise ValueError("E530 qid bootstrap requires at least two question groups.")
    gain_sums = np.empty(len(unique_qids), dtype=np.float64)
    counts = np.empty(len(unique_qids), dtype=np.int64)
    for index, qid in enumerate(unique_qids):
        mask = qid_array == qid
        gain_sums[index] = float((prior_loss[mask] - candidate_loss[mask]).sum())
        counts[index] = int(mask.sum())
    rng = np.random.default_rng(seed)
    gains = np.empty(n_replicates, dtype=np.float64)
    for replicate in range(n_replicates):
        draw = rng.integers(0, len(unique_qids), size=len(unique_qids))
        gains[replicate] = float(gain_sums[draw].sum() / counts[draw].sum())
    return {
        "resampler": "test_qid",
        "groups": len(unique_qids),
        "replicates": n_replicates,
        "seed": seed,
        "support_positive_log_loss_gain": float((gains > 0.0).mean()),
        "mean_log_loss_gain": float(gains.mean()),
        "ci_2_5": float(np.quantile(gains, 0.025)),
        "ci_97_5": float(np.quantile(gains, 0.975)),
    }


def external_gate(
    metrics: dict[str, object],
    prior_metrics: dict[str, object],
    bootstrap: dict[str, object],
) -> dict[str, object]:
    class_f1 = {str(key): float(value) for key, value in metrics["class_f1"].items()}
    clauses = {
        "accuracy_at_least_0_60": float(metrics["accuracy"]) >= 0.60,
        "macro_f1_at_least_0_55": float(metrics["macro_f1"]) >= 0.55,
        "every_class_f1_at_least_0_45": min(class_f1.values()) >= 0.45,
        "top_label_ece_10_at_most_0_15": (
            float(metrics["top_label_ece_10"]) <= 0.15
        ),
        "log_loss_gain_at_least_0_20": (
            float(prior_metrics["log_loss"]) - float(metrics["log_loss"]) >= 0.20
        ),
        "multiclass_brier_gain_at_least_0_05": (
            float(prior_metrics["multiclass_brier"])
            - float(metrics["multiclass_brier"])
            >= 0.05
        ),
        "qid_bootstrap_support_at_least_0_95": (
            float(bootstrap["support_positive_log_loss_gain"]) >= 0.95
        ),
    }
    return {
        "clauses": clauses,
        "passes_external_gate": bool(all(clauses.values())),
    }


def run_mathdial_tutor_move_transfer(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    paths = discover_project_paths(project_root)
    cache_result = prepare_mathdial_tutor_move_cache(project_root)
    frame = pd.read_parquet(cache_result["cache_path"])
    train = frame.loc[frame["split"].eq("train")].reset_index(drop=True)
    test = frame.loc[frame["split"].eq("test")].reset_index(drop=True)
    if set(train["qid"]).intersection(test["qid"]):
        raise RuntimeError("E530 train/test question leakage detected after cache load.")

    word, char = build_vectorizers()
    train_word = word.fit_transform(train["model_text"])
    test_word = word.transform(test["model_text"])
    train_char = char.fit_transform(train["model_text"])
    test_char = char.transform(test["model_text"])
    train_matrix = _combine_blocks(train_word, train_char)
    test_matrix = _combine_blocks(test_word, test_char)
    classifier = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=1_000,
        random_state=SEED,
    )
    classifier.fit(train_matrix, train["move"])
    if tuple(classifier.classes_) != CLASS_ORDER:
        raise RuntimeError(
            f"E530 classifier class order changed: {tuple(classifier.classes_)}"
        )
    probability = classifier.predict_proba(test_matrix)
    if probability.shape != (len(test), len(CLASS_ORDER)):
        raise RuntimeError(f"E530 probability shape changed: {probability.shape}")
    if not np.isfinite(probability).all():
        raise RuntimeError("E530 produced non-finite probabilities.")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-10):
        raise RuntimeError("E530 class probabilities do not sum to one.")

    train_counts = np.asarray(
        [(train["move"] == label).sum() for label in CLASS_ORDER], dtype=np.float64
    )
    prior = train_counts / train_counts.sum()
    prior_probability = np.repeat(prior[None, :], len(test), axis=0)
    metrics = _multiclass_metrics(test["move"], probability, CLASS_ORDER)
    prior_metrics = _multiclass_metrics(
        test["move"], prior_probability, CLASS_ORDER
    )
    bootstrap = _qid_bootstrap(
        test["qid"],
        test["move"],
        probability,
        prior_probability,
        CLASS_ORDER,
    )
    gate = external_gate(metrics, prior_metrics, bootstrap)

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_mathdial_tutor_move_transfer")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model_path = run_dir / "mathdial_tutor_move_model.joblib"
    artifact = {
        "candidate": "E530_mathdial_tutor_move_transfer",
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "move_taxonomy": MOVE_TAXONOMY,
        "class_order": CLASS_ORDER,
        "word_vectorizer": word,
        "char_vectorizer": char,
        "classifier": classifier,
        "block_scale": 1.0 / np.sqrt(2.0),
        "seed": SEED,
    }
    joblib.dump(artifact, model_path, compress=3)
    prediction_frame = test[
        ["dialogue_id", "qid", "source_row", "turn_index", "move"]
    ].copy()
    for index, label in enumerate(CLASS_ORDER):
        prediction_frame[f"prob_{label}"] = probability[:, index]
    prediction_path = run_dir / "test_predictions.parquet"
    prediction_frame.to_parquet(prediction_path, index=False)
    bootstrap_path = run_dir / "qid_bootstrap.json"
    bootstrap_path.write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    elapsed = time.perf_counter() - started
    report = {
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "candidate": "E530_mathdial_tutor_move_transfer",
        "configuration": {
            "source_commit": SOURCE_COMMIT,
            "source_sha256": SOURCE_SHA256,
            "input": "[STUDENT] previous_non_teacher\\n[TUTOR] teacher_text",
            "word_vectorizer": {
                "analyzer": "word",
                "ngram_range": [1, 2],
                "max_features": WORD_MAX_FEATURES,
                "min_df": 2,
                "lowercase": True,
                "strip_accents": "unicode",
                "sublinear_tf": True,
                "norm": "l2",
            },
            "char_vectorizer": {
                "analyzer": "char_wb",
                "ngram_range": [3, 5],
                "max_features": CHAR_MAX_FEATURES,
                "min_df": 2,
                "lowercase": True,
                "strip_accents": "unicode",
                "sublinear_tf": True,
                "norm": "l2",
            },
            "block_scale": 1.0 / np.sqrt(2.0),
            "classifier": {
                "C": 1.0,
                "solver": "liblinear",
                "max_iter": 1_000,
                "random_state": SEED,
                "class_weight": None,
            },
        },
        "data_audit": cache_result["metadata"]["audit"],
        "cache_sha256": cache_result["metadata"]["cache_sha256"],
        "train_matrix_shape": list(train_matrix.shape),
        "test_matrix_shape": list(test_matrix.shape),
        "word_vocabulary_size": len(word.vocabulary_),
        "char_vocabulary_size": len(char.vocabulary_),
        "classifier_classes": classifier.classes_.tolist(),
        "classifier_iterations": classifier.n_iter_.tolist(),
        "test_metrics": metrics,
        "train_prior": {
            label: float(value) for label, value in zip(CLASS_ORDER, prior)
        },
        "prior_metrics": prior_metrics,
        "paired_qid_bootstrap": bootstrap,
        "external_gate": gate,
        "runtime_seconds": elapsed,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "model_file": model_path.name,
        "model_bytes": model_path.stat().st_size,
        "model_sha256": _sha256(model_path),
        "prediction_file": prediction_path.name,
        "prediction_sha256": _sha256(prediction_path),
        "bootstrap_file": bootstrap_path.name,
        "bootstrap_sha256": _sha256(bootstrap_path),
        "competition_cache_authorized": bool(gate["passes_external_gate"]),
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "passes_external_gate": bool(gate["passes_external_gate"]),
    }
    return result


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen E530 MathDial tutor-move transfer."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--stage", choices=("prepare", "run", "pipeline"), default="pipeline"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.stage == "prepare":
        result = prepare_mathdial_tutor_move_cache(args.project_root)
    else:
        result = run_mathdial_tutor_move_transfer(args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
