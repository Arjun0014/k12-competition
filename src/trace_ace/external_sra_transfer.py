from __future__ import annotations

import argparse
import hashlib
import json
import random
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from trace_ace.io import discover_project_paths
from trace_ace.nli_cross_encoder_cache import mastery_hypothesis
from trace_ace.semantic_cache import compact_objective_context


SEED = 20260722
MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
MODEL_DIRECTORY = "nli-deberta-v3-small"
MAX_LENGTH = 256
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
EPOCHS = 1
ENCODER_LEARNING_RATE = 2e-5
HEAD_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.10
UNFROZEN_LAYERS = 2
EXPECTED_LABELS = {0: "contradiction", 1: "entailment", 2: "neutral"}
SRA_TO_NLI = {
    "correct": 1,
    "partially_correct_incomplete": 2,
    "contradictory": 0,
    "irrelevant": 0,
    "non_domain": 0,
}
EXPECTED_SPLIT_ROWS = {
    "train": 8910,
    "test-unseen-answers": 979,
    "test-unseen-questions": 1552,
    "test-unseen-domains": 4562,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_frame_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    digest = hashlib.sha256()
    for row in frame[columns].itertuples(index=False, name=None):
        digest.update(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def normalize_sra_label(value: str) -> str:
    return "_".join(str(value).strip().lower().replace("-", " ").split())


def canonical_reference_answers(question: ET.Element) -> list[str]:
    rows: list[tuple[str, str]] = []
    for answer in question.findall("./referenceAnswers/referenceAnswer"):
        text = " ".join((answer.text or "").split()).strip()
        if text:
            rows.append((str(answer.attrib.get("category", "")).upper(), text))
    best = [text for category, text in rows if category == "BEST"]
    selected = best or [text for _, text in rows]
    return list(dict.fromkeys(selected))


def parse_sem_eval_2013(dataset_root: str | Path) -> pd.DataFrame:
    root = Path(dataset_root) / "semeval-5way"
    if not root.is_dir():
        raise FileNotFoundError(f"SemEval five-way directory not found: {root}")
    records: list[dict[str, object]] = []
    for corpus in ("beetle", "sciEntsBank"):
        corpus_root = root / corpus
        if not corpus_root.is_dir():
            raise FileNotFoundError(f"Missing SemEval corpus: {corpus_root}")
        for split_dir in sorted(path for path in corpus_root.iterdir() if path.is_dir()):
            if split_dir.name == "reliability":
                continue
            core = split_dir / "Core"
            if not core.is_dir():
                continue
            for xml_path in sorted(core.glob("*.xml")):
                question = ET.parse(xml_path).getroot()
                question_text = " ".join(
                    (question.findtext("questionText", default="") or "").split()
                ).strip()
                references = canonical_reference_answers(question)
                if not question_text or not references:
                    raise ValueError(f"Missing question/reference text in {xml_path}")
                hypothesis = " [OR] ".join(references)
                for answer in question.findall("./studentAnswers/studentAnswer"):
                    answer_id = str(answer.attrib.get("id", "")).strip()
                    label = normalize_sra_label(answer.attrib.get("accuracy", ""))
                    answer_text = " ".join((answer.text or "").split()).strip()
                    if not label:
                        continue
                    if label not in SRA_TO_NLI:
                        raise ValueError(f"Unknown SemEval label {label!r} in {xml_path}")
                    if not answer_id or not answer_text:
                        raise ValueError(f"Missing student answer ID/text in {xml_path}")
                    records.append(
                        {
                            "student_answer_id": answer_id,
                            "corpus": corpus,
                            "split": split_dir.name,
                            "module": str(question.attrib.get("module", "")),
                            "question_id": str(question.attrib.get("id", "")),
                            "question": question_text,
                            "reference_answer": hypothesis,
                            "student_answer": answer_text,
                            "sra_label": label,
                            "nli_label": int(SRA_TO_NLI[label]),
                            "source_file": xml_path.relative_to(root).as_posix(),
                        }
                    )
    frame = pd.DataFrame(records).sort_values(
        ["split", "corpus", "question_id", "student_answer_id"], kind="mergesort"
    )
    if frame.empty:
        raise ValueError("SemEval parser produced no labeled rows.")
    duplicates = frame.loc[frame["student_answer_id"].duplicated(keep=False)]
    if not duplicates.empty:
        conflicts = duplicates.groupby("student_answer_id").agg(
            labels=("sra_label", "nunique"),
            answers=("student_answer", "nunique"),
            splits=("split", "nunique"),
        )
        if (conflicts > 1).any(axis=None):
            raise ValueError("Conflicting duplicate SemEval student-answer IDs detected.")
        frame = frame.drop_duplicates("student_answer_id", keep="first")
    frame = frame.reset_index(drop=True)
    actual_counts = frame["split"].value_counts().sort_index().to_dict()
    if Path(dataset_root).name == "TalkMoves" and actual_counts != EXPECTED_SPLIT_ROWS:
        raise ValueError(
            f"Canonical SemEval split counts changed: {actual_counts} != {EXPECTED_SPLIT_ROWS}"
        )
    return frame


def prepare_sem_eval_cache(project_root: str | Path) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    frame = parse_sem_eval_2013(paths.root / "Datasets" / "TalkMoves")
    output_path = paths.cache_dir / "sem_eval_2013_task7_canonical.parquet"
    metadata_path = paths.cache_dir / "sem_eval_2013_task7_canonical.metadata.json"
    frame.to_parquet(output_path, index=False)
    columns = [
        "student_answer_id",
        "corpus",
        "split",
        "question_id",
        "question",
        "reference_answer",
        "student_answer",
        "sra_label",
        "nli_label",
    ]
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "SemEval-2013 Task 7 Student Response Analysis",
        "license": "CC BY-SA 3.0",
        "organizer_sharealike_ruling_supplied_on": "2026-07-22",
        "parser": "canonical five-way Core XML; reliability excluded; answer-ID dedupe",
        "rows": int(len(frame)),
        "split_rows": {
            key: int(value) for key, value in frame["split"].value_counts().items()
        },
        "label_rows": {
            key: int(value) for key, value in frame["sra_label"].value_counts().items()
        },
        "ordered_content_sha256": _ordered_frame_sha256(frame, columns),
        "parquet_sha256": _sha256(output_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "cache_path": str(output_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }


def _premises(frame: pd.DataFrame) -> list[str]:
    return [
        f"[QUESTION] {question} [STUDENT ANSWER] {answer}"
        for question, answer in zip(frame["question"], frame["student_answer"])
    ]


def _freeze_deberta(model, unfrozen_layers: int) -> dict[str, int]:
    if not hasattr(model, "deberta"):
        raise TypeError("E400 expects a DeBERTa-v2-family sequence classifier.")
    layers = model.deberta.encoder.layer
    if not 1 <= int(unfrozen_layers) <= len(layers):
        raise ValueError("unfrozen_layers must be between one and the encoder depth.")
    for parameter in model.parameters():
        parameter.requires_grad = False
    first_trainable = len(layers) - int(unfrozen_layers)
    for layer in layers[first_trainable:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for module in (model.pooler, model.classifier):
        for parameter in module.parameters():
            parameter.requires_grad = True
    return {
        "total_encoder_layers": int(len(layers)),
        "frozen_encoder_layers": int(first_trainable),
        "unfrozen_encoder_layers": int(unfrozen_layers),
    }


def _tokenize(tokenizer, frame: pd.DataFrame):
    import torch

    encoded = tokenizer(
        _premises(frame),
        frame["reference_answer"].astype(str).tolist(),
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


def _evaluate_external(model, keys, tensors, labels: np.ndarray) -> tuple[np.ndarray, dict]:
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
            inputs = {key: value for key, value in zip(keys, batch)}
            logits = model(**inputs).logits
            batches.append(logits.cpu().numpy().astype(np.float64))
            if batch_number % 25 == 0 or batch_number == len(loader):
                print(f"external_eval_batch={batch_number}/{len(loader)}", flush=True)
    logits = np.concatenate(batches)
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(axis=1, keepdims=True)
    predicted = probability.argmax(axis=1)
    correct = labels == 1
    metrics = {
        "rows": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "multiclass_log_loss": float(log_loss(labels, probability, labels=[0, 1, 2])),
        "correct_vs_rest_auroc": float(roc_auc_score(correct.astype(int), probability[:, 1])),
    }
    return probability, metrics


def _trainable_state_dict(model) -> dict[str, object]:
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = model.state_dict()
    result = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in state.items()
        if name in trainable
    }
    if set(result) != trainable:
        raise RuntimeError("Trainable tensors are missing from the E400 state dictionary.")
    return result


def train_external_sra_transfer(project_root: str | Path) -> dict[str, object]:
    import torch
    from safetensors.torch import load_file, save_file
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    paths = discover_project_paths(project_root)
    cache_path = paths.cache_dir / "sem_eval_2013_task7_canonical.parquet"
    metadata_path = paths.cache_dir / "sem_eval_2013_task7_canonical.metadata.json"
    if not cache_path.exists() or not metadata_path.exists():
        prepare_sem_eval_cache(project_root)
    frame = pd.read_parquet(cache_path)
    train = frame.loc[frame["split"].eq("train")].reset_index(drop=True)
    unseen_questions = frame.loc[
        frame["split"].eq("test-unseen-questions")
    ].reset_index(drop=True)
    unseen_domains = frame.loc[
        frame["split"].eq("test-unseen-domains")
    ].reset_index(drop=True)
    if len(train) != EXPECTED_SPLIT_ROWS["train"]:
        raise ValueError("Unexpected E400 external training-row count.")

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
        model_path, local_files_only=True
    )
    actual_labels = {
        int(key): str(value).lower() for key, value in model.config.id2label.items()
    }
    if actual_labels != EXPECTED_LABELS:
        raise ValueError(f"Unexpected base NLI labels: {actual_labels}")
    freeze_summary = _freeze_deberta(model, UNFROZEN_LAYERS)

    train_keys, train_tensors = _tokenize(tokenizer, train)
    question_keys, question_tensors = _tokenize(tokenizer, unseen_questions)
    domain_keys, domain_tensors = _tokenize(tokenizer, unseen_domains)
    if not (train_keys == question_keys == domain_keys):
        raise RuntimeError("E400 tokenizer fields differ across external splits.")
    baseline_question_probability, baseline_question_metrics = _evaluate_external(
        model,
        question_keys,
        question_tensors,
        unseen_questions["nli_label"].to_numpy(dtype=np.int64),
    )
    baseline_domain_probability, baseline_domain_metrics = _evaluate_external(
        model,
        domain_keys,
        domain_tensors,
        unseen_domains["nli_label"].to_numpy(dtype=np.int64),
    )
    labels = torch.tensor(train["nli_label"].to_numpy(), dtype=torch.long)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(*train_tensors, labels),
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
            if batch_number % 25 == 0 or batch_number == len(loader):
                print(
                    f"external_epoch={epoch + 1} step={batch_number}/{len(loader)} "
                    f"train_loss={running_loss / seen:.6f}",
                    flush=True,
                )
        training_rows.append(
            {"epoch": epoch + 1, "steps": global_step, "train_loss": running_loss / seen}
        )

    question_probability, question_metrics = _evaluate_external(
        model,
        question_keys,
        question_tensors,
        unseen_questions["nli_label"].to_numpy(dtype=np.int64),
    )
    domain_probability, domain_metrics = _evaluate_external(
        model,
        domain_keys,
        domain_tensors,
        unseen_domains["nli_label"].to_numpy(dtype=np.int64),
    )
    clauses = {
        "unseen_questions_correct_auroc_at_least_0_70": (
            question_metrics["correct_vs_rest_auroc"] >= 0.70
        ),
        "unseen_domains_correct_auroc_at_least_0_68": (
            domain_metrics["correct_vs_rest_auroc"] >= 0.68
        ),
        "unseen_questions_macro_f1_at_least_0_45": question_metrics["macro_f1"]
        >= 0.45,
        "unseen_questions_correct_auroc_not_regressed": (
            question_metrics["correct_vs_rest_auroc"]
            >= baseline_question_metrics["correct_vs_rest_auroc"]
        ),
        "unseen_domains_correct_auroc_not_regressed": (
            domain_metrics["correct_vs_rest_auroc"]
            >= baseline_domain_metrics["correct_vs_rest_auroc"]
        ),
    }
    passes_gate = bool(all(clauses.values()))

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_external_sra_transfer")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    delta_path = run_dir / "sra_deberta_delta.safetensors"
    delta = _trainable_state_dict(model)
    save_file(delta, str(delta_path))
    loaded = load_file(str(delta_path), device="cpu")
    if set(loaded) != set(delta):
        raise RuntimeError("E400 serialized delta key mismatch.")
    max_difference = max(
        float(torch.max(torch.abs(delta[name] - loaded[name])).item()) for name in delta
    )
    if max_difference != 0.0:
        raise RuntimeError("E400 delta serialization is not exact.")

    pd.DataFrame(training_rows).to_csv(
        run_dir / "training_metrics.csv", index=False, lineterminator="\n"
    )
    metrics = pd.DataFrame(
        [
            {"split": "test-unseen-questions", **question_metrics},
            {"split": "test-unseen-domains", **domain_metrics},
        ]
    )
    metrics.to_csv(run_dir / "external_metrics.csv", index=False, lineterminator="\n")
    prediction_frames = []
    for split_frame, probability in (
        (unseen_questions, question_probability),
        (unseen_domains, domain_probability),
    ):
        prediction_frames.append(
            pd.DataFrame(
                {
                    "student_answer_id": split_frame["student_answer_id"],
                    "split": split_frame["split"],
                    "nli_label": split_frame["nli_label"],
                    "prob_contradiction": probability[:, 0],
                    "prob_entailment": probability[:, 1],
                    "prob_neutral": probability[:, 2],
                }
            )
        )
    for split_frame, probability in (
        (unseen_questions, baseline_question_probability),
        (unseen_domains, baseline_domain_probability),
    ):
        prediction_frames.append(
            pd.DataFrame(
                {
                    "student_answer_id": split_frame["student_answer_id"],
                    "split": split_frame["split"].astype(str) + "-base-nli",
                    "nli_label": split_frame["nli_label"],
                    "prob_contradiction": probability[:, 0],
                    "prob_entailment": probability[:, 1],
                    "prob_neutral": probability[:, 2],
                }
            )
        )
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        run_dir / "external_predictions.parquet", index=False
    )
    dataset_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    report = {
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "candidate": "E400_sem_eval_adapted_deberta",
        "model": MODEL_NAME,
        "model_license": "Apache-2.0",
        "external_dataset": "SemEval-2013 Task 7 Student Response Analysis",
        "external_dataset_license": "CC BY-SA 3.0",
        "dataset_content_sha256": dataset_metadata["ordered_content_sha256"],
        "training_rows": int(len(train)),
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "batch_size": TRAIN_BATCH_SIZE,
        "encoder_learning_rate": ENCODER_LEARNING_RATE,
        "head_learning_rate": HEAD_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_fraction": WARMUP_FRACTION,
        "seed": SEED,
        "freeze_summary": freeze_summary,
        "external_metrics": {
            "test-unseen-questions": question_metrics,
            "test-unseen-domains": domain_metrics,
        },
        "base_nli_external_metrics": {
            "test-unseen-questions": baseline_question_metrics,
            "test-unseen-domains": baseline_domain_metrics,
        },
        "external_gate_clauses": clauses,
        "passes_external_gate": passes_gate,
        "delta_file": delta_path.name,
        "delta_bytes": delta_path.stat().st_size,
        "delta_sha256": _sha256(delta_path),
        "trainable_tensor_count": len(delta),
        "max_serialization_difference": max_difference,
        "V_final_accessed": False,
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"run_id": run_id, "run_dir": str(run_dir), "report": report}


def _load_adapted_model(model_path: Path, delta_path: Path):
    from safetensors.torch import load_file
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    )
    delta = load_file(str(delta_path), device="cpu")
    state = model.state_dict()
    unknown = sorted(set(delta).difference(state))
    if unknown:
        raise ValueError(f"Unknown E400 delta tensors: {unknown}")
    state.update(delta)
    model.load_state_dict(state, strict=True)
    return model


def build_competition_session_cache(
    project_root: str | Path,
    transfer_run_id: str,
    *,
    batch_size: int = EVAL_BATCH_SIZE,
    chunk_size: int = 256,
    candidate_name: str = "E400_sem_eval_adapted_deberta_session",
    experiment_label: str = "E400",
) -> dict[str, object]:
    import torch
    from transformers import AutoTokenizer

    paths = discover_project_paths(project_root)
    run_dir = paths.experiments_dir / "runs" / transfer_run_id
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"{experiment_label} report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not bool(report.get("passes_external_gate")):
        raise RuntimeError(
            f"{experiment_label} external gate failed; competition caching is forbidden."
        )
    delta_path = run_dir / str(report["delta_file"])
    if _sha256(delta_path) != report["delta_sha256"]:
        raise ValueError(
            f"{experiment_label} delta SHA-256 changed before competition caching."
        )

    frame = pd.read_parquet(
        paths.cache_dir / "modeling_base.parquet",
        columns=["response_id", "learning_objective"],
    )
    contexts = pd.read_parquet(
        paths.cache_dir / "response_objective_context.parquet",
        columns=["response_id", "objective_context"],
    )
    if list(frame["response_id"]) != list(contexts["response_id"]):
        raise ValueError(
            f"{experiment_label} competition context rows are misaligned."
        )
    premises = [
        compact_objective_context(text) for text in contexts["objective_context"]
    ]
    hypotheses = [mastery_hypothesis(value) for value in frame["learning_objective"]]

    torch.manual_seed(SEED)
    torch.set_num_threads(min(6, max(1, torch.get_num_threads())))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    model_path = paths.root / "assets" / "pretrained" / MODEL_DIRECTORY
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = _load_adapted_model(model_path, delta_path)
    model.eval()
    if int(model.config.hidden_size) != 768:
        raise ValueError(f"Unexpected {experiment_label} hidden dimension.")

    prefix = str(report["delta_sha256"])[:12]
    pooled_path = paths.cache_dir / f"sra_deberta_session_pooled_256_{prefix}.npy"
    logits_path = paths.cache_dir / f"sra_deberta_session_logits_256_{prefix}.npy"
    progress_path = paths.cache_dir / f"sra_deberta_session_256_{prefix}.progress.json"
    metadata_path = paths.cache_dir / f"sra_deberta_session_256_{prefix}.metadata.json"
    pooled_shape = (len(frame), 768)
    logits_shape = (len(frame), 3)
    if pooled_path.exists() or logits_path.exists():
        if not (pooled_path.exists() and logits_path.exists() and progress_path.exists()):
            raise RuntimeError(
                f"Incomplete {experiment_label} cache artifacts cannot be resumed safely."
            )
        pooled_values = np.lib.format.open_memmap(pooled_path, mode="r+")
        logits_values = np.lib.format.open_memmap(logits_path, mode="r+")
        if pooled_values.shape != pooled_shape or logits_values.shape != logits_shape:
            raise ValueError(f"Existing {experiment_label} cache shape changed.")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("delta_sha256") != report["delta_sha256"]:
            raise ValueError(
                f"{experiment_label} progress belongs to another checkpoint."
            )
        completed = int(progress.get("completed_rows", 0))
    else:
        pooled_values = np.lib.format.open_memmap(
            pooled_path, mode="w+", dtype=np.float32, shape=pooled_shape
        )
        logits_values = np.lib.format.open_memmap(
            logits_path, mode="w+", dtype=np.float32, shape=logits_shape
        )
        completed = 0

    for chunk_start in range(completed, len(frame), int(chunk_size)):
        chunk_end = min(len(frame), chunk_start + int(chunk_size))
        pooled_batches: list[np.ndarray] = []
        logits_batches: list[np.ndarray] = []
        for batch_start in range(chunk_start, chunk_end, int(batch_size)):
            batch_end = min(chunk_end, batch_start + int(batch_size))
            tokenized = tokenizer(
                premises[batch_start:batch_end],
                hypotheses[batch_start:batch_end],
                padding=True,
                truncation="only_first",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            with torch.inference_mode():
                outputs = model(
                    **tokenized,
                    output_hidden_states=True,
                    return_dict=True,
                )
                pooled = model.pooler(outputs.hidden_states[-1])
            pooled_batches.append(pooled.cpu().numpy().astype(np.float32))
            logits_batches.append(outputs.logits.cpu().numpy().astype(np.float32))
        pooled_values[chunk_start:chunk_end] = np.concatenate(pooled_batches)
        logits_values[chunk_start:chunk_end] = np.concatenate(logits_batches)
        pooled_values.flush()
        logits_values.flush()
        progress = {
            "transfer_run_id": transfer_run_id,
            "delta_sha256": report["delta_sha256"],
            "completed_rows": int(chunk_end),
            "total_rows": int(len(frame)),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        progress_path.write_text(
            json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"competition_cache_rows={chunk_end}/{len(frame)}", flush=True)

    pooled = np.asarray(np.load(pooled_path, mmap_mode="r"))
    logits = np.asarray(np.load(logits_path, mmap_mode="r"))
    if not np.isfinite(pooled).all() or not np.isfinite(logits).all():
        raise RuntimeError(
            f"{experiment_label} competition cache contains non-finite values."
        )
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": candidate_name,
        "transfer_run_id": transfer_run_id,
        "delta_sha256": report["delta_sha256"],
        "rows": int(len(frame)),
        "pooled_shape": list(pooled.shape),
        "logits_shape": list(logits.shape),
        "pooled_sha256": _sha256(pooled_path),
        "logits_sha256": _sha256(logits_path),
        "max_length": MAX_LENGTH,
        "batch_size": int(batch_size),
        "V_final_accessed": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "pooled_path": str(pooled_path),
        "logits_path": str(logits_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare, train, or cache the preregistered E400 SRA transfer model."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--stage", choices=("prepare", "train", "cache", "pipeline"))
    parser.add_argument("--transfer-run-id")
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.stage == "prepare":
        result = prepare_sem_eval_cache(args.project_root)
    elif args.stage == "train":
        result = train_external_sra_transfer(args.project_root)
    elif args.stage == "cache":
        if not args.transfer_run_id:
            raise ValueError("--transfer-run-id is required for the cache stage.")
        result = build_competition_session_cache(
            args.project_root, args.transfer_run_id, batch_size=args.batch_size
        )
    else:
        prepare_sem_eval_cache(args.project_root)
        trained = train_external_sra_transfer(args.project_root)
        if trained["report"]["passes_external_gate"]:
            cached = build_competition_session_cache(
                args.project_root, trained["run_id"], batch_size=args.batch_size
            )
            result = {"training": trained, "cache": cached}
        else:
            result = {"training": trained, "cache": None}
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
