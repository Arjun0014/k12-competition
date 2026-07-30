from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score, log_loss

from trace_ace.io import discover_project_paths
from trace_ace.source_robust_validation import _current_rss_bytes


PROTOCOL_ID = "P880_external_llm_tutor_move_v1"
LABELS = ("focus", "generic", "probing", "telling")
LABEL_TEXT = tuple(f" {label}" for label in LABELS)
MODEL_KEY_DEFAULT = "qwen3_0_6b"
MAX_LENGTH = 256
STUDENT_TOKEN_CAP = 48
TUTOR_TOKEN_CAP = 64
BATCH_SIZE = 16
CPU_THREADS = 6
RESOURCE_ROWS = 32
SCREEN_ROWS_PER_CLASS = 128
BOOTSTRAP_REPLICATES = 2_000
SEED = 20260730
MAX_EXTERNAL_HOURS = 2.0
MAX_LOCAL_CACHE_HOURS = 168.0
LOCAL_SESSION_COUNT = 22_821
WINDOWS_PER_SESSION = 16
MAX_RSS_BYTES = 8 * 1024**3
MIN_MACRO_F1 = 0.55
MIN_MACRO_F1_GAIN = 0.03
MIN_CLASS_F1 = 0.40
MIN_BOOTSTRAP_SUPPORT = 0.95
MIN_CONFIRM_ACCURACY = 0.57
MATHDIAL_SHA256 = (
    "7d34f5fce9c55fda7fb11109156e74763da8eaa25315c4e4e8eb9fbd02138b98"
)
E530_PREDICTIONS_SHA256 = (
    "caa34f7134db99cf750f53bc2991ef9bf23b80dbf093815639e014d02b3b06d5"
)

SYSTEM_PROMPT = "Classify the tutor move in a math tutoring exchange."
USER_TEMPLATE = """Labels:
focus: redirect or narrow the student toward the relevant step without giving the answer.
generic: greeting, acknowledgement, praise, or neutral management without substantive math guidance.
probing: ask the student to explain, reason, or produce a next step.
telling: directly provide a correction, method, explanation, or answer.

Previous student:
{student_marker}
Tutor:
{tutor_marker}
Choose exactly one label: focus, generic, probing, or telling."""
STUDENT_MARKER = "ZXQW_STUDENT_MOVE_MARKER_880"
TUTOR_MARKER = "ZXQW_TUTOR_MOVE_MARKER_880"
ASSISTANT_PREFIX = "The label is"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    directory: str
    revision: str
    model_sha256: str
    config_sha256: str
    tokenizer_sha256: str
    license_sha256: str


MODEL_SPECS: Mapping[str, ModelSpec] = {
    "qwen3_0_6b": ModelSpec(
        key="qwen3_0_6b",
        name="Qwen/Qwen3-0.6B",
        directory="Qwen3-0.6B",
        revision="c1899de289a04d12100db370d81485cdf75e47ca",
        model_sha256=(
            "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
        ),
        config_sha256=(
            "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
        ),
        tokenizer_sha256=(
            "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
        ),
        license_sha256=(
            "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"
        ),
    ),
    "qwen2_5_1_5b": ModelSpec(
        key="qwen2_5_1_5b",
        name="Qwen/Qwen2.5-1.5B-Instruct",
        directory="Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        model_sha256=(
            "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
        ),
        config_sha256=(
            "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670"
        ),
        tokenizer_sha256=(
            "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
        ),
        license_sha256=(
            "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"
        ),
    ),
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _paths(project_root: str | Path, model_key: str) -> dict[str, Path]:
    root = discover_project_paths(project_root).root
    spec = MODEL_SPECS[model_key]
    model = root / "assets" / "pretrained" / spec.directory
    e530 = (
        root
        / "experiments"
        / "runs"
        / "20260724T164257Z_mathdial_tutor_move_transfer"
    )
    return {
        "model": model / "model.safetensors",
        "config": model / "config.json",
        "tokenizer": model / "tokenizer.json",
        "license": model / "LICENSE",
        "mathdial": root / "data_cache" / "mathdial_tutor_moves_e530.parquet",
        "e530_predictions": e530 / "test_predictions.parquet",
        "benchmark": (
            root
            / "data_cache"
            / f"llm_tutor_move_{PROTOCOL_ID}_{model_key}_benchmark.json"
        ),
    }


def verify_sources(project_root: str | Path, model_key: str) -> dict[str, str]:
    spec = MODEL_SPECS[model_key]
    paths = _paths(project_root, model_key)
    expected = {
        "model": spec.model_sha256,
        "config": spec.config_sha256,
        "tokenizer": spec.tokenizer_sha256,
        "license": spec.license_sha256,
        "mathdial": MATHDIAL_SHA256,
    }
    expected["e530_predictions"] = E530_PREDICTIONS_SHA256
    observed = {key: _sha256(paths[key]) for key in expected}
    failures = [
        f"{key}: expected {expected[key]}, observed {observed[key]}"
        for key in expected
        if observed[key] != expected[key]
    ]
    if failures:
        raise ValueError("P880 source drift: " + "; ".join(failures))
    return observed


def _runtime_contract() -> dict[str, object]:
    if sys.version_info[:3] != (3, 12, 8) or sklearn.__version__ != "1.8.0":
        raise RuntimeError(
            "P880 requires project .venv Python 3.12.8 and scikit-learn 1.8.0."
        )
    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "executable": sys.executable,
        "platform": platform.platform(),
    }


def _load_model(project_root: str | Path, model_key: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = MODEL_SPECS[model_key]
    model_dir = (
        discover_project_paths(project_root).root
        / "assets"
        / "pretrained"
        / spec.directory
    )
    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.float32,
    )
    model.eval()
    observed = tuple(
        tokenizer.encode(value, add_special_tokens=False) for value in LABEL_TEXT
    )
    if any(len(value) != 1 for value in observed):
        raise ValueError(f"P880 label tokenization drifted: {observed}")
    return tokenizer, model, tuple(value[0] for value in observed)


def _normalized(value: object) -> str:
    return " ".join(str(value).split())


def prompt_token_ids(tokenizer, previous_student: object, teacher_text: object) -> list[int]:
    user = USER_TEMPLATE.format(
        student_marker=STUDENT_MARKER,
        tutor_marker=TUTOR_MARKER,
    )
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    rendered += ASSISTANT_PREFIX
    if rendered.count(STUDENT_MARKER) != 1 or rendered.count(TUTOR_MARKER) != 1:
        raise ValueError("P880 chat template did not preserve input markers.")
    left, remainder = rendered.split(STUDENT_MARKER)
    middle, right = remainder.split(TUTOR_MARKER)
    student_ids = tokenizer.encode(_normalized(previous_student), add_special_tokens=False)[
        -STUDENT_TOKEN_CAP:
    ]
    tutor_ids = tokenizer.encode(_normalized(teacher_text), add_special_tokens=False)[
        :TUTOR_TOKEN_CAP
    ]
    result = [
        *tokenizer.encode(left, add_special_tokens=False),
        *student_ids,
        *tokenizer.encode(middle, add_special_tokens=False),
        *tutor_ids,
        *tokenizer.encode(right, add_special_tokens=False),
    ]
    if len(result) > MAX_LENGTH:
        raise RuntimeError(f"P880 prompt exceeded {MAX_LENGTH} tokens: {len(result)}")
    return result


def _score_rows(
    tokenizer,
    model,
    label_token_ids: Sequence[int],
    frame: pd.DataFrame,
    *,
    progress: bool = False,
) -> np.ndarray:
    import torch

    required = {"previous_student", "teacher_text"}
    if not required.issubset(frame.columns):
        raise ValueError(f"P880 input is missing {sorted(required - set(frame.columns))}")
    output = np.empty((len(frame), len(LABELS)), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(frame), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(frame))
            rows = frame.iloc[start:stop]
            sequences = [
                prompt_token_ids(tokenizer, student, teacher)
                for student, teacher in zip(
                    rows["previous_student"],
                    rows["teacher_text"],
                    strict=True,
                )
            ]
            width = max(map(len, sequences))
            pad_id = int(tokenizer.pad_token_id)
            input_ids = np.full((len(sequences), width), pad_id, dtype=np.int64)
            attention = np.zeros((len(sequences), width), dtype=np.int64)
            for row_index, sequence in enumerate(sequences):
                input_ids[row_index, -len(sequence) :] = sequence
                attention[row_index, -len(sequence) :] = 1
            logits = model(
                input_ids=torch.from_numpy(input_ids),
                attention_mask=torch.from_numpy(attention),
                use_cache=False,
            ).logits[:, -1, list(label_token_ids)]
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            output[start:stop] = probabilities
            if progress:
                print(f"P880 scored {stop}/{len(frame)} rows", flush=True)
    if (
        output.shape != (len(frame), len(LABELS))
        or not np.isfinite(output).all()
        or not np.allclose(output.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("P880 probabilities failed audit.")
    return output


def _stable_order(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(
        lambda row: hashlib.sha256(
            f"{row.qid}|{row.source_row}|{row.turn_index}".encode()
        ).hexdigest(),
        axis=1,
    )


def select_screen_rows(frame: pd.DataFrame) -> pd.DataFrame:
    test = frame.loc[frame["split"].eq("test")].copy()
    if set(test["move"]) != set(LABELS):
        raise ValueError("P880 held-out label set changed.")
    test["_stable_order"] = _stable_order(test)
    selected = (
        test.sort_values(["move", "_stable_order"], kind="mergesort")
        .groupby("move", sort=True, group_keys=False)
        .head(SCREEN_ROWS_PER_CLASS)
        .sort_values("_stable_order", kind="mergesort")
        .drop(columns="_stable_order")
        .reset_index(drop=True)
    )
    expected = SCREEN_ROWS_PER_CLASS * len(LABELS)
    counts = selected["move"].value_counts().to_dict()
    if len(selected) != expected or set(counts.values()) != {SCREEN_ROWS_PER_CLASS}:
        raise ValueError(f"P880 screen selection changed: {counts}")
    return selected


def _metric_summary(target: Sequence[str], probabilities: np.ndarray) -> dict[str, object]:
    truth = np.asarray(target, dtype=object)
    prediction = np.asarray(LABELS, dtype=object)[probabilities.argmax(axis=1)]
    class_f1 = f1_score(
        truth,
        prediction,
        labels=list(LABELS),
        average=None,
        zero_division=0,
    )
    return {
        "rows": int(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(
            f1_score(
                truth,
                prediction,
                labels=list(LABELS),
                average="macro",
                zero_division=0,
            )
        ),
        "log_loss": float(log_loss(truth, probabilities, labels=list(LABELS))),
        "per_class_f1": {
            label: float(value) for label, value in zip(LABELS, class_f1, strict=True)
        },
        "prediction_counts": {
            label: int((prediction == label).sum()) for label in LABELS
        },
    }


def _paired_qid_bootstrap(
    frame: pd.DataFrame,
    candidate: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, object]:
    qids = np.asarray(sorted(frame["qid"].astype(str).unique()))
    positions = {
        qid: np.flatnonzero(frame["qid"].astype(str).to_numpy() == qid)
        for qid in qids
    }
    truth = frame["move"].astype(str).to_numpy()
    candidate_label = np.asarray(LABELS)[candidate.argmax(axis=1)]
    baseline_label = np.asarray(LABELS)[baseline.argmax(axis=1)]
    generator = np.random.default_rng(SEED)
    gains = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = generator.choice(qids, size=len(qids), replace=True)
        indices = np.concatenate([positions[qid] for qid in sampled])
        candidate_f1 = f1_score(
            truth[indices],
            candidate_label[indices],
            labels=list(LABELS),
            average="macro",
            zero_division=0,
        )
        baseline_f1 = f1_score(
            truth[indices],
            baseline_label[indices],
            labels=list(LABELS),
            average="macro",
            zero_division=0,
        )
        gains[replicate] = candidate_f1 - baseline_f1
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": SEED,
        "qid_groups": int(len(qids)),
        "mean_macro_f1_gain": float(gains.mean()),
        "interval_95": np.quantile(gains, [0.025, 0.975]).tolist(),
        "positive_gain_support": float(np.mean(gains > 0)),
    }


def _baseline_probabilities(project_root: str | Path, frame: pd.DataFrame) -> np.ndarray:
    path = _paths(project_root, MODEL_KEY_DEFAULT)["e530_predictions"]
    if _sha256(path) != E530_PREDICTIONS_SHA256:
        raise ValueError("P880 E530 prediction source drifted.")
    baseline = pd.read_parquet(path)
    keys = ["qid", "source_row", "turn_index"]
    columns = [f"prob_{label}" for label in LABELS]
    merged = frame[keys].merge(
        baseline[keys + ["move", *columns]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(frame) or merged[columns].isna().any().any():
        raise ValueError("P880 E530 alignment failed.")
    if list(merged["move"].astype(str)) != list(frame["move"].astype(str)):
        raise ValueError("P880 E530 labels do not align.")
    return merged[columns].to_numpy(dtype=np.float64)


def benchmark(project_root: str | Path, model_key: str) -> dict[str, object]:
    runtime = _runtime_contract()
    sources = verify_sources(project_root, model_key)
    paths = _paths(project_root, model_key)
    if paths["benchmark"].exists():
        result = json.loads(paths["benchmark"].read_text(encoding="utf-8"))
        if result.get("source_hashes") != sources:
            raise ValueError("P880 cached benchmark source binding changed.")
        return result
    columns = [
        "split",
        "qid",
        "source_row",
        "turn_index",
        "previous_student",
        "teacher_text",
    ]
    frame = pd.read_parquet(paths["mathdial"], columns=columns)
    frame = frame.loc[frame["split"].eq("test")].drop(columns="split")
    frame["_stable_order"] = _stable_order(frame)
    frame = (
        frame.sort_values("_stable_order", kind="mergesort")
        .head(RESOURCE_ROWS)
        .drop(columns="_stable_order")
        .reset_index(drop=True)
    )
    tokenizer, model, label_ids = _load_model(project_root, model_key)
    started = time.perf_counter()
    _score_rows(tokenizer, model, label_ids, frame, progress=True)
    elapsed = time.perf_counter() - started
    seconds_per_row = elapsed / len(frame)
    external_hours = seconds_per_row * 3_664 / 3600
    local_cache_hours = (
        seconds_per_row * LOCAL_SESSION_COUNT * WINDOWS_PER_SESSION / 3600
    )
    peak_rss = _current_rss_bytes()
    gates = {
        "external_hours_at_most_2": external_hours <= MAX_EXTERNAL_HOURS,
        "local_cache_hours_at_most_168": (
            local_cache_hours <= MAX_LOCAL_CACHE_HOURS
        ),
        "rss_below_8_gib": peak_rss < MAX_RSS_BYTES,
    }
    result = {
        "protocol_id": PROTOCOL_ID,
        "stage": "target_free_resource_benchmark",
        "model": MODEL_SPECS[model_key].__dict__,
        "prompt_sha256": _canonical_sha256(
            {
                "system": SYSTEM_PROMPT,
                "user": USER_TEMPLATE,
                "assistant_prefix": ASSISTANT_PREFIX,
                "labels": LABELS,
                "max_length": MAX_LENGTH,
                "student_token_cap": STUDENT_TOKEN_CAP,
                "tutor_token_cap": TUTOR_TOKEN_CAP,
            }
        ),
        "rows": int(len(frame)),
        "elapsed_seconds": elapsed,
        "seconds_per_row": seconds_per_row,
        "projected_external_hours": external_hours,
        "projected_local_cache_hours": local_cache_hours,
        "peak_rss_bytes": peak_rss,
        "gates": gates,
        "proceed": bool(all(gates.values())),
        "runtime": runtime,
        "source_hashes": sources,
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    paths["benchmark"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def screen(project_root: str | Path, model_key: str) -> dict[str, object]:
    started = time.perf_counter()
    runtime = _runtime_contract()
    sources = verify_sources(project_root, model_key)
    resource = benchmark(project_root, model_key)
    if not resource["proceed"]:
        raise RuntimeError("P880 resource gate did not authorize the external screen.")
    frame = pd.read_parquet(_paths(project_root, model_key)["mathdial"])
    selected = select_screen_rows(frame)
    tokenizer, model, label_ids = _load_model(project_root, model_key)
    probabilities = _score_rows(
        tokenizer,
        model,
        label_ids,
        selected,
        progress=True,
    )
    baseline = _baseline_probabilities(project_root, selected)
    candidate_metrics = _metric_summary(selected["move"], probabilities)
    baseline_metrics = _metric_summary(selected["move"], baseline)
    bootstrap = _paired_qid_bootstrap(selected, probabilities, baseline)
    macro_gain = (
        float(candidate_metrics["macro_f1"]) - float(baseline_metrics["macro_f1"])
    )
    per_class = candidate_metrics["per_class_f1"]
    assert isinstance(per_class, dict)
    gates = {
        "macro_f1_at_least_0_55": (
            float(candidate_metrics["macro_f1"]) >= MIN_MACRO_F1
        ),
        "macro_f1_gain_at_least_0_03": macro_gain >= MIN_MACRO_F1_GAIN,
        "all_class_f1_at_least_0_40": min(per_class.values()) >= MIN_CLASS_F1,
        "bootstrap_support_at_least_0_95": (
            float(bootstrap["positive_gain_support"]) >= MIN_BOOTSTRAP_SUPPORT
        ),
    }
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime(
        f"%Y%m%dT%H%M%SZ_llm_tutor_move_{model_key}_external_screen"
    )
    run_dir = discover_project_paths(project_root).experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    prediction_frame = selected[
        ["dialogue_id", "qid", "source_row", "turn_index", "move"]
    ].copy()
    for index, label in enumerate(LABELS):
        prediction_frame[f"prob_{label}"] = probabilities[:, index]
        prediction_frame[f"baseline_prob_{label}"] = baseline[:, index]
    prediction_frame.to_parquet(
        run_dir / "predictions.parquet", index=False, compression="zstd"
    )
    (run_dir / "qid_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "stage": "held_out_external_screen",
        "model": MODEL_SPECS[model_key].__dict__,
        "rows": int(len(selected)),
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "macro_f1_gain_vs_e530": macro_gain,
        "paired_qid_bootstrap": bootstrap,
        "gates": gates,
        "decision": "pass_to_full_external_confirmation"
        if all(gates.values())
        else "reject_exact_model_prompt",
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": _current_rss_bytes(),
        "runtime": runtime,
        "source_hashes": sources,
        "resource_benchmark_sha256": _sha256(
            _paths(project_root, model_key)["benchmark"]
        ),
        "artifacts": {
            "predictions_sha256": _sha256(run_dir / "predictions.parquet"),
            "bootstrap_sha256": _sha256(run_dir / "qid_bootstrap.json"),
        },
        "competition_outcomes_accessed": False,
        "V_seen_accessed": False,
        "V_objective_accessed": False,
        "V_style_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen external LLM tutor-move probe."
    )
    parser.add_argument("command", choices=("benchmark", "screen"))
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), default=MODEL_KEY_DEFAULT)
    args = parser.parse_args(argv)
    if args.command == "benchmark":
        result = benchmark(args.project_root, args.model)
    else:
        result = screen(args.project_root, args.model)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
