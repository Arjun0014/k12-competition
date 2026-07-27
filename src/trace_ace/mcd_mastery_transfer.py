from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trace_ace.io import discover_project_paths


PROTOCOL_ID = "E610_mcd_crosslingual_mastery_transfer_v1"
SEED = 20260727
MODEL_DIRECTORY = "mdeberta-v3-base"
MAX_LENGTH = 384
MAX_TURNS = 24
MAX_TURN_CHARACTERS = 24
N_LABELS = 3
TRAIN_ROWS = 4_242
TEST_ROWS = 984
TRAIN_GROUPS = 399
TEST_GROUPS = 95
SOURCE_HASHES = {
    "labels": "153d77a2324cdcb73d7c41f87fc216821cb9cc9a5284a02459cb90362c5d1cf3",
    "transcripts": "103c9f9cc878dc491e67668f7e9e574b066fa72504efcf0a6ab1fff8a73c4157",
    "published_split": "34e5f6c610ed65b4be0f5ea3d8e714995dc2274914bcdb80e2252d4a8f82e37d",
    "model": "6f89419baf0f1aaad5cab7d53901e36a8c1af8f6b4ab58b15db9af32df656ead",
    "config": "bcffcd343dc5efa5ef2d5a58d2b405eed108f01cc45b48d0a907b333ec41801f",
    "spm": "13c8d666d62a7bc4ac8f040aab68e942c861f93303156cc28f5c7e885d86d6e3",
    "tokenizer_config": "3f3978e0c036f2c2588cac34a6047cbb0af0b0dc1814254e291028529805496d",
}
EXPECTED_RUNTIME = {
    "python": "3.12.8",
    "numpy": "2.5.1",
    "scikit_learn": "1.8.0",
    "sentencepiece": "0.2.1",
    "protobuf": "6.33.5",
    "torch": "2.13.0+cpu",
    "transformers": "5.14.1",
}
TASK_SUFFIX = (
    "Task: classify demonstrated math mastery as Apprentice, Understanding, "
    "or Mastery."
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_frame_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame[
        ["example_id", "source_group", "split", "text", "label"]
    ].itertuples(index=False, name=None):
        digest.update(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _paths(project_root: str | Path) -> dict[str, Path]:
    paths = discover_project_paths(project_root)
    source = paths.root / "Datasets" / "MCD" / "data" / "dataset"
    model = paths.root / "assets" / "pretrained" / MODEL_DIRECTORY
    return {
        "labels": source / "df_feature_num_label-3.csv",
        "transcripts": source / "item_dict_anonymized.json",
        "published_split": source / "train_dev_test.json",
        "model": model / "pytorch_model.bin",
        "config": model / "config.json",
        "spm": model / "spm.model",
        "tokenizer_config": model / "tokenizer_config.json",
        "model_dir": model,
        "cache": paths.cache_dir / "mcd_mastery_e610.parquet",
        "metadata": paths.cache_dir / "mcd_mastery_e610.metadata.json",
        "benchmark": paths.cache_dir / "mcd_mastery_e610_benchmark.json",
    }


def verify_sources(project_root: str | Path) -> dict[str, str]:
    paths = _paths(project_root)
    for name, expected in SOURCE_HASHES.items():
        path = paths[name]
        if not path.exists():
            raise FileNotFoundError(f"Missing E610 {name}: {path}")
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"E610 {name} SHA-256 changed: {observed}")
    return dict(SOURCE_HASHES)


def assert_runtime() -> dict[str, str]:
    import google.protobuf
    import sentencepiece
    import sklearn
    import torch
    import transformers

    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "sentencepiece": sentencepiece.__version__,
        "protobuf": google.protobuf.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    if observed != EXPECTED_RUNTIME:
        raise RuntimeError(f"E610 runtime differs from frozen .venv: {observed}")
    return observed


def source_group(example_id: object) -> str:
    value = str(example_id)
    if "_" not in value:
        raise ValueError(f"E610 example ID lacks source-group separator: {value}")
    group, suffix = value.split("_", 1)
    if not group or not suffix:
        raise ValueError(f"E610 example ID is malformed: {value}")
    return group


def split_for_group(group: str) -> str:
    digest = hashlib.sha256(f"E610|{group}".encode("utf-8")).hexdigest()
    return "test" if int(digest[:16], 16) % 5 == 0 else "train"


def compact_dialogue(turns: object) -> str:
    if not isinstance(turns, list):
        raise ValueError("E610 transcript is not a turn list.")
    lines: list[str] = []
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError("E610 transcript turn is not a mapping.")
        text = " ".join(str(turn.get("text", "")).split())
        if not text:
            continue
        role = str(turn.get("who", "")).strip().lower()
        if role == "teacher":
            prefix = "Tutor"
        elif role == "student":
            prefix = "Student"
        else:
            raise ValueError(f"E610 received unknown speaker: {role!r}")
        lines.append(f"{prefix}: {text[:MAX_TURN_CHARACTERS]}")
    if not lines:
        raise ValueError("E610 transcript has no non-empty turns.")
    return "\n".join([*lines[-MAX_TURNS:], TASK_SUFFIX])


def build_canonical_cache(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    source_hashes = verify_sources(project_root)
    paths = _paths(project_root)
    labels = pd.read_csv(paths["labels"])
    if (
        list(labels.columns)[0] != "new_id"
        or "label" not in labels
        or len(labels) != TRAIN_ROWS + TEST_ROWS
        or labels["new_id"].astype(str).duplicated().any()
        or set(labels["label"].tolist()) != {0, 1, 2}
    ):
        raise ValueError("E610 label table violates the frozen contract.")
    transcripts = json.loads(paths["transcripts"].read_text(encoding="utf-8"))
    transcript_ids = set(map(str, transcripts))
    label_ids = set(labels["new_id"].astype(str))
    if transcript_ids != label_ids:
        raise ValueError("E610 transcript and label IDs do not align.")

    records: list[dict[str, object]] = []
    for row in labels.itertuples(index=False):
        example_id = str(row.new_id)
        group = source_group(example_id)
        records.append(
            {
                "example_id": example_id,
                "source_group": group,
                "split": split_for_group(group),
                "text": compact_dialogue(transcripts[example_id]),
                "label": int(row.label),
            }
        )
    frame = pd.DataFrame(records).sort_values(
        ["split", "source_group", "example_id"], kind="mergesort"
    ).reset_index(drop=True)
    train = frame.loc[frame["split"].eq("train")]
    test = frame.loc[frame["split"].eq("test")]
    if (
        len(train) != TRAIN_ROWS
        or len(test) != TEST_ROWS
        or train["source_group"].nunique() != TRAIN_GROUPS
        or test["source_group"].nunique() != TEST_GROUPS
        or set(train["source_group"]) & set(test["source_group"])
        or frame["text"].duplicated().any()
    ):
        raise ValueError("E610 canonical split or uniqueness audit failed.")
    ordered_hash = _ordered_frame_sha256(frame)
    frame.to_parquet(paths["cache"], index=False)
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(frame),
        "split_rows": frame["split"].value_counts().sort_index().to_dict(),
        "split_groups": frame.groupby("split")["source_group"].nunique().to_dict(),
        "split_labels": {
            split: {
                str(label): int(count)
                for label, count in subset["label"]
                .value_counts()
                .sort_index()
                .items()
            }
            for split, subset in frame.groupby("split", sort=True)
        },
        "ordered_content_sha256": ordered_hash,
        "cache_sha256": _sha256(paths["cache"]),
        "source_sha256": source_hashes,
        "runtime": runtime,
        "configuration": {
            "group_rule": (
                'test iff int(SHA256("E610|" + source_group)[:16],16) mod 5 == 0'
            ),
            "label_map": {
                "0": "Apprentice",
                "1": "Understanding",
                "2": "Mastery",
            },
            "max_turns": MAX_TURNS,
            "max_turn_characters": MAX_TURN_CHARACTERS,
            "task_suffix": TASK_SUFFIX,
            "max_tokens": MAX_LENGTH,
            "truncation_side": "left",
            "fix_mistral_regex": True,
        },
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load_tokenizer(project_root: str | Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        _paths(project_root)["model_dir"],
        fix_mistral_regex=True,
    )
    tokenizer.truncation_side = "left"
    return tokenizer


def initialize_model(project_root: str | Path):
    import torch
    from transformers import AutoModelForSequenceClassification

    torch.manual_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        _paths(project_root)["model_dir"],
        num_labels=N_LABELS,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    for layer in model.deberta.encoder.layer[10:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for module in (model.pooler, model.classifier):
        for parameter in module.parameters():
            parameter.requires_grad = True
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    if (
        not trainable
        or any(name.startswith("deberta.encoder.layer.0.") for name in trainable)
        or not any(name.startswith("deberta.encoder.layer.10.") for name in trainable)
        or not any(name.startswith("deberta.encoder.layer.11.") for name in trainable)
    ):
        raise ValueError("E610 trainable-layer contract failed.")
    return model


def trainable_state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the frozen E610 MCD mastery-transfer branch."
    )
    parser.add_argument("stage", choices=("prepare", "audit-model"))
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.stage == "prepare":
        result = build_canonical_cache(args.project_root)
    else:
        model = initialize_model(args.project_root)
        tokenizer = load_tokenizer(args.project_root)
        result = {
            "runtime": assert_runtime(),
            "tokenizer": type(tokenizer).__name__,
            "model": type(model).__name__,
            "parameters": int(model.num_parameters()),
            "trainable_parameters": int(
                sum(value.numel() for value in model.parameters() if value.requires_grad)
            ),
            "trainable_initialization_sha256": trainable_state_sha256(model),
            "V_joint_accessed": False,
            "V_final_accessed": False,
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
