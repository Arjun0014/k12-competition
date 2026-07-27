from __future__ import annotations

import pandas as pd

from trace_ace.multicorpus_correctness_transfer import (
    GSM_TRAIN_QUESTIONS,
    SELECTED_QUESTION_SHA256,
    _question_hash_order,
    build_gsm_rows,
    corrupt_final_integer,
    final_integer,
    prepare_multicorpus_cache,
)


def test_final_integer_and_corruption() -> None:
    assert final_integer("work\n#### 1,234") == 1234
    assert corrupt_final_integer("work\n#### 5") == ("work\n#### 6", 5, 6)
    assert corrupt_final_integer("work\n#### -3") == ("work\n#### -4", -3, -4)


def test_gsm_rows_are_paired_and_change_only_final_integer() -> None:
    frame = pd.DataFrame(
        {
            "question": ["q1", "q2"],
            "answer": ["reason\n#### 4", "other\n#### -2"],
        }
    )
    rows = build_gsm_rows(frame, split="unit", selected_rows=None)
    assert len(rows) == 4
    assert rows.groupby("source_row")["nli_label"].agg(list).tolist() == [
        [1, 0],
        [1, 0],
    ]
    assert set(rows["variant"]) == {"correct", "corrupted"}


def test_real_gsm_selection_contract() -> None:
    frame = pd.read_parquet(
        "Datasets/Grade School Math (GSM8k)/main/train.parquet"
    )
    order = _question_hash_order(frame)[:GSM_TRAIN_QUESTIONS]
    import hashlib

    observed = hashlib.sha256(
        "\n".join(frame.iloc[order]["question"].astype(str)).encode()
    ).hexdigest()
    assert observed == SELECTED_QUESTION_SHA256
    assert all(final_integer(value) is not None for value in frame["answer"])


def test_real_multicorpus_cache_contract() -> None:
    result = prepare_multicorpus_cache(".")
    metadata = result["metadata"]
    assert metadata["training_rows"] == 17_820
    assert metadata["training_rows_by_corpus"] == {
        "SemEval": 8_910,
        "GSM8K": 8_910,
    }
    assert metadata["gsm_test_rows"] == 2_638
