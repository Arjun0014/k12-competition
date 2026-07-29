import numpy as np
import pandas as pd

from trace_ace.mathdial_contrastive_alignment import (
    EXPECTED_COUNTS,
    QUERY_PREFIX,
    epoch_training_rows,
    load_mathdial_alignment_source,
    objective_query,
    qid_bootstrap,
    retrieval_rows,
    summarize_retrieval,
)


def test_e820_source_is_pinned_and_question_disjoint() -> None:
    train, test, audit = load_mathdial_alignment_source(".")
    assert len(train) == EXPECTED_COUNTS["legal_train_rows"]
    assert len(test) == EXPECTED_COUNTS["test_rows"]
    assert train["qid_hash"].nunique() == EXPECTED_COUNTS["legal_train_qids"]
    assert test["qid_hash"].nunique() == EXPECTED_COUNTS["test_qids"]
    assert set(train["qid_hash"]).isdisjoint(set(test["qid_hash"]))
    assert audit["prohibited_fields_consumed"] is False
    assert audit["outcome_fields_consumed"] == []


def test_e820_epoch_sampler_has_one_row_per_qid() -> None:
    train, _, _ = load_mathdial_alignment_source(".")
    first = epoch_training_rows(train, epoch=0)
    second = epoch_training_rows(train, epoch=1)
    assert len(first) == EXPECTED_COUNTS["legal_train_qids"]
    assert not first["qid_hash"].duplicated().any()
    assert not second["qid_hash"].duplicated().any()
    assert set(first["qid_hash"]) == set(second["qid_hash"])
    assert list(first["row_hash"]) != list(second["row_hash"])


def test_e820_query_contract() -> None:
    value = objective_query(" Adding fractions. ")
    assert value == f"{QUERY_PREFIX}Adding fractions."


def test_e820_retrieval_metrics_and_group_bootstrap() -> None:
    documents = np.eye(3, dtype=np.float32)
    queries = np.asarray(
        [
            [0.1, 0.9, 0.0],
            [0.0, 0.1, 0.9],
            [0.9, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    qids = ["a", "b", "c"]
    base = retrieval_rows(queries, documents, qids, qids)
    adapted = retrieval_rows(documents, documents, qids, qids)
    assert summarize_retrieval(base)["mrr"] < 1.0
    assert summarize_retrieval(adapted)["mrr"] == 1.0
    bootstrap = qid_bootstrap(base, adapted, replicates=200, seed=7)
    assert bootstrap["qids"] == 3
    assert bootstrap["positive_gain_support"] == 1.0


def test_e820_qid_bootstrap_averages_repeated_dialogues() -> None:
    base = pd.DataFrame(
        {
            "qid_hash": ["a", "a", "b"],
            "reciprocal_rank": [0.5, 0.25, 0.5],
        }
    )
    adapted = pd.DataFrame(
        {
            "qid_hash": ["a", "a", "b"],
            "reciprocal_rank": [1.0, 0.5, 1.0],
        }
    )
    result = qid_bootstrap(base, adapted, replicates=100, seed=11)
    assert result["qids"] == 2
    assert result["observed_gain"] > 0.0
