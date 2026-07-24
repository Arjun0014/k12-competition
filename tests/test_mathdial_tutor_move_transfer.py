import numpy as np

from trace_ace.mathdial_tutor_move_transfer import (
    CLASS_ORDER,
    EXPECTED_CLASS_COUNTS,
    EXPECTED_DIALOGUES,
    EXPECTED_MOVE_ROWS,
    MOVE_TAXONOMY,
    _multiclass_metrics,
    _qid_bootstrap,
    build_classifier,
    build_vectorizers,
    external_gate,
    parse_mathdial_tutor_moves,
)


def test_mathdial_tutor_move_source_and_split_contract() -> None:
    frame, audit = parse_mathdial_tutor_moves(".")
    assert tuple(audit["taxonomy"]) == MOVE_TAXONOMY
    assert tuple(audit["class_order"]) == CLASS_ORDER
    assert audit["qid_overlap_count"] == 314
    for split in ("train", "test"):
        split_frame = frame.loc[frame["split"].eq(split)]
        assert audit["splits"][split]["eligible_dialogues"] == EXPECTED_DIALOGUES[split]
        assert len(split_frame) == EXPECTED_MOVE_ROWS[split]
        assert audit["splits"][split]["class_counts"] == EXPECTED_CLASS_COUNTS[split]
        assert split_frame["model_text"].str.startswith("[STUDENT] ").all()
        assert split_frame["model_text"].str.contains("\n[TUTOR] ", regex=False).all()
        for move in MOVE_TAXONOMY:
            assert not split_frame["teacher_text"].str.casefold().str.startswith(
                f"({move})"
            ).any()
    train_qids = set(frame.loc[frame["split"].eq("train"), "qid"])
    test_qids = set(frame.loc[frame["split"].eq("test"), "qid"])
    assert not train_qids.intersection(test_qids)


def test_e530_vectorizer_contract() -> None:
    word, char = build_vectorizers()
    assert word.analyzer == "word"
    assert word.ngram_range == (1, 2)
    assert word.max_features == 50_000
    assert char.analyzer == "char_wb"
    assert char.ngram_range == (3, 5)
    assert char.max_features == 75_000
    for vectorizer in (word, char):
        assert vectorizer.min_df == 2
        assert vectorizer.lowercase is True
        assert vectorizer.strip_accents == "unicode"
        assert vectorizer.sublinear_tf is True
        assert vectorizer.norm == "l2"


def test_e530_classifier_contract() -> None:
    classifier = build_classifier()
    assert classifier.estimator.C == 1.0
    assert classifier.estimator.solver == "liblinear"
    assert classifier.estimator.max_iter == 1_000
    assert classifier.estimator.random_state == 20260724
    assert classifier.estimator.class_weight is None


def test_e530_metrics_bootstrap_and_gate_are_deterministic() -> None:
    labels = np.asarray(
        ["focus", "generic", "probing", "telling"] * 4, dtype=object
    )
    probability = np.full((len(labels), len(CLASS_ORDER)), 0.01, dtype=np.float64)
    class_to_index = {label: index for index, label in enumerate(CLASS_ORDER)}
    for row, label in enumerate(labels):
        probability[row, class_to_index[str(label)]] = 0.97
    probability /= probability.sum(axis=1, keepdims=True)
    prior = np.full_like(probability, 0.25)
    metrics = _multiclass_metrics(labels, probability, CLASS_ORDER)
    prior_metrics = _multiclass_metrics(labels, prior, CLASS_ORDER)
    qids = np.asarray([f"q{index // 2}" for index in range(len(labels))])
    first = _qid_bootstrap(
        qids, labels, probability, prior, CLASS_ORDER, n_replicates=5_000, seed=19
    )
    second = _qid_bootstrap(
        qids, labels, probability, prior, CLASS_ORDER, n_replicates=5_000, seed=19
    )
    assert first == second
    assert first["support_positive_log_loss_gain"] == 1.0
    gate = external_gate(metrics, prior_metrics, first)
    assert gate["passes_external_gate"] is True
    assert all(gate["clauses"].values())
