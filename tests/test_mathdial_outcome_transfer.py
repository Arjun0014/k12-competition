from trace_ace.mathdial_outcome_transfer import LABEL_MAP, parse_mathdial


def test_mathdial_label_contract() -> None:
    assert LABEL_MAP == {
        "Yes": 1,
        "Yes, but I had to reveal the answer": 0,
        "No": 0,
    }


def test_canonical_mathdial_rows() -> None:
    frame = parse_mathdial(".")
    assert frame["split"].value_counts().to_dict() == {"train": 2253, "test": 595}
    assert frame["example_id"].is_unique
    assert set(frame["outcome_label"]) == {0, 1}
    train_qids = set(frame.loc[frame["external_train_eligible"], "qid"])
    test_qids = set(frame.loc[frame["external_test_eligible"], "qid"])
    assert not train_qids.intersection(test_qids)
