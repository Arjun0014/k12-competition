from pathlib import Path

from trace_ace.external_sra_transfer import (
    canonical_reference_answers,
    normalize_sra_label,
    parse_sem_eval_2013,
)


def test_label_normalization() -> None:
    assert normalize_sra_label("non-domain") == "non_domain"
    assert normalize_sra_label("Partially Correct Incomplete") == (
        "partially_correct_incomplete"
    )


def test_parser_prefers_best_reference_and_maps_labels(tmp_path: Path) -> None:
    core = tmp_path / "semeval-5way" / "beetle" / "train" / "Core"
    core.mkdir(parents=True)
    xml = """<?xml version="1.0"?>
<question id="q1" module="m1">
  <questionText>Why?</questionText>
  <referenceAnswers>
    <referenceAnswer category="BEST">Because it is correct.</referenceAnswer>
    <referenceAnswer category="MINIMAL">Because.</referenceAnswer>
  </referenceAnswers>
  <studentAnswers>
    <studentAnswer id="a1" accuracy="correct">It is correct.</studentAnswer>
    <studentAnswer id="a2" accuracy="partially_correct_incomplete">Because.</studentAnswer>
  </studentAnswers>
</question>
"""
    (core / "q1.xml").write_text(xml, encoding="utf-8")
    other = tmp_path / "semeval-5way" / "sciEntsBank"
    other.mkdir(parents=True)

    frame = parse_sem_eval_2013(tmp_path)
    assert len(frame) == 2
    assert set(frame["nli_label"]) == {1, 2}
    assert set(frame["reference_answer"]) == {"Because it is correct."}


def test_canonical_reference_falls_back_without_best() -> None:
    import xml.etree.ElementTree as ET

    question = ET.fromstring(
        """<question><referenceAnswers>
        <referenceAnswer>First answer.</referenceAnswer>
        <referenceAnswer>Second answer.</referenceAnswer>
        </referenceAnswers></question>"""
    )
    assert canonical_reference_answers(question) == ["First answer.", "Second answer."]
