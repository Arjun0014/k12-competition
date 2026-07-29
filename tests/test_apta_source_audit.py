from __future__ import annotations

import json
import zipfile
from pathlib import Path

from trace_ace.apta_source_audit import audit_apta_source, write_report


def _write_ready_source(root: Path, learners: int = 55) -> None:
    root.mkdir(parents=True)
    transaction_lines = [
        "\t".join(
            [
                "Anon Student Id",
                "Session Id",
                "Transaction Id",
                "Time",
                "Role",
                "Condition",
                "Problem Name",
                "Step Name",
                "KC (Default)",
                "Student Response Type",
                "Tutor Response Type",
                "Outcome",
                "Text",
            ]
        )
    ]
    outcome_lines = ["student_id,pretest_score,posttest_score"]
    for index in range(learners):
        student = f"learner-{index:03d}"
        condition = "dynamic" if index % 2 else "standard"
        transaction_lines.append(
            "\t".join(
                [
                    student,
                    f"session-{index:03d}",
                    f"tx-{index:03d}",
                    "2026-01-01 10:00:00",
                    "solver" if index % 2 else "tutor",
                    condition,
                    "linear-equations",
                    "solve-x",
                    "combine-like-terms",
                    "ATTEMPT",
                    "RESULT",
                    "CORRECT" if index % 2 else "INCORRECT",
                    "deidentified message",
                ]
            )
        )
        outcome_lines.append(f"{student},{0.2 + index / 1000:.3f},{0.4 + index / 1000:.3f}")
    (root / "transactions.tsv").write_text(
        "\n".join(transaction_lines) + "\n", encoding="utf-8"
    )
    (root / "prepost.csv").write_text(
        "\n".join(outcome_lines) + "\n", encoding="utf-8"
    )


def test_ready_source_links_transactions_to_prepost_without_values(tmp_path: Path) -> None:
    source = tmp_path / "Datasets" / "APTA"
    _write_ready_source(source)

    report = audit_apta_source(source)

    assert report["discovery_ready"] is True
    assert (
        report["privacy_safe_linkage_counts"][
            "linked_transaction_and_learning_outcome_students"
        ]
        == 55
    )
    assert report["raw_identifiers_emitted"] is False
    assert report["outcome_values_emitted"] is False
    serialized = json.dumps(report)
    assert "learner-000" not in serialized
    assert "CORRECT" not in serialized


def test_transaction_correctness_is_not_a_learning_outcome(tmp_path: Path) -> None:
    source = tmp_path / "Datasets" / "APTA"
    _write_ready_source(source)
    (source / "prepost.csv").unlink()

    report = audit_apta_source(source)

    assert report["gates"]["transaction_outcome_present"] is True
    assert report["gates"]["pre_post_or_learning_outcome_present"] is False
    assert report["discovery_ready"] is False


def test_unsafe_archive_is_reported_and_not_opened(tmp_path: Path) -> None:
    source = tmp_path / "Datasets" / "APTA"
    source.mkdir(parents=True)
    archive_path = source / "download.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.tsv", "Anon Student Id\tOutcome\nx\tCORRECT\n")

    report = audit_apta_source(source)

    assert report["archives"][0]["safe"] is False
    assert report["gates"]["archives_safe"] is False
    assert report["tables"] == []
    assert report["discovery_ready"] is False


def test_report_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "Datasets" / "APTA"
    _write_ready_source(source)
    report = audit_apta_source(source)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_hash = write_report(report, first)
    second_hash = write_report(report, second)

    assert first_hash == second_hash
    assert first.read_bytes() == second.read_bytes()
