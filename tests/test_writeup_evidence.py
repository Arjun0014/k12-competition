from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_ace.writeup_evidence import (
    CLAIMS,
    PROTECTED_ZIP,
    PROTECTED_ZIP_SHA256,
    build_writeup_evidence,
    nested_value,
)


def test_nested_value_requires_complete_path() -> None:
    report = {"outer": {"metric": 0.125}}
    assert nested_value(report, ("outer", "metric")) == 0.125
    with pytest.raises(KeyError, match="outer.missing"):
        nested_value(report, ("outer", "missing"))


def test_claim_ids_and_sources_are_frozen() -> None:
    claim_ids = [claim.claim_id for claim in CLAIMS]
    assert len(claim_ids) == len(set(claim_ids))
    assert len(CLAIMS) >= 15
    assert all(claim.run_id.startswith("202607") for claim in CLAIMS)


def test_builder_rejects_changed_protected_zip(tmp_path: Path) -> None:
    zip_path = tmp_path / PROTECTED_ZIP
    zip_path.parent.mkdir(parents=True)
    zip_path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="Protected ZIP hash changed"):
        build_writeup_evidence(tmp_path)


def test_live_evidence_builder_is_deterministic() -> None:
    project_root = Path(__file__).resolve().parents[1]
    first = build_writeup_evidence(project_root)
    manifest_path = (
        project_root / "reports" / "generated" / "writeup_evidence_manifest.json"
    )
    first_bytes = manifest_path.read_bytes()
    second = build_writeup_evidence(project_root)
    second_bytes = manifest_path.read_bytes()
    assert first == second
    assert first_bytes == second_bytes
    assert first["claim_count"] == len(CLAIMS)
    assert first["protected_zip"]["sha256"] == PROTECTED_ZIP_SHA256
    assert first["V_final_accessed"] is False
    assert all(
        item["V_final_accessed"] is False for item in first["reports"].values()
    )
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded == first
