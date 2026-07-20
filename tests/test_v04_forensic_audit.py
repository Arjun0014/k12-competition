from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from trace_ace.v04_forensic_audit import (
    EXPECTED_WIDTHS,
    GATE_A_COMPONENTS,
    _matrix_check,
    _text_check,
    audit_oof_integrity,
    calibration_metrics,
    gate_a_decision,
    import_packaged_main,
    prior_shift_sensitivity,
)


def _component_parity() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for component in GATE_A_COMPONENTS.values():
        for index in range(100):
            rows.append(
                {
                    "response_id": f"r{index:03d}",
                    "component": component,
                    "expected_width": EXPECTED_WIDTHS[component],
                    "offline_width": EXPECTED_WIDTHS[component],
                    "runtime_width": EXPECTED_WIDTHS[component],
                    "offline_probability": 0.4,
                    "runtime_probability": 0.4000002,
                    "absolute_probability_difference": 2e-7,
                }
            )
    return pd.DataFrame(rows)


def _package_verification() -> dict[str, bool]:
    return {
        "packaged_local_artifact_match": True,
        "packaged_local_supervised_delta_match": True,
        "encoder_asset_hashes_match": True,
        "replay_integrity_verified": True,
    }


def _passing_preprocessing() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "check": ["objective_context", "supervised_input_ids", "behavior_raw"],
            "comparison_type": ["text", "matrix", "matrix"],
            "offline_shape": ["[100]", "[100, 192]", "[100, 16]"],
            "runtime_shape": ["[100]", "[100, 192]", "[100, 16]"],
            "max_abs_difference": [0.0, 0.0, 1.5e-6],
            "exact": [True, True, False],
        }
    )


def test_import_packaged_main_executes_the_exact_source(tmp_path: Path):
    source = tmp_path / "main.py"
    source.write_text("SENTINEL = 'packaged-v04'\n", encoding="utf-8")

    module = import_packaged_main(source)

    assert module.SENTINEL == "packaged-v04"
    assert Path(module.__file__).resolve() == source.resolve()


def test_parity_checks_detect_exact_and_changed_preprocessing():
    exact_text = _text_check("context", ["a", "b"], ["a", "b"])
    changed_text = _text_check("context", ["a", "b"], ["a", "c"])
    left = sparse.csr_matrix([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    exact_matrix = _matrix_check("features", left, left.copy())
    changed_matrix = _matrix_check(
        "features", left, sparse.csr_matrix([[0.0, 1.0], [2.1, 0.0]], dtype=np.float32)
    )

    assert exact_text["exact"] is True
    assert exact_text["offline_sha256"] == exact_text["runtime_sha256"]
    assert changed_text["exact"] is False
    assert changed_text["mismatch_count"] == 1
    assert exact_matrix["exact"] is True
    assert exact_matrix["max_abs_difference"] == 0.0
    assert changed_matrix["exact"] is False
    assert changed_matrix["max_abs_difference"] > 0.09


def test_matrix_check_exposes_non_finite_difference():
    check = _matrix_check(
        "features",
        np.array([[1.0, np.nan]]),
        np.array([[1.0, 2.0]]),
    )

    assert check["exact"] is False
    assert check["max_abs_difference"] is None


def test_calibration_and_prior_shift_keep_fidelity_labels():
    target = np.tile([0, 1], 50)
    probability = np.where(target == 1, 0.72, 0.34).astype(float)
    oof = pd.DataFrame(
        {
            "protocol": "semantic_k50_s0",
            "response_id": [f"r{index}" for index in range(len(target))],
            "target": target,
            "fold": np.tile([0, 1, 2, 3, 4], 20),
            "probability": probability,
            "model": "v04_proxy",
            "fidelity": "proxy_not_final_refit",
            "source": "synthetic",
        }
    )

    calibration = calibration_metrics(oof)
    shifted = prior_shift_sensitivity(oof, target_priors=(0.40, 0.70))

    assert set(calibration["fold"]) == {"pooled", "0", "1", "2", "3", "4"}
    assert set(calibration["fidelity"]) == {"proxy_not_final_refit"}
    assert np.isfinite(calibration["log_loss"]).all()
    assert shifted["assumed_positive_rate"].tolist() == [0.40, 0.70]
    assert set(shifted["fidelity"]) == {"proxy_not_final_refit"}
    assert np.isfinite(shifted["weighted_log_loss"]).all()


def test_gate_a_requires_tolerance_aware_preprocessing_and_local_asset_identity():
    components = _component_parity()
    package = _package_verification()
    passing_preprocessing = _passing_preprocessing()
    failing_preprocessing = passing_preprocessing.copy()
    failing_preprocessing.loc[2, "max_abs_difference"] = 2.1e-6

    passed = gate_a_decision(
        components, passing_preprocessing, package, {"passed": True}
    )
    failed = gate_a_decision(
        components, failing_preprocessing, package, {"passed": True}
    )

    assert passed["passed"] is True
    assert passed["component_set_exact"] is True
    assert passed["component_rows_exact"] is True
    assert passed["component_coverage_exact"] is True
    assert passed["preprocessing_bitwise_exact"] is False
    assert passed["preprocessing_within_tolerance"] is True
    assert passed["final_output_maximum_probability_difference"] == 2e-7
    assert passed["classification"].startswith(
        "no parity defect observed on preserved 100-row smoke sample"
    )
    assert failed["passed"] is False
    assert failed["preprocessing_within_tolerance"] is False


def test_gate_a_rejects_missing_component_and_incomplete_row_count():
    components = _component_parity()
    missing_component = components.loc[
        ~components["component"].eq(GATE_A_COMPONENTS["supervised"])
    ]
    incomplete = components.drop(
        components.loc[
            components["component"].eq(GATE_A_COMPONENTS["role"])
        ].index[-1]
    )

    missing_result = gate_a_decision(
        missing_component,
        _passing_preprocessing(),
        _package_verification(),
        {"passed": True},
    )
    incomplete_result = gate_a_decision(
        incomplete,
        _passing_preprocessing(),
        _package_verification(),
        {"passed": True},
    )

    assert missing_result["passed"] is False
    assert missing_result["component_set_exact"] is False
    assert incomplete_result["passed"] is False
    assert incomplete_result["component_rows_exact"] is False
    assert incomplete_result["component_row_counts"][GATE_A_COMPONENTS["role"]] == 99


def test_gate_a_rejects_nan_probability_and_nan_numeric_preprocessing():
    components = _component_parity()
    components.loc[0, "offline_probability"] = np.nan
    preprocessing = _passing_preprocessing()
    preprocessing.loc[2, "max_abs_difference"] = np.nan

    result = gate_a_decision(
        components,
        preprocessing,
        _package_verification(),
        {"passed": True},
    )

    assert result["passed"] is False
    assert result["component_probabilities_finite"] is False
    assert result["numeric_preprocessing_differences_finite"] is False
    assert result["numeric_preprocessing_maximum_difference"] is None


def test_gate_a_rejects_duplicate_or_gapped_component_coverage():
    duplicate = _component_parity()
    role_rows = duplicate.index[
        duplicate["component"].eq(GATE_A_COMPONENTS["role"])
    ]
    duplicate.loc[role_rows[-1], "response_id"] = "r000"
    gapped = _component_parity()
    supervised_rows = gapped.index[
        gapped["component"].eq(GATE_A_COMPONENTS["supervised"])
    ]
    gapped.loc[supervised_rows[-1], "response_id"] = "outside-smoke-sample"

    duplicate_result = gate_a_decision(
        duplicate,
        _passing_preprocessing(),
        _package_verification(),
        {"passed": True},
    )
    gapped_result = gate_a_decision(
        gapped,
        _passing_preprocessing(),
        _package_verification(),
        {"passed": True},
    )

    assert duplicate_result["passed"] is False
    assert duplicate_result["duplicate_component_response_ids"] == 1
    assert gapped_result["passed"] is False
    assert gapped_result["component_coverage_exact"] is False
    assert gapped_result["component_coverage_gap_count"] == 2


def test_oof_integrity_rejects_cross_source_target_or_fold_inconsistency():
    first = pd.DataFrame(
        {
            "protocol": ["p", "p"],
            "response_id": ["r1", "r2"],
            "target": [0, 1],
            "fold": [0, 1],
            "probability": [0.2, 0.8],
            "model": ["a", "a"],
            "fidelity": ["exact", "exact"],
            "source": ["source-a", "source-a"],
        }
    )
    second = first.copy()
    second["model"] = "b"
    second["source"] = "source-b"
    second.loc[0, "target"] = 1
    second.loc[1, "fold"] = 0

    with pytest.raises(ValueError, match="target/fold values conflict"):
        audit_oof_integrity(pd.concat([first, second], ignore_index=True))


def test_oof_integrity_rejects_duplicate_response_within_source():
    oof = pd.DataFrame(
        {
            "protocol": ["p", "p"],
            "response_id": ["r1", "r1"],
            "target": [0, 0],
            "fold": [0, 0],
            "probability": [0.2, 0.3],
            "model": ["a", "a"],
            "fidelity": ["exact", "exact"],
            "source": ["source-a", "source-a"],
        }
    )

    with pytest.raises(ValueError, match="duplicate response IDs"):
        audit_oof_integrity(oof)
