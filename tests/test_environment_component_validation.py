from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import trace_ace.environment_component_validation as component_module
from trace_ace.environment_component_validation import (
    ALLOWED_ENVIRONMENTS,
    BASELINE_CANDIDATE,
    CANDIDATE_REGISTRY,
    CANDIDATE_REGISTRY_SHA256,
    DEVELOPMENT_ENVIRONMENTS,
    EXPECTED_CANDIDATE_REGISTRY_SHA256,
    PLAN_AMENDMENT,
    _new_run_dir,
    apply_platt,
    candidate_probability,
    create_locked_candidate,
    fit_constrained_platt,
    gate_c_decision,
    inner_environment_folds,
    load_or_fit_environment_components,
    load_or_fit_nested_calibration,
    paired_bootstrap_summary,
    prior_shrink_prediction,
    run_development_stage,
    run_environment_component_validation,
    run_joint_stage,
    validate_candidate_registry,
    validate_development_assignment_contract,
    verify_locked_candidate,
    write_or_verify_joint_assignment_binding,
    write_or_verify_joint_opened,
    write_or_verify_phase_contract,
)
from trace_ace.validation_environments import assignment_sha256


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _self_hashed_contract(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "test",
        "code_sha256": HASH_A,
        "modeling_fingerprint_sha256": HASH_B,
        "feature_cache_hashes": {"feature": HASH_C},
        "selection_assignment_sha256": HASH_A,
        "protocol_sha256": HASH_B,
        "V_final_accessed": False,
    }
    payload.update(updates)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["phase_contract_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _lock_bindings(contract: dict[str, object]) -> dict[str, object]:
    return {
        "phase_contract_sha256": contract["phase_contract_sha256"],
        "development_oof_sha256": HASH_C,
        "code_sha256": contract["code_sha256"],
        "modeling_fingerprint_sha256": contract["modeling_fingerprint_sha256"],
        "feature_cache_hashes": contract["feature_cache_hashes"],
        "selection_assignment_sha256": contract["selection_assignment_sha256"],
        "protocol_sha256": contract["protocol_sha256"],
    }


def _test_lock(
    *,
    candidate: str = "v02_feedback20",
    calibration: str = "prior_90",
    contract: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    contract = contract or _self_hashed_contract()
    deployment: dict[str, object] = {
        "rule": "fixed_prior_shrink_with_full_training_prevalence",
        "model_weight": 0.90,
        "full_training_prevalence": 0.55,
        "source_prediction_sha256": HASH_A,
        "source_rows": 20,
    }
    if calibration == "raw":
        deployment = {
            "rule": "no_probability_calibration",
            "source_prediction_sha256": HASH_A,
            "source_rows": 20,
        }
    elif calibration == "nested_platt":
        deployment = {
            "rule": "single_platt_fit_on_full_v_seen_cross_fitted_raw_predictions",
            "intercept": -0.1,
            "slope": 0.9,
            "source_prediction_sha256": HASH_A,
            "source_rows": 20,
        }
    nested_parameters = (
        {
            environment: [
                {"outer_fold": fold, "intercept": 0.0, "slope": 1.0}
                for fold in range(5)
            ]
            for environment in DEVELOPMENT_ENVIRONMENTS
        }
        if calibration == "nested_platt"
        else {}
    )
    lock = create_locked_candidate(
        candidate,
        calibration,
        {
            "candidate": candidate,
            "calibration": calibration,
            "mean_delta_log_loss_vs_v02": -0.003,
            "passes_development_guard": True,
        },
        nested_parameters,
        deployment_rule=deployment,
        bindings=_lock_bindings(contract),
    )
    return lock, contract


def test_fixed_candidate_registry_and_formulas_are_exact_and_mutation_fails():
    validate_candidate_registry()
    assert CANDIDATE_REGISTRY_SHA256 == EXPECTED_CANDIDATE_REGISTRY_SHA256
    assert PLAN_AMENDMENT["formulas"] == list(CANDIDATE_REGISTRY)
    assert set(PLAN_AMENDMENT["excluded"]) == {
        "supervised_bge_small",
        "full_v04",
    }
    assert CANDIDATE_REGISTRY["v02"] == {
        "full": 0.25,
        "role": 0.25,
        "bge_small": 0.50,
    }
    assert CANDIDATE_REGISTRY["bge_nb30"] == {
        "full": 0.175,
        "role": 0.175,
        "bge_base": 0.35,
        "nbsvm": 0.30,
    }
    components = {
        "full": np.array([0.2, 0.4]),
        "role": np.array([0.4, 0.6]),
        "bge_small": np.array([0.6, 0.8]),
        "bge_base": np.array([0.5, 0.7]),
        "feedback_sgd": np.array([0.1, 0.9]),
        "nbsvm": np.array([0.3, 0.8]),
    }
    np.testing.assert_allclose(
        candidate_probability(components, "v02"),
        0.25 * components["full"]
        + 0.25 * components["role"]
        + 0.50 * components["bge_small"],
    )
    np.testing.assert_allclose(
        candidate_probability(components, "bge_feedback20"),
        0.20 * components["full"]
        + 0.20 * components["role"]
        + 0.40 * components["bge_base"]
        + 0.20 * components["feedback_sgd"],
    )
    mutated = copy.deepcopy(CANDIDATE_REGISTRY)
    mutated["v02"]["full"] += 0.01
    mutated["v02"]["role"] -= 0.01
    with pytest.raises(ValueError, match="digest differs"):
        validate_candidate_registry(mutated)


def _assignment_for_frame(
    frame: pd.DataFrame,
    environments: tuple[str, ...] = ALLOWED_ENVIRONMENTS,
    suite: str = "development",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for environment in environments:
        for index, row in frame.reset_index(drop=True).iterrows():
            family = index % 6
            style = index % 4
            rows.append(
                {
                    "schema_version": "test",
                    "suite": suite,
                    "split_seed": 1,
                    "environment": environment,
                    "response_id": str(row["response_id"]),
                    "session_id": str(row["session_id"]),
                    "learning_objective_id": str(row["learning_objective_id"]),
                    "fold": index % 5,
                    "semantic_family": family,
                    "style_cell": style,
                    "joint_cell": f"family_{family:03d}__style_{style:03d}",
                    "evaluation_eligible": True,
                }
            )
    return pd.DataFrame(rows)


def _tiny_modeling_frame(rows: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "response_id": [f"r{index:03d}" for index in range(rows)],
            "session_id": [f"s{index:03d}" for index in range(rows)],
            "learning_objective_id": [f"o{index % 10:02d}" for index in range(rows)],
            "target": np.asarray([index % 2 for index in range(rows)], dtype=np.int8),
        }
    )


def test_assignment_contract_hard_refuses_final_suite_and_wrong_environment_set():
    frame = _tiny_modeling_frame(10)
    assignments = _assignment_for_frame(frame, suite="V_final")
    with pytest.raises(ValueError, match="development.*only"):
        validate_development_assignment_contract(
            assignments,
            expected_sha256=assignment_sha256(assignments),
            expected_environments=ALLOWED_ENVIRONMENTS,
        )
    selection = _assignment_for_frame(frame, environments=DEVELOPMENT_ENVIRONMENTS)
    with pytest.raises(ValueError, match="Unexpected development environments"):
        validate_development_assignment_contract(
            selection,
            expected_sha256=assignment_sha256(selection),
            expected_environments=ALLOWED_ENVIRONMENTS,
        )


def test_inner_calibration_folds_preserve_environment_group_and_session_purge():
    frame = _tiny_modeling_frame(36)
    assignments = _assignment_for_frame(frame)
    for environment, group_column in {
        "V_seen": "session_id",
        "V_objective": "semantic_family",
        "V_style": "style_cell",
        "V_joint": "joint_cell",
    }.items():
        aligned = assignments.loc[
            assignments["environment"].eq(environment)
        ].reset_index(drop=True)
        folds = inner_environment_folds(
            frame, aligned, environment, seed=73, n_splits=3
        )
        groups = (
            frame["session_id"].astype(str).to_numpy()
            if group_column == "session_id"
            else aligned[group_column].astype(str).to_numpy()
        )
        for fold in range(3):
            validation = folds == fold
            train = ~validation
            validation_sessions = set(frame.loc[validation, "session_id"])
            train &= ~frame["session_id"].isin(validation_sessions).to_numpy()
            assert not set(frame.loc[train, "session_id"]).intersection(
                validation_sessions
            )
            assert not set(groups[train]).intersection(set(groups[validation]))


def test_prior_shrink_and_constrained_platt_are_finite_and_monotone():
    probability = np.array([0.05, 0.20, 0.35, 0.65, 0.80, 0.95])
    target = np.array([0, 0, 0, 1, 1, 1])
    prior = np.full_like(probability, 0.5)
    shrunk = prior_shrink_prediction(probability, prior, model_weight=0.9)
    np.testing.assert_allclose(shrunk, 0.9 * probability + 0.1 * prior)
    intercept, slope = fit_constrained_platt(target, probability)
    calibrated = apply_platt(probability, intercept, slope)
    assert slope > 0.0
    assert np.isfinite(calibrated).all()
    assert np.all(np.diff(calibrated) > 0.0)


def test_locked_candidate_hash_contract_and_raw_v02_guards():
    lock, contract = _test_lock()
    verify_locked_candidate(lock, contract=contract)
    changed = copy.deepcopy(lock)
    changed["calibration"] = "raw"
    with pytest.raises(ValueError, match="hash verification"):
        verify_locked_candidate(changed, contract=contract)
    stale_contract = dict(contract)
    stale_contract["code_sha256"] = HASH_C
    with pytest.raises(ValueError, match="stale code_sha256"):
        verify_locked_candidate(lock, contract=stale_contract)
    with pytest.raises(ValueError, match="Raw v0.2"):
        create_locked_candidate(
            BASELINE_CANDIDATE,
            "raw",
            {
                "candidate": BASELINE_CANDIDATE,
                "calibration": "raw",
                "passes_development_guard": True,
            },
            {},
            deployment_rule={
                "rule": "none",
                "source_prediction_sha256": HASH_A,
            },
            bindings=_lock_bindings(contract),
        )


def test_phase_contract_and_run_directory_are_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    contract = _self_hashed_contract()
    written = write_or_verify_phase_contract(tmp_path, contract, resume=False)
    assert written == contract
    with pytest.raises(FileExistsError):
        write_or_verify_phase_contract(tmp_path, contract, resume=False)
    changed = _self_hashed_contract(code_sha256=HASH_C)
    with pytest.raises(ValueError, match="immutable"):
        write_or_verify_phase_contract(tmp_path, changed, resume=True)

    experiments = tmp_path / "experiments"
    monkeypatch.setattr(
        component_module,
        "discover_project_paths",
        lambda _: SimpleNamespace(experiments_dir=experiments),
    )
    run_id = "20260720T000000Z_environment_component_validation"
    _, run_dir = _new_run_dir(tmp_path, run_id, resume=False)
    assert run_dir.is_dir()
    with pytest.raises(FileExistsError):
        _new_run_dir(tmp_path, run_id, resume=False)
    assert _new_run_dir(tmp_path, run_id, resume=True)[1] == run_dir


def _fake_component_output(frame: pd.DataFrame, environment: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "environment": environment,
            "response_id": frame["response_id"].astype(str),
            "session_id": frame["session_id"].astype(str),
            "learning_objective_id": frame["learning_objective_id"].astype(str),
            "semantic_family": np.arange(len(frame)) % 6,
            "fold": np.arange(len(frame)) % 5,
            "evaluation_eligible": True,
            "target": frame["target"].to_numpy(dtype=np.int8),
            "fold_prior": np.full(len(frame), 0.5),
            "pred_full": np.linspace(0.2, 0.8, len(frame)),
        }
    )


def test_component_checkpoint_binds_contract_schema_order_and_finite_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    frame = _tiny_modeling_frame(10)
    assignments = _assignment_for_frame(frame, environments=("V_seen",))
    contract = _self_hashed_contract(selection_assignment_sha256=assignment_sha256(assignments))
    monkeypatch.setattr(
        component_module,
        "fit_environment_components",
        lambda frame, environment_assignments, blocks, required: _fake_component_output(
            frame, "V_seen"
        ),
    )
    first = load_or_fit_environment_components(
        tmp_path,
        frame,
        assignments,
        object(),
        "V_seen",
        {"full"},
        contract,
    )
    second = load_or_fit_environment_components(
        tmp_path,
        frame,
        assignments,
        object(),
        "V_seen",
        {"full"},
        contract,
    )
    pd.testing.assert_frame_equal(first, second)
    metadata_path = tmp_path / "checkpoints" / "V_seen.components.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["phase_contract_sha256"] = HASH_C
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Stale or incompatible"):
        load_or_fit_environment_components(
            tmp_path,
            frame,
            assignments,
            object(),
            "V_seen",
            {"full"},
            contract,
        )


def test_nested_calibration_uses_one_bound_checkpoint_per_outer_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    frame = _tiny_modeling_frame(10)
    assignments = _assignment_for_frame(frame, environments=("V_seen",))
    component_frame = _fake_component_output(frame, "V_seen")
    component_frame["pred_role"] = np.linspace(0.3, 0.7, len(frame))
    component_frame["pred_bge_small"] = np.linspace(0.25, 0.75, len(frame))
    contract = _self_hashed_contract()

    def fake_masks(
        modeling: pd.DataFrame, environment_assignments: pd.DataFrame, fold: int
    ) -> tuple[np.ndarray, np.ndarray]:
        validation = environment_assignments["fold"].to_numpy(dtype=int) == fold
        return ~validation, validation

    def fake_outer(
        modeling: pd.DataFrame,
        environment_assignments: pd.DataFrame,
        blocks: object,
        raw_component_frame: pd.DataFrame,
        candidate: str,
        *,
        outer_fold: int,
        inner_splits: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        _, validation = fake_masks(modeling, environment_assignments, outer_fold)
        return (
            validation,
            np.full(int(validation.sum()), 0.35 + 0.05 * outer_fold),
            {
                "outer_fold": outer_fold,
                "intercept": 0.0,
                "slope": 1.0,
                "inner_splits": inner_splits,
            },
        )

    monkeypatch.setattr(component_module, "purged_fold_masks", fake_masks)
    monkeypatch.setattr(component_module, "nested_platt_outer_fold", fake_outer)
    prediction, parameters = load_or_fit_nested_calibration(
        tmp_path,
        frame,
        assignments,
        object(),
        component_frame,
        "V_seen",
        BASELINE_CANDIDATE,
        contract,
    )
    assert np.isfinite(prediction).all()
    assert len(parameters) == 5
    for fold in range(5):
        stem = f"V_seen.v02.nested_platt.fold_{fold}"
        assert (tmp_path / "checkpoints" / f"{stem}.parquet").exists()
        assert (tmp_path / "checkpoints" / f"{stem}.json").exists()

    metadata_path = (
        tmp_path / "checkpoints" / "V_seen.v02.nested_platt.fold_2.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["raw_component_input_sha256"] = HASH_C
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="raw_component_input_sha256"):
        load_or_fit_nested_calibration(
            tmp_path,
            frame,
            assignments,
            object(),
            component_frame,
            "V_seen",
            BASELINE_CANDIDATE,
            contract,
        )


def _bootstrap_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(17)
    for environment_index, environment in enumerate(ALLOWED_ENVIRONMENTS):
        for index in range(120):
            target = index % 2
            baseline = 0.68 if target else 0.32
            candidate = 0.74 if target else 0.26
            rows.append(
                {
                    "environment": environment,
                    "response_id": f"r{index:03d}",
                    "session_id": f"s{index // 2:03d}",
                    "semantic_family": index % 10,
                    "candidate": "bge_nb30",
                    "calibration": "nested_platt",
                    "evaluation_eligible": True,
                    "target": target,
                    "pred_v02": baseline + rng.normal(0, 0.002),
                    "prediction": candidate
                    + rng.normal(0, 0.002 + environment_index * 0.0001),
                }
            )
    return pd.DataFrame(rows)


def test_paired_bootstrap_is_deterministic_shared_and_fully_bound():
    frame = _bootstrap_fixture()
    first = paired_bootstrap_summary(frame, n_replicates=5_000, seed=71)
    second = paired_bootstrap_summary(frame, n_replicates=5_000, seed=71)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["resampler"]) == {"session", "objective_family"}
    assert set(first["candidate"]) == {"bge_nb30"}
    assert set(first["calibration"]) == {"nested_platt"}
    assert (first["replicates"] == 5_000).all()
    assert (first["unique_replicates"] == 5_000).all()
    for _, group in first.groupby("resampler"):
        assert group["shared_resample_sha256"].nunique() == 1
    macro = first[first["environment"].eq("ALL_MACRO")]
    assert (macro["probability_gain_positive"] > 0.99).all()
    with pytest.raises(ValueError, match="At least 5000"):
        paired_bootstrap_summary(frame, n_replicates=4_999, seed=71)


def _gate_fixtures() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for environment in ALLOWED_ENVIRONMENTS:
        rows.append(
            {
                "environment": environment,
                "candidate": "bge_nb30",
                "calibration": "nested_platt",
                "delta_log_loss_vs_v02": -0.003,
                "delta_roc_auc_vs_v02": 0.004,
                "delta_brier_score_vs_v02": -0.001,
                "delta_ece_10_vs_v02": -0.002,
                "calibration_intercept": 0.1,
                "calibration_slope": 0.9,
                "calibration_sanity_distance": 0.1,
                "delta_calibration_sanity_vs_v02": -0.01,
                "prediction_input_sha256": HASH_A,
                "lock_sha256": HASH_B,
                "phase_contract_sha256": HASH_C,
            }
        )
    bootstrap_rows: list[dict[str, object]] = []
    for environment in [*ALLOWED_ENVIRONMENTS, "ALL_MACRO"]:
        for resampler, stream_hash in (
            ("session", HASH_A),
            ("objective_family", HASH_B),
        ):
            bootstrap_rows.append(
                {
                    "environment": environment,
                    "resampler": resampler,
                    "candidate": "bge_nb30",
                    "calibration": "nested_platt",
                    "prediction_input_sha256": HASH_A,
                    "replicates": 5_000,
                    "unique_replicates": 5_000,
                    "shared_resample_sha256": stream_hash,
                    "probability_gain_positive": 0.95,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def _gate(metrics: pd.DataFrame, bootstrap: pd.DataFrame) -> dict[str, object]:
    return gate_c_decision(
        metrics,
        bootstrap,
        candidate="bge_nb30",
        calibration="nested_platt",
        expected_prediction_input_sha256=HASH_A,
        expected_lock_sha256=HASH_B,
        expected_phase_contract_sha256=HASH_C,
    )


def test_gate_c_requires_every_original_clause_and_per_environment_calibration():
    metrics, bootstrap = _gate_fixtures()
    passed = _gate(metrics, bootstrap)
    assert passed["passes_gate_c"] is True
    assert passed["calibration_sanity_non_regression_in_every_environment"] is True
    degraded = metrics.copy()
    degraded.loc[
        degraded["environment"].eq("V_style"), "delta_log_loss_vs_v02"
    ] = 0.001
    failed = _gate(degraded, bootstrap)
    assert failed["passes_gate_c"] is False
    assert failed["clauses"]["worst_environment_regression_at_most_0_0005"] is False
    ece_regression = metrics.copy()
    ece_regression.loc[
        ece_regression["environment"].eq("V_seen"), "delta_ece_10_vs_v02"
    ] = 1e-6
    assert _gate(ece_regression, bootstrap)["passes_gate_c"] is False


def test_gate_c_rejects_duplicates_nonfinite_stale_and_under_5000():
    metrics, bootstrap = _gate_fixtures()
    duplicate = pd.concat([metrics, metrics.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="exactly four unique"):
        _gate(duplicate, bootstrap)
    nonfinite = metrics.copy()
    nonfinite.loc[0, "delta_log_loss_vs_v02"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _gate(nonfinite, bootstrap)
    stale = bootstrap.copy()
    stale["prediction_input_sha256"] = HASH_C
    with pytest.raises(ValueError, match="prediction-input binding"):
        _gate(metrics, stale)
    too_few = bootstrap.copy()
    too_few["replicates"] = 4_999
    with pytest.raises(ValueError, match="at least 5,000"):
        _gate(metrics, too_few)
    repeated_draws = bootstrap.copy()
    repeated_draws["unique_replicates"] = 4_999
    with pytest.raises(ValueError, match="unique bootstrap"):
        _gate(metrics, repeated_draws)


def test_joint_requires_lock_and_development_rerun_is_forbidden_after_joint(
    tmp_path: Path,
):
    contract = _self_hashed_contract()
    with pytest.raises(FileNotFoundError, match="development stage"):
        run_joint_stage(
            tmp_path,
            tmp_path,
            _tiny_modeling_frame(10),
            object(),
            contract,
            n_bootstrap=5_000,
        )
    (tmp_path / "joint_opened.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden after"):
        run_development_stage(
            tmp_path,
            tmp_path,
            _tiny_modeling_frame(10),
            pd.DataFrame(),
            object(),
            contract,
        )


def test_joint_opened_sentinel_binds_lock_contract_and_assignments(tmp_path: Path):
    lock, contract = _test_lock()
    frame = _tiny_modeling_frame(10)
    assignments = _assignment_for_frame(frame, environments=("V_joint",))
    sentinel = write_or_verify_joint_opened(tmp_path, lock, contract)
    assert sentinel["V_final_accessed"] is False
    assert write_or_verify_joint_opened(tmp_path, lock, contract) == sentinel
    binding = write_or_verify_joint_assignment_binding(
        tmp_path, sentinel, assignments
    )
    assert binding["joint_assignment_sha256"] == assignment_sha256(assignments)
    changed = assignments.copy()
    changed.loc[0, "fold"] = 4
    with pytest.raises(ValueError, match="differs on resume"):
        write_or_verify_joint_assignment_binding(tmp_path, sentinel, changed)


def test_final_stage_is_not_a_callable_path():
    with pytest.raises(ValueError, match="V_final is forbidden"):
        run_environment_component_validation(".", stage="final")
