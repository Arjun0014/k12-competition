from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import trace_ace.validation_environments as validation_module
from trace_ace.validation_environments import (
    CONFIRMATION_ENVIRONMENT,
    DEVELOPMENT_SEED,
    DEVELOPMENT_SUITE,
    ENVIRONMENT_DEFINITIONS,
    ENVIRONMENTS,
    JOINT_CONFIRMATION_ASSIGNMENT_PATH,
    PROTOCOL_VERSION,
    SELECTION_ASSIGNMENT_PATH,
    SELECTION_ENVIRONMENTS,
    assignment_sha256,
    build_validation_environments,
    build_validation_manifest,
    evaluation_mask_for_fold,
    load_development_selection_assignments,
    load_joint_confirmation_assignments,
    purged_fold_masks,
    run_validation_environment_build,
    validate_environment_assignments,
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ValidationEnvironmentTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(440)
        objective_centers = rng.normal(size=(12, 10))
        objective_centers /= np.linalg.norm(objective_centers, axis=1, keepdims=True)
        rows: list[dict[str, object]] = []
        embeddings: list[np.ndarray] = []
        behavior: list[dict[str, object]] = []
        for session in range(72):
            duration = 120 + (session % 17) * 15
            utterances = 14 + (session % 13)
            student_fraction = 0.25 + 0.04 * (session % 7)
            tutor_fraction = 0.92 - student_fraction
            behavior.append(
                {
                    "session_id": f"s{session:03d}",
                    "role_switches": 4 + session % 11,
                    "tutor_words": 80 + session % 29,
                    "student_words": 40 + session % 23,
                }
            )
            for offset in range(2):
                objective = (session + offset * 5) % 12
                rows.append(
                    {
                        "response_id": f"r{session:03d}_{offset}",
                        "session_id": f"s{session:03d}",
                        "learning_objective_id": f"o{objective:02d}",
                        "learning_objective": f"objective {objective}",
                        "target": (session // 2 + offset + objective) % 2,
                        "n_utterances": utterances,
                        "duration_seconds": duration,
                        "content_chars": 500 + session * 3,
                        "student_fraction": student_fraction,
                        "tutor_fraction": tutor_fraction,
                        "background_fraction": 0.01 * (session % 6),
                        "unclear_fraction": 0.015 * (session % 5),
                    }
                )
                embeddings.append(objective_centers[objective])
        self.frame = pd.DataFrame(rows)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.behavior = pd.DataFrame(behavior)

    def build(self, frame=None, embeddings=None, suite=DEVELOPMENT_SUITE):
        return build_validation_environments(
            self.frame if frame is None else frame,
            self.embeddings if embeddings is None else embeddings,
            seed=DEVELOPMENT_SEED,
            suite_name=suite,
            n_splits=3,
            n_semantic_families=6,
            n_style_cells=9,
            session_behavior=self.behavior,
        )

    def validate(self, assignments, frame=None, **kwargs):
        return validate_environment_assignments(
            self.frame if frame is None else frame,
            assignments,
            n_splits=3,
            min_validation_rows=10,
            min_class_rows=2,
            **kwargs,
        )

    def _manifest(self, assignments, diagnostics=None, root=None):
        return build_validation_manifest(
            development_assignments=assignments,
            source_hashes={"training": "abc123"},
            development_diagnostics=[] if diagnostics is None else diagnostics,
            n_splits=3,
            n_semantic_families=6,
            n_style_cells=9,
            min_validation_rows=10,
            min_class_rows=2,
            session_behavior_used=True,
            project_root=root,
        )

    def test_assignments_are_deterministic_target_independent_and_row_invariant(self):
        first = self.build()
        changed = self.frame.copy()
        changed["target"] = 1 - changed["target"]
        second = self.build(frame=changed)
        pd.testing.assert_frame_equal(first, second)

        permutation = np.random.default_rng(91).permutation(len(self.frame))
        permuted = self.build(
            frame=self.frame.iloc[permutation].reset_index(drop=True),
            embeddings=self.embeddings[permutation],
        )
        pd.testing.assert_frame_equal(first, permuted)
        self.assertEqual(len(first), len(self.frame) * len(ENVIRONMENTS))
        self.assertFalse(
            {"target", "is_correct", "probability", "prediction", "log_loss"}.intersection(
                first.columns
            )
        )

    def test_objective_text_and_embedding_must_be_invariant_per_id(self):
        changed_text = self.frame.copy()
        changed_text.loc[0, "learning_objective"] = "different objective text"
        with self.assertRaisesRegex(ValueError, "text varies"):
            self.build(frame=changed_text)

        changed_embedding = self.embeddings.copy()
        changed_embedding[0, 0] += np.float32(0.01)
        with self.assertRaisesRegex(ValueError, "embeddings vary"):
            self.build(embeddings=changed_embedding)

    def test_all_environments_purge_sessions_and_prohibit_group_leakage(self):
        assignments = self.build()
        diagnostics = self.validate(assignments)
        self.assertEqual(len(diagnostics), len(ENVIRONMENTS) * 3)
        for environment in ENVIRONMENTS:
            subset = assignments[assignments["environment"].eq(environment)]
            for fold in range(3):
                train, held_out = purged_fold_masks(self.frame, subset, fold)
                self.assertFalse(
                    set(self.frame.loc[train, "session_id"]).intersection(
                        set(self.frame.loc[held_out, "session_id"])
                    )
                )

    def test_assignment_contract_rejects_identity_group_and_shape_tampering(self):
        assignments = self.build()
        mutations: list[tuple[str, pd.DataFrame, str]] = []

        changed = assignments.copy()
        changed.loc[0, "session_id"] = "wrong-session"
        mutations.append(("session", changed, "session_id"))

        changed = assignments.copy()
        changed.loc[0, "learning_objective_id"] = "wrong-objective"
        mutations.append(("objective", changed, "learning_objective_id"))

        changed = assignments.copy()
        changed.loc[0, "semantic_family"] += 1
        mutations.append(("family", changed, "identity fields|semantic_family"))

        changed = assignments.copy()
        changed.loc[0, "style_cell"] += 1
        mutations.append(("style", changed, "identity fields|style_cell"))

        changed = assignments.copy()
        changed.loc[0, "joint_cell"] = "family_999__style_999"
        mutations.append(("joint", changed, "identity fields|joint_cell"))

        changed = assignments.copy()
        changed.loc[changed["environment"].eq(CONFIRMATION_ENVIRONMENT), "environment"] = (
            "V_unknown"
        )
        mutations.append(("environment", changed, "known environments"))

        changed = assignments.copy()
        changed.loc[0, "suite"] = "other"
        mutations.append(("suite", changed, "suite='development'"))

        changed = pd.concat([assignments, assignments.iloc[[0]]], ignore_index=True)
        changed.loc[len(changed) - 1, "response_id"] = "extra-response"
        mutations.append(("extra row", changed, "expected exactly"))

        for name, mutation, message in mutations:
            with self.subTest(name=name), self.assertRaisesRegex(
                (ValueError, PermissionError), message
            ):
                self.validate(mutation)

    def test_v_seen_scoring_excludes_ineligible_rows_but_purges_their_sessions(self):
        frame = self.frame.copy()
        embeddings = self.embeddings.copy()
        frame.loc[0, "learning_objective_id"] = "singleton-objective"
        frame.loc[0, "learning_objective"] = "singleton objective"
        embeddings[0] = np.linspace(-1.0, 1.0, embeddings.shape[1], dtype=np.float32)
        assignments = self.build(frame=frame, embeddings=embeddings)
        seen = assignments.loc[assignments["environment"].eq("V_seen")].copy()
        singleton = seen.loc[seen["response_id"].eq(frame.loc[0, "response_id"])].iloc[0]
        self.assertFalse(bool(singleton["evaluation_eligible"]))
        fold = int(singleton["fold"])

        train, held_out = purged_fold_masks(frame, seen, fold)
        scoring = evaluation_mask_for_fold(frame, seen, fold)
        row = int(frame.index[frame["response_id"].eq(singleton["response_id"])][0])
        self.assertTrue(bool(held_out[row]))
        self.assertFalse(bool(scoring[row]))
        self.assertFalse(bool(train[row]))
        held_out_sessions = set(frame.loc[held_out, "session_id"].astype(str))
        self.assertFalse(frame.loc[train, "session_id"].astype(str).isin(held_out_sessions).any())

        diagnostics = self.validate(assignments, frame=frame)
        summary = next(
            item
            for item in diagnostics
            if item["environment"] == "V_seen" and item["fold"] == fold
        )
        self.assertEqual(summary["held_out_rows"], int(held_out.sum()))
        self.assertEqual(summary["validation_rows"], int(scoring.sum()))
        self.assertGreaterEqual(summary["ineligible_rows_excluded_from_scoring"], 1)

    def test_class_adequacy_failure_is_detected_on_scoring_mask(self):
        assignments = self.build()
        broken = self.frame.copy()
        seen = assignments[assignments["environment"].eq("V_seen")]
        score = evaluation_mask_for_fold(broken, seen, 0)
        broken.loc[score, "target"] = 0
        with self.assertRaisesRegex(ValueError, "class adequacy"):
            self.validate(assignments, frame=broken)

    def test_v_joint_is_documented_as_combination_not_marginal_exclusion(self):
        definition = ENVIRONMENT_DEFINITIONS["V_joint"]
        self.assertEqual(definition["display_name"], "interaction-combination holdout")
        self.assertIn("exact family/style combination", definition["guarantee"])
        self.assertEqual(len(definition["non_guarantees"]), 2)

    def test_sealed_final_suite_is_outside_build_and_validation_apis(self):
        with self.assertRaisesRegex(PermissionError, "outside this API"):
            self.build(suite="V_final")
        with self.assertRaisesRegex(PermissionError, "development seed only"):
            build_validation_environments(
                self.frame,
                self.embeddings,
                seed=DEVELOPMENT_SEED + 1,
                suite_name=DEVELOPMENT_SUITE,
                n_splits=3,
                n_semantic_families=6,
                n_style_cells=9,
                session_behavior=self.behavior,
            )

        assignments = self.build()
        assignments["suite"] = "V_final"
        with self.assertRaisesRegex(PermissionError, "suite='development'"):
            self.validate(assignments)

    def test_manifest_versions_full_protocol_and_discloses_no_final_artifact(self):
        assignments = self.build()
        diagnostics = self.validate(
            assignments, diagnostic_environments=SELECTION_ENVIRONMENTS
        )
        manifest = self._manifest(assignments, diagnostics)
        development = manifest["development"]
        self.assertEqual(manifest["protocol_version"], PROTOCOL_VERSION)
        self.assertEqual(
            development["aggregate_assignment_sha256"], assignment_sha256(assignments)
        )
        self.assertEqual(
            development["selection"]["environments"], list(SELECTION_ENVIRONMENTS)
        )
        self.assertEqual(
            development["joint_confirmation"]["environments"],
            [CONFIRMATION_ENVIRONMENT],
        )
        final_policy = manifest["sealed_final_holdout"]
        self.assertTrue(final_policy["managed_outside_developer_api"])
        self.assertFalse(final_policy["normal_developer_api_access"])
        self.assertFalse(
            {"assignment_path", "assignment_sha256", "seed"}.intersection(final_policy)
        )

        protocol = manifest["protocol"]
        self.assertEqual(protocol["passed_parameters"]["n_splits"], 3)
        self.assertEqual(protocol["seeds"]["development_base"], DEVELOPMENT_SEED)
        self.assertIn("objective_family_clustering", protocol["algorithms"])
        self.assertIn("dependencies", protocol["provenance"])
        self.assertIn("code_sha256", protocol["provenance"])
        self.assertEqual(manifest["provenance"], protocol["provenance"])
        self.assertEqual(manifest["protocol_sha256"], _canonical_sha256(protocol))

    def _write_split_artifacts(self, root: Path, assignments: pd.DataFrame):
        selection = assignments.loc[
            assignments["environment"].isin(SELECTION_ENVIRONMENTS)
        ].reset_index(drop=True)
        joint = assignments.loc[
            assignments["environment"].eq(CONFIRMATION_ENVIRONMENT)
        ].reset_index(drop=True)
        selection_path = root / SELECTION_ASSIGNMENT_PATH
        joint_path = root / JOINT_CONFIRMATION_ASSIGNMENT_PATH
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        selection.to_parquet(selection_path, index=False)
        joint.to_parquet(joint_path, index=False)
        manifest = self._manifest(assignments, root=root)
        manifest["development"]["selection"]["file_sha256"] = validation_module.sha256_file(
            selection_path
        )
        manifest["development"]["joint_confirmation"]["file_sha256"] = (
            validation_module.sha256_file(joint_path)
        )
        manifest_path = root / "data_cache" / "validation_environments.manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return selection, joint, manifest, manifest_path

    def test_selection_loader_cannot_load_or_expose_joint_before_lock(self):
        assignments = self.build()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, _, _, _ = self._write_split_artifacts(root, assignments)
            paths = SimpleNamespace(root=root, cache_dir=root / "data_cache")
            reads: list[str] = []
            real_read = pd.read_parquet

            def guarded_read(path, *args, **kwargs):
                reads.append(str(path))
                if "joint_confirmation" in str(path):
                    raise AssertionError("normal loader crossed the confirmation boundary")
                if "final" in str(path).lower():
                    raise AssertionError("normal loader crossed the final boundary")
                return real_read(path, *args, **kwargs)

            with patch.object(validation_module, "discover_project_paths", return_value=paths), patch.object(
                validation_module.pd, "read_parquet", side_effect=guarded_read
            ):
                loaded, sanitized = load_development_selection_assignments(root)
            pd.testing.assert_frame_equal(loaded, selection)
            self.assertEqual(len(reads), 1)
            self.assertNotIn("joint_confirmation", sanitized["development"])
            self.assertNotIn("final", reads[0].lower())

    def test_joint_loader_requires_valid_locked_candidate_sentinel(self):
        assignments = self.build()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, joint, manifest, _ = self._write_split_artifacts(root, assignments)
            paths = SimpleNamespace(root=root, cache_dir=root / "data_cache")
            invalid_path = root / "invalid-lock.json"
            invalid_path.write_text("{}", encoding="utf-8")
            with patch.object(validation_module, "discover_project_paths", return_value=paths):
                with self.assertRaisesRegex(ValueError, "hash verification"):
                    load_joint_confirmation_assignments(
                        root, locked_candidate_sentinel=invalid_path
                    )

            payload = {
                "candidate": "candidate-a",
                "calibration": "raw",
                "development_assignment_sha256": manifest["development"][
                    "aggregate_assignment_sha256"
                ],
                "selection_environments": list(SELECTION_ENVIRONMENTS),
                "confirmation_environment": CONFIRMATION_ENVIRONMENT,
                "V_final_accessed": False,
            }
            sentinel = dict(payload)
            sentinel["lock_sha256"] = _canonical_sha256(payload)
            sentinel_path = root / "locked-candidate.json"
            sentinel_path.write_text(json.dumps(sentinel), encoding="utf-8")
            with patch.object(validation_module, "discover_project_paths", return_value=paths):
                loaded, _ = load_joint_confirmation_assignments(
                    root, locked_candidate_sentinel=sentinel_path
                )
            pd.testing.assert_frame_equal(loaded, joint)

    def test_build_never_reads_final_and_requires_versioned_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "data_cache"
            cache.mkdir(parents=True)
            modeling_path = cache / "modeling_base.parquet"
            embeddings_path = cache / "bge_small_objective_256.npy"
            behavior_path = cache / "session_behavior.parquet"
            self.frame.to_parquet(modeling_path, index=False)
            np.save(embeddings_path, self.embeddings)
            self.behavior.to_parquet(behavior_path, index=False)
            final_trap = cache / "validation_environments_final.parquet"
            legacy_combined = cache / "validation_environments_development.parquet"
            paths = SimpleNamespace(root=root, cache_dir=cache)
            real_read = pd.read_parquet
            reads: list[str] = []

            def guarded_read(path, *args, **kwargs):
                reads.append(str(path))
                if Path(path) == final_trap:
                    raise AssertionError("development build attempted to load V_final")
                return real_read(path, *args, **kwargs)

            kwargs = {
                "n_splits": 3,
                "n_semantic_families": 6,
                "n_style_cells": 9,
                "min_validation_rows": 10,
                "min_class_rows": 2,
            }
            with patch.object(validation_module, "discover_project_paths", return_value=paths), patch.object(
                validation_module.pd, "read_parquet", side_effect=guarded_read
            ):
                result = run_validation_environment_build(root, **kwargs)
            self.assertFalse(result["V_final_accessed"])
            self.assertNotIn("final_path", result)
            self.assertNotIn("joint_confirmation_path", result)
            self.assertTrue((root / SELECTION_ASSIGNMENT_PATH).exists())
            self.assertTrue((root / JOINT_CONFIRMATION_ASSIGNMENT_PATH).exists())
            self.assertNotIn(str(final_trap), reads)

            frozen_paths = (
                root / SELECTION_ASSIGNMENT_PATH,
                root / JOINT_CONFIRMATION_ASSIGNMENT_PATH,
                cache / "validation_environments.manifest.json",
            )
            before = {
                path: (validation_module.sha256_file(path), path.stat().st_mtime_ns)
                for path in frozen_paths
            }
            with patch.object(
                validation_module, "discover_project_paths", return_value=paths
            ):
                repeated = run_validation_environment_build(root, **kwargs)
            after = {
                path: (validation_module.sha256_file(path), path.stat().st_mtime_ns)
                for path in frozen_paths
            }
            self.assertEqual(repeated["protocol_sha256"], result["protocol_sha256"])
            self.assertEqual(after, before)

            # Legacy generated files make the normal boundary fail closed.  An
            # explicit new protocol version removes their exact verified paths
            # without ever deserializing the final artifact.
            final_trap.write_bytes(b"not a parquet file")
            legacy_combined.write_bytes(b"not a parquet file")
            with patch.object(
                validation_module, "discover_project_paths", return_value=paths
            ):
                with self.assertRaisesRegex(RuntimeError, "legacy.*must be removed"):
                    run_validation_environment_build(root, **kwargs)
            reads.clear()
            with patch.object(
                validation_module, "discover_project_paths", return_value=paths
            ), patch.object(
                validation_module.pd, "read_parquet", side_effect=guarded_read
            ):
                migrated = run_validation_environment_build(
                    root,
                    **kwargs,
                    protocol_version_override="2026-07-20-test-v5",
                )
            self.assertEqual(migrated["protocol_version"], "2026-07-20-test-v5")
            self.assertFalse(final_trap.exists())
            self.assertFalse(legacy_combined.exists())
            self.assertNotIn(str(final_trap), reads)
            migration_manifest = json.loads(
                (cache / "validation_environments.manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                set(migration_manifest["legacy_artifact_cleanup"]["removed"]),
                {
                    "data_cache/validation_environments_development.parquet",
                    "data_cache/validation_environments_final.parquet",
                },
            )
            self.assertTrue(
                migration_manifest["legacy_artifact_cleanup"]["legacy_artifacts_absent"]
            )

            changed = self.frame.copy()
            changed.loc[0, "target"] = 1 - int(changed.loc[0, "target"])
            changed.to_parquet(modeling_path, index=False)
            with patch.object(validation_module, "discover_project_paths", return_value=paths):
                with self.assertRaisesRegex(RuntimeError, "source hashes changed"):
                    run_validation_environment_build(root, **kwargs)
                with self.assertRaisesRegex(RuntimeError, "new version"):
                    run_validation_environment_build(
                        root,
                        **kwargs,
                        protocol_version_override="2026-07-20-test-v5",
                    )
                rebuilt = run_validation_environment_build(
                    root,
                    **kwargs,
                    protocol_version_override="2026-07-20-test-v6",
                )
            self.assertEqual(rebuilt["protocol_version"], "2026-07-20-test-v6")
            manifest = json.loads(
                (cache / "validation_environments.manifest.json").read_text(encoding="utf-8")
            )
            self.assertIn("supersedes", manifest)


if __name__ == "__main__":
    unittest.main()
