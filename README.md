# Trace the Ace

Reproducible training and code-submission project for the K-12 AI Infrastructure **Trace the Ace** competition.

The model predicts the probability that a student answers an aligned follow-up question correctly after a tutoring session. Model selection uses binary log loss and session-grouped validation.

## Project status

- Competition, rules, data, runtime, and external-license audit completed.
- Raw competition data is preserved in place and excluded from Git.
- Foundation pipeline builds source manifests, transcript caches, and frozen grouped folds.
- Baseline and hard-validation pipelines produce session-grouped and objective-disjoint OOF predictions, metrics, calibration diagnostics, and reusable feature caches.
- The promoted v0.2 offline runner blends full-transcript hashing, tutor/student role and behavior features, and MIT-licensed BGE-small semantic interactions.
- `submission_builds/final_ensemble_v02.zip` passed all local/runtime gates and scored public log loss `0.6081` / AUROC `0.6147`, moving the entry to rank 16 at the time of submission.
- v0.4 completed successfully but regressed to public log loss `0.6158`; it is rejected, and v0.2 remains the confirmed champion.
- Current development follows a gated forensic, validation-redesign, and recovery process; active experiment notes remain private until the competition closes.

A dated local research log preserves decisions, negative results, and evidence for the final competition write-up without publishing active strategy or competition-derived details.

## Expected raw inputs

The loaders discover the randomized file suffixes automatically and require exactly one of each:

```text
train_features_*.csv
train_labels_*.csv
submission_format_*.csv       # one 100-row smoke format and one full format
Train transcripts/*.csv
```

Do not rename, edit, or commit these files.

## Local setup

Python 3.12 is the target runtime and is selected by `.python-version`. The serialized sparse-model artifact must be built with scikit-learn 1.8.0 to match the official competition container.

```powershell
uv sync
uv run python scripts/build_foundation.py --project-root .
uv run python scripts/train_baselines.py --project-root .
uv run python scripts/train_sparse_model.py --project-root .
uv run python scripts/build_role_cache.py --project-root .
uv run python scripts/build_objective_retrieval.py --project-root .
uv run python scripts/build_semantic_cache.py --project-root .
uv run python scripts/build_multiview_cache.py --project-root .
uv run python scripts/run_multiview_validation.py --project-root . --quick
uv run python scripts/run_semantic_knn_validation.py --project-root . --quick
uv run python scripts/build_encoder_upgrade_cache.py --project-root .
uv run python scripts/run_encoder_upgrade_validation.py --project-root . --quick
uv run python scripts/run_session_weight_validation.py --project-root . --quick
uv run python scripts/build_long_context_cache.py --project-root .
uv run python scripts/run_long_context_validation.py --project-root . --quick
uv run python scripts/build_nli_cross_encoder_cache.py --project-root .
uv run python scripts/run_nli_cross_encoder_validation.py --project-root . --quick
uv run python scripts/run_qwen_pilot.py build --project-root .
uv run python scripts/run_qwen_pilot.py validate --project-root .
uv run python scripts/run_nbsvm_validation.py --project-root . --quick
uv run python scripts/build_ordered_features.py --project-root .
uv run python scripts/run_feedback_hash_validation.py --project-root . --quick
uv run python scripts/audit_candidate.py --help
uv run python scripts/run_dense_tree_validation.py --project-root . --quick
uv run python scripts/run_robust_validation.py --project-root .
uv run python scripts/train_final_ensemble.py --project-root .
uv run python scripts/train_feedback_upgrade.py --project-root .
uv run python scripts/build_submission.py --project-root .
uv run python scripts/verify_project.py --project-root .
uv run python -m unittest discover -s tests -v
```

Generated caches, trained artifacts, run outputs, and ZIP files are intentionally ignored by Git.

## License

Project source code is released under the [MIT License](LICENSE). Competition data, transcripts, downloaded model weights, and other third-party resources are not included and retain their respective terms.

## Reproducibility rules

- Use the frozen fold assignments in `data_cache/response_folds.parquet`.
- Split by `session_id`, never by response row.
- Fit every learned preprocessing component using only training folds.
- Do not use counts or aggregates from other test samples during inference.
- Register the exact source, version, license, and hash of every external resource.
- Never log test transcript text, objective text, or test-derived summaries.
