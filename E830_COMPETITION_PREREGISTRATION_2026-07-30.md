# E830 competition preregistration: curriculum-progression objective head

Status at freeze: the target-free source gate passed. No E830 competition
outcome, selection-environment score, `V_joint`, or `V_final` has been read.

## Bound target-free lineage

- Target-free protocol:
  `E830_curriculum_progression_target_free_v1`.
- Feature-cache SHA-256:
  `2391a8f5d807323c2802086e8b8f78d126b0115f4ad71531deaac267c453ec19`.
- Target-free report SHA-256:
  `5b53d3f6fd3d34179c4a2f885d6592538d82d77090a74f1682bdb53d93ca9a2b`.
- The report must say every target-free clause passed and every outcome,
  selection environment, `V_joint`, and `V_final` access flag is false.
- Competition component lineage remains immutable run
  `20260720T075633Z_environment_component_validation`, including its exact
  component hashes and 4,096 response rows per environment as enforced by the
  existing loader.

## Frozen features and estimator

Align the 398 target-free rows many-to-one by `learning_objective_id`. Use
exactly these 20 numeric inputs:

- eight `curriculum_score_*` values;
- eight `curriculum_probability_*` values;
- expected stage, maximum similarity, top-two margin, and entropy.

Within every existing fold, fit `StandardScaler` followed by
`LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000,
random_state=20260730)`. Each training row receives inverse training-objective
frequency weight, normalized to mean one, so the head estimates
cross-objective difficulty rather than allowing frequent objectives to
dominate. No objective ID, nearest-standard text/domain, transcript feature,
raw-v0.5 prediction, rejected-model output, calibration sweep, or
environment-specific parameter enters the head.

The candidate objective prediction is blended over raw v0.5, fixed as
`0.25*pred_full + 0.25*pred_role + 0.50*pred_bge_base`. Evaluate only
curriculum weights `0.10`, `0.20`, and `0.30`. Selection is lowest
equal-environment, equal-fold macro log loss; a tie uses the smaller weight.
No post-hoc weight or rescue is allowed.

## Frozen validation and acceptance gates

Run exactly once on `V_seen`, `V_objective`, and `V_style`, using each
environment's existing five folds and validation eligibility. For every
preregistered weight, report log loss, AUROC, Brier score, ECE, fold deltas,
and 5,000-replicate deterministic positive-gain bootstraps independently over
session and semantic-family groups.

The selected weight must pass every clause:

- equal-environment/equal-fold macro log-loss gain at least `0.0016`;
- all three selection environments improve and none regresses;
- worst fold log-loss regression at most `0.0005`;
- macro AUROC non-regression;
- macro Brier non-regression;
- macro ECE non-regression;
- session bootstrap positive-gain support at least `0.95`;
- semantic-family bootstrap positive-gain support at least `0.95`.

Any failed clause rejects exact E830. Do not try another C, weighting scheme,
feature subset, stage smoother, target transform, seed, blend, calibration,
split, or gate.

## Conditional confirmation and backup gate

Only if every selection clause passes may the already-defined `V_joint` be
opened once at the selected weight. It must independently pass:

- log-loss gain at least `0.0010`;
- worst-fold regression at most `0.0005`;
- AUROC/Brier/ECE non-regression;
- both 5,000-replicate bootstrap supports at least `0.95`.

`V_final` remains sealed. A passed E830 still requires the existing
conservative backup and projected-public gates before any new package. v0.5
and its protected ZIP remain unchanged. No upload or competition submission
is authorized.
