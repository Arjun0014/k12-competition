# E810 preregistration: productive numeric elaboration

Date frozen: 2026-07-29  
Branch: `codex/v04-recovery`  
Parent HEAD before E810 work: `d51681cabbdebf8c63fd5516ec549ea4595e5d9f`  
Status at freeze: target-free discovery passed; no E810 competition outcome has
been read or scored.

## Decision and independence

E810 is authorized for exactly one validation invocation. It implements the
three-feature productive-math-talk reference published by DrivenData on
2026-07-27:

- `n_student_words`
- `numeric_turns_per_word`
- `digit_chars_per_word`

The exact implementation reproduces the organizer's 35,062 nonmissing response
rows, all three published means and standard deviations to six decimals, 107
quiet sessions, and 159 quiet response rows. These exact ratios were not
implemented in the repository or learning log. In particular, the existing
role cache counts numeric tokens but does not count number-bearing student
turns or digit characters with the shared student-word denominator.

The frozen target-free audit passed every clause:

- response coverage `35062 / 35072 = 0.999714`;
- augmented dense-design rank `34` versus `31`, a gain of three;
- cross-fit R-squared from existing v0.5 dense role/session features is
  `0.622637` for numeric turns per word and `0.672253` for digit characters per
  word;
- nearest absolute Spearman correlations are `0.517249` and `0.604735`;
- the fixed synthetic residual benchmark improves log loss by `0.067527`;
- the published fixed numeric score has Spearman correlation `0.270501`,
  `0.395153`, and `0.270433` with raw v0.5 predictions on `V_seen`,
  `V_objective`, and `V_style`.

The feature cache SHA-256 is
`560e82ceb07dc22ef0511da2b92d361f6e52564be997a2364a282e8c6c1eca27`.
The target-free report SHA-256 is
`ed202b92697ca7d11c1115378f1b8825761901d9ee60092526ff56c5a445afac`.
The discovery runtime was 24.129 seconds under the project `.venv`. It accessed
no competition outcome, `V_joint`, or `V_final`.

E810 is independent of rejected correctness, external-transfer, tutor-move,
uptake, timing, challenge-state, session-mastery, coherence, objective
interaction, and generic calibration branches. It does not relabel or rescue
any rejected experiment. The organizer's one-split reference improved a
constant baseline from 0.6089 to 0.6054, while the target-free correlation audit
shows residual variation relative to v0.5. That provides a plausible, not
guaranteed, path to the required robust 0.0016 log-loss gain.

## Frozen data lineage and splits

- Numeric features come only from
  `data_cache/utterances.parquet`, itself derived from the authorized training
  transcripts.
- Identity alignment uses `response_id` and `session_id` only.
- Outcome validation uses the immutable component OOF files from run
  `20260720T075633Z_environment_component_validation`.
- Selection environments are exactly `V_seen`, `V_objective`, and `V_style`.
- Each environment uses its existing five folds. For every fold, all rows from
  every validation session are excluded from model training by `_fold_masks`.
- `V_joint` is confirmation-only and may be opened only if every selection gate
  passes. `V_final` remains sealed in all cases.

## Frozen feature extraction

- Student word regex: `[a-z0-9]+(?:'[a-z]+)?`, case insensitive.
- Digit regex: `\d`.
- `n_student_words`: total regex words across student turns.
- `numeric_turns_per_word`: student turns containing at least one digit divided
  by student words.
- `digit_chars_per_word`: digit characters across student turns divided by
  student words.
- Sessions with fewer than 100 student words are excluded from fitting. Their
  validation prediction is the corresponding full fold-training target prior.
- No number-word expansion, tutor feature, timing feature, learning-objective
  feature, post-hoc threshold, or alternate feature definition is authorized.

## Frozen estimator and blends

For each environment and fold:

- fit `StandardScaler` on the three eligible training features only;
- fit scikit-learn 1.8.0 `LogisticRegression` with `C=1.0`, solver `lbfgs`,
  `max_iter=1000`, and seed `20260729`;
- produce a leakage-safe numeric-head probability for the validation fold;
- define raw v0.5 exactly as `0.25*pred_full + 0.25*pred_role +
  0.50*pred_bge_base`;
- evaluate only numeric-head blend weights `0.10`, `0.20`, and `0.30`.

Select the lowest equal-environment, equal-fold macro log loss. An exact tie
uses the smaller numeric-head weight. No post-hoc sweep, calibration, coefficient
constraint, feature removal, threshold adjustment, or rescue is authorized.

## Frozen selection gates

The selected candidate passes only if every clause is true:

1. mean macro log-loss gain over raw v0.5 is at least `0.0016`;
2. all three selection environments improve;
3. no selection environment regresses in macro log loss;
4. the worst individual fold log-loss regression is at most `0.0005`;
5. macro AUROC does not regress;
6. macro Brier score does not regress;
7. macro 10-bin ECE does not regress;
8. paired 5,000-replicate session-cluster bootstrap support for positive
   log-loss gain is at least `0.95`;
9. paired 5,000-replicate semantic-family-cluster bootstrap support is at least
   `0.95`.

Any failed clause rejects E810 literally, with no rescue.

## Frozen V_joint confirmation

Only after all selection gates pass, evaluate the already selected weight once
on `V_joint`. Confirmation requires:

- macro log-loss gain at least `0.0010`;
- worst-fold log-loss regression at most `0.0005`;
- AUROC, Brier score, and 10-bin ECE non-regression;
- session and semantic-family bootstrap support each at least `0.95` using
  5,000 paired replicates.

A failed confirmation rejects E810. `V_joint` cannot choose a weight or modify
the model. `V_final` remains sealed.

## Packaging and submission boundary

No cache beyond the authorized target-free cache and no submission artifact is
built before E810 passes all gates. If it passes, a production implementation
and ZIP may be built only after the frozen backup/conservative top-five gate is
applied. The protected v0.5 ZIP must remain byte-identical at SHA-256
`65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.
Codex must never upload or submit any artifact; the participant makes every
competition submission manually.
