# E830 preregistration: official curriculum-progression objective prior

Status at freeze: target-free implementation only. No E830 target-free score,
competition outcome, `V_seen`, `V_objective`, `V_style`, `V_joint`, or
`V_final` has been read.

## Independent mechanism and reason to try it

Raw v0.5 remains weakest when the objective ID is unseen and the objective is
poorly covered by retrieved transcript language. Objective difficulty is the
strongest simple signal in the project, but the rejected BGE nearest-objective
prior transfers difficulty only by semantic proximity. It has no explicit
model of pedagogical sequence: for example, same-denominator fractions precede
different-denominator fractions, and place-value limits progress from 1,000 to
10,000, 1,000,000, and 10,000,000.

E830 is a structured, target-free curriculum prior built from the statutory
English mathematics programme of study. It is not another encoder,
contrastive-retrieval, tutor-move, student-state, timing, numeric-talk,
objective-token-cross, objective-alignment, or outcome-labelled k-nearest
branch. No rejected checkpoint or prediction is consumed.

## Frozen lineage

- Source: Department for Education, *National curriculum in England:
  mathematics programmes of study*, updated 28 September 2021.
- URL:
  `https://www.gov.uk/government/publications/national-curriculum-in-england-mathematics-programmes-of-study/national-curriculum-in-england-mathematics-programmes-of-study`
- Licence: Open Government Licence v3.0.
- Raw HTML path:
  `Datasets/UK_National_Curriculum/national_curriculum_mathematics_programmes_of_study.html`.
- Raw SHA-256:
  `d8499da572edb7f1dbf9c42284dae265e121a8215692cff204c3a5d73cd6c490`.
- Competition input is restricted to the three non-outcome columns
  `response_id`, `learning_objective_id`, and `learning_objective` from the
  immutable training feature CSV. The label CSV is not opened.
- Existing BGE-base objective embeddings may be read only for a target-free
  redundancy audit. No target, fitted outcome model, rejected prediction, or
  validation environment is read.

## Frozen feature construction

Parse exactly eight stages: years 1-6, key stage 3, and key stage 4. Preserve
each stage's heading-3 domain and all unique paragraph/list statements of at
least 20 normalized characters.

Fit two unsupervised TF-IDF spaces jointly on the official statements and the
398 competition objective strings:

- word unigrams/bigrams, English stop words, sublinear TF, L2 norm;
- character-within-word 3-5 grams, minimum document frequency 2, sublinear TF,
  L2 norm.

Similarity is fixed at 55% word plus 45% character cosine. For each stage use
the maximum statement similarity. Convert the eight scores to a posterior with
temperature `0.08`; emit all scores, probabilities, expected stage, best stage,
maximum similarity, top-two margin, entropy, and the single nearest official
statement/domain. There is no learned competition-outcome parameter.

## Frozen target-free evaluation

1. Leave each official statement out of its own retrieval candidates and
   recover its stage from every other statement.
2. Evaluate 24 manually frozen competition-objective anchors spanning early
   primary through KS4. Anchor ranges are constants in the committed source and
   cannot be changed after a score.
3. Measure coverage of all 398 objectives.
4. Predict expected curriculum stage from the untouched row-aligned BGE-base
   objective cache with five-fold shuffled objective-level ridge regression,
   fixed alpha `100`; excessive predictability means E830 is redundant.
5. Run a deterministic synthetic proper-score benchmark in which correctness
   probability increases with the true official stage. Compare length/digit
   controls with the same controls plus leave-one-out curriculum features.

Every clause must pass:

- all eight stages and at least 15 statements per stage;
- official leave-one-out stage Spearman at least `0.30`, mean absolute expected
  stage error at most `2.00`, within-one best-stage accuracy at least `0.45`,
  and exact best-stage accuracy at least `0.18`;
- competition-objective median maximum similarity at least `0.20` and 10th
  percentile at least `0.10`;
- at least 60% of anchors in their frozen range, at least 80% within one stage,
  and anchor midpoint/expected-stage Spearman at least `0.70`;
- at least six distinct best stages used across competition objectives;
- BGE fixed-ridge cross-validated R-squared below `0.85`;
- synthetic log-loss gain at least `0.01`.

Any failed clause rejects exact E830 before competition outcomes. There is no
rescue, threshold change, alternate corpus, parser, weighting, temperature,
anchor set, smoother, encoder, prompt, seed, or hyperparameter sweep.

## Conditional competition continuation

Passing this source gate is necessary but not sufficient. Before any
competition outcome is opened, a separate committed preregistration must bind
the exact feature-cache and report hashes, existing immutable 4,096-row
component lineage, frozen folds/environments, fold-local estimator,
regularization, seed, blend weights, proper-score gates, fold bound, and both
bootstrap rules.

That screen must require at least `0.0016` equal-environment macro log-loss
gain over raw v0.5, improvement in all three selection environments,
non-regression of AUROC/Brier/ECE, a bounded worst fold, and at least `0.95`
positive-gain support in both session and semantic-family bootstraps.
`V_joint` remains confirmation-only and `V_final` remains sealed. No ZIP,
upload, or submission is authorized by this preregistration.
