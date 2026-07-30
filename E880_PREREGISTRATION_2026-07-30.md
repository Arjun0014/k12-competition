# E880 preregistration: ALTER-Math problem-solving-success transfer

Date frozen: 2026-07-30 (Asia/Kolkata)

## Question and independence

Can a direction learned from real K-12 mathematics discussions whose entire
problem-solving outcome is labeled improve raw v0.5's prediction of whether the
student answers the next conceptual assessment correctly?

The source is the public
`ALTER-Math/annotated-math-tutoring-dataset`, revision
`efaaa64e2dd08c67d9ebef3ab3141d7de28c5d40`, created 2026-07-27. Its retained
raw CSV is 23,995,917 bytes with SHA-256
`50370bde7e0bb4ed691ca3a1bcf7533f7494966b63f3d6086224c66136d7d63b`.
It contains 2,318 real tutoring threads annotated for whole-discussion
problem-solving success. The dataset card declares Apache-2.0.

This is not a rescue or rename of a closed branch:

- P880/P881 classified individual tutor moves with zero-shot LLMs; E880 uses a
  real thread-level success outcome and a frozen linear transfer direction.
- E760 modeled ordered uptake and overreliance inside competition transcripts;
  E880 does not hand-code uptake, copying, or tutor moves.
- E770 used hand-authored competition tutor-move state features; E880 uses only
  a learned external outcome direction.
- E840 predicted later StudyChat exam scores from longitudinal behavior counts;
  E880 is transcript-level, mathematics-specific, and has a binary discussion
  resolution outcome.
- E580/E590 transferred SRA or process-correctness labels from different tasks.
  E880's source is newly released, real K-12 tutoring in the same broad
  mathematics-discussion domain, and its target is whole-thread success.

The earlier source audits could not have evaluated a dataset created on
2026-07-27. Search of the repository and learning log found no prior ALTER-Math
or LEVI experiment.

## Frozen lineage and preprocessing

Only `C:\Competition\K12\.venv` is authorized. Expected runtime is Python
3.12.8, NumPy 2.5.1, pandas 2.3.3, scikit-learn 1.8.0, PyTorch 2.13.0+cpu, and
Transformers 5.14.1.

One source row is created per `id`. The implementation must verify:

- 24,116 raw rows and 26 exact source columns;
- 2,318 unique `id` values and unique `(id,id2)` pairs;
- `Success` is binary and constant within every `id`;
- the reconstructed `*document*` transcript is constant within every `id`;
- no normalized transcript is duplicated across distinct IDs;
- every session receives exactly one fixed fold
  `int(sha256("E880|" + id)[:16], 16) % 5`.

The raw CSV is never changed. Each document is mapped to BGE text as:

1. fixed task line:
   `Task: represent whether the student's mathematics problem was successfully resolved during the discussion.`
2. the last eight turns, in chronological order, under
   `End of mathematics discussion:`;
3. the first four turns not already selected, in chronological order, under
   `Earlier mathematics discussion:`.

Roles are frozen as `[STUDENT]` for `u*`, `[EXPERT_TUTOR]` for `e*`, and
`[PEER_TUTOR]` for `p*`; unknown prefixes become `[OTHER]`. Each selected turn
is capped at its first 16 whitespace-delimited words. No tutoring-strategy,
knowledge-state, or utterance-level annotation column is a model input.

Encoder: packaged `BAAI/bge-base-en-v1.5`, MIT license, frozen CLS pooling,
normalized 768-dimensional embeddings, maximum length 256, batch size 16, CPU
threads 6. Model/source hashes are checked by the existing BGE verifier.

## Pre-outcome requirements

Before any predictive outcome score:

1. focused unit tests must pass;
2. Ruff must pass on all new Python files;
3. a 32-document synthetic, target-free encoder benchmark must complete with
   correct shape, finite unit-normalized values, projected full-source time
   under 30 minutes, and peak RSS under 8 GiB;
4. this preregistration and the frozen implementation must be committed and
   pushed.

The full 2,318-row encoding run is target-free. Its benchmarked duration is
expected to be below one hour and therefore remains in the foreground without a
scheduler.

## Frozen external estimator and gate

Estimator: five fixed hash folds, unweighted `LogisticRegression(C=0.1,
solver="lbfgs", max_iter=1000, random_state=20260730)` on normalized embeddings.
The comparator for each validation fold is that fold's legal training
prevalence. No grid, seed sweep, calibration, class weighting, or alternate
text view is authorized.

The exact external branch continues only if all clauses pass:

- pooled candidate log loss at most 0.59;
- pooled log-loss gain over the fold-prior comparator at least 0.07;
- pooled AUROC at least 0.75;
- pooled Brier gain at least 0.02;
- pooled macro-F1 at threshold 0.5 at least 0.65;
- pooled ECE-10 at most 0.08;
- positive log-loss gain in all five folds;
- worst-fold log-loss gain at least 0.01;
- 5,000-session paired bootstrap support for positive gain at least 0.99;
- bootstrap 95% lower bound at least 0.03.

Any failure rejects this exact branch. It does not authorize a new `C`, text
view, threshold, calibration, blend, or rescue.

If the external gate passes, one all-source model is fit with the same
estimator. Let `m` be its decision margin. The median and IQR of its margins on
the full external source are frozen into the checkpoint. For any competition
context embedding:

`z = clip((m - external_median) / external_IQR, -4, 4)`

`evidence = tanh(z / 2)`

The target-free competition applicability gate requires finite evidence for all
35,072 rows, evidence standard deviation at least 0.10, evidence IQR at least
0.20, absolute median at most 0.75, and at most 20% of rows with absolute
evidence above 0.95. Any failure rejects E880 without reading competition
outcomes.

## Frozen competition validation

Before competition outcomes are read, the first passing external run's exact
report, checkpoint, embedding-cache, and evidence-cache hashes must be written
to an E880 competition-binding JSON, committed, and pushed.

Raw v0.5 is unchanged:

`p05 = 0.25 * pred_full + 0.25 * pred_role + 0.50 * pred_bge_base`

Exactly three strengths are preregistered:

`candidate_gamma = sigmoid(logit(clip(p05)) + gamma * evidence)`

for `gamma in (0.10, 0.20, 0.30)`. This is fixed external evidence in log-odds
space, not a competition-fitted stacker or calibration model. No competition or
test aggregate defines the direction, centering, scale, or strength.

One and only one development execution may evaluate these three candidates on
already authorized `V_seen`, `V_objective`, and `V_style`. For every gamma, use
equal-fold metrics within each environment, then equal-environment selection.
Each gamma receives a predeclared 5,000-replicate paired bootstrap by session
and by semantic family. A gamma passes only if:

- equal-environment mean log-loss gain over raw v0.5 is at least 0.0016;
- all three environments improve log loss;
- worst environment has no log-loss regression;
- mean AUROC does not regress;
- mean Brier score does not regress;
- mean ECE-10 does not regress;
- worst environment-fold log-loss regression is at most 0.0005;
- session and semantic-family bootstrap support are both at least 0.95;
- both bootstrap 95% lower bounds are positive.

If multiple strengths pass, select the one with the largest mean log-loss gain;
ties within `1e-12` choose the smaller gamma. If none pass, E880 is rejected
literally.

Only a development-passing strength may access confirmation-only `V_joint`.
It must have non-regressing log loss, AUROC, Brier, and ECE-10, worst-fold
log-loss regression at most 0.0005, and at least 0.90 session and semantic-family
bootstrap support with positive lower bounds. Failure rejects E880. `V_final`
remains sealed.

## Packaging and submission boundary

Raw v0.5 and
`submission_builds/final_ensemble_v05_bge_backup.zip` remain unchanged; the
protected ZIP must retain SHA-256
`65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.

A new clearly named ZIP may be built only after every external, applicability,
development, confirmation, reproducibility, parity, format, and backup gate
passes. No competition upload or submission is authorized; the user performs
every submission manually.
