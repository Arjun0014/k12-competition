# E910 preregistration: ALTER-Math learner knowledge-state transfer

Date frozen: 2026-07-30 (Asia/Kolkata)

## Question and independence

Can three human-annotated, mathematics-specific learner knowledge processes
provide a calibrated trajectory signal that complements raw v0.5's broad
semantic representation?

E910 predicts only ALTER-Math's utterance-level `computational skill`,
`conceptual knowledge`, and `strategic knowledge` annotations. It never reads
the source `Success` column as input or target and consumes no E880 checkpoint,
margin, probability, calibration, or whole-thread representation.

This is distinct from closed branches:

- E680 predicted a single CIMA `Guess` action from a complete dialogue history;
  E910 predicts three fine-grained human knowledge-process annotations from the
  current student utterance.
- E770 used ten hand-authored binary lexical student states plus tutor moves;
  E910 uses no E770 rule, feature, move, interaction, cache, or outcome and
  learns continuous probabilities from external human annotations.
- E220's broad reasoning/uncertainty aggregates and E760's lexical uptake do not
  distinguish computational, conceptual, and strategic knowledge processes.
- E880's exact whole-thread success direction is closed. E910 does not change
  E880's C, text view, calibration, threshold, or source outcome.

The overlap risk is acknowledged: E770 included broad `expressed_reasoning` and
`objective_or_math_evidence` flags. E910 is authorized only if its external
proper-score gate shows reliable discrimination of the finer human taxonomy.
No E770 rescue is allowed.

The public source is pinned to Hugging Face revision
`efaaa64e2dd08c67d9ebef3ab3141d7de28c5d40`, Apache-2.0. The unchanged raw CSV
has 23,995,917 bytes and SHA-256
`50370bde7e0bb4ed691ca3a1bcf7533f7494966b63f3d6086224c66136d7d63b`.

## Frozen source and leakage controls

Only `C:\Competition\K12\.venv` is authorized: Python 3.12.8, NumPy 2.5.1,
pandas 2.3.3, and scikit-learn 1.8.0.

The implementation reads only `id`, `id2`, `content`, and the three frozen
knowledge labels. The `Success` column is header-verified but excluded by
`usecols`. The current utterance is the text before `*document*`; the replicated
full document after that delimiter is never model input. Only roles beginning
with `u` are retained.

Normalize a student utterance only for leakage detection by lowercasing,
replacing non-alphanumeric runs with spaces, and collapsing whitespace. Remove
every row whose normalized text appears in more than one source session. This
leaves exactly 8,573 rows in 2,304 sessions, with positive counts
252/486/130. No text normalization is applied to classifier input.

The ordered source-input SHA-256 is
`02da071a600d751e10644eb0ae9048a89cc855ad899ddd8dafb0c91d417d2c0a`;
the canonical Parquet SHA-256 is
`72cc15248e0446593e2419b848a722cd6d2764f901f6af205447b12962ed06d1`;
and the source-audit JSON SHA-256 is
`ee871ac8cff5809e00639a926af7b8cce335ace207b96c6eedd205800cf4dbcf`.

Assign the complete session to
`int(sha256("E910|" + session_id)[:16], 16) % 5`. Frozen fold
row/session/positive counts are encoded in the implementation. No session or
cross-session exact normalized utterance may cross a fold.

`linguistic knowledge` is excluded because it has only 21 positives.
`affective control` is excluded because it has only 93 positives and overlaps
the closed uncertainty/affect family. Neither exclusion may be reversed after
scoring.

## Frozen external representation, estimator, and gate

Each fold fits its vocabulary on training text only. The feature union is:

- word TF-IDF `(1,2)` n-grams, `min_df=2`, at most 60,000 features,
  sublinear TF, float32;
- `char_wb` TF-IDF `(3,5)` n-grams, `min_df=2`, at most 100,000 features,
  sublinear TF, float32;
- equal feature-union weights, no stopword list.

Each label uses an independent, unweighted
`LogisticRegression(C=1.0, l1_ratio=0, solver="liblinear",
max_iter=2000, random_state=20260730)`. The legal comparator is the label's
training-fold prevalence. No class weight, calibration, threshold, C,
vocabulary, label mixture, or alternate text view is authorized.

Exactly one five-fold external score may run after tests, the synthetic
target-free benchmark, and commit/push. It passes only if every clause holds:

1. every label AUROC is at least 0.70;
2. every label average precision is at least three times its prevalence;
3. macro average-precision gain over prevalence is at least 0.08;
4. every label log-loss gain is at least 0.005;
5. macro log-loss gain is at least 0.015;
6. every label Brier gain is positive and macro Brier gain is at least 0.003;
7. every label ECE-10 is at most 0.03;
8. at least 13 of 15 label-fold log-loss gains are positive;
9. worst label-fold log-loss gain is at least -0.003;
10. a 5,000-replicate session-cluster bootstrap has positive-gain support at
    least 0.99 and 95% lower bound at least 0.0075.

Thresholded F1 is intentionally not a gate: the only downstream quantities are
continuous probabilities, and the source prevalences are 1.5%-5.7%. Any failed
clause rejects exact E910 without rescue.

## Frozen conditional competition features

Only a fully passing external report may authorize an all-source model with the
same vectorizer and estimators. For each already-authorized competition
objective context, use only chronological `[STUDENT]` lines. For each of the
three labels, emit exactly four values:

1. mean probability over all student turns;
2. mean over the first `ceil(n/2)` turns;
3. mean over the remaining turns, or the final turn when `n=1`;
4. late mean minus early mean.

With zero student turns, the three means equal the all-source label prevalence
and the difference is zero. These are exactly 12 features. No threshold,
outcome-aware event definition, source-success signal, tutor-turn feature,
objective text feature, maximum, count, or alternate phase split is allowed.

The target-free applicability gate requires all 35,072 response rows aligned
exactly once, finite features, every feature nonconstant, and at least 99%
student-turn coverage. Every probability feature must lie in `[0,1]`; every
difference in `[-1,1]`. At least six of the twelve features must have standard
deviation at least 0.01. Any failure rejects E910 before competition outcomes.

## Frozen competition validation

Before competition scoring, bind the passing external report, OOF, all-source
model, metadata, and competition-feature hashes in a committed and pushed
manifest. Raw v0.5 remains

`p05 = 0.25 * pred_full + 0.25 * pred_role + 0.50 * pred_bge_base`.

On each of the five existing session-disjoint folds, fit an unweighted
`LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000,
random_state=20260730)` on fold-local standardized 12-feature inputs. Evaluate
only probability blends 10%, 20%, and 30% with raw v0.5. One development
execution may score all three on V_seen, V_objective, and V_style. Selection is
minimum equal-fold, equal-environment mean log loss; ties within `1e-12` choose
the smaller weight.

A weight passes only if:

- equal-environment log-loss gain is at least 0.0016;
- all three environments improve;
- mean AUROC, Brier, and ECE-10 do not regress;
- worst environment-fold log-loss regression is at most 0.0005;
- 5,000-replicate paired session and semantic-family bootstraps both have
  positive-gain support at least 0.95 and positive 95% lower bounds.

Any failure rejects exact E910 without a different C, phase, label, feature,
weight, calibration, or post-hoc blend.

Only a passing development weight may access confirmation-only V_joint, where
log loss, AUROC, Brier, and ECE-10 must not regress, worst-fold regression must
be at most 0.0005, and both bootstrap supports must be at least 0.90 with
positive lower bounds. V_final remains sealed.

## Resource, packaging, and submission boundary

Before the external score, focused tests and Ruff must pass, and a 512-row
synthetic-label benchmark must produce finite probabilities, projected
five-fold runtime below one hour, and RSS below 8 GiB. Source annotations and
`Success` are not benchmark labels.

The frozen benchmark passed in `0.0659548` seconds with a `512 x 5,526` sparse
matrix, 41,445 nonzeros, finite `512 x 3` probabilities, projected external
runtime `5.5217822` seconds, and observed RSS 188,792,832 bytes. Benchmark
SHA-256 is
`a5cb8209d355351aa50e97205bb3ec2ab95af75c4308472371581187cda7caf1`.

A clearly named ZIP may be built only after every external, applicability,
development, confirmation, reproducibility, parity, format, and backup gate
passes. The protected v0.5 ZIP remains unchanged with SHA-256
`65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.
Codex will never upload or submit; the participant submits manually.
