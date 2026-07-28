# E760 preregistration — directional teacher uptake

Date frozen: 2026-07-28

Protocol: `E760_conversational_uptake_transfer_v1`

Predecessor HEAD: `6b313b800e58e78ef61f47bae0c012d61ba928f9`

Branch: `codex/v04-recovery`

## Decision context

The participant-provided leaderboard screenshot records:

- first place: public log loss `0.6008`, AUROC `0.6283`;
- fifth place: public log loss `0.6040`, AUROC `0.6229`;
- tenth place: public log loss `0.6053`, AUROC `0.6137`;
- protected v0.5: public log loss `0.6054`, therefore approximately eleventh at the
  time of the screenshot.

The first-place gap is `0.0046`; the safety target `0.6005` requires a `0.0049`
public loss gain. No such gain is claimed locally.

## Scientific hypothesis and independence

E760 asks whether tutor uptake of an immediately preceding student contribution
adds a robust process signal beyond v0.5. It is directionally and empirically
separate from rejected E710:

- E710 used competition-self-supervision for `tutor -> next student` coherence.
- E760 uses external expert supervision for `student -> next teacher` uptake.
- E760 uses none of E710's cache, model, scores, predictions, or validation result.
- A failure closes the entire uptake family; E710 cannot be blended into a rescue.

The mechanism follows Demszky et al., *Measuring Conversational Uptake: A Case
Study on Student-Teacher Interactions*, ACL 2021. The authors report that uptake
includes repetition, reformulation, acknowledgment, and question answering.

## Frozen external source and legal boundary

Repository: `https://github.com/ddemszky/conversational-uptake.git`

Git revision: `67fb30cf7c3ea6f619487e33d2f99692dca107d0`

Repository license: root MIT `LICENSE`

Immutable source hashes:

| File | SHA-256 |
|---|---|
| `data/uptake_data.csv` | `c6bb9e5ff69b6c8ae8981b9d613c41bda1c1ed5685e9e8a74f97d3c1bddb642d` |
| `LICENSE` | `c4fcfe38abef13391b7546f3a3ec9f8c8fc7a480255f888655a57e49076c0429` |
| `README.md` | `2a2895477a4308d4214a5010dfa162797bc236667f4b02a1e1f508cdc1bdc0a7` |

The repository root license is treated as covering the checked-out repository
contents. Its notice must be retained with any substantial copied portion. The
authors' Hugging Face checkpoint is separately marked `CC-BY-NC-ND-4.0`; it is
excluded and must not be downloaded, used, modified, trained from, or packaged.

The source has 2,246 exchanges and 774 observation groups. There are 1,998
non-null `uptake_zscore` labels. The human ordinal majority counts are 171 low,
744 middle, 780 high, and 551 missing.

## Frozen canonical cache and split

Canonical cache: `data_cache/conversational_uptake_e760.parquet`

- ordered-content SHA-256:
  `c2ef8d92db3f22247e1a16ec8c81d5acb2058e3eec6aa889d65ead1629546d82`
- Parquet SHA-256:
  `368b5cee4035251838bd50be09904374024c2e81f1cb2fe504df9acadb30d542`
- sort: stable `obs_id`, then `exchange_idx`;
- identity: unique `(obs_id, exchange_idx)`;
- evaluation rows: only non-null `uptake_zscore`;
- splitter: scikit-learn 1.8.0 `GroupKFold(n_splits=5, shuffle=True,
  random_state=20260728)`;
- group: `obs_id`;
- source-observation overlap between training and validation: exactly zero in
  every fold.

No competition target, V_joint row, or V_final row was read while building this
cache or split.

## Frozen external estimators

Both estimators are five-fold grouped out-of-fold ridge regressions with
`alpha=10.0`, `solver="lsqr"`, and seed/split `20260728`.

The repetition comparator has exactly ten fold-standardized features:

1. log student token count;
2. log teacher token count;
3. log shared unique-token count;
4. Jaccard overlap;
5. student-token recall;
6. teacher-token precision;
7. student question marker;
8. teacher question marker;
9. teacher acknowledgment prefix;
10. exact student-substring reuse.

The E760 candidate adds one non-negative `2^17`-wide feature hash containing:

- role-prefixed student and teacher unigrams;
- role-prefixed student and teacher bigrams;
- shared-token indicators;
- ordered `student_token > teacher_token` cross features.

Each role is truncated to 48 normalized tokens; directional crosses use the
first 24 unique tokens per role. A fold-local
`TfidfTransformer(norm="l2", sublinear_tf=True)` is fit on training rows only.
The same ten repetition controls are appended. No model or preprocessing
hyperparameter may change after the external result.

Ordinal predictions use fold-training-only midpoints between the mean
`uptake_zscore` of the low, middle, and high expert-majority classes.

## Frozen external gate

Exactly one grouped external outcome validation is authorized. E760 proceeds
only if every clause passes:

1. pooled candidate Pearson correlation is at least `0.30`;
2. pooled candidate Spearman correlation is at least `0.30`;
3. pooled Spearman gain over repetition is at least `0.05`;
4. pooled RMSE gain over repetition is at least `0.015`;
5. quadratic-weighted ordinal kappa is at least `0.15`;
6. candidate Spearman is positive in all five folds;
7. candidate Spearman beats repetition in at least four of five folds;
8. a 2,000-replicate observation-group bootstrap with seed `20260728` gives at
   least `0.90` support for positive Spearman gain.

Failure of any clause rejects E760 literally. There is no altered n-gram,
dimension, regularization, baseline, threshold, fold, label, or checkpoint
rescue.

## Pre-score verification

Focused tests: `4 passed`.

The first synthetic benchmark attempt failed before any external or competition
outcome score because signed hashing produced negative values that are invalid
for sublinear TF-IDF. The representation was corrected to non-negative hashing
and an explicit non-negativity regression assertion was added.

The corrected 512-row target-free benchmark passed every frozen clause:

- Python `3.12.8`;
- scikit-learn `1.8.0`;
- finite matrices and predictions;
- synthetic Spearman `0.9990861109`;
- feature time `0.0597855` seconds;
- projected 2,246-row feature time `0.2622622` seconds.

This external validation is safely below one hour and must run in the active
foreground turn. The short scheduler wake card is not considered proven until
its exact wake token arrives; therefore no over-one-hour unattended run is
authorized.

## Conditional competition phase

This section authorizes no competition outcome score yet. It becomes active only
if all external clauses pass.

The next phase must first freeze and test a competition target-free feature
cache, its exact session/objective aggregations, estimator, weights, seed,
runtime, and hashes. It must be committed and pushed before exactly one
`V_seen`/`V_objective`/`V_style` selection validation.

That selection uses only a fixed fold-local linear uptake component and
preregistered 10%, 20%, and 30% probability blends over raw v0.5. The selected
blend must:

- gain at least `0.0016` equal-environment mean log loss;
- improve all three environments;
- have no fold regression above `0.0005`;
- not regress macro AUROC, Brier, or ECE;
- have at least `0.95` positive-gain support in both session and
  semantic-family bootstraps.

V_joint remains confirmation-only. V_final remains sealed. No submission,
platform upload, or new ZIP is authorized by this preregistration.
