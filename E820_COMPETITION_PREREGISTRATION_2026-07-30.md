# E820 competition preregistration

Freeze date: 2026-07-30 (Asia/Kolkata)  
Parent commit containing the target-free freeze:
`f647f8608cd0eac2a14723e7fed45032752cd69e`  
Status at this freeze: external target-free gate passed; no E820 competition
outcome, `V_joint`, or `V_final` has been accessed.

## Bound external checkpoint

The only eligible checkpoint is the final E820 delta from target-free run
`20260729T190028Z_mathdial_contrastive_alignment`.

- delta SHA-256:
  `c7caf144e91db5aedd427ab27c860cd5ae091cbb1e466621b0b563f88e6b9248`;
- external report SHA-256:
  `d3f6f8c5bb5ec832c959db91d2777f93b58330b8fd129afe5c49db78b3d02fad`;
- epoch losses: `0.0329653421` and `0.0035182768`;
- held-out confusion MRR gain: `0.1157348316`;
- held-out confusion recall@5 gain: `0.1252086811`;
- 2,000-question-ID bootstrap mean MRR gain:
  `0.1213716071`, interval `[0.0966970774, 0.1473972389]`, support `1.0`;
- problem MRR/recall@5 gains:
  `0.0115870000/0.0066777963`;
- off-diagonal document cosine mean/std:
  `0.1968484074/0.0878590420`.

All eight external clauses passed. No alternate epoch, checkpoint, seed,
prompt, pooling, layer, or loss is eligible.

## Target-free competition cache

- Rows are exactly the immutable sorted 4,096-row pilot selected by
  `data_cache/qwen3_pilot_indices.npy`, SHA-256
  `4e62045be2bd473ef41e8aaf5f4b351b7432760ed6c1bbd7ccd88ca112d1efba`.
- Context text is the existing `compact_objective_context` transform over the
  immutable objective-retrieval cache, maximum 256 tokens.
- Objective query is exactly
  `Represent this K-12 math learning target for retrieving its tutoring dialogue: `
  plus the current row's learning-objective text, maximum 96 tokens.
- Encoder is the packaged BGE-base model plus only the bound E820 delta.
  Pooling is normalized CLS. Base AutoModel encoding must match the existing
  BGE-base context cache with maximum absolute difference at most `2e-5` and
  minimum sample cosine at least `0.99999`.
- Build one context cache and one objective cache. Before outcomes, create and
  commit `E820_COMPETITION_CACHE_BINDING_2026-07-30.json` with exact cache,
  metadata, and target-free-report SHA-256 values. Validation refuses an
  absent or mismatched committed binding.
- Cache construction may read response identity, objective text, and
  objective-context text only. It may not read any target, component
  prediction, environment result, rejected output, `V_joint`, or `V_final`.

The frozen resource benchmark passed before cache construction. Base parity on
32 rows had maximum absolute difference `1.2293458e-7` and minimum cosine
`0.9999999404`. Encoding 64 documents and 64 queries took
`12.4862565/1.7284089` seconds and projects `805.7099749` seconds for 4,096
contexts plus 244 unique objective queries. Peak RSS was `965,009,408` bytes.
Benchmark SHA-256 is
`6ba487d36d6208f7e7c1d0b85866c15c45debd85c66a87891cff25e91e1ebcd6`.
This is a sub-hour foreground cache build; no scheduler wake is appropriate.

## Frozen selection validation

Selection environments are exactly `V_seen`, `V_objective`, and `V_style`.
Restrict each immutable component OOF file to the bound 4,096 pilot rows.
Use its existing five folds and the repository `_fold_masks` session purge.

For each environment:

- candidate semantic geometry is exactly `0.7*context`,
  `0.7*objective`, `6.0*(context*objective)`, and
  `0.7*abs(context-objective)`;
- dense controls are the existing behavior/retrieval block with its final
  cosine replaced by E820 context-objective cosine;
- fit exactly fold-local scikit-learn `LogisticRegression` through the
  existing `semantic_logistic_oof`, `C=0.1`, five folds;
- seed is `20260730`;
- raw v0.5 is exactly `0.25*pred_full + 0.25*pred_role +
  0.50*pred_bge_base`;
- evaluate only E820 semantic-head probability blend weights `0.10`, `0.20`,
  and `0.30`.

Select the lowest equal-environment, equal-fold macro log loss; an exact tie
uses the smaller weight. No post-hoc sweep, calibration, alternate C, feature
removal, environment exclusion, threshold, or rescue is allowed.

The selected candidate passes only if every clause is true:

1. mean macro log-loss gain over raw v0.5 is at least `0.0016`;
2. all three selection environments improve;
3. no selection environment regresses in macro log loss;
4. worst individual fold log-loss regression is at most `0.0005`;
5. macro AUROC does not regress;
6. macro Brier score does not regress;
7. macro 10-bin ECE does not regress;
8. 5,000-replicate session-cluster bootstrap support is at least `0.95`;
9. 5,000-replicate semantic-family bootstrap support is at least `0.95`.

Any failed clause rejects exact E820 without rescue.

## V_joint confirmation

Only after every selection clause passes, evaluate the already selected
weight once on `V_joint`, using the same bound rows, geometry, controls,
estimator, and folds. Require:

- macro log-loss gain at least `0.0010`;
- worst-fold regression at most `0.0005`;
- AUROC, Brier, and ECE non-regression;
- session and semantic-family bootstrap support each at least `0.95`.

`V_joint` cannot select a weight or modify the candidate. A failed
confirmation rejects E820. `V_final` remains sealed.

## After the screen

A pass authorizes only a separately frozen full-cache/production and backup
audit. It does not authorize a ZIP, upload, or submission. A new ZIP may be
built only if production parity and conservative backup gates pass while the
protected v0.5 ZIP remains byte-identical at SHA-256
`65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.
Codex never uploads or submits; the participant makes every submission.
