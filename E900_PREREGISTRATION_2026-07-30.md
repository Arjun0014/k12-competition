# E900 preregistration: ColBERT token-level late interaction

## Scientific question

Can token-level objective-to-dialogue matching expose objective coverage that
the publicly successful BGE-base single-vector component loses when it
compresses a long tutoring conversation?

E900 is not another sentence-encoder swap. ColBERT creates one normalized
128-dimensional vector per token and scores a query by summing, over its fixed
32 query positions, the maximum cosine match to any of 180 document positions.
This is the released MaxSim late-interaction mechanism.

## Independence from closed branches

- E420, E450, E470, and E890 are single-vector encoder replacements.
- E540 uses four BGE-base sentence vectors and fixed scalar attention.
- E550 changes pooled sentence readout.
- E720 hashes objective/role token conjunctions without contextual token
  matching.
- E820 adapts a BGE sentence encoder with contrastive MathDial retrieval.

E900 loads none of their rejected checkpoints, predictions, heads, or scores.
It uses an untouched released ColBERT checkpoint and a token-level interaction
operator. Repository and learning-log searches found no prior ColBERT,
late-interaction, or MaxSim implementation.

The 64-row target-free discovery screen produced confusion MRR `0.5342764` and
recall@5 `0.6093750`, versus BGE-base `0.3294654` and `0.4531250`. This is the
plausibility evidence for a possible `0.0016` robust competition gain; it is
not a competition outcome score.

## Stage A: frozen external target-free gate

### Lineage

- ColBERT model: `colbert-ir/colbertv2.0`
- Revision: `c1e84128e85ef755c096a95bdb06b47793b13acf`
- Declared license: MIT
- Safetensors SHA-256:
  `3f58890b1dfdfec066ef12ba431fa9d56992da9e30a53489242c4156e37e9017`
- Official implementation reference:
  `stanford-futuredata/ColBERT@cc4f3dc91c0b45d2d08c251d9d95178285c65f1c`
- External evaluation: all 599 rows of official MathDial test at revision
  `b06c020a0a1f57a87577fec33e657b63e7eb476e`
- Consumed fields: `qid`, `question`, `teacher_described_confusion`,
  `conversation`
- No MathDial outcome/correctness field is consumed.

### Representation

- CPU threads: 6
- Query length: 32
- Document length: 180
- Query marker: `[unused0]`, token ID 1
- Document marker: `[unused1]`, token ID 2
- Projection: released bias-free `128 x 768` matrix
- Token vectors: L2-normalized
- Document punctuation and padding: masked exactly as in the official code
- Interaction: sum of query-token maximum document-token cosine
- Comparator: untouched packaged BGE-base with the already frozen E820 query
  prefix and 96/256 query/document lengths

### External clauses

All clauses are conjunctive:

1. Teacher-confusion MRR gain over BGE-base is at least `0.08`.
2. Teacher-confusion recall@5 gain is at least `0.08`.
3. A 2,000-replicate question-ID bootstrap gives at least `0.95` support for
   positive teacher-confusion MRR gain.
4. Original-problem MRR regression is no worse than `0.005`.
5. Original-problem recall@5 regression is no worse than `0.005`.

The resource gate is maximum projected runtime 3,600 seconds and peak RSS below
8 GiB. The benchmark uses 32 deterministic conversation-length quantiles and
contains no competition text or outcome.

The frozen benchmark passed before the external screen. It took `1.1372753`
seconds for 32 ColBERT queries, `4.7337609` seconds for 32 ColBERT documents,
`0.0273112` seconds for their score matrix, and `8.3511233` seconds for the
BGE comparator encodes. The measured all-599 projection is `306.6486504`
seconds, peak RSS is `1,448,837,120` bytes, and benchmark SHA-256 is
`eadf85c06e8ae2d26dbba4eea0daa43d5310f19cbc9b57964f024f9f595c3edf`.
The external screen is therefore a sub-hour foreground task; no scheduler wake
is appropriate.

Any failed clause rejects exact Stage A without prompt, token length,
punctuation, marker, pooling, projection, model, query view, fusion,
fine-tuning, threshold, or benchmark rescue.

## Conditional Stage B

Stage B does not yet exist. If and only if Stage A passes, freeze in a separate
preregistration before competition outcomes:

- the immutable competition sample and three environments;
- deterministic early/late student/tutor evidence extraction from one
  response's own objective context;
- per-response ColBERT token-match profile features;
- estimator, folds, seed, scales, candidate blend weights;
- log-loss/AUROC/Brier/ECE gates, fold bound, session bootstrap, and backup
  gate;
- target-free cache hashes and runtime plan.

No cross-test-row feature, test aggregation, pseudo-label, or inference-time
fit is permitted. `V_joint` stays confirmation-only and `V_final` sealed.

`competition_text_accessed=false`; `competition_outcomes_accessed=false`;
`V_seen_accessed=false`; `V_objective_accessed=false`;
`V_style_accessed=false`; `V_joint_accessed=false`;
`V_final_accessed=false`.
