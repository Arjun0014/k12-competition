# E860 preregistration: target-free ASR/spoken-math normalization

Status at freeze: aggregate opportunity counts and synthetic unit tests only.
No E860 resource benchmark, NCTE corruption score, competition target-free
semantic score, competition outcome, `V_joint`, or `V_final` has been read.

## Independent mechanism

The local corpus is long spoken-math ASR: every session contains `[unclear]`,
approximately 24% of utterances begin with a speech disfluency, 10% contain
number words, and 11% contain explicit spoken operators. Yet no prior branch
canonicalized spoken numbers/operators before objective matching or semantic
encoding. Character n-grams tested surface robustness but did not create a
shared representation for “twelve divided by three” and `12 / 3`.

E860 is a deterministic parallel text view. It removes `[unclear]`, strips only
turn-initial fillers, collapses only adjacent exact repetitions, and maps
unambiguous number expressions and explicit math operators to canonical tokens.
It does not correct `to/two`, `for/four`, `ate/eight`, `won/one`, `some/sum`,
or any other ambiguous homophone. It does not infer missing speech, speaker
role, correctness, mastery, difficulty, misconception, tutor move, or outcome.

This differs from E820's encoder fine-tuning, E830's curriculum prior, E850's
speaker reassignment, the failed character-hash view, and all correctness,
state, timing, calibration, or aggregation branches. It uses the unchanged
packaged BGE-base model only to test whether the deterministic view preserves
meaning and repairs controlled ASR corruption.

## Frozen sources

- NCTE single utterances SHA-256
  `bbfa37ac857991e5d12b1677216896f32d7cf4c2d95b618b67afa7b3bf23b3d7`;
  target-free text, video, turn, speaker, and length columns only.
- Competition response/objective-context cache SHA-256
  `218d5b041f78c6803f28078c35b7fa9e37b725f456c4052974e5150b2afa43f6`;
  identity, text, and retrieval-coverage columns only.
- Packaged MIT BGE-base safetensors SHA-256
  `c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7`.

## Frozen samples and perturbation

- Select 2,048 NCTE utterances of 5–80 words containing an explicit number or
  operator by a namespaced SHA-256 ordering of `video_id|turn_idx`.
- Deterministically verbalize integer digits and operator symbols, prepend
  `Um, uh`, duplicate one hash-selected token, and insert one `[unclear]`.
- Select 2,048 competition response contexts by namespaced response hash,
  exactly 512 from each quartile of target-free retrieval-term coverage.
- Remove the `[OBJECTIVE]` header only when measuring transcript/objective
  alignment. Do not alter the existing cached context.

BGE-base uses CPU float32, six threads, packaged normalized CLS pooling,
maximum length 256, and batch 16. No prompt, model, layer, pooling, length,
sample, perturbation, or normalization sweep exists.

## Frozen target-free gate

Every clause must pass:

- at least 100 NCTE videos;
- exact normalized clean/corrupt equality on at least 90% of rows;
- mean token-Jaccard recovery gain at least `0.20`;
- mean BGE clean/corrupt cosine gain at least `0.02`;
- mean normalized clean/corrupt cosine at least `0.95`;
- positive BGE corruption gain on at least 90% of rows;
- at least 25% of competition objectives and 90% of contexts change;
- raw/normalized BGE semantic preservation at least `0.95` for objectives and
  `0.90` for contexts;
- mean normalized transcript/objective cosine gain at least `0.005`;
- positive transcript/objective cosine gain on at least 60% of rows;
- mean gain at least `0.007` in the lowest retrieval-coverage quartile.

Any failure rejects exact E860 before competition outcomes. Do not add
homophones, change canonical tokens, relax thresholds, alter perturbations,
change the sample/model, or retain only favorable subgroups.

## Conditional continuation

A complete target-free pass authorizes only a separately committed competition
preregistration. That continuation may build one normalized compact-context and
objective BGE-base cache, use the existing frozen interaction formula and
fold-local `C=0.1` logistic head, and evaluate only preregistered 10/20/30%
blends over raw v0.5. It may not normalize or replace the deployed sparse,
feedback, raw semantic, calibration, or prior components.

Promotion still requires at least `0.0016` robust mean log-loss gain, every
selection environment improving, proper-score non-regression, the frozen fold
bound, and both 5,000-replicate bootstrap rules. `V_joint` stays
confirmation-only and `V_final` sealed. v0.5 and its protected ZIP remain
unchanged. No upload or submission is authorised.
