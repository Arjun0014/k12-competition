# E820 preregistration: target-free MathDial contrastive objective alignment

Status at freeze: implementation only; no E820 retrieval score, competition
outcome, `V_joint`, or `V_final` has been read.

## Why this is the next independent branch

The remaining legal weakness of v0.5 is short learning-objective text aligned
to long tutoring evidence, especially when objective IDs are unseen and
lexical coverage is low. E820 trains no mastery, correctness, tutor-move,
student-state, timing, numeric, or competition-outcome head. It instead adapts
the untouched packaged BGE-base encoder with target-free positive pairs:
MathDial's original K-12 math problem and the tutoring dialogue grounded in
that problem.

This is not E430 (MathDial self-correction classification), E530 (MathDial
tutor-move classification), E720 (competition-label sparse token
conjunctions), or E730 (fold-local linear Procrustes on competition
embeddings). E820 consumes none of those checkpoints, predictions, labels, or
fitted maps. It learns a nonlinear retrieval representation only from an
external question-dialogue relation.

## Frozen source and split

- Official `eth-nlped/mathdial`, revision
  `b06c020a0a1f57a87577fec33e657b63e7eb476e`, CC BY-SA 4.0.
- `train.jsonl` SHA-256
  `96980babee081a3da48ed0f1fb3068ab52f23ddc67e63bfd5ec99ad701fd29cc`.
- `test.jsonl` SHA-256
  `28d1e537d65a6e6ff7b8bde602a2c7e2493b93d6bc785d1848506a650997f122`.
- Keep all 599 official-test dialogues. Purge from training every question ID
  appearing in test. The frozen legal training set is 1,677 dialogues across
  717 question IDs; test has 599 dialogues across 394 question IDs.
- Consume only `qid`, `question`, `teacher_described_confusion`, and
  `conversation`. Do not consume `self-correctness`, ground truth, incorrect
  solution, typicality ratings, competition labels, or rejected-model output.
- Training selects exactly one dialogue per legal question ID per epoch by
  deterministic row-hash rotation. This prevents same-question in-batch false
  negatives.

## Frozen model and optimization

- Base: `BAAI/bge-base-en-v1.5`, revision
  `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, MIT; local safetensors SHA-256
  `c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7`.
- Shared query/document encoder, normalized CLS pooling.
- Query prefix exactly:
  `Represent this K-12 math learning target for retrieving its tutoring dialogue: `
- Query/document maximum lengths: 96/256 tokens.
- Freeze embeddings and lower eight encoder layers; train only the upper four
  encoder layers.
- Symmetric in-batch InfoNCE, temperature 0.05, batch 8, two epochs,
  AdamW learning rate `2e-5`, weight decay `0.01`, 10% linear warmup,
  gradient cap `1.0`, seed `20260729`.
- No hard-negative mining, generated negative, model selection, checkpoint
  choice, alternate prompt, alternate pooling, or hyperparameter sweep.

## Frozen target-free evaluation

Evaluate the untouched base and the single final adapted checkpoint on every
official-test dialogue. Candidate documents are all 599 test dialogues. Each
query is correct when it retrieves any dialogue with the same question ID.
Use two fixed query views:

1. original problem text;
2. teacher-described confusion, which is the more objective-like transfer
   view and is never used for training.

Report MRR and recall at 1/5/10. Bootstrap the adapted-minus-base confusion MRR
over 394 question-ID groups with 2,000 deterministic replicates.

Every clause must pass:

- confusion MRR gain at least `0.04`;
- confusion recall@5 gain at least `0.04`;
- confusion grouped-bootstrap positive-gain support at least `0.95`;
- problem MRR and recall@5 regress by no more than `0.005` each;
- adapted document off-diagonal cosine mean below `0.90` and standard
  deviation above `0.02`;
- epoch-two mean training loss is below epoch-one mean training loss.

A failed clause rejects exact E820 before any competition outcome. No rescue,
second seed, alternate epoch, prompt, layer count, loss, batch, checkpoint,
query view, or threshold is permitted.

## Competition continuation, frozen before any score

Passing the external gate does not authorize a score. It only authorizes a
separate committed competition preregistration using the final E820 delta,
the existing immutable 4,096-row BGE-base screen lineage, the existing frozen
five environment folds, and exactly 10%, 20%, and 30% blends over raw v0.5.
That later document must freeze cache construction, fold-local estimator,
proper-score gates, worst-fold bound, session and semantic-family bootstraps,
and the `>=0.0016` mean loss-gain requirement before `V_seen`,
`V_objective`, or `V_style` is opened.

`V_joint` remains confirmation-only. `V_final` remains sealed. No ZIP may be
built unless every backup gate passes. No upload or competition submission is
authorized.
