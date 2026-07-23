# Trace the Ace Top-5 Recovery Plan

Date: 2026-07-22 (Asia/Kolkata)  
Status: Active local research; no platform submission is authorized.

## Objective and score target

The current public champion is v0.2 at log loss `0.6081`, AUROC `0.6147`, and rank `#20`.
The live leaderboard cutoff on 2026-07-22 is:

| Rank | Log loss | AUROC |
|---:|---:|---:|
| #1 | 0.6013 | 0.6309 |
| #3 | 0.6032 | 0.6288 |
| #4 | 0.6033 | 0.6332 |
| #5 | 0.6042 | 0.6229 |
| #6 | 0.6046 | 0.6232 |

The operational target is public log loss `<=0.6038`, not merely a displayed tie at `0.6042`.
That requires an improvement of at least `0.0043` over v0.2. Because local gains can shrink or reverse
under hidden source shift, no candidate is called top-5-ready from a fragile pooled improvement.

## Non-negotiable execution rules

1. Codex never uploads or submits a model. The participant performs every platform submission manually.
2. v0.2 remains immutable. Rejected v0.4 components are not revived through weight tuning.
3. `V_final` stays sealed until one complete architecture, feature transform, blend, and calibration rule are locked.
4. Every experiment records its data lineage, code/config hash, runtime, metrics, failures, projected public score,
   projected rank bracket, and next decision in `PROJECT_LEARNING_LOG.md`.
5. A consistent but sub-top-5 gain may produce a clearly named backup ZIP. A backup is not a promotion.
6. Long jobs receive at most one halfway checkpoint and one expected-completion checkpoint. The checkpoint time
   is calculated from a measured pilot or completed-fold rate. There is no minute-by-minute polling.
7. A runner must enforce its own finite-loss, memory, checkpoint, and stale-resume checks so monitoring is not
   needed for safety.

## Evidence carried forward

- v0.2 is the only model with strong hidden-test evidence.
- Replacing BGE-small with BGE-base improved all four hardened environments by mean log loss `0.001272`, but did
  not clear the `0.0025` robustness gate. Its raw public projection is about `0.6068`, currently around rank #14.
- Calibration transforms selected on pooled behavior were unstable across environments.
- Ordered feedback hashing and NB-SVM add ranking signal but reverse in objective-frequency regimes.
- Internal outcome fine-tuning produced attractive selected folds but severe hidden calibration failure in v0.4.
- D100 worst-group training weakened the semantic ranking boundary and was rejected.
- The remaining plausible source of a step change is a more transferable representation of student mastery,
  trained without competition provider/objective shortcuts.

## External-data decision

The organizer has explicitly stated that CC BY-SA data is sufficiently open for this competition. The local
SemEval-2013 Task 7 Student Response Analysis corpus is therefore admitted, subject to attribution and exact
lineage logging. It is much more relevant than the other available corpora because it contains manually graded
short student answers, questions, reference answers, and unseen-question/unseen-domain evaluation splits.

Excluded or deferred resources:

- Bridge, DrawEduMath, SciQ, and actual TalkMoves remain excluded because of non-commercial restrictions.
- MRBench remains excluded because of Bridge ancestry and unresolved upstream licensing.
- GSM8K and FairytaleQA are permissive but do not provide aligned mastery labels; they are not used in E400.
- Hosted APIs do not receive competition transcripts.

## E400: SemEval-adapted educational entailment

### Data contract

- Parse only canonical SemEval five-way XML.
- Deduplicate by `studentAnswer.id`; prefer `Core` over duplicate `Extra` records.
- Exclude reliability re-annotations and rows with missing labels.
- Preserve official train, unseen-answers, unseen-questions, and unseen-domains splits.
- Normalize `non-domain` to `non_domain`.
- Map labels onto the existing DeBERTa NLI head without reinitializing it:
  - `correct -> entailment`;
  - `partially_correct_incomplete -> neutral`;
  - `contradictory`, `irrelevant`, and `non_domain -> contradiction`.
- Premise: question plus student answer. Hypothesis: the concatenated canonical reference answer(s).

### Training contract

- Base checkpoint: local `cross-encoder/nli-deberta-v3-small` (Apache-2.0).
- Train only the top two of six encoder layers, pooler, and existing three-class head.
- One epoch, batch 16, maximum length 256 with student premise truncated before reference hypothesis.
- AdamW; encoder learning rate `2e-5`, head learning rate `1e-4`, weight decay `0.01`, warmup `10%`, gradient norm `1.0`.
- Seed `20260722`; CPU threads capped at six.
- No learning-rate, epoch, prompt, layer-count, or label-map sweep after results are visible.

### External continuation gate

The frozen checkpoint must achieve all of the following on official external holdouts:

- correct-vs-rest AUROC `>=0.70` on unseen questions;
- correct-vs-rest AUROC `>=0.68` on unseen domains;
- three-way macro F1 `>=0.45` on unseen questions; and
- no correct-vs-rest AUROC regression versus the untouched base NLI checkpoint on either holdout; and
- no non-finite output or label/schema inconsistency.

Failure stops this external checkpoint. Passing permits competition feature caching, not promotion.

### Competition feature views

1. `E400_session`: one compact objective-conditioned session premise paired with the mastery hypothesis.
2. `E410_turns`, only after the external gate passes: deterministic first/highest-overlap, longest/high-overlap,
   and last/high-overlap student evidence turns, each paired with the objective. Aggregate entailment, neutral,
   contradiction, entropy, disagreement, and temporal-change features. Missing views receive explicit masks.

The external encoder is frozen for all competition rows. Competition labels train only fold-local linear probes.
No objective ID, provider ID, target prior, or test-batch aggregate is permitted.

## Validation and promotion gates

Candidate development uses `V_seen`, `V_objective`, and `V_style`. The already opened `V_joint` may be reported as
development evidence, but cannot be described as untouched confirmation for E400/E410. `V_final` remains sealed.

Each fixed external feature view is evaluated as a fold-local linear component and in a preregistered blend with
the stable BGE-base replacement. Candidate weights may be selected only within the development environments and
must be locked before any new confirmation suite or `V_final` is opened.

### Backup gate

A candidate may receive a verified backup ZIP when it:

- improves mean log loss by at least `0.0010` across the hardened environments;
- improves at least three environments;
- regresses no environment by more than `0.0005`;
- does not worsen macro AUROC, Brier score, or ECE; and
- has at least `90%` paired session-bootstrap support.

The ZIP name must include `experimental-backup` or `validated-backup`, its measured gain, date, and artifact hash.

### Top-5 gate

A candidate becomes top-5-ready only when it:

- improves mean log loss by at least `0.0043` versus v0.2;
- improves every hardened environment;
- regresses no fold materially and no legal subgroup by more than `0.0010`;
- improves or preserves macro AUROC, Brier score, ECE, and calibration sanity;
- has at least `95%` session- and semantic-family-bootstrap support;
- passes one locked confirmation protocol not used for its architecture or weight choice; and
- then passes the still-sealed `V_final` with at least `0.0035` loss gain and no metric reversal.

Raw public projection is `0.6081 - robust_mean_gain`. Reports also show a conservative projection using only
60% of the local gain. The live leaderboard bracket is reported, but never used to tune predictions.

## Bounded next branches if E400/E410 fail

1. `E420_long_context`: a 4,096-row, fixed ModernBERT long-context screen using objective-conditioned full-session
   evidence. Continue only on a material same-sample gain without AUROC or calibration regression.
2. `E430_transfer_then_outcome`: initialize the internal outcome encoder from the externally validated educational
   entailment checkpoint and run legal fold-local adaptation. This expensive branch is allowed only if frozen
   E400/E410 features already show stable complementarity.
3. `E440_external_ensemble`: combine independently validated session and turn evidence only if each is useful alone;
   no rescue stacking of failed components.

## E430 amendment frozen before MathDial modeling

The official MathDial repository was discovered after E400 launch and before any MathDial model was fit. Commit
`b06c020a0a1f57a87577fec33e657b63e7eb476e` is CC BY-SA 4.0 and contains 2,861 math tutoring dialogues with
teacher-annotated end-of-dialogue `self-correctness`. This is promoted ahead of the earlier generic internal-transfer
idea because it supplies a genuinely external tutoring-outcome label.

- `Yes` maps to positive independent correction.
- `No` maps to negative.
- `Yes, but I had to reveal the answer` maps to negative because the competition asks whether the student can
  answer independently after tutoring.
- Missing labels are excluded and reported.
- The official test split is immutable and used only for the external continuation gate. A source audit found that
  the published train/test files share question IDs, so every train row whose `qid` appears in test is purged before
  fitting. This frozen correction prevents question-text leakage while retaining the complete official test set.
- Any competition candidate remains a fold-local probe and a preregistered 10/20/30% blend over raw BGE-base
  replacement. No public result, objective prior, provider ID, or `V_final` row may select it.
- E430 is evaluated only after E400 completes. It is an independent branch, not a rescue re-fit of E400 after seeing
  competition outcomes.

The plan stops a branch immediately when its frozen gate fails. It does not stop the overall research program until
a top-5-ready local candidate exists or a genuine external/compute blocker requires participant action.

## E420 detailed freeze after E430 rejection and before ModernBERT scoring

E430 completed and failed all three of its frozen external clauses, so it is rejected without a competition cache.
E420 is now the next bounded branch. The following contract is frozen before downloading or scoring ModernBERT:

- Model: official `answerdotai/ModernBERT-base` revision
  `8949b909ec900327062f0ebf497f51aef5e6f0c8`, Apache-2.0, loaded as the frozen base encoder.
- Screen rows: the exact existing 4,096-row fold/label-balanced encoder sample in
  `qwen3_pilot_indices.npy`, SHA-256 `4e62045be2bd473ef41e8aaf5f4b351b7432760ed6c1bbd7ccd88ca112d1efba`.
  This sample and its `semantic_k50_s0` fold assignments were fixed before E420 existed.
- Input: the complete cached role-marked session transcript as the first sequence and the fixed hypothesis
  `The student demonstrates mastery of this learning objective: <objective>.` as the second sequence.
  `only_first` truncation preserves the complete objective hypothesis. No target-dependent retrieval, objective ID,
  provider ID, prior, test aggregate, or external annotation is used.
- Operational context-length selection is label-free. Benchmark 16 deterministic pilot rows spanning token-length
  quantiles at 2,048, 4,096, and 8,192 tokens, batch size one, six CPU threads. Freeze the longest length whose
  measured projection for 4,096 rows is at most eight hours and whose process RSS stays below 8 GB. This benchmark
  may choose runtime length only; it cannot inspect targets or model-quality metrics.
- Representation: concatenate the separately L2-normalized final-layer first-token vector and attention-mask mean
  vector, then divide by `sqrt(2)`. Cache float32 vectors with exact response-order, model-revision, source-hash,
  token-length, finiteness, and norm audits.
- Probe: one fold-local logistic regression with fixed `C=0.1`, the existing session purge, and the existing legal
  dense controls standardized only on each legal outer-training fold. There is no C, pooling, layer, prompt, or
  blend sweep in the screen.
- Fair comparator: BGE-base interaction features at the same fixed `C=0.1`, fit on the identical 4,096 rows,
  folds, purge masks, labels, and dense controls.
- Continue to a full competition cache only if ModernBERT either improves log loss by at least `0.0015` with
  non-regression in AUROC, Brier, and ECE-10, or improves AUROC by at least `0.0050` with non-regression in log loss,
  Brier, and ECE-10. It must also reach at least 90% paired session-bootstrap support for positive log-loss gain.
  Failure rejects this exact E420 representation without a prompt, layer, pooling, C, or weight rescue.
- A passing screen earns only full-cache construction and the frozen fold-local `V_seen`, `V_objective`, and
  `V_style` evaluation. Competition blends remain exactly 10%, 20%, and 30% over raw BGE-base replacement.
  `V_joint` and `V_final` retain their existing restrictions, and no platform submission is authorized.

## E450 freeze after E420 rejection and before BGE-large scoring

E420's raw ModernBERT representation regressed log loss, AUROC, Brier, and ECE with only 0.06% session-bootstrap
support. It is rejected without rescue. The surviving positive encoder evidence is the sentence-trained BGE family:
BGE-base improved every hardened environment over BGE-small. E450 therefore isolates one further capacity step
without changing text selection, token length, pooling, or probe architecture.

- Model: official `BAAI/bge-large-en-v1.5` revision
  `d4aa6901d3a41ba39fb536a557fa166f842b0e09`, MIT, 24-layer BERT, 1,024-dimensional
  SentenceTransformers CLS pooling. Download only exact safetensors/config/tokenizer/module/model-card files.
- Sample: reuse the exact 4,096-row encoder sample and `semantic_k50_s0` fold assignments with index SHA-256
  `4e62045be2bd473ef41e8aaf5f4b351b7432760ed6c1bbd7ccd88ca112d1efba`.
- Text: use the existing compact objective context and objective text exactly as the BGE-base cache did. Encode both
  separately at maximum length 256, L2 normalize, and construct the identical scaled
  `[context, objective, product, absolute-difference]` interaction. There is no instruction prefix or new view.
- Screen: one fixed `C=0.1` fold-local logistic probe with the same session purge and legal dense controls.
  The fair comparator is BGE-base at fixed `C=0.1` on the identical rows, folds, labels, purge masks, and controls.
- Operational benchmark: 32 deterministic pilot contexts spanning BGE-large token-length quantiles, batch size one,
  six CPU threads. Proceed only if 4,096 context rows plus unique objectives project below eight hours and RSS below
  8 GB. Batch size and token length cannot change after the benchmark.
- Continuation gate: either at least `0.0015` log-loss gain with AUROC/Brier/ECE non-regression, or at least `0.0050`
  AUROC gain with log-loss/Brier/ECE non-regression, plus at least 90% paired session-bootstrap support for positive
  log-loss gain. Failure rejects BGE-large under this exact representation without C, prompt, pooling, context, or
  blend rescue.
- Passing earns full 35,072-row context/objective caches and hardened fold-local evaluation only. Blend weights remain
  exactly 10%, 20%, and 30% over raw BGE-base replacement. `V_joint` and `V_final` restrictions remain unchanged.
  No platform upload or submission is authorized.
