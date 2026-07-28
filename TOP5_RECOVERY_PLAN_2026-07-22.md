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

## E460 freeze after E450 rejection and before binary SemEval training

E450 regressed loss, AUROC, and Brier with zero bootstrap support and is rejected without rescue. E400 also remains
rejected: its three-class checkpoint failed its frozen macro-F1 gate and may not be loaded or reinterpreted. E460 is
a new checkpoint from the untouched base NLI model and a different, competition-aligned external objective:
distinguish a fully correct student answer from every response that is not fully correct.

- Data: reuse the canonical SemEval-2013 Task 7 cache with content SHA-256
  `7a51ebe3b8c6d1537648c511836eb54360831006053eb65980f7736bd6351dfb`. Fit all 8,910 official training
  answers; evaluate the untouched 1,552 unseen-question and 4,562 unseen-domain answers separately. Label only
  `correct` as positive; partial, contradictory, irrelevant, and non-domain answers are negative.
- Model: start from the original `cross-encoder/nli-deberta-v3-small`, never the E400 delta. Replace its three-way
  classifier with a fresh two-logit head under seed `20260724`. Required initial weight SHA-256 is
  `696ad4a9b5d7c57ccf4906b3aca4a9e9d85921429b92ca7f096aacd6c4e6c55c`; the zero bias SHA-256 is
  `af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc`.
- Training: the existing fixed `[QUESTION] ... [STUDENT ANSWER] ...` premise and official reference-answer
  hypothesis, maximum length 256 with premise-only truncation, one epoch, batch size 16, unweighted cross-entropy,
  top two of six encoder layers plus pooler/head trainable, encoder/head learning rates `2e-5/1e-4`, weight decay
  `0.01`, 10% linear warmup, gradient clipping at 1.0, and six CPU threads. No E400 weights or soft labels enter.
- External gate: on both official evaluation splits, binary macro-F1 must be at least `0.65`, ECE-10 at most `0.10`,
  log loss must beat the fixed training-prior predictor by at least `0.02`, Brier must beat it by at least `0.01`,
  and question-ID bootstrap support for positive log-loss gain must be at least 95%. Unseen-question AUROC must be
  at least `0.72`; unseen-domain AUROC must be at least `0.76`. Every clause is required. Failure rejects this exact
  binary branch without threshold, epoch, class-weight, head, prompt, or checkpoint rescue.
- Passing earns only a competition cache from this new checkpoint, leakage-safe fold-local probes, and the frozen
  `V_seen`, `V_objective`, and `V_style` evaluation. The only candidate weights are 10%, 20%, and 30% over raw
  BGE-base replacement. `V_joint` and `V_final` restrictions, backup/top-five gates, and the manual-only submission
  boundary remain literal.

## E470 freeze after E460 rejection and before MPNet scoring

E460 created external ranking signal but failed its group-bootstrap, calibration, and unseen-domain AUROC clauses;
it is rejected without calibration or checkpoint rescue. E470 returns to the only robust positive evidence—the
sentence-embedding path—but changes the encoder architecture and contrastive pretraining family rather than scale.

- Model: official `sentence-transformers/all-mpnet-base-v2` revision
  `e8c3b32edf5434bc2275fc9bab85f82640a19130`, Apache-2.0, 768-dimensional normalized mean pooling. Download
  only the exact safetensors/config/tokenizer/module/model-card files.
- Sample and text: the exact 4,096-row `qwen3_pilot_indices.npy` sample, `semantic_k50_s0` folds, compact objective
  contexts, raw objective strings, maximum length 256, no instruction prefix, and separately normalized context and
  objective embeddings.
- Probe: the identical scaled `[context, objective, product, absolute-difference]` interaction, identical legal
  dense controls, fixed `C=0.1`, five fold-local logistic fits, and the existing session purge. The fair comparator
  is BGE-base on the identical rows, folds, labels, masks, and controls.
- Operational gate: 32 deterministic context token-length quantiles, batch size one, six CPU threads. Proceed only
  if 4,096 contexts plus unique objectives project below eight hours and peak RSS remains below 8 GB. Batch size,
  token length, pooling, and text remain fixed after this label-free benchmark.
- Continuation gate: either log-loss gain at least `0.0015` with AUROC/Brier/ECE non-regression, or AUROC gain at
  least `0.0050` with log-loss/Brier/ECE non-regression, plus at least 90% paired session-bootstrap support for
  positive loss gain. Failure rejects this exact MPNet representation without a C, pooling, prompt, context, token,
  or blend rescue.
- Passing earns a full 35,072-row cache and hardened fold-local validation. Competition blend weights remain only
  10%, 20%, and 30% over raw BGE-base replacement; `V_joint`, `V_final`, backup/top-five gates, and manual-only
  platform submission remain unchanged.

## E490 freeze after E470 rejection and before instruction-model scoring

E470 regressed loss/AUROC/Brier with zero bootstrap support, closing blind generic encoder replacement. The next
independent representation adds explicit outcome reasoning while remaining label-free at cache time.

- Model: official Apache-2.0 `Qwen/Qwen3-0.6B` revision
  `c1899de289a04d12100db370d81485cdf75e47ca`. Use the exact safetensors/config/tokenizer/chat-template files,
  CPU float32, six threads, evaluation mode, and no generated tokens.
- Sample: exact 4,096-row prior encoder sample and `semantic_k50_s0` folds. For every row, use the existing compact
  objective context and objective string only. Fixed system message: `You assess whether a student has mastered a
  K-12 learning objective after tutoring.` Fixed user message: `Learning objective: {objective}\nTutoring
  evidence:\n{context}\n\nWill the student answer the next assessment question correctly? Answer Yes or No.`
  Apply the official chat template with thinking disabled and truncate the evidence only to a 384-token total prompt.
- Frozen representation: L2-normalized final prompt-token hidden state plus the uncalibrated next-token
  `logit(Yes)-logit(No)` scalar. Cache construction is label-free; there is no generation, prompt variant, layer
  selection, or answer parsing.
- Operational gate: benchmark 16 deterministic prompt-length quantiles at batch size one. Proceed only if the full
  4,096-row cache projects below eight hours and RSS stays below 8 GB. Batch size, prompt, length, dtype, and threads
  remain fixed afterward.
- Fair screen: one fixed `C=0.1` fold-local logistic probe with the same legal dense controls and session purge,
  compared with BGE-base on identical rows/folds. Continue only through the existing dual loss/AUROC path with
  Brier/ECE non-regression and at least 90% paired session-bootstrap support. Failure rejects this exact
  representation without prompt, token, layer, C, calibration, or blend rescue.
- Passing earns full-cache and hardened-environment work only; 10/20/30% blends over raw BGE-base, sealed-evaluation
  restrictions, backup/top-five gates, and manual-only submission remain unchanged.

## E500 freeze after E490 rejection and before instruction-scale benchmark

E490 proves the fixed 0.6B instruction representation is inadequate and remains rejected. E500 isolates only model
scale using official Apache-2.0 `Qwen/Qwen2.5-1.5B-Instruct` revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. It inherits E490's exact system/user messages, 384-token
evidence-only truncation, official non-reasoning chat template, final-token normalized hidden state, Yes-minus-No
margin, 4,096-row sample/folds, legal controls, fixed `C=0.1`, fair BGE-base comparator, bootstrap, and continuation
gate. No generated tokens or prompt changes are allowed.

- Run only the same 16-row prompt-length-quantile CPU benchmark first. Batch size one, float32, six threads, less
  than eight projected hours, and less than 8 GB RSS are mandatory. Failure stops E500 before target scoring.
- A passing benchmark authorizes one resumable 4,096-row label-free cache and the unchanged fair screen. Failure of
  that screen rejects instruction-model scaling under this protocol without prompt, length, dtype, token, layer,
  C, calibration, or blend rescue.
- Only a passing screen can earn full-cache/hardened evaluation; all sealed-environment, backup/top-five, and
  manual-submission restrictions remain literal.

## E510 freeze after E500 rejection and before timing-dynamics scoring

E420/E450/E470/E490/E500 all show that another blind representation-capacity change is not justified. A complete
raw-field coverage audit found one competition-legal input family not yet modeled beyond coarse total duration:
the relative utterance timestamps. The target-free audit covered all 6,139,854 cached utterances and 22,821
sessions. Timestamps are complete and monotonic, every session has at least 10 distinct timestamps, and positive
inter-utterance gaps have median 6 seconds and 90th percentile 24 seconds. E510 therefore isolates interaction
timing without changing any text encoder or inspecting any new target result.

- Source: `data_cache/utterances.parquet`, SHA-256
  `80a231d18dbb0641989ebb972f08988f0ddd3cb7c4fb769ad2fe90b5a98ba5a4`. Use only `session_id`,
  `utterance_id`, relative `HH:MM:SS` timestamp, and the supplied role. No response label, objective ID, provider ID,
  test aggregate, inferred identity, or external annotation enters cache construction.
- Fixed session features: distinct-time share; zero-gap share; log1p mean/median/p90/p95/p99/maximum/standard
  deviation of positive gaps; shares of gaps over 10 and 30 seconds; the same fixed mean/median/p90/zero-gap
  summaries for tutor-to-student and student-to-tutor transitions; their median-latency asymmetry; early and late
  median positive gaps and log difference; early and late tutor-to-student medians and log difference; log1p total,
  student, tutor, and background turns per minute; and entropy of the fixed gap bins
  `[0], [1], [2], [3-5], [6-10], [11-30], [31+]`. Missing transition subsets receive deterministic zeros.
- Probe: one timing-only `LogisticRegression(C=0.1, solver="liblinear", max_iter=1000, random_state=20260724)`.
  Standardization is fit inside each legal outer-training fold only. The existing frozen `V_seen`, `V_objective`,
  and `V_style` assignments and session purge are used exactly; no feature selection, C sweep, nonlinear model,
  target calibration, or label-dependent cache statistic is allowed.
- Comparator: reconstruct raw BGE-base replacement exactly as
  `0.25 * pred_full + 0.25 * pred_role + 0.50 * pred_bge_base` from immutable Phase C component OOF artifacts.
  The development source run is `20260720T075633Z_environment_component_validation`; its
  `development_candidate_oof.parquet` SHA-256 is
  `07430f29fef800edd04800fc04392d5ca655e25897eb3712899d7cc0bf1d60a0`.
- Candidate weights are exactly 10%, 20%, and 30% timing component over raw BGE-base replacement. Select the lowest
  equal-fold-macro development log loss, with weight as the only selection dimension. No weight interpolation or
  post-hoc sweep is permitted.
- Development continuation gate: mean loss gain at least `0.00075` across `V_seen`, `V_objective`, and `V_style`;
  at least two environments improved; worst environment regression at most `0.0005`; macro AUROC, Brier, and ECE
  non-regression; and at least `90%` paired session-bootstrap support for positive macro loss gain. Failure rejects
  this exact timing component and all three frozen weights without opening any additional assignment or rescue.
- Passing locks the selected weight and permits one evaluation on the already-opened Phase C `V_joint` component
  artifact. It does not make `V_joint` untouched evidence. Promotion then applies the existing backup and top-five
  gates literally across all four hardened environments. `V_final` remains sealed unless the full top-five gate
  authorizes it. No platform upload or submission is authorized.

## E520 freeze after E510 rejection and before BGE-base multi-view scoring

E510 proves that additional session-style metadata is harmful. The remaining bounded representation gap is that
the validated BGE-base encoder was used only on the original mixed objective context, while the deterministic
student-only and ordered answer-feedback evidence views were screened only with weaker BGE-small. E520 isolates
whether BGE-base capacity makes those already-fixed views useful; it changes neither the view extractor nor the
encoder family.

- Sources: `response_multiview_texts.parquet` schema `2026-07-17-v2-budgeted`, SHA-256
  `e7b28d679220672ed379ed0de8f6cef9ece38cbeb7b0473cae11ae80584d4607`; exact 4,096-row pilot index SHA-256
  `4e62045be2bd473ef41e8aaf5f4b351b7432760ed6c1bbd7ccd88ca112d1efba`; and the existing
  `semantic_k50_s0` fold assignments. Student and feedback texts are already target-free, response-aligned,
  non-empty, and capped below 256 words by the frozen extractor.
- Encoder: the packaged MIT `BAAI/bge-base-en-v1.5` assets already used by the verified backup. Exact local
  safetensors/config/module/pooling SHA-256 values are
  `c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7`,
  `bc00af31a4a31b74040d73370aa83b62da34c90b75eb77bfa7db039d90abd591`,
  `84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf`, and
  `c9bef85e8bbf4b2eab4941b3fb62bd33f88686748b478f2e264d256472d9643b`.
  Use CPU float32, six threads, maximum length 256, packaged CLS pooling, L2 normalization, and batch 16.
- The label-free benchmark sampled 32 fixed length quantiles per view. Batch 16 was fastest:
  `0.1912` seconds/row for student evidence and `0.2339` for feedback, projecting about 29 minutes for both
  4,096-row caches. Proceed only below two projected hours and 8 GB RSS.
- Candidate representations are exactly: student evidence/objective interaction, feedback evidence/objective
  interaction, and the L2-normalized arithmetic mean of student and feedback embeddings/objective interaction.
  Every interaction is the existing fixed `[0.7*context, 0.7*objective, 6.0*product,
  0.7*absolute_difference]`; every probe uses fixed `C=0.1`, the same legal dense controls, training-fold-only
  standardization, and session purge. No alternate view, pooling, prompt, token length, layer, or C is allowed.
- Fair comparator: BGE-base original-context interaction at fixed `C=0.1` on the identical 4,096 rows, frozen
  folds, labels, purge masks, and controls. Each of the three candidate probes is evaluated only at preregistered
  10%, 20%, and 30% blends over that comparator. The lowest pilot log loss selects one representation/weight;
  there is no interpolation or post-hoc rescue.
- Continuation requires either at least `0.0015` log-loss gain with AUROC/Brier/ECE non-regression or at least
  `0.0050` AUROC gain with log-loss/Brier/ECE non-regression, plus at least `90%` paired session-bootstrap support
  for positive loss gain. Failure rejects all nine frozen combinations. Passing earns only the selected full cache
  and hardened `V_seen`, `V_objective`, and `V_style` evaluation with weights still limited to 10/20/30% over raw
  BGE-base. Existing `V_joint`, `V_final`, backup/top-five, and manual-only submission restrictions remain literal.

## E530 freeze after E520 rejection and before MathDial tutor-move transfer

E430's dialogue-level self-correction label failed, but this does not test MathDial's dense teacher-annotated move
taxonomy. Ikram, Scarlatos, and Lan (2025) report that tutor-move annotations complement dialogue text for outcome
prediction and that MathDial's `generic`, `probing`, `focus`, and `telling` moves have different outcome
associations. E530 transfers only this taxonomy and never loads the rejected E430 checkpoint.

- Data: official CC BY-SA 4.0 MathDial commit `b06c020a0a1f57a87577fec33e657b63e7eb476e`, local train/test JSONL
  SHA-256 `96980babee081a3da48ed0f1fb3068ab52f23ddc67e63bfd5ec99ad701fd29cc` and
  `28d1e537d65a6e6ff7b8bde602a2c7e2493b93d6bc785d1848506a650997f122`. Purge every official-train dialogue
  whose `qid` appears in official test, exactly as E430. Keep the full official test split immutable. Parse only
  non-empty `Teacher: (move) text` turns with move in the fixed four-class taxonomy; pair each with the most recent
  preceding non-teacher utterance. Report every excluded row and class count.
- Representation: fixed input `[STUDENT] {previous_student}\n[TUTOR] {teacher_text}`. Combine two independently
  L2-normalized `TfidfVectorizer` blocks with equal `1/sqrt(2)` weights: word 1-2 grams, 50,000 maximum features,
  and `char_wb` 3-5 grams, 75,000 maximum features; both use lowercase, Unicode accent stripping, `min_df=2`,
  sublinear TF, and training-only IDF.
- Classifier: one `LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=20260724)`, no class
  weights, threshold change, calibration, vocabulary change, feature sweep, or neural rescue.
- External gate on all non-empty official-test tutor turns: accuracy at least `0.60`, macro-F1 at least `0.55`,
  every class F1 at least `0.45`, top-label ECE-10 at most `0.15`, log loss at least `0.20` better and multiclass
  Brier at least `0.05` better than the fixed leakage-safe training-prior predictor, and at least `95%` test-`qid`
  bootstrap support for positive log-loss gain. Every clause is required.
- Failure rejects this exact move-transfer branch without weakening thresholds. Passing permits a target-free
  competition cache only: apply the fixed external classifier independently to tutor turns, then aggregate fixed
  move probabilities/fractions, early-late changes, and transitions both over the session and the existing
  objective-conditioned tutor-evidence view. Competition labels may train only fold-local linear probes; candidate
  weights remain exactly 10%, 20%, and 30% over raw BGE-base replacement. `V_joint`, `V_final`, backup/top-five,
  and manual-only submission restrictions remain literal.

## Public evidence amendment after the BGE-base submission and before E530 scoring

The participant manually submitted the unchanged verified BGE-base backup as full job `id-2518`. The supplied
platform log verifies completed execution, exit code `0`, root-level output generation, and public log loss
`0.6054`; the participant observed rank `#8`. Public AUROC was not included in the supplied log and is not inferred.
Runtime was approximately 242.6 seconds.

This replaces v0.2 as the public champion and is a `0.0027` public loss improvement over `0.6081`, larger than the
four-environment local gain of `0.001272`. The conservative `0.6038` top-five target is now `0.0016` away in public
loss. This public result is confirmation evidence only: it does not alter E530's data, representation, classifier,
external gate, fixed 10/20/30% competition weights, or sealed-evaluation rules, and it cannot authorize any
post-hoc calibration or weight sweep. The literal top-five local gate remains `0.0043` robust gain versus v0.2.

## E530 parser correction and final pre-score freeze

The earlier pause handoff quoted `11,139/3,699` move turns from a preliminary raw-teacher-turn audit. The exact
frozen parser, tested before any external prediction metric, applies every written exclusion and the exact E430
dialogue subset:

- Exclude the same nine official-train and four official-test dialogues with missing `self-correctness` used by
  E430, then purge the 314 labeled test `qid` values from labeled training. This yields exactly 1,679 training
  dialogues and 595 test dialogues with no question overlap.
- Initialize each dialogue's preceding non-teacher evidence with its supplied `student_incorrect_solution`, which
  is the student's documented state immediately before tutoring. Thereafter, update it after every non-teacher
  turn. This makes the first tutor move well-defined without inventing text.
- Split turns only on the official `|EOM|` marker. Remove the leading move annotation from tutor text before
  vectorization. Exclude one malformed training move marker, 42 empty training tutor texts, and 13 empty test tutor
  texts.
- The resulting immutable screen has 11,106 training turns
  (`focus=4,102`, `generic=2,611`, `probing=2,567`, `telling=1,826`) and 3,664 official-test turns
  (`focus=1,241`, `generic=884`, `probing=946`, `telling=593`).

These are corrections to preliminary counting, not model or threshold changes. The TF-IDF, classifier, external
gate, bootstrap, continuation, sealed-evaluation, and manual-submission contracts above remain unchanged.

## E530 scikit-learn 1.8 compatibility correction before scoring

The first E530 launch stopped after 12.3 seconds during classifier construction, before any fitted model,
prediction, external metric, bootstrap, report, or reusable checkpoint existed. Competition-aligned scikit-learn
1.8.0 rejects direct multiclass use of the `liblinear` solver instead of applying its historical one-vs-rest
behavior automatically.

To preserve the frozen solver rather than substitute a different optimizer, E530 now wraps the unchanged
`LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=20260724)` in scikit-learn's
`OneVsRestClassifier`. This fits one fixed liblinear binary estimator per move and normalizes their probabilities
for the exclusive four-class output. It changes no row, text, vocabulary rule, block weight, base estimator,
regularization, seed, metric, threshold, bootstrap, or gate. The failed pre-fit launch is not evidence and the
corrected run must start from zero.

## E530 completed result and rejection

Corrected run `20260724T164257Z_mathdial_tutor_move_transfer` completed from zero in 12.8 seconds under `.venv`
Python 3.12.8 and scikit-learn 1.8.0. It fit 11,106 training turns and evaluated all 3,664 immutable test turns
over 391 question IDs with a 97,930-column sparse matrix.

- Accuracy `0.536572 < 0.60`.
- Macro-F1 `0.498263 < 0.55`.
- Class F1: focus `0.566291`, generic `0.755182`, probing `0.250188`, telling `0.421390`; therefore the
  every-class `>=0.45` clause fails.
- Top-label ECE-10 `0.035565` passes.
- Log loss `1.026673` versus prior `1.357012` gains `0.330338`, passing the `0.20` clause.
- Summed multiclass Brier `0.567969` versus prior `0.735933` gains `0.167965`, passing the `0.05` clause.
- The 5,000-replicate test-`qid` bootstrap has `1.0000` positive-gain support, mean `0.330261`, and 95% interval
  `[0.311361, 0.349291]`, passing its clause.

Three of seven mandatory clauses fail. Reject this exact E530 branch without class weighting, threshold changes,
calibration, vocabulary/C/solver changes, alternate contexts, neural rescue, or selective class use. It does not
authorize a competition cache, fold-local outcome probe, hardened-environment evaluation, blend, package, or
public projection. `V_joint` and `V_final` were not accessed.

## E540 freeze after E530 rejection and before multi-instance BGE-base encoding

E530 is rejected and cannot supply move features. E540 returns to the only representation with confirmed public
transfer, BGE-base, but tests a genuinely different information bottleneck: the current model compresses up to 70
chronological objective-relevant transcript lines into one short string before encoding. E540 encodes four
chronological evidence instances independently and performs target-free semantic attention afterward.

- Immutable sources: `response_objective_context.parquet` SHA-256
  `218d5b041f78c6803f28078c35b7fa9e37b725f456c4052974e5150b2afa43f6`; exact 4,096-row pilot index
  SHA-256 `4e62045be2bd473ef41e8aaf5f4b351b7432760ed6c1bbd7ccd88ca112d1efba`; modeling, baseline OOF,
  BGE-base context, and objective hashes remain
  `ea49463819e387b3a61aeafda5007938e39948940eac5b134f607ab13e3d6ede`,
  `1693717193bae4bed75765d48b6c85de8f240eba8f5ac92b0724f499e7fe7ba6`,
  `b5f0066cc8d659e6593596ac47967bd14f2d0f02cedc82319e2a2313cf564d1a`, and
  `6cc754a287f9ec303aa5f81dd98ab0321c7c5299b2a4e251d6f86f63afe6eb27`.
- Fixed text transform: retain the existing objective header and split its already-selected discussion lines into
  four contiguous chronological bins with `numpy.array_split`. For a bin with more than eight lines, keep its
  first four and last four; retain all lines otherwise. Compact every retained line to its first 16 whitespace
  words. No target, role-specific rescore, cue lexicon, alternate retrieval, or learned selector enters.
- Label-free audit on all pilot texts: discussion-line count min/median/p95/max is `3/45/65/70`; all 16,384
  segments are non-empty and 16,355 are unique. BGE-base token-count min/median/p95/p99/max is
  `8/133/176/192/232`; none reaches the fixed 256-token limit.
- Encoder: the exact packaged MIT `BAAI/bge-base-en-v1.5` assets and hashes already verified for v0.5, CPU
  float32, six threads, batch 16, maximum length 256, packaged CLS pooling, and L2 normalization. Benchmark 32
  fixed length quantiles from each segment position before building. Proceed only below two projected hours for
  all 16,384 pilot segments and below 8 GB peak RSS; no batch/length/pooling change follows the benchmark.
- Fixed attention: for unit segment embeddings `e_i` and unit objective embedding `o`, compute
  `a_i = softmax(10 * dot(e_i, o))` across the four chronological instances, then L2-normalize
  `sum_i a_i e_i`. Temperature `10`, four bins, eight-line cap, and compaction are frozen; no alternative is
  scored.
- Fair screen: construct the existing
  `[0.7*context, 0.7*objective, 6.0*product, 0.7*absolute_difference]` interaction and the same legal dense controls,
  replacing only their final cosine with the attended-context/objective cosine. Fit fixed fold-local
  `C=0.1` logistic probes with session purge on the exact five `semantic_k50_s0` folds. Reconstruct the raw
  BGE-base original-context comparator on the identical rows and folds.
- Candidate weights are exactly 10%, 20%, and 30% attended probe over the fair BGE-base comparator; lowest pilot
  log loss selects the weight. Continue only through the existing E520 dual path: at least `0.0015` loss gain with
  AUROC/Brier/ECE non-regression, or at least `0.0050` AUROC gain with loss/Brier/ECE non-regression, plus at least
  90% paired session-bootstrap support for positive loss gain.
- Failure rejects this exact multi-instance attention branch without changing temperature, segment count,
  line allocation, compaction, C, pooling, encoder, dense controls, or blend weights. Passing earns full target-free
  cache construction and frozen `V_seen`/`V_objective`/`V_style` evaluation only. `V_joint`, `V_final`,
  backup/top-five gates, and manual-only platform submission remain literal.

### E540 benchmark authorization

The committed implementation passes seven focused E540/E520 tests and Ruff in the competition environment.
The fixed 128-row benchmark took `38.648` wall-clock seconds and projects `2,825.139` seconds (`0.784761` hours)
for all 16,384 segment encodes. Peak RSS was `1,084,813,312` bytes. Both frozen resource clauses pass, so the
exact cache build is authorized with one calculated completion checkpoint and no interim polling. Benchmark
SHA-256 is `7bbc44f76fb0c932e1b566f185b9ff0e48f97b7303af1fff01061ef26d9e6310`.
No target metric, `V_joint`, or `V_final` was accessed.

### E540 completed result and rejection

The cache build completed in `1,920.789` seconds with peak RSS `1,163,173,888` bytes. Cache SHA-256 is
`c5d54f1f69baf17a640fcff8f8f292cc2dbbe579c529257734885890520d2c48`; segment-text SHA-256 is
`6d9c04624f8a4a31e30d4bbe9dcb6ff43e03dd527ee186ddeebba9074cd13b9f`.

Frozen run `20260724T180101Z_semantic_attention` selected the preregistered `10%` blend, but it regressed the
identical-fold BGE comparator on every metric: log loss `0.595732349` versus `0.595607488`
(`+0.000124861` worse), AUROC `0.608344076` versus `0.608965679` (`-0.000621603`), Brier `0.203629751`
versus `0.203578114` (`+0.000051637`), and ECE-10 `0.024429701` versus `0.024219608` (`+0.000210093`).
The 5,000-replicate paired session bootstrap has mean log-loss gain `-0.000124980`, 95% interval
`[-0.000184901,-0.000065013]`, and zero positive-gain support. All three continuation clauses fail.

Reject E540 exactly without post-hoc attention, segmentation, compaction, C, pooling, feature, or blend changes.
No full cache, hardened evaluation, ZIP, or public projection is authorized. Report SHA-256 is
`5ef6d959574f362c36fb30d24d8a91b65d7b6211d1ad66636d6d6f89aa66ca02`.
`V_joint_accessed=false`; `V_final_accessed=false`.

## E620 freeze after E610 resource rejection and before frozen multilingual mastery scoring

E610 is closed and its trainable mDeBERTa model may not be shortened or substituted. E620 is a separately
preregistered question: whether the aligned MCD mastery labels are already linearly available in a small,
fully-frozen cross-lingual sentence representation. It performs no encoder adaptation and cannot inherit an E610
checkpoint.

- Reuse only E610's immutable MCD raw sources, canonical compaction, label map, and 4,242-row/399-group training
  versus 984-row/95-group test split. Canonical ordered-content and Parquet SHA-256 values remain
  `6336d7d377d67c34c8ea5471d2a7f6e3c119b8d0448b35f118bfa4f01401d4f9` and
  `4180bbb1829214c569f084e3387eb38352d5b241e03f217736305b0c97d4dc2f`.
- Encoder: official MIT `intfloat/multilingual-e5-small` revision
  `614241f622f53c4eeff9890bdc4f31cfecc418b3`, safetensors SHA-256
  `1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477`.
  Prefix every canonical text with `passage: `, tokenize with maximum 384 and left truncation, run CPU float32,
  six threads, batch 16, attention-mask mean pool, and L2 normalize. Every encoder parameter remains frozen.
- Fit exactly one multinomial `LogisticRegression(C=1.0, solver="lbfgs", max_iter=400,
  random_state=20260727)` on legal training embeddings, with no scaling, class/sample weights, calibration,
  threshold, or feature addition. Evaluate the complete group-disjoint test.
- Before full encoding, benchmark 32 fixed SHA-ordered train rows and 32 fixed SHA-ordered test rows. Project all
  5,226 external plus 4,096 competition-pilot encodes; proceed only below four hours and 8 GiB RSS.
- External gate requires accuracy at least `0.50`, macro-F1 at least `0.42`, every-class F1 at least `0.30`,
  macro one-vs-rest AUROC at least `0.65`, quadratic-weighted kappa at least `0.25`, log loss at most `1.00`,
  summed multiclass Brier at most `0.60`, log-loss gain at least `0.10` over the legal train-prior predictor, and
  at least 90% positive-gain support in 2,000 source-group bootstraps. Every clause is mandatory.
- Failure rejects E620 without model, prefix, length, pooling, C, scaling, class weight, calibration, or
  threshold rescue. Passing permits the exact 4,096-row `semantic_k50_s0` competition pilot only. Encode
  `passage: Tutoring evidence: {objective_context}\nObjective: {learning_objective}` under the same contract;
  concatenate the 384-dimensional unit embedding, three frozen MCD logits, and the existing 35 controls. Fit the
  legal fold-local `C=0.1` probe and evaluate only 10%, 20%, and 30% blends over the fair raw BGE comparator.
  Continue only through the unchanged `0.0015` loss or `0.0050` AUROC path with Brier/ECE non-regression and
  90% session-bootstrap support. V_joint remains confirmation-only, V_final sealed, and submission manual-only.

### E620 benchmark, external result, and rejection

The 64-row frozen-encoder benchmark completed in `5.919467` seconds and projected `862.207336` seconds
(`0.239502` hours) for all 9,322 conditional external/pilot encodes at `1,021,825,024` bytes RSS. Benchmark
SHA-256 is `7702b40a05de1185e0749b5e03acb9e4b74ce3a9a6c4cfff4014219cb1679504`.
The authorized external cache then completed all 5,226 rows in `476.882965` seconds; its shape is
`[5226,384]`, cache SHA-256 is `6cd5b733970ddbfa94b9507c047c6915bd83b664bddbea99a4a923fb36f16a1f`,
and metadata SHA-256 is `04d87c4f1675bb8b3f7b805fe12967ad868d044dfae52fa36b13326c34979990`.

The frozen group-disjoint test produced accuracy `0.659552846`, macro-F1 `0.469735660`, class F1
`[0.761372706,0.000000000,0.647834275]`, macro one-vs-rest AUROC `0.761372780`, quadratic kappa
`0.494533948`, log loss `0.820439933`, summed multiclass Brier `0.470915542`, and reporting-only top-label
ECE-10 `0.056854967`. Loss improved over the legal train-prior predictor by `0.193923616`. The 2,000-draw
95-source-group bootstrap has mean gain `0.164471962`, 95% interval `[0.122465764,0.205161815]`, and `1.0000`
positive support. Eight clauses pass, but the mandatory every-class F1 clause fails because the Understanding
class F1 is zero.

Reject E620 exactly without model, prefix, length, pooling, C, scaling, class weight, calibration, threshold, or
label rescue. No competition pilot/cache, hardened validation, ZIP, or public projection is authorized. External
report SHA-256 is `5cd14a0cf186e95af59c163d2174fa35913c874b4a0b1383f1ee2bef6ff87467`;
ordered prediction SHA-256 is `d0bac62924fc25033af91cdffade92c14b443bc717b4dd1362c66bc6bcf192b6`.
`V_joint_accessed=false`; `V_final_accessed=false`.

## E560 freeze after E550 rejection and before nonlinear-head scoring

E520, E540, and E550 show that new BGE-base views and pooling do not improve the fair pilot. E560 keeps the exact
publicly confirmed CLS representation and asks one different question: whether the fixed linear outcome boundary
is the remaining bottleneck. It uses no new text, model, external data, pooling, or target-derived test feature.

- Inputs are the exact deployed BGE-base interaction block
  `[0.7*context, 0.7*objective, 6.0*product, 0.7*absolute_difference]` and the same 35 legal controls,
  standardized only inside each legal outer-training fold and scaled by `0.08`. Exact pilot rows, five
  `semantic_k50_s0` folds, validation-session purge, and fair `C=0.1` logistic comparator remain unchanged.
- Candidate head: a 32-unit GELU residual MLP with direct linear logit, 15% dropout on the nonlinear path, and one
  scalar bias initialized to the legal fold-training logit prior. Direct and nonlinear output weights initialize
  at zero; hidden weights use Xavier uniform. Train exactly 30 epochs with batch 128, deterministic fold shuffle,
  AdamW learning rate `5e-4`, weight decay `1e-3`, gradient-norm cap `1.0`, unweighted BCE-with-logits, six CPU
  threads, and seed `20260725 + fold`. No early stopping, checkpoint selection, class weighting, or calibration.
- Evaluate exactly 10%, 20%, and 30% residual-head probability over the fair raw BGE-base comparator and select
  the lowest pilot loss. Continue only through the same dual path: at least `0.0015` loss gain with
  AUROC/Brier/ECE non-regression, or at least `0.0050` AUROC gain with loss/Brier/ECE non-regression, plus at
  least 90% positive-gain support in 5,000 paired session bootstraps.
- Failure rejects this exact nonlinear head without hidden-width/depth, activation, dropout, initialization,
  epoch, batch, optimizer, learning-rate, weight-decay, seed-ensemble, input-scaling, calibration, or blend rescue.
  Passing earns only frozen `V_seen`, `V_objective`, and `V_style` evaluation; `V_joint` remains confirmation-only
  and `V_final` sealed until the literal top-five gate. Platform submission remains manual-only.

### E560 benchmark authorization

The deterministic label-free benchmark trained one legal-fold-sized residual head for one epoch in `1.878458`
seconds using index-parity labels and no outcomes. This projects `281.769` seconds (`0.078269` hours) for the
frozen five folds and 30 epochs. Peak RSS was `1,647,194,112` bytes. Both operational gates pass, authorizing
exactly one unattended pilot validation with one calculated completion checkpoint and no interim polling.
Benchmark SHA-256 is `dd91b8e4f908620ce0b7a0dbb8afab182f0291f7cd933eb0b835e3bbe16e21ec`.
`V_joint_accessed=false`; `V_final_accessed=false`.

### E560 completed result and rejection

Frozen run `20260724T183952Z_residual_head` selected the preregistered `30%` blend. Log loss improved by
`0.001047773`, Brier by `0.000390984`, and ECE-10 by `0.006621537`, but AUROC regressed by `0.002874485`.
The 5,000-draw paired session bootstrap has mean gain `0.001044909`, 95% interval
`[-0.000512153,+0.002581618]`, and `0.9120` positive-gain support. Bootstrap passes, but loss gain is below the
`0.0015` threshold and AUROC non-regression fails; both continuation paths therefore fail.

Reject E560 exactly without architecture, optimizer, epoch, seed, input, calibration, or blend rescue. No hardened
evaluation, ZIP, or public projection is authorized. Report SHA-256 is
`80351ff8af062e37edd9aaf206c15174af79b811c98a6d7f472e7bc1124366a4`.
`V_joint_accessed=false`; `V_final_accessed=false`.

## E570 freeze after E560 rejection and before pairwise-ranker scoring

E570 returns to a convex linear logit and isolates a new learning objective: transferable within-objective
ordering of transcript evidence. It does not reuse E560's nonlinear architecture, dropout, minibatch optimizer,
or checkpoint.

- Inputs and evaluation rows are the exact BGE-base interaction block and 35 fold-local standardized controls on
  the fixed 4,096-row `semantic_k50_s0` pilot, with the same validation-session purge and fair `C=0.1` logistic
  comparator.
- Pair rule: inside each legal outer-training fold, sort by `response_id`. For every `learning_objective_id` with
  both outcomes, create `max(n_positive,n_negative)` positive-negative pairs, repeating the smaller class
  cyclically. Never pair across objectives; do not mine, weight, filter, or select pairs by prediction.
- Model/optimizer: zero weight and legal fold-prior bias linear logit, full-batch PyTorch L-BFGS, maximum 100
  iterations, history 20, strong-Wolfe line search, `1e-7` gradient tolerance, `1e-9` change tolerance.
  Minimize exactly `0.5 * mean BCE + 0.5 * mean softplus(-(positive_logit-negative_logit)) +
  0.0015 * ||w||^2`.
- Evaluate exactly 10%, 20%, and 30% ranker probability over the fair BGE comparator. Continue only through the
  unchanged `0.0015` loss path or `0.0050` AUROC path with loss/Brier/ECE non-regression and at least 90%
  positive-gain support in 5,000 paired session bootstraps.
- Failure rejects E570 without loss-mixture, pair construction, margin, L2, iteration, optimizer, input,
  calibration, or blend-weight rescue. Passing earns frozen hardened selection evaluation only; `V_joint` remains
  confirmation-only and `V_final` sealed until the literal top-five gate. Platform submission remains manual-only.

### E570 benchmark authorization

The label-free index-parity benchmark completed one full fold fit in `1.754113` seconds with 11 L-BFGS iterations
and 15 closure evaluations, projecting `8.770566` seconds for five folds. Peak RSS was `1,643,012,096` bytes.
Both resource gates pass; the frozen validation is authorized directly. Benchmark SHA-256 is
`ecc439a519bcda6becc718916519edc6eb75842311230cec235ed64846a15c35`.
No outcome score, `V_joint`, or `V_final` was accessed.

### E570 completed result and rejection

Frozen run `20260724T185443Z_objective_pairwise_ranker` selected the preregistered `30%` blend. It improved log
loss by `0.000440043`, AUROC by `0.003218648`, and Brier by `0.000198833`, but worsened ECE-10 by
`0.006165258`. The 5,000-draw paired session bootstrap has mean gain `0.000442064`, 95% interval
`[-0.000014790,+0.000908053]`, and `0.9706` positive-gain support. Bootstrap passes, but both magnitude paths
and ECE non-regression fail.

Reject E570 exactly without loss-mixture, pairing, regularization, optimizer, calibration, or blend rescue.
No hardened evaluation, ZIP, or public projection is authorized. Report SHA-256 is
`22aca753689c361d83561d45dbc0c62911619b42e6803088e62fb6df109f3601`.
`V_joint_accessed=false`; `V_final_accessed=false`.

## E580 freeze after E570 rejection and before multi-corpus correctness training

E580 is a new external-transfer checkpoint trained from the untouched original DeBERTa NLI base. It does not load
or continue the rejected E400 delta. Canonical SemEval supplies natural correct/partial/incorrect responses;
MIT GSM8K adds explicit mathematical final-answer verification.

- Immutable SemEval cache SHA-256 is
  `527256be4e5f2e509f725b302ab8b163cc57760560a58621b2c02880ebe5dbb1`; use its exact 8,910 training rows and
  official unseen-question/unseen-domain sets. GSM8K main train/test parquet SHA-256 values are
  `ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d` and
  `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59`.
- All GSM final answers are integers after `####`. Select 4,455 training questions by
  `(SHA256(question), original_row)` order; selected-question SHA-256 is
  `2bbcc18022ccf879c60a2cbaf46d8487e89e74c22c723fe4130a9f8ea1bb4092`.
  Create one entailment row from the original solution and one contradiction row by changing only the final
  integer `n` to `n+1` for nonnegative `n` or `n-1` for negative `n`. Fixed hypothesis:
  `The correct final answer is n.` Use all 1,319 official test questions as 2,638 untouched paired rows.
- Training contains exactly 8,910 SemEval plus 8,910 GSM rows, shuffled once with seed `20260725`. Start from the
  original `cross-encoder/nli-deberta-v3-small`; one epoch, batch 16, maximum 256, only top two of six encoder
  layers plus pooler/head trainable, encoder/head learning rates `2e-5/1e-4`, weight decay `0.01`, 10% warmup,
  gradient cap `1.0`. No class/corpus weighting or checkpoint selection.
- The original E400 clauses remain unchanged: unseen-question correct-vs-rest AUROC at least `0.70`, unseen-domain
  AUROC at least `0.68`, unseen-question macro-F1 at least `0.45`, and no AUROC regression versus untouched base
  on either split. Additional GSM clauses require correct-vs-corrupt AUROC at least `0.95`, two-class-normalized
  log loss at most `0.35`, and no AUROC or loss regression versus the untouched base.
- Any failure rejects the exact E580 checkpoint without corpus weight, selection, corruption, prompt, label,
  epoch, layer, optimizer, learning-rate, calibration, or checkpoint rescue. Passing earns competition cache
  construction and the already frozen fold-local/hardened protocol only. `V_final` stays sealed; submissions
  remain manual-only.

### E580 cache and benchmark authorization

The competition-aligned implementation passes Ruff and seven focused E580/E400 tests. The canonical cache has
`17,820` training rows and `2,638` official paired GSM8K test rows; ordered-content SHA-256 is
`1ef624e1cf53708d52628cc3e05da117986b7e34a623dbe8427ed06ce8cac78c` and Parquet SHA-256 is
`2ed6dbaa3f02e8333735c1c7005f17c28e6d1181a4670a47781736b634d89003`.

The corrected two-training-batch/two-evaluation-batch benchmark took `11.425449/19.504990` seconds,
projects `9,036.159` seconds (`2.510044` hours) for 1,114 training steps plus both frozen base/candidate external
passes, and observed `1,851,256,832` bytes RSS. Both six-hour and 8 GiB resource clauses pass. Benchmark SHA-256
is `abe6e69b01a7ee297eb48b183fbd1dac5b8a4ebbfff5ddfd446793e3da48d8dc`.
Authorize exactly one E580 material run from step zero with one calculated completion heartbeat and no inspection
of its worker or logs beforehand. No outcome score, `V_joint`, or `V_final` was accessed by the benchmark.

### E580 completed external result and continuation

Run `20260727T085628Z_multicorpus_correctness_transfer` completed all `1,114/1,114` steps in approximately
`10,770.321` seconds and passed all nine frozen external clauses:

- unseen-question AUROC `0.711945848`, macro-F1 `0.455755785`, and multiclass loss `1.041258151`;
- unseen-domain AUROC `0.760500763`, macro-F1 `0.469804458`, and multiclass loss `0.997024533`;
- GSM paired AUROC `0.994636045`, log loss `0.111028344`, Brier `0.033980069`, ECE-10 `0.024192835`, and
  accuracy `0.950720243`.

The untouched base scored unseen-question/unseen-domain AUROC `0.596582994/0.724288305` and GSM AUROC/loss
`0.778627064/0.648353903`, so every non-regression clause also passes. Delta SHA-256 is
`13a2b5aa4d4fefa122b65c15ece3407719a29df8903047926dfe7acbb0116e9a`; report SHA-256 is
`acbff2ab6af75cddd68c021d90e45e8451adf8e10c864d8d5945e832fb25201f`.

Accept E580 through its external gate only. This authorizes target-free competition cache construction and the
already frozen leakage-safe fold-local probe protocol; it does not yet authorize a backup ZIP, `V_final`, or a
platform submission. No post-hoc corpus, prompt, model, optimizer, or checkpoint change is permitted.

### E580 competition result and rejection

The target-free 35,072-row cache completed in `5,732.478` seconds. Pooled/logit shapes are
`[35072,768]/[35072,3]`; SHA-256 values are
`c02bc9774e95803567feb4e25a8541f1b69e5d361e9bc027a9b522cb825c1685` and
`822a7f5e845d8c84c992138f5d5fa16d453cb64f3f55a9e58cca146b8e2d6722`.

Frozen run `20260727T105659Z_external_sra_validation` evaluated only the preregistered 10%, 20%, and 30% blends.
None passed the three-environment development guard:

- 10% mean loss gain `-0.000219870`, only one environment improved, worst regression `0.001443243`, mean AUROC
  gain `0.001348021`, Brier gain `-0.000085859`, and ECE gain `-0.001373519`;
- 20% mean loss gain `-0.002116213`, worst regression `0.004246710`;
- 30% mean loss gain `-0.004322192`, worst regression `0.007410665`.

At 10%, V_objective improved loss by `0.002137427`, but V_seen and V_style regressed by
`0.001353793/0.001443243`; V_style AUROC also regressed `0.000162913`. Therefore no weight was selected,
V_joint was not opened, and bootstrap/top-five projection are not applicable. Reject E580 exactly without
corpus, cache-view, prompt, probe, calibration, or blend rescue. No ZIP or submission is authorized; V_final
remains sealed. Validation report SHA-256 is
`6c3b9ef0cccf950067a222c2e4f2be6677c432b45dbc7815c44ecdfa2017586c`.

## E590 freeze after E580 rejection and before PRM800K scoring

E580 proved that explicit mathematical correctness can preserve SemEval transfer, but its synthetic final-answer
corruption learned an objective-shift-only signal and regressed seen/style calibration. E590 is a fresh checkpoint
from the untouched NLI base using human step-level correctness judgments, not an E580 checkpoint or data-weight
rescue.

- Source: OpenAI PRM800K at commit `7ecc794703b2877f63226f2477a49b34f9b25163`, MIT license SHA-256
  `f213be7e9bf1040b5407cdc7b55c24a053c8fe7bc9b9f755541d822ff8af814b`. Phase-2 train/test SHA-256 values
  are `1110237feeb51d1bc200cb37b8f965cfdc1036eac7d506094049366fe7dc1089` and
  `6b172efa884ac8341a946dd82e06947c135b7254109fb3f7aa907c715d98aaad`; official MATH train/test split
  hashes are `90d96daeac3fe343ebb1e22ce93dd99690f75983e957f88de42f87cffe1e8076` and
  `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`.
- Split: admit only phase-2 train examples whose problem occurs in the official 11,999-problem train split;
  exclude quality-control and initial-screening rows. The complete phase-2 test set maps to 458 distinct official
  held-out problems with zero train-question overlap. Exclude flagged, blank, or unrated completions. Hash the
  exact `(problem, ground truth, prior chosen steps, candidate step)` input; remove all inputs with conflicting
  ratings and deduplicate exact repeats.
- Labels map human rating `-1/0/+1` to contradiction/neutral/entailment. From the remaining training pool, select
  exactly 2,970 unique inputs per class by SHA-256 order, for 8,910 PRM rows. Selected content SHA-256 is
  `73eed21b2a1fe1d26a5ab93b9751dd616b8892687294d4d1796147f4d3bddc3d`.
  Evaluate every 25,530 unique non-conflicting phase-2 test input: 5,810 contradiction, 1,938 neutral, and
  17,782 entailment; content SHA-256 is
  `456ecb3150c941b2d457bfb37476e687b47a8ad5847f100d17a87c72a9d74774`.
- Fixed PRM premise order is candidate step, problem, ground-truth solution, then prior reasoning, compacted to
  first `64/64/80` whitespace tokens and the last 48 prior-reasoning tokens respectively. Fixed hypothesis:
  `The candidate reasoning step is mathematically correct and advances the solution.` No alternative text,
  context budget, label mapping, sample balance, or data mixture may be scored.
- Train one fresh checkpoint on the canonical 8,910 SemEval rows plus 8,910 PRM rows, shuffled once with seed
  `20260727`. Preserve E580's original model, one epoch, batch 16, maximum length 256, top two trainable encoder
  layers plus pooler/head, learning rates `2e-5/1e-4`, weight decay `0.01`, 10% warmup, gradient cap `1.0`, and
  no weighting or checkpoint selection.
- Preserve every SemEval clause from E580. On the full question-disjoint PRM test set require macro-F1 at least
  `0.50`, contradiction-vs-rest AUROC at least `0.75`, multiclass log loss at most `0.90`, log-loss improvement
  of at least `0.10` versus the untouched base, no contradiction-AUROC regression, and at least `0.95`
  held-out-question-bootstrap support for positive log-loss gain using 2,000 replicates.
- Any failed clause rejects E590 without prompt, context, sample, class/corpus weight, label, layer, optimizer,
  calibration, or checkpoint rescue. A pass earns only the same target-free session cache and frozen 10%/20%/30%
  fold-local competition screen. `V_final` remains sealed and all submission remains manual-only.

### E590 cache and benchmark authorization

The canonical 43,350-row cache reproduces both frozen PRM content hashes. Ordered-content SHA-256 is
`bcb3781c4c46b97a9404ba20dc8f54e196fe5a9f534fc462cd5abccb1322e974`; Parquet SHA-256 is
`9f48c85106c5dfc46aad9b4aeaf12d7cdf20ac2b44f785b046bd3cbdf15282e6`.

The two-training-batch/two-evaluation-batch benchmark took `12.813906/20.604873` seconds and projects
`17,336.757` seconds (`4.815766` hours) for all 1,114 steps and two complete 31,644-row external passes.
Observed RSS was `1,929,891,840` bytes. Both the frozen six-hour and 8 GiB gates pass. Benchmark SHA-256 is
`e286fc0ee006cb6fe1f73cbb0ecae6d81c7b3b1e115d9fe365583977ef72e47c`.
Authorize exactly one E590 material run from zero with one calculated completion heartbeat and no process/log
inspection beforehand. No prediction metric, competition outcome, V_joint, or V_final was accessed.

### E590 completed external result and rejection

Run `20260727T154320Z_prm_correctness_transfer` completed all 1,114 steps in `16,183.689` seconds. It improved
PRM test log loss from `1.669510424` to `1.079224649`; the 2,000-replicate held-out-question bootstrap mean gain
was `0.590862304`, 95% interval `[0.519872697,0.663775560]`, with `1.0000` positive support. Nevertheless,
PRM macro-F1 `0.303501964`, contradiction AUROC `0.581801001`, and loss `1.079224649` failed their
`0.50/0.75/0.90` clauses.

SemEval ranking remained strong—unseen-question/domain AUROC `0.717038025/0.752461658`—but unseen-question
macro-F1 was `0.417453866 < 0.45`. Four frozen clauses therefore fail. Reject E590 exactly without corpus,
sampling, context, prompt, label, class weight, optimizer, threshold, or checkpoint rescue. No competition cache,
ZIP, V_joint, V_final, or submission is authorized. Delta/report SHA-256 values are
`400b98d5de925bd720a2aa7311ac8d11413d704ffaa3a5960251f6649f00d3f5` and
`7fb4fbad0c642151a7a19c6e1c8c893b1e09e72fa9a0225d8d05336ae140153e`.

## E600 freeze after E590 rejection and before session-bagged scoring

The external correctness programs improved shifted educational ranking but repeatedly failed classification or
seen/style transfer. E600 returns to the only publicly confirmed BGE-base feature family and tests estimator
variance rather than a new encoder, view, head, objective, or calibration.

- Use the exact 4,096-row `semantic_k50_s0` pilot, frozen five outer folds, session purge, BGE-base interaction
  block, 35 legal controls, fold-local standardization scaled by `0.08`, `C=0.1` L-BFGS logistic estimator, and
  identical single-fit BGE comparator from E560/E570.
- Inside each legal outer-training fold, assign every training session to one of five immutable slices using
  `int(SHA256("E600|" + session_id)[:16],16) mod 5`. Fit five estimators; estimator `k` excludes only slice `k`.
  Each estimator fits its own dense scaler on its legal retained rows. Average their five validation
  probabilities with equal 20% weights. No label stratification, bootstrap resampling, class weighting, or
  prediction calibration is allowed.
- The candidate is the bagged probability itself, not a blend-weight sweep. Continue only if it improves pilot
  log loss by at least `0.0015` with AUROC/Brier/ECE non-regression, or AUROC by at least `0.0050` with
  loss/Brier/ECE non-regression, and 5,000 paired session bootstraps give at least 90% positive-loss-gain support.
- Failure rejects this exact bagging estimator without bag count, slice hash, retained fraction, C, scaling,
  feature, class weight, calibration, or blend rescue. Passing earns only frozen V_seen/V_objective/V_style
  evaluation; V_joint remains confirmation-only and V_final sealed. Submission remains manual-only.

### E600 completed result and rejection

Frozen run `20260727T164822Z_session_bagging` worsened the identical single-fit BGE comparator on every required
metric. Candidate-minus-comparator changes were log loss `+0.001110363`, AUROC `-0.001700568`, Brier
`+0.000450354`, and ECE `+0.005814786`; all five outer folds regressed in loss. The 5,000-replicate paired
session bootstrap estimated mean log-loss gain `-0.001108170`, 95% interval
`[-0.001456518,-0.000759177]`, and zero positive-gain support.

Reject E600 exactly without any bag-count, session-hash, retained-fraction, C, scale, feature, class-weight,
calibration, or blend rescue. No hardened evaluation, ZIP, public projection, or submission is authorized.
Report SHA-256 is `a0bddb097db481989044ac751c0c088e7793685b5399f2abb64be0f1d676f1c6`.
`V_joint_accessed=false`; `V_final_accessed=false`.

## E610 freeze after E600 rejection and before MCD cross-lingual mastery transfer

E600 confirms that estimator variance is not the missing signal. The external audit therefore changes the
supervision source, not the rejected BGE estimator. E610 uses the closest permissively licensed public task found:
authentic one-to-one grade-8 mathematics dialogue labeled by demonstrated mastery level.

- Source: public MIT repository `ai4ed/MCD` at commit
  `7ffc5e97948654d91e1cd368a3ecce96e0f4763a`, with its raw Google Drive payload linked by that repository.
  Immutable raw SHA-256 values are
  `153d77a2324cdcb73d7c41f87fc216821cb9cc9a5284a02459cb90362c5d1cf3`
  (`df_feature_num_label-3.csv`),
  `103c9f9cc878dc491e67668f7e9e574b066fa72504efcf0a6ab1fff8a73c4157`
  (`item_dict_anonymized.json`), and
  `34e5f6c610ed65b4be0f5ea3d8e714995dc2274914bcdb80e2252d4a8f82e37d`
  (`train_dev_test.json`). There are 5,226 unique transcripts and no exact transcript duplicates.
- The published split is not used for scoring because the identifier prefix shared by related source recordings
  overlaps heavily across its train/dev/test partitions. Define `source_group` as the substring of `new_id`
  before the first underscore. Assign group test iff
  `int(SHA256("E610|" + source_group)[:16],16) mod 5 == 0`; all other groups train. This yields 4,242 rows
  over 399 train groups and 984 rows over 95 disjoint test groups, with zero group or exact-transcript overlap.
  Supplied integer labels are kept in their documented ordinal order
  `0=Apprentice, 1=Understanding, 2=Mastery`; no handcrafted numeric features are model inputs.
- Each dialogue input uses the last 24 non-empty turns in original order. Prefix each with `Tutor:` or `Student:`,
  keep the first 24 Unicode characters of that turn, then append
  `Task: classify demonstrated math mastery as Apprentice, Understanding, or Mastery.` Tokenizer truncation is
  from the left at 384 tokens so the latest evidence and task remain. No translation, target lexicon, metadata,
  timestamps, official-split indicator, or numeric feature enters.
- Backbone: official MIT `microsoft/mdeberta-v3-base` revision
  `a0484667b22365f84929a935b5e50a51f71f159d`, multilingual CC100 pretraining, 12 layers and hidden size 768.
  Freeze embeddings and encoder layers 0-9; train only layers 10-11, pooler, and a fresh three-logit classifier
  initialized once under seed `20260727`. Train exactly one epoch, batch 8, gradient accumulation 2, AdamW
  learning rate `2e-5`, weight decay `0.01`, linear warmup 6%, gradient cap `1.0`, unweighted cross-entropy,
  deterministic row shuffle, CPU float32 and six threads. No checkpoint selection or class weighting.
- Before material training, benchmark two fixed train batches, two fixed test batches, and the matching backward
  passes; project the complete base-feature pass, 266 optimizer steps, and base/candidate test evaluation.
  Proceed only below 10 hours and 8 GiB RSS.
- External comparator: one fold-safe multinomial
  `LogisticRegression(C=0.1, solver="lbfgs", max_iter=400, random_state=20260727)` fitted on untouched
  mDeBERTa normalized CLS vectors from the 4,242 legal training rows. E610 must achieve test accuracy at least
  `0.55`, macro-F1 at least `0.48`, every-class F1 at least `0.35`, macro one-vs-rest AUROC at least `0.70`,
  quadratic-weighted kappa at least `0.35`, log loss at most `0.95`, and summed multiclass Brier at most `0.55`.
  It must not regress from the frozen-CLS comparator in log loss, macro AUROC, or kappa, and a 2,000-replicate
  source-group bootstrap must give at least 90% support for positive paired log-loss gain. Every clause is
  mandatory.
- Failure rejects E610 without source-group, label, compaction, length, layer, epoch, optimizer, class-weight,
  calibration, threshold, checkpoint, or prompt rescue. Passing permits only a target-free 4,096-row pilot
  cache: normalized candidate CLS plus three raw logits from
  `Tutoring evidence: {objective_context}\nObjective: {learning_objective}` at the same 384-token left-truncated
  contract. Fit the exact legal fold-local `C=0.1` probe on those 771 values plus the existing 35 controls and
  evaluate only 10%, 20%, and 30% blends over the fair raw BGE comparator. Continue only through the unchanged
  `0.0015` loss or `0.0050` AUROC path with Brier/ECE non-regression and 90% session-bootstrap support.
  A pilot pass earns full target-free caching and frozen V_seen/V_objective/V_style evaluation; V_joint remains
  confirmation-only and V_final sealed. Platform submission remains manual-only.

### E610 tokenizer compatibility correction before cache construction

Transformers 5.14.1 requires the official SentencePiece parser and flags the legacy tokenizer regex embedded in
this older checkpoint. Before any tokenization or prediction, pin `sentencepiece==0.2.1` and
`protobuf==6.33.5`, and load the unchanged official tokenizer with `fix_mistral_regex=True`. This is the
library's explicit compatibility correction; it changes no model/data source, split, text, maximum length,
training setting, metric, or gate. The downloaded PyTorch and SentencePiece payloads exactly match published
SHA-256 values `6f89419baf0f1aaad5cab7d53901e36a8c1af8f6b4ab58b15db9af32df656ead`
and `13c8d666d62a7bc4ac8f040aab68e942c861f93303156cc28f5c7e885d86d6e3`.

### E610 canonical-cache authorization

The source-bound canonical builder reproduces the frozen 4,242/984 rows and 399/95 disjoint groups. Ordered
content SHA-256 is `6336d7d377d67c34c8ea5471d2a7f6e3c119b8d0448b35f118bfa4f01401d4f9`;
Parquet SHA-256 is `4180bbb1829214c569f084e3387eb38352d5b241e03f217736305b0c97d4dc2f`.
Two clean initializations reproduce trainable-state SHA-256
`4e51494114193795988a96c8313394dcda333ff7e696053f9363e087d43a4b0f` across exactly 14,768,643 trainable
parameters. No prediction metric or competition outcome was accessed. The frozen target-free resource benchmark
is authorized.

### E610 resource result and rejection

The two fixed synthetic-label training batches took `361.492176` seconds and the two target-free evaluation
batches took `106.720519` seconds. This projects `137,437.094` seconds (`38.176971` hours) for the frozen
base-feature pass, 531 physical training batches/266 optimizer steps, and candidate test evaluation. Peak RSS was
`1,465,061,376` bytes. The memory clause passes, but the frozen ten-hour ceiling fails by a factor of 3.82.

Reject E610 exactly at its resource gate without sequence-length, compaction, layer, batch, epoch, optimizer,
checkpoint, quantization, or model substitution rescue. No MCD outcome prediction, external metric, competition
cache, hardened validation, ZIP, or public projection is authorized. Benchmark SHA-256 is
`3de48a2b8282bed89bfac15e30a4d73c32896bbe793c4cfb42ad6d80ba6e74d6`.
`V_joint_accessed=false`; `V_final_accessed=false`.

## E550 freeze after E540 rejection and before BGE-base masked-mean encoding

E540 shows that selecting more instances with the existing CLS geometry is harmful. E550 tests a different,
label-free encoder readout while retaining the only model family with confirmed public transfer. The exact
packaged BGE-base forward pass supplies both its deployed normalized final-layer CLS vector and a separately
normalized attention-mask mean of the same final hidden states. This is not an E540 attention, segmentation, or
weight rescue.

- Immutable sources: exact 4,096-row `qwen3_pilot_indices.npy`; `response_objective_context.parquet`;
  `modeling_base.parquet`; baseline `20260716T183434Z_robust_validation/oof_predictions.parquet`; deployed
  BGE-base CLS context/objective caches; and packaged `BAAI/bge-base-en-v1.5` model/config/modules/pooling files.
  Their hashes remain exactly those bound by E540.
- Text and encoder: use the unchanged compact `objective_context` and raw `learning_objective`, maximum 256
  tokens, CPU float32, six threads, batch 16, and the exact packaged MIT BGE-base weights. Mean pooling is
  `sum(last_hidden_state * attention_mask) / sum(attention_mask)`, including every non-padding special token,
  followed by row L2 normalization. No layer mix, token exclusion, prompt, prefix, context change, or alternate
  pooling is scored.
- Parity/resource preflight: on 32 fixed token-length quantiles from contexts and 32 from objectives, the
  normalized first-token output from the raw forward pass must match packaged SentenceTransformers CLS output
  within `2e-6` maximum absolute error. Project all 8,192 pilot mean embeddings from the measured forward-pass
  time; proceed only below two hours and 8 GB peak RSS.
- Fair pilot: reconstruct the exact original BGE-base `C=0.1` session-purged five-fold probe on
  `semantic_k50_s0`. The dual candidate concatenates the existing CLS interaction block and the masked-mean
  interaction block, each divided by `sqrt(2)`, and adds only masked-mean context/objective cosine to the same
  fold-local standardized dense controls. Regularization remains `C=0.1`.
- Evaluate exactly `10%`, `20%`, and `30%` dual-pooling probe over the fair raw BGE-base comparator and select
  the lowest pilot loss. Continue only if either loss improves by at least `0.0015` with AUROC/Brier/ECE
  non-regression, or AUROC improves by at least `0.0050` with loss/Brier/ECE non-regression, and the 5,000-draw
  paired session bootstrap gives at least 90% support for positive loss gain.
- Failure rejects this exact dual-pooling representation without a layer, token, CLS/mean scale, C, dense-feature,
  calibration, or blend-weight sweep. Passing earns full target-free cache construction followed by the frozen
  `V_seen`, `V_objective`, and `V_style` protocol. `V_joint` remains confirmation-only as already opened by the
  recovery plan; `V_final` remains sealed until the literal top-five gate authorizes it. Platform submission
  remains manual-only.

### E550 benchmark authorization

The 64-row dual-forward benchmark plus independent packaged-CLS parity check completed in `35.120` wall-clock
seconds. Raw-forward normalized CLS differs from the deployed SentenceTransformers result by at most
`1.0430813e-7`, passing the frozen `2e-6` parity tolerance. The measured raw forward projects `865.293` seconds
(`0.240359` hours) for all 8,192 pilot encodes; peak RSS was `1,247,465,472` bytes. Both operational gates pass.
The exact cache build is authorized with one calculated completion checkpoint and no interim polling. Benchmark
SHA-256 is `8216a509731064e0a20aafae6a260c9aa92cc31d8403eedcc9b1f2e50eccfd75`.
No outcome score, `V_joint`, or `V_final` was accessed.

### E550 completed result and rejection

The cache build completed in `805.088` seconds with peak RSS `1,388,027,904` bytes. Context/objective mean-cache
SHA-256 values are `0ffa5e1092f4a40e745f704fc0f43d5cd178c73ecd35f44db83b6fe8d897ad86` and
`074641a7cb358e29dfb6212bd4ad730668196e0e3ceeec71140ad979ca13c3ed`.

Frozen run `20260724T183326Z_bge_base_dual_pooling` selected the preregistered `10%` blend, but improved the fair
BGE comparator by only `0.000000551` log loss and `0.000077843` AUROC. Brier improved by `0.000000599` and
ECE-10 by `0.000166763`. The 5,000-replicate paired session bootstrap has mean log-loss gain `0.000000253`,
95% interval `[-0.000028217,+0.000029632]`, and `0.5000` positive-gain support. Both magnitude paths and the
bootstrap clause fail.

Reject E550 exactly without layer, token, pooling-scale, C, dense-feature, calibration, or blend rescue. No full
cache, hardened evaluation, ZIP, or public projection is authorized. Report SHA-256 is
`010505a7b883ebb89fddd1923069bec8206894b6a947e8a5400e51cff5911ad5`.
`V_joint_accessed=false`; `V_final_accessed=false`.

## E630 freeze after E620 rejection and before Dialogue-KT next-response transfer

E620 confirms that frozen multilingual embeddings expose broad mastery but collapse MathDial's middle class.
E630 does not collapse that label, reweight E620, or reuse E430/E580/E620 weights. It changes the external task to
the directly deployment-aligned Dialogue Knowledge Tracing objective: given dialogue history through the tutor's
current question and a learning objective, predict whether the student's *next* response will be correct.

- Source: official MIT `umass-ml4ed/dialogue-kt` repository at commit
  `c61f335f89005161b6ef439872cc1735bee26745`; repository-license SHA-256 is
  `d51d440899b6671ff0d71fc859fa74ec3a3f095c9a8aa74da5adcc194cbba1f2`. Its annotated MathDial train/test
  SHA-256 values are `bd3906355e98d357cbfa5e0e67e3b778e976f6f1a340cef35928a33325b7e6ad` and
  `f9d583489b6fe57f62b2a5239194a013f6b0662bcbcfe5b5218cf97b0a1efe63`. MathDial remains CC BY-SA 4.0
  under the participant-supplied organizer ruling. The evaluation-only CoMTA file and license hash to
  `fdfebc0108bec3f6eab0839a5ccc8b58f24ba4c0e55e2511e67c7116845de9e1` and
  `d047c993977d0017314176b237da232de7525bc6b2faad8341bbbb242debc761`; CoMTA may be used only as an
  internal untouched external evaluation set and must never train, tune, package, redistribute, or enter a
  submission artifact.
- Leakage-safe source split: keep all 595 official MathDial test dialogues and purge from training every one of
  the 314 test `qid` values, exactly as E430. This leaves 1,679 legal training dialogues over 720 question IDs.
  Parse the released `dialogue` and `annotation` fields only. Match the official Dialogue-KT correction by using
  MathDial's human `self-correctness` on an eligible final turn (`Yes=true`, `No=false`, answer-revealed=missing)
  and CoMTA's human `expected_result` on its eligible final turn. Exclude malformed annotations, missing
  correctness, empty knowledge-component lists, and empty current tutor turns. The audited usable counts before
  canonical serialization are 7,980 MathDial train turns, 2,573 MathDial official-test turns, and 623 CoMTA
  evaluation-only turns.
- Input contract: for turn `t`, include dialogue history only through the tutor utterance at `t`; never include
  the student response whose correctness is the label. Preserve prior tutor/student pairs in order and end on the
  current tutor utterance. The second sequence is the released knowledge-component descriptions, deduplicated in
  source order. Use tokenizer left truncation at 384 tokens so the current tutor question and objective survive.
  No problem text, ground-truth solution, initial incorrect solution, self-correctness field, annotation rationale,
  dialogue metadata, competition label, source-split marker, or current student answer may enter the model input.
- Backbone: a clean official Apache-2.0 `cross-encoder/nli-deberta-v3-small` base, never an E400/E460/E580/E590
  checkpoint. Replace its three-way head with a freshly seeded two-logit head and verify the exact initialization
  before training. Freeze embeddings and encoder layers 0-3; train only encoder layers 4-5, pooler, and classifier
  for exactly one epoch, batch 16, AdamW encoder/head learning rates `2e-5/1e-4`, weight decay `0.01`, 10% linear
  warmup, gradient cap `1.0`, unweighted cross-entropy, deterministic shuffle, CPU float32, and six threads.
  There is no checkpoint selection, validation split, class weighting, threshold fit, calibration, or prompt/model
  rescue.
- Resource gate: after canonical-cache and focused-test completion, benchmark two fixed training batches plus two
  fixed evaluation batches. Project the full 499-step epoch and both external evaluations; proceed only below
  three hours and 8 GiB RSS. The benchmark uses real tokenized inputs but synthetic alternating labels and may not
  compute any outcome metric.
- Frozen external gate: on all 2,573 legal MathDial official-test turns require AUROC at least `0.70`, macro-F1
  at least `0.60`, log loss at most `0.66`, ECE-10 at most `0.10`, log-loss gain at least `0.02` and Brier gain
  at least `0.005` versus the untouched legal-train prior, plus at least `0.95` support for positive loss gain in
  2,000 `qid` bootstraps. On all 623 evaluation-only CoMTA turns require AUROC at least `0.60`, macro-F1 at least
  `0.55`, log loss at most `0.68`, ECE-10 at most `0.15`, no log-loss or Brier regression versus the same frozen
  MathDial train prior, and at least `0.90` positive-loss-gain support in 2,000 dialogue bootstraps. Every clause
  is mandatory.
- Any failed clause rejects E630 without source mixture, final-label rule, input/history change, current-response
  leakage, prompt, length, layer, optimizer, class weight, threshold, calibration, checkpoint, or gate rescue.
  A complete pass permits one target-free competition cache using the same history/objective contract, followed by
  only leakage-safe fold-local probes and the frozen 10%/20%/30% blends over v0.5 BGE-base. Selection remains
  limited to `V_seen`, `V_objective`, and `V_style`; `V_joint` is confirmation-only as already authorized and
  `V_final` stays sealed until the literal top-five gate. No platform upload or submission is authorized.

### E630 canonical-cache and initialization authorization

The source-bound builder reproduces 7,980/2,573/623 usable train/MathDial-test/CoMTA-evaluation turns, with
zero legal MathDial `qid` overlap and zero CoMTA training rows. Ordered-content SHA-256 is
`f6d5cf38716bf5b6a8fabade6346a17d9d3a4cbd7966a600907b4bb1ffa45458`; Parquet SHA-256 is
`2d7647576e681527c90be80db9bb7dac0794b864314e6c79ce3aeb3fae7d9152`. Two clean seed-`20260727`
model constructions reproduce binary classifier weight/bias SHA-256 values
`c06504ffa78c64e79d3a6ecbad025b542a50bc7188463f7f4b1fd07ca267abb1` and
`af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc`.
No prediction metric, competition outcome, `V_joint`, or `V_final` was accessed. Authorize only the frozen
two-train/two-evaluation-batch resource benchmark.

### E630 pre-benchmark objective-length correction

The first benchmark attempt stopped during tokenizer construction before a forward pass or prediction because some
released multi-KC second sequences were too long for `only_first` truncation at 384 tokens. Freeze a deterministic
source-only compaction before rebuilding the canonical cache: keep the first three unique KC descriptions in
released order and the first 180 Unicode characters of each, joined by ` | `. This preserves the no-current-answer
contract and keeps second-sequence semantics while guaranteeing capacity for dialogue history. It changes no label,
split, model, optimizer, gate, or evaluation rule. Recompute and bind the canonical hashes before rerunning the
benchmark; the failed attempt produced no benchmark artifact and no outcome metric.

The corrected canonical rebuild preserves all 7,980/2,573/623 rows and zero-overlap invariants. Its bound
ordered-content/Parquet SHA-256 values are
`f6d5cf38716bf5b6a8fabade6346a17d9d3a4cbd7966a600907b4bb1ffa45458` and
`2d7647576e681527c90be80db9bb7dac0794b864314e6c79ce3aeb3fae7d9152`.

### E630 resource result and material-run authorization

Two fixed real-input/synthetic-label training batches took `21.114765` seconds and two real-input evaluation
batches took `35.974071` seconds. This projects `6,185.473` seconds (`1.718187` hours) for all 499 training and
51 external-evaluation batches. Observed RSS was `2,072,903,680` bytes. Both frozen three-hour and 8 GiB gates
pass. Benchmark SHA-256 is `0465f7c414223b1b01442f1ccf9751666f5ce6f651b5db8efa26bee5b22dd4be`.
No outcome metric was computed. Authorize exactly one material E630 run from the verified fresh binary head, with
one calculated completion heartbeat and no worker/log inspection before it.

### E630 completed external result and literal rejection

Run `20260727T192934Z_dialogue_kt_transfer` completed all 499 training steps and both external evaluations in
`5,227.613` seconds (`1.452115` hours), below the measured resource projection. Final mean training loss was
`0.697319`. The fresh binary-head, canonical source, runtime, zero-CoMTA-training, and no-current-student-response
contracts all matched their frozen hashes; `V_final_accessed=false`.

On all 2,573 official MathDial test turns, candidate accuracy/macro-F1/AUROC were
`0.540614069/0.537010413/0.563515689`; log loss, Brier, and ECE-10 were
`0.693397132/0.249964660/0.056300784`. Versus the legal MathDial-train prior, log-loss/Brier gains were only
`0.000901865/0.000611182`. The 2,000-replicate bootstrap across 386 test `qid` groups had mean gain
`0.000971313`, 95% interval `[-0.005854962,0.008285809]`, and `0.6055` positive support. Only the ECE clause
passes; AUROC, macro-F1, absolute loss, both proper-score gains, and bootstrap support fail.

On all 623 evaluation-only CoMTA turns, accuracy/macro-F1/AUROC were
`0.524879615/0.516709997/0.523795944`; log loss, Brier, and ECE-10 were
`0.690553032/0.248749411/0.063438432`. Log loss and Brier regress the unchanged MathDial-train prior by
`0.000052909/0.000072801`. The 2,000-replicate bootstrap across 151 dialogues had mean gain
`-0.000029578`, 95% interval `[-0.012999494,0.013406494]`, and `0.4950` positive support. Again only the ECE
clause passes; all classification, loss, non-regression, and bootstrap clauses fail.

Delta/prediction/metrics/training/report SHA-256 values are
`ccda1ea41fb0629aacbabb7e425e9355b03e40d6b1f224f51855570695451d4c`,
`549e776fe21595b7a5c6bfccdc7e76edbd91abcc57fdb9585dfc7c3ab961a6ed`,
`988009bcaccb93dc9dae56ad10db02f590386a1fa2ffc732aa02089ab090927e`,
`dfe924a06fd5ac95d895545061b18b8f89e715507b12b7c5a8e60dee394918d5`, and
`a5b6f53d7fc456b4209d083c6b913a8794728dfb88f72a1a7773e8499ef4a79c`.

Reject E630 exactly without history/current-response leakage, objective compaction, source mixture, final-label
rule, prompt, length, layer, epoch, optimizer, class weight, threshold, calibration, checkpoint, or external-gate
rescue. No competition cache, hardened environment, ZIP, or public projection is authorized. Champion remains
v0.5 at public `0.6054`, participant-observed rank `#8`; no platform action occurred.

### Post-E630 source-screen result

`umass-ml4ed/Difficulty-Aware-DialogKT` was inspected at commit
`3a9362e12eb51675897a5b5b458b0fc709d2ff63`. The repository currently exposes only a 71-byte README and has no
code, data, model weights, experimental result, or license. Reject it as a usable independent branch until the
authors publish reproducible, licensed material. This screening accessed no competition validation environment;
`V_final` remains sealed.

## E640 freeze after E630 rejection and before SimulatorArena outcome scoring

E430/E580/E590/E630 tested synthetic self-correction, answer/process correctness, and next-response correctness.
E640 instead uses a newly released source with a directly aligned real outcome: whether a human student answered a
math problem correctly after an actual multi-turn AI tutoring interaction. Microsoft SimulatorArena is MIT
licensed at immutable commit `e9f677c4975496fdd37f28bb6343ec3c1c54c8b4`. The redacted 450-dialogue annotation
file SHA-256 is `b2909d037da14ebfafd72e66ad8985ab700f0f164bc5d94713ca6fc43aaf651a`; the repository
license SHA-256 is `9906940f61b1f0b533fa7d99baf55178b2808fbe113ea51dfbfad8572ccd5f2b`.

- Keep the 449 rows whose released `problem_1_correctness` is exactly `correct` or `incorrect`; exclude the one
  `unknown` row. The binary target is `correct=1`. Use only paired `user_queries` and `ai_responses` through
  `problem_1_turns`, or all pairs when the released turn count is non-positive.
- Never input the separate final answer or solution, initial solution, `solve_or_not`, second-problem content,
  correctness label, rating, feedback, strength/weakness, stop reason, model identity, worker/user identity,
  problem ID, source path, difficulty/type, expertise, or extracted profile. These fields may support lineage,
  grouping, and audits only. The redacted problem/solution is not restored.
- Canonical text is chronological `Student:`/`Tutor:` dialogue with whitespace collapsed, followed by the fixed
  task line `Task: predict whether the student will answer the held-out assessment correctly after tutoring.`
  Encode with the exact packaged MIT `BAAI/bge-base-en-v1.5` v0.5 assets, normalized final-layer CLS pooling,
  CPU float32, six threads, batch 16, maximum 256 tokens, and left truncation so the end-state evidence and task
  survive. No E430/E580/E590/E620/E630 checkpoint is loaded.
- Assign each released `workerId` to one of five immutable outer folds with
  `int(SHA256("E640|" + workerId)[:16],16) mod 5`. For outer fold `k`, validate every row from workers in `k`;
  train only on other workers and additionally purge every row whose `problem_id` occurs in validation. This
  yields 449 OOF predictions with zero worker and problem overlap in every fit. Fit one unweighted
  `LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000, random_state=20260728)` on normalized embeddings per
  fold. No class weight, threshold fitting, calibration, prompt/view, C, pooling, length, or fold sweep exists.
- Compare against each fold's legal training-positive-rate probability. Every clause is mandatory: AUROC at
  least `0.60`; macro-F1 at threshold 0.5 at least `0.55`; log loss at most `0.65`; log-loss gain versus the
  fold prior at least `0.02`; Brier gain at least `0.008`; ECE-10 at most `0.15`; positive log-loss gain in at
  least four of five folds; and at least `0.90` positive-gain support in separate 2,000-replicate worker and
  problem bootstraps.
- Any failed clause rejects E640 without row, label, prompt, truncation, pooling, fold, C, class-weight, threshold,
  calibration, checkpoint, or gate rescue. A complete pass permits one all-source refit, one target-free
  competition session cache using the identical text/encoder contract, and only leakage-safe fold-local probes.
  Competition selection remains limited to `V_seen`, `V_objective`, and `V_style`; `V_joint` is
  confirmation-only. Candidate weights are exactly 10%, 20%, and 30% over raw v0.5 BGE-base, with no post-hoc
  sweep. `V_final` remains sealed until the literal top-five gate, and no upload or submission is authorized.

### E640 canonical-source authorization

The canonical builder reproduces 449 usable rows with label counts `incorrect=153`, `correct=296`; every outer
fit has 240-298 training rows and 70-124 validation rows with zero worker or problem overlap. Ordered-content
SHA-256 is `950c0341583c6721b2e51fdca28c53d2a00d33d2a3d46d5ce2122c05671cf447`; Parquet SHA-256 is
`e743a05e140b10f154ad1f8e2d8d431d90cade9883c9e8e8f4c4988d55d57f60`. These hashes are bound in code before
the resource benchmark or any candidate prediction. Authorize only the fixed 32-row, label-free resource
benchmark next.

### E640 resource result and external-cache authorization

The fixed 32-row benchmark completed in `6.393618` seconds (`0.199801` seconds/row) and projects `89.710453`
seconds (`0.024920` hours) for all 449 external rows. Peak RSS was `927,973,376` bytes. The frozen one-hour and
8 GiB gates pass under Python `3.12.8`, scikit-learn `1.8.0`, Torch `2.13.0+cpu`, and Transformers `5.14.1`.
Benchmark SHA-256 is `f259ee1d75563fd7a7128c04ce04cb25ee5976a1f6e883855c19cb6784ca3bf0`. No label or prediction metric was
accessed. Authorize exactly one external-cache build with one calculated active completion heartbeat and no
worker/log inspection before it.

### E640 completed external result and literal rejection

The 449-row cache completed in `86.786697` seconds versus the measured `89.710453`-second projection. Its shape is
`449 x 768` and SHA-256 is `e1ed9007a7076fcf1ff3bfc2f0b65bbf8d1b580ef7a9c9602d525b09460e4a1e`;
all source/canonical/runtime bindings matched and no competition outcome was accessed.

Frozen OOF metrics were log loss `0.665055158`, AUROC `0.416821233`, Brier `0.235266026`, ECE-10 `0.060792718`,
accuracy `0.659242762`, and macro-F1 `0.397315436`. Versus legal fold priors, loss/Brier gains were only
`0.001035734/0.000440221`. Four of five folds had positive loss gain, but fold 0 regressed by `0.000250800`.
The 2,000-replicate worker bootstrap had mean gain `0.001697009`, 95% interval
`[-0.000682468,0.003942501]`, and `0.9260` support; the problem bootstrap had mean gain `0.001276332`, interval
`[-0.000501634,0.003000442]`, and `0.9280` support.

Only ECE, positive-fold count, and both bootstrap-support clauses pass. AUROC, macro-F1, absolute loss,
loss-gain, and Brier-gain clauses fail. Reject E640 exactly without row, label, prompt, truncation, pooling, fold,
C, class-weight, threshold, calibration, checkpoint, or gate rescue. No competition cache, hardened validation,
ZIP, public projection, upload, or submission is authorized. Prediction/fold-model/report SHA-256 values are
`6f1f47f56dc7f3485dbfb4a83939b9f9547c24e4e1427f24ef64adc477811eac`,
`b52737c8d7b50549e9ead9efbfc2f450479880e7c80632c91a5e27994df9c2d2`, and
`5c2c72bac733b258ed553b170fa3c92c70ce9f3bd92d222865d882619fbc9a70`. `V_joint_accessed=false`;
`V_final_accessed=false`.

### Completion-heartbeat scheduling correction

The E640 cache itself finished normally, but the app continued emitting the same three-minute heartbeat despite
its `COUNT=2` RRULE until the task could execute and delete it. Treat `COUNT` as ineffective for suppressing
already queued short-interval heartbeats in this environment. Never again use a minute-frequency recurrence for
a completion checkpoint. Future calculated wakes must use one explicit DTSTART with a daily-frequency,
single-count RRULE, must be verified persisted ACTIVE, and must self-delete before inspection. This operational
correction changes no experiment result.

## E650 freeze after E640 rejection and before SimulatorArena quality scoring

E640 established that a dialogue-only BGE head does not robustly predict mathematical correctness. E650 is not a
correctness-label rescue: it tests a different, independently released supervision target from the same
MIT-licensed real interactions—the human student's 1-10 overall assessment of tutoring quality. This directly
targets the pedagogical-quality signal that v0.5 semantic similarity does not explicitly model.

- Use all 450 immutable SimulatorArena rows at commit `e9f677c4975496fdd37f28bb6343ec3c1c54c8b4`, source
  SHA-256 `b2909d037da14ebfafd72e66ad8985ab700f0f164bc5d94713ca6fc43aaf651a`. Target is exactly
  `(overall_rating - 1) / 9`, with released ratings 1-10; mean/std are `0.703457/0.280880`.
- Input is exactly E640's chronological first-problem `Student:`/`Tutor:` dialogue and fixed task line, except the
  task line becomes `Task: represent the pedagogical quality of this tutoring interaction.` The rating itself,
  final answer/solution, correctness, self-report, feedback, identities, model, problem/difficulty, profiles, and
  second problem remain forbidden as inputs.
- Use the exact packaged v0.5 BGE-base normalized CLS encoder, 256-token left truncation, CPU float32, and no
  rejected E640 fold head. Build a new target-free cache because the fixed quality task line differs from E640's
  correctness task line. No competition target or rejected external checkpoint enters.
- Assign workers with `int(SHA256("E650|" + workerId)[:16],16) mod 5`; validate each worker fold and purge every
  training row sharing a validation `problem_id`. Fit one unweighted
  `Ridge(alpha=10.0, fit_intercept=True, solver="lsqr", max_iter=1000, tol=1e-6)` per fold on normalized
  embeddings; clip predictions to `[0,1]`. No alpha, transform, prompt, pooling, threshold, or calibration sweep.
- Compare with each fold's legal training-mean rating. Every clause is mandatory: Pearson and Spearman correlation
  at least `0.25`; normalized RMSE at most `0.27`; RMSE and MAE gains versus the fold mean at least `0.01` each;
  positive RMSE gain in at least four of five folds; and at least `0.90` positive squared-error-gain support in
  separate 2,000-replicate worker and problem bootstraps.
- Any failed clause rejects E650 without target binning, row/prompt/view, alpha, clipping, fold, weighting,
  nonlinear head, calibration, checkpoint, or gate rescue. A complete pass permits an all-source quality head,
  one target-free competition session cache, leakage-safe fold-local probes, and only the frozen 10%, 20%, and
  30% blends over v0.5. Selection remains `V_seen`/`V_objective`/`V_style`; `V_joint` is confirmation-only and
  `V_final` remains sealed. No upload or submission is authorized.

### E650 canonical-source authorization

The canonical builder reproduces all 450 rows and every frozen zero-overlap fold count. Ordered-content SHA-256
is `0d493508458104eebf05810762b704397052841b28e785bae5eac12fe12bc202`; Parquet SHA-256 is
`85b27544e041459e038ed6d302fec112c5d870873b450d0a05e6a656eed4c0de`. These hashes are bound before any
quality-model prediction. Authorize only the fixed 32-row, target-free resource benchmark next.

### E650 resource result and cache authorization

The fixed 32-row benchmark completed in `7.220233` seconds and projects `101.534532` seconds (`0.028204` hours)
for all 450 rows. Peak RSS was `1,023,414,272` bytes; the frozen one-hour/8 GiB clauses pass. Benchmark SHA-256
is `c35ccfda504420f8a477f08080bbb0b1fa41087bfc9638aeb89e1e5cb5ce1186`. No quality target or prediction
metric was accessed. Authorize one clean external-cache build and one explicit-DTSTART completion heartbeat.

### E650 completed external result and literal rejection

The 450-row quality cache completed in `90.711856` seconds versus the measured `101.534532`-second projection.
Its finite float32 shape is `450 x 768`; cache and metadata SHA-256 values are
`1d7cbb544d4a71acbcf9bb6826f632a38f3fb3d187cf7a4a00b80205a604916e` and
`181ed305c658da5526bd85832ee55334d934e6e984288973ad939a31add189c1`. Runtime was Python `3.12.8`,
scikit-learn `1.8.0`, Torch `2.13.0+cpu`, and Transformers `5.14.1`; all frozen source, canonical, and encoder
bindings matched.

Frozen OOF RMSE/MAE were `0.280129688/0.232232784`, with Pearson/Spearman only
`0.076133341/0.081796906`. Against the legal fold-training-mean baseline (`0.282441971/0.234790075`), RMSE and
MAE gains were only `0.002312282/0.002557291`. All five folds had small positive RMSE gains:
`[0.000612010,0.004783110,0.002687334,0.001674522,0.003661786]`.

The 2,000-replicate worker bootstrap estimated mean squared-error gain `0.001144557`, 95% interval
`[-0.000215033,0.002596338]`, and `0.9515` support. The problem bootstrap estimated `0.001708759`,
`[0.000684896,0.002749808]`, and `0.9995` support. Positive-fold count and both bootstrap-support clauses pass,
but Pearson, Spearman, absolute RMSE, RMSE-gain, and MAE-gain clauses fail.

Reject E650 literally without target binning, row/prompt/view, alpha, clipping, fold, weighting, nonlinear head,
calibration, checkpoint, or gate rescue. No all-source fit, competition cache, hardened validation, blend, ZIP,
public projection, upload, or submission is authorized. Prediction/fold-model/report SHA-256 values are
`4b826055f534f9713017337c112fee6227ecbd7188724a5e9e56b5ee967aac01`,
`0bdd526cf17535d13181681f6d9e853747844d711e8075b751419c816087de2c`, and
`7581e2f3478c705e50f32985753137d5bde6a85e10274dc89fabdc2f74e6193b`. The v0.5 backup ZIP remains unchanged
at SHA-256 `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`;
`V_joint_accessed=false`; `V_final_accessed=false`.

## E660 freeze after E650 rejection and before ConvoLearn scoring

E650 showed that a 450-row post-hoc satisfaction rating does not yield a transferable dialogue-quality boundary.
E660 does not rescue that dataset, target, representation, fold, or estimator. It uses the independently released
MIT ConvoLearn corpus: 2,134 unique tutor-student dialogues collected from credentialed teachers interacting with
a simulated seventh-grade student, with explicit pedagogical effectiveness, completeness, and knowledge-building
dimension annotations.

- Immutable source: `masharma/convolearn` commit `f250e930356f2092462c4d1cd6bb2aae85689b1f`; Parquet SHA-256
  `c1599655c3a2a3ef5fd199906200f02afd6b26a1bd4b5fbc16a18d6b784d5c04`; MIT dataset-card SHA-256
  `63d12a3c43645e188d03599f4e500a49309048f77670e17209628cdea56f3c09`. Use all 2,134 rows, all four
  released science topics, all 21 pedagogical subdimensions, and all six broad dimensions. No competition row or
  outcome enters canonicalization or embedding.
- Canonical text is target-free. Collapse each non-empty `Student:`/`Teacher:` line, split the chronological line
  indices into five contiguous bins, retain the first and last line of every bin without duplication, truncate
  each retained line to its first 16 whitespace tokens, and append exactly
  `Task: represent the demonstrated tutoring quality and pedagogical approach.` Encode with the unchanged
  packaged MIT BGE-base normalized final-layer CLS representation, 256-token left truncation, CPU float32, six
  threads, and batch 16. Ratings, dimension names, subdimension names, topic, exchange count, and source metadata
  are forbidden from model input.
- Fit three fixed fold-local heads from the same embedding: Ridge `alpha=10` for normalized effectiveness
  `(rating-1)/4`; Ridge `alpha=10` for normalized completeness `(rating-1)/2`; and unweighted multinomial
  `LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000, random_state=20260728)` for the six broad
  pedagogical dimensions. No alpha/C, prompt, pooling, compaction, clipping, transform, weighting, calibration, or
  threshold sweep exists.
- Evaluate two separately frozen shift protocols. `subdimension_disjoint` assigns each of the 21 released
  subdimensions by `SHA256("E660|subdimension") mod 5`, so validation subdimensions never enter training.
  `topic_disjoint` uses four leave-one-science-topic-out folds. Compare regression with each fold's legal training
  mean and dimension classification with its legal smoothed training prior.
- Every external clause is mandatory in both protocols: effectiveness Pearson and Spearman at least `0.25`,
  effectiveness RMSE gain at least `0.015`; completeness Pearson and Spearman at least `0.20`, completeness RMSE
  gain at least `0.010`; dimension macro-F1 at least `0.45`, dimension log-loss gain at least `0.10`; positive
  effectiveness and completeness RMSE gain in every fold; and at least `0.90` support for positive effectiveness
  squared-error gain in a 2,000-replicate bootstrap over the held-out grouping unit.
- Any failed clause rejects E660 without label binning, source mixture, row/view, task line, segment rule, alpha,
  C, pooling, classifier, weighting, calibration, or gate rescue. A complete pass permits one all-source
  multi-head fit and one target-free competition session cache. Competition labels may train only leakage-safe
  fold-local linear probes. Candidate blends remain exactly 10%, 20%, and 30% over raw v0.5; selection uses only
  `V_seen`, `V_objective`, and `V_style`, while `V_joint` remains confirmation-only. The backup and top-five gates
  remain literal, `V_final` remains sealed, and no upload or submission is authorized.

### E660 source audit and canonical authorization

The immutable source has no missing fields or duplicated conversation text. Effectiveness uses nine half-point
values from 1-5; completeness uses five half-point values from 1-3. Topic counts are 794, 604, 371, and 365;
the six broad-dimension counts range from 189 to 589. Raw dialogues have 13/21/106 minimum/median/maximum lines
and 141/456/9,557 minimum/median/maximum whitespace tokens. The frozen subdimension folds contain
`[520,565,385,374,290]` rows from `[5,5,4,4,3]` held-out subdimensions. This audit authorizes canonical-cache
construction only; no candidate prediction metric has been generated.

Canonicalization reproduced all 2,134 rows, both rating scales, six dimensions, four topics, and every frozen
zero-overlap split. Ordered-content SHA-256 is
`952587761234e94183217b77f5637d131f11cc22aa25f59e013ef7dadb0717b8`; Parquet SHA-256 is
`664eabd969d837cb0fba8f490caa897f9ac2f9c28290b8e8c94d960d7d513941`. Normalized effectiveness
mean/std are `0.588918/0.270404`; completeness mean/std are `0.625937/0.308970`. Bind both hashes in code
before any embedding benchmark or candidate prediction. Authorize only the fixed 32-row target-free resource
benchmark next.

### E660 resource result and external-cache authorization

The fixed 32-row benchmark completed in `6.491799` seconds and projects `432.921812` seconds
(`0.120256` hours) for all 2,134 canonical rows. Peak RSS was `887,513,088` bytes. The frozen one-hour and
8 GiB clauses pass under Python `3.12.8`, scikit-learn `1.8.0`, Torch `2.13.0+cpu`, and Transformers
`5.14.1`. Benchmark SHA-256 is
`45d4271f38fcbcb8af806ef41d34f2db0049003fc763dbecec84f5484e78fefb`. No rating, dimension, candidate
prediction, competition outcome, `V_joint`, or `V_final` was accessed. Authorize exactly one external-cache build
with one calculated ACTIVE completion heartbeat and no worker/log inspection before it.

### E660 completed external result and literal rejection

The target-free cache completed `2,134/2,134` rows in `394.637652` seconds versus the `432.921812`-second
projection. Its finite float32 shape is `2134 x 768`, and SHA-256 is
`fd25b9d68f660460fad476a556964c4502af1fa18596fec260a5007e7835cbec`. Runtime was Python `3.12.8`,
scikit-learn `1.8.0`, Torch `2.13.0+cpu`, and Transformers `5.14.1`; every frozen source, license, canonical,
and encoder binding matched.

Subdimension-disjoint effectiveness RMSE/MAE/Pearson/Spearman were
`0.266721623/0.236146846/0.164661252/0.169305434`, with prior RMSE/gain `0.273285629/0.006564006`.
Completeness RMSE/MAE/Pearson/Spearman were `0.305745197/0.260942124/0.146388226/0.148762441`, with prior
RMSE/gain `0.313131537/0.007386340`. Effectiveness fold gains were
`[0.007032679,0.008125151,0.004301792,0.005536436,0.007639791]`; completeness fold gains were
`[0.005049050,0.010228058,0.004667493,0.006948443,0.010776400]`. Dimension macro-F1/log loss/prior
loss/gain were `0.085655981/1.984642534/1.981505576/-0.003136958`. The 2,000-replicate grouped bootstrap
estimated effectiveness mean squared-error gain `0.003523123`, 95% interval `[0.002184970,0.004688000]`, and
support `1.0000`.

Topic-disjoint effectiveness RMSE/MAE/Pearson/Spearman were
`0.266249456/0.236397645/0.196367640/0.205097257`, with prior RMSE/gain `0.271380392/0.005130936`.
Completeness RMSE/MAE/Pearson/Spearman were `0.305628633/0.263395310/0.156287978/0.173845319`, with prior
RMSE/gain `0.310501595/0.004872962`. Effectiveness fold gains were
`[0.006126791,0.003386168,0.007849288,0.004720660]`; completeness fold gains were
`[0.007013741,0.002820100,0.007372854,0.003243892]`. Dimension macro-F1/log loss/prior loss/gain were
`0.072720199/1.714333949/1.718619705/0.004285755`. The 2,000-replicate grouped bootstrap estimated
effectiveness mean squared-error gain `0.002931742`, interval `[0.002198736,0.003675220]`, and support `1.0000`.

Both protocols pass only positive gain in every regression fold and bootstrap support. Both fail every frozen
effectiveness/completeness correlation and RMSE-gain threshold and both dimension macro-F1/log-loss-gain
thresholds. Reject E660 exactly without label, source, text, segment, prompt, pooling, alpha, C, classifier,
weighting, calibration, fold, threshold, checkpoint, or gate rescue. No all-source fit, competition cache,
hardened validation, blend, ZIP, public improvement, upload, or submission is authorized. Prediction,
fold-model, and report SHA-256 values are
`f5589837f6e7c71221b0b6e8bf8419e169fef31353068adc54781e4576a00a79`,
`1676429ab113590b4572f3f3b5056ada37beb44a39c91efdb97c7df6b62b7db8`, and
`cd31b109e008a939b8923d44972d9c9f24f0cbba3e2616ac8ff7d86afe5cd469`. The v0.5 score/rank evidence remains
`0.6054` and approximately `#8`; the preserved ZIP remains unchanged at
`65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`. `competition_outcomes_accessed=false`;
`V_joint_accessed=false`; `V_final_accessed=false`.

## E670 freeze after E660 rejection and before Bridge candidate embedding

E640-E660 tested outcome, satisfaction, and generic pedagogical-quality targets. E670 does not rescue any of
those sources, labels, representations, estimators, or gates. It uses Bridge's paired cognitive-task-analysis
supervision: for the same demonstrated student math mistake, a real novice tutor response is paired with an
experienced teacher's rewritten remediation response. This directly targets response choice at a learning
opportunity rather than dialogue-level quality or released correctness.

- Immutable source: `rose-e-wang/bridge` Hugging Face dataset commit
  `8f469883aa7d7a5c1d64e5c961a033ce71d21f5e`, licensed CC BY-NC 4.0. Source-card SHA-256 is
  `279b838af2dc1b73fe69d2757ccb4b8e00251708621d1f66d69ce78703584e63`. Local train/validation/test
  JSON SHA-256 values are `870fa10d299711d315718d1995f234988582d7168fa9c054c267ededd89ba260`,
  `1d87ffece5fba05b6f5e91a2bcd0e5d556afb00973dd47e09969c875576bf327`, and
  `7c1f5eccce6f635924ca495439c7a9d70885c73122dd4caf8ad9066c657265e9`.
- Use all 700 paired rows: 419 released train, 71 validation, and 210 test rows; 459 unique `c_id` values,
  383 session roots, and 208 lesson topics. Every row has exactly four history turns and nonempty original
  novice `c_r` and expert `c_r_` responses; no pair is text-identical. Released error, strategy, intention,
  split, `c_revision`, and all later student/tutor turns are forbidden as features.
- Construct two target-free texts per pair. Preserve the four `c_h` turns chronologically as `Student:` or
  `Tutor:` lines, append either the full original or expert response as chronological `Tutor:` lines, and append
  exactly `Task: assess how effectively the candidate tutor response remediates the demonstrated mathematical
  misunderstanding.` Encode with the unchanged packaged MIT BGE-base normalized final-layer CLS representation,
  256-token left truncation, CPU float32, six threads, and batch 16.
- For each pair form `d = expert_embedding - novice_embedding`. Fit the fixed antisymmetric model by stacking
  `d` with label 1 and `-d` with label 0 and using unweighted
  `LogisticRegression(C=0.1, solver="lbfgs", fit_intercept=False, max_iter=1000,
  random_state=20260728)`. No C, prompt, context, turn, pooling, normalization, weighting, calibration,
  threshold, or nonlinear-model sweep exists.
- Evaluate two frozen five-fold protocols. `session_disjoint` assigns the `c_id` prefix before the final
  underscore by `SHA256("E670|session|" + session_root) mod 5`. `lesson_disjoint` assigns the exact
  `lesson_topic` by `SHA256("E670|lesson|" + lesson_topic) mod 5`. All duplicate `c_id` and lesson rows remain
  together. Compare with the fixed 0.5 preference prior: log loss `0.693147181` and Brier `0.25`.
- Every external clause is mandatory in both protocols: expert-preference accuracy at least `0.60`; balanced
  AUROC at least `0.65`; log-loss gain at least `0.025`; Brier gain at least `0.010`; ECE-10 at most `0.10`;
  positive log-loss gain in every fold; and at least `0.95` support for positive log-loss gain in a
  2,000-replicate bootstrap over the held-out grouping unit.
- Any failed clause rejects E670 without row filtering, expert-label reinterpretation, response selection,
  source mixture, context window, task line, C, pooling, estimator, weighting, calibration, fold, threshold, or
  gate rescue. A complete pass permits one all-source ranking vector and one target-free competition cache.
- If permitted, select at most eight tutor utterances per competition session using deterministic evenly spaced
  chronological indices including the first and last tutor utterance. For each selected utterance, use exactly
  the preceding four transcript utterances plus that candidate tutor utterance and the frozen task line. Score
  its normalized BGE-base CLS embedding with the all-source Bridge vector and aggregate exactly mean, standard
  deviation, minimum, maximum, first, last, least-squares chronological slope, and fraction of positive logits.
  No competition outcome enters this cache.
- Competition outcomes may train only leakage-safe fold-local unweighted logistic probes over those eight fixed
  aggregates. Evaluate `V_seen`, `V_objective`, and `V_style`; use `V_joint` only for the plan's allowed locked
  confirmation. Candidate blends remain exactly 10%, 20%, and 30% over raw v0.5. The backup/top-five gates
  remain literal, `V_final` remains sealed, and no upload or submission is authorized.

### E670 source audit and canonicalization authorization

The immutable source audit found 700 complete pairs, 383 distinct session roots, 208 lessons, no empty candidate
response, no identical novice/expert pair, and one duplicated response pair. Novice responses contain
`2/17/92` minimum/median/maximum whitespace tokens; expert responses contain `1/16/79`. The 241 repeated
four-turn contexts are expected multiple expert annotations and are protected by both frozen grouping schemes.
This audit authorizes canonicalization only. Bind the ordered canonical-content and Parquet hashes before any
embedding benchmark or candidate prediction, then run only the fixed first-32-pair target-free resource
benchmark. The resource gate is projected full-cache runtime at most one hour and peak RSS at most 8 GiB.

### E670 canonical-source authorization

The first canonical attempt found a released source edge case before any embedding: 16 empty history turns and
three empty candidate-response turns. Preserve each as its chronological empty speaker marker; require the
complete candidate response to contain nonempty text. No row was removed and no target, embedding, or prediction
was accessed. The corrected canonicalizer reproduces all 700 pairs and all source/split counts.

Session-disjoint validation folds contain `[126,160,143,121,150]` pairs from `[68,82,75,70,88]` held-out
session roots. Lesson-disjoint folds contain `[242,72,145,101,140]` pairs from `[47,36,42,36,47]` held-out
lessons. Every fold has zero held-out-group or pair overlap. Ordered-content SHA-256 is
`1a6a300fe87904aa83b9a3ce6e804f6583d763b5e9805566c55d5e7c9dea2b32`; canonical Parquet SHA-256 is
`a460bfa5d864c29dfd02e08e9b92e8173bfad9829fc69ebd1fa205707aa734f1`. Bind both hashes in code before
any candidate embedding or preference score. Authorize only the frozen first-32-pair target-free resource
benchmark next.

### E670 resource result and external-cache authorization

The fixed 32-pair/64-text benchmark completed in `6.851101` seconds and projects `149.867837` seconds
(`0.041630` hours) for all 1,400 candidate texts. Peak RSS was `861,392,896` bytes, so the frozen one-hour and
8 GiB gates pass under Python `3.12.8`, scikit-learn `1.8.0`, Torch `2.13.0+cpu`, and Transformers
`5.14.1`. Benchmark SHA-256 is
`297f2a17fa3713b3f024d48cdba1df962a31f3020f7631b3aba8932ccd6ea1c7`. No preference label, candidate
score, competition outcome, `V_joint`, or `V_final` was accessed. Authorize exactly one external-cache build
with one calculated ACTIVE completion heartbeat and no worker/log inspection before it.

### E670 completed external result and literal rejection

The 1,400-text cache completed in `184.038498` seconds versus the measured `149.867837`-second projection. Its
finite float32 shape is `1400 x 768`; SHA-256 is
`661a2738eec75de96927a1f186c556d5987c3d41f4ffa91b3b692086d6dfd08e`. Runtime was Python `3.12.8`,
scikit-learn `1.8.0`, Torch `2.13.0+cpu`, and Transformers `5.14.1`; every frozen source, canonical, encoder,
and pairing contract matched.

Session-disjoint accuracy/AUROC/log loss/Brier/ECE-10 were
`0.824285714/0.890438776/0.612076049/0.210171626/0.269725834`. Against the fixed balanced prior, log-loss and
Brier gains were `0.081071132/0.039828374`. All five fold log-loss gains were positive:
`[0.097642705,0.077088901,0.087545607,0.070036514,0.074127649]`. The 2,000-replicate session-root
bootstrap estimated mean gain `0.081085948`, 95% interval `[0.071528575,0.090445693]`, and support `1.0000`.

Lesson-disjoint accuracy/AUROC/log loss/Brier/ECE-10 were
`0.821428571/0.896487755/0.613850040/0.210980307/0.269271586`. Log-loss and Brier gains were
`0.079297141/0.039019693`. All five fold log-loss gains were positive:
`[0.072599614,0.084107513,0.080696261,0.079334278,0.086924506]`. The 2,000-replicate lesson bootstrap
estimated mean gain `0.079034157`, 95% interval `[0.069725769,0.087187616]`, and support `1.0000`.

Both protocols pass accuracy, AUROC, log-loss gain, Brier gain, every-fold gain, and bootstrap support, but both
fail the frozen `ECE-10 <= 0.10` clause by a wide margin. Because every clause was mandatory, reject E670
without Platt, temperature, isotonic, prior, scale, intercept, C, threshold, probability reinterpretation, or any
other calibration/gate rescue. No all-source ranking vector, competition cache, hardened validation, blend, ZIP,
public improvement, upload, or submission is authorized.

Prediction/fold-model/report SHA-256 values are
`b7692b100ae9c221499b6eb622289527956963b9046bcf183c329784a278d5f1`,
`a087a568f1c2c9913176a82309129e3ea40785baae4e64646a4e9eb38590a4b4`, and
`cb88571f457d4dd64f584c0b437feea7e152c05bcd7d906c7ac95c9196d8202f`. The v0.5 score/rank evidence remains
`0.6054` and approximately `#8`; its protected ZIP remains unchanged at
`65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.
`competition_outcomes_accessed=false`; `V_joint_accessed=false`; `V_final_accessed=false`.

## E680 freeze after E670 rejection and before CIMA candidate embedding

E670 is closed and none of its Bridge data, pairwise checkpoint, coefficient, probability, or calibration
evidence may enter E680. E680 uses a different supervision family from the independently released CIMA corpus:
the student's own epistemic action after a tutoring exchange. The binary target is the released `Guess` action
bit, distinguishing an attempted answer from help-seeking, affirmation, or other student behavior.

- Immutable source: CIMA repository commit `3fa48593c001046893f5f1320ab031f7237e7998`, released under Creative
  Commons Attribution 2.5. Dataset/README SHA-256 values are
  `544dfc50dd05b14579e1a04b4751f4ffda8a72d3d398410d1aba0bcc2dd57405` and
  `23680d5c37de5b69436172372f14f88ddc31376a27738256d07bcaa78642897b`.
- Use all 1,135 `prepDataset` rows. Every history contains 2, 4, 6, 8, or 10 alternating turns beginning with
  the tutor and ending with the student. The `Guess` bit is positive in 514 rows and negative in 621. Preserve
  the complete chronological `past_convo` as alternating `Tutor:`/`Student:` lines and append exactly
  `Task: represent whether the final student turn is an independent answer attempt rather than help-seeking or
  acknowledgment.` All `studentActions`, `tutorActions`, candidate tutor responses, tutor keys, images, Italian
  answer fields, grammar rules, and English concept fields are forbidden from model input.
- Encode with the unchanged packaged MIT BGE-base normalized final-layer CLS representation, 256-token left
  truncation, CPU float32, six threads, and batch 16. Fit only an unweighted
  `LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000, random_state=20260728)` on the released Guess bit.
  No C, prompt, context, turn, pooling, weighting, target transform, threshold, calibration, or nonlinear-model
  sweep exists.
- Evaluate two frozen five-fold protocols. `exercise_disjoint` assigns the exact released `img` identifier by
  `SHA256("E680|exercise|" + img) mod 5`. `concept_disjoint` assigns the exact
  `(engPrep, engObj, engColor)` triple by `SHA256("E680|concept|" + compact_json_triple) mod 5`. All duplicate
  histories and group identities remain together. Each fold comparator is the legal training-fold Guess prior.
- Every clause is mandatory in both protocols: AUROC at least `0.70`; macro-F1 at least `0.65`; log-loss gain
  at least `0.030`; Brier gain at least `0.010`; ECE-10 at most `0.10`; positive log-loss gain in every fold;
  and at least `0.95` support for positive log-loss gain in a 2,000-replicate bootstrap over the held-out
  grouping unit.
- Any failed clause rejects E680 without row/action filtering, multilabel reinterpretation, source mixture,
  history length, role order, task line, C, pooling, estimator, weighting, calibration, threshold, fold, or gate
  rescue. A complete pass permits one all-source Guess head and one target-free competition student-state cache.
- If permitted, select at most eight student utterances per competition session using deterministic evenly
  spaced chronological indices including the first and last student utterance. For each selected utterance,
  encode exactly the preceding nine transcript utterances plus that student utterance with their released roles
  and the frozen task line. Score with the all-source CIMA head and aggregate exactly mean, standard deviation,
  minimum, maximum, first, last, least-squares chronological slope, and fraction at least 0.5. No competition
  outcome enters this cache.
- Competition outcomes may train only leakage-safe fold-local unweighted logistic probes over those eight fixed
  aggregates. Evaluate `V_seen`, `V_objective`, and `V_style`; use `V_joint` only for allowed locked
  confirmation. Candidate blends remain exactly 10%, 20%, and 30% over raw v0.5. Backup/top-five gates remain
  literal, `V_final` remains sealed, and no upload or submission is authorized.

### E680 source audit and canonicalization authorization

The immutable source contains 1,135 usable preparation contexts and an empty unused `shapeDataset`. Released
history lengths are 2/4/6/8/10 turns with counts `[373,325,230,125,82]`; every history is even-length and ends
in a nonempty student utterance. The Guess/Question/Affirmation/Other positive counts are
`[514,551,162,2]`; 88 rows have two active actions and three have three, but the frozen target remains the
released Guess bit without reinterpretation. There are 225 exercise images, 123 concept triples, and three
duplicate complete histories, all protected by group assignment. This audit authorizes canonicalization only.
Bind ordered-content and Parquet hashes before the fixed first-32-row target-free resource benchmark.
