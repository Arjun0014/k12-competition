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
