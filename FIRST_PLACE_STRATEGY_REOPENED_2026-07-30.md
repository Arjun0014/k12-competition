# Trace the Ace: reopened first-place strategy

**Date:** 2026-07-30 (Asia/Kolkata)
**Branch:** `codex/v04-recovery`
**Starting HEAD:** `136dd46aa755cbb5df685eb20480b40a74faa000`
**Public champion:** v0.5, public log loss `0.6054`
**Current rank:** participant-reported `#12`; not independently re-read in this audit
**Protected ZIP:** `submission_builds/final_ensemble_v05_bge_backup.zip`
**Protected ZIP SHA-256:** `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`
**Submission boundary:** Codex never uploads or submits. The participant makes every platform submission manually.

## Executive decision

The project is not out of useful work, but it is out of evidence for another
immediate competition-outcome experiment. No E880 is preregistered.

The best route to an overall first-place prize now has two simultaneous goals:

1. defend a final top-15 leaderboard position and pursue a score improvement
   only when a genuinely independent candidate clears the existing gates; and
2. build a judge-ready research contribution now, because the competition
   explicitly awards first through third using both final leaderboard position
   and write-up quality.

This is not a retreat from modeling. It is a correction to the objective. The
official rubric gives 35% to relevance, 35% to generalizability, 15% to
communication, and 15% to rigor. Rank 12 is therefore inside the current
eligibility band, while public leaderboard first place by itself would still
not guarantee the overall prize.

The next score branch remains conditional APTA. While access is pending, the
productive action is to construct the report evidence package and preserve the
validation budget. The APTA monitor has been deleted at the participant's
request; access will be checked again only after the participant reports that
it has been granted.

## Why twelve entries can be ahead without proving that our process is wrong

The last participant-supplied board showed first at `0.6008`, fifth at `0.6040`,
and tenth at `0.6053`. Relative to v0.5:

- tenth was only `0.0001` better;
- fifth was `0.0014` better;
- first was `0.0046` better.

The ranks are therefore compressed. Moving one ten-thousandth can move several
places near v0.5. New entries can lower the rank even when the model and its
score are unchanged.

There is no public evidence about the architectures used by the teams above
us. The rules prohibit private code sharing, and the leaderboard reports only
scores. Claims that another team used Claude, GPT, an LLM judge, a particular
encoder, or a leakage shortcut would be speculation.

Our process also deliberately rejects some candidates that could improve an
ordinary average split but look unsafe under shift:

- the BGE-base plus ordered-feedback candidate improved mean local loss by
  about `0.001851` and AUROC by `0.009059`;
- it nevertheless had a worst-fold loss regression of `0.002414` and a worst
  descriptive subgroup regression of `0.005273`;
- the later E770/E810/E820/E830 heads each regressed equal-environment loss by
  roughly `0.00167-0.00205`.

That conservatism can sacrifice a public-board gamble. It also avoids selecting
the same kind of objective/provider-specific model that previously achieved
very optimistic grouped local loss and then failed publicly. The private
leaderboard may differ from the public leaderboard, so robustness remains a
real competitive asset rather than paperwork.

## What v0.5 captures and what remains missing

v0.5 is a fixed probability average:

- 25% full-session word hashing;
- 25% role-separated transcript, objective, behavior, and alignment evidence;
- 50% BGE-base objective-conditioned semantic interaction.

It captures topic, broad language, role, session behavior, retrieval coverage,
and semantic alignment. It does not contain the rejected ordered-feedback
NB-SVM or supervised BGE-small heads.

The strongest durable positive after the original sparse model was encoder
capacity: replacing BGE-small with BGE-base improved hardened loss by
`0.0012721180`, improved all four environments in the frozen Phase C audit,
and then improved the public score to `0.6054`.

The strongest complementary process signal was ordered answer-to-feedback
text. It improved average ranking and loss, but its probability quality
reversed in hard objective-frequency regimes. This yields the central research
finding:

> Lesson-content semantics and local tutor feedback carry complementary
> outcome signal, but process heads that look strong on average can reverse
> under objective and transcript-style shift.

The remaining first-place score gap is therefore unlikely to be closed by
another count, blend weight, calibration map, objective prior, generic
pedagogical-quality score, or small encoder swap. It requires either real
outcome-linked process supervision with deployable features or a genuinely
new representation that demonstrates stable transfer before competition
outcomes.

## New primary-source audit

### Pedagogical reward model: legal and interesting, but not independent

`eth-nlped/Qwen2.5-1.5B-pedagogical-rewardmodel` is public under CC BY 4.0. It
is a float32, approximately two-billion-parameter sequence classifier trained
on MathDial and MRBench preference pairs. Its accompanying MathTutorBench paper
reports held-out Bridge expert-versus-novice ranking accuracy of `0.84`.

It is not an E880 candidate:

- E670 already tested the same Bridge expert-remediation construct;
- E670 reached about `0.824/0.821` accuracy on session/lesson-disjoint screens
  and strong AUROC, then failed the frozen calibration gate;
- the reward model's reported result is only marginally stronger and does not
  supply a learning-outcome target;
- its official input depends on a math problem and reference solution that the
  competition does not provide;
- the local machine has CPU-only Torch and about 11.4 GB total RAM, while the
  unquantized checkpoint is about 6.19 GB before runtime activations.

Downloading or running it would spend substantial time and memory on a closed
pedagogical-ranking family without a plausible robust `0.0016` outcome path.
No checkpoint was downloaded.

Primary sources:

- https://huggingface.co/eth-nlped/Qwen2.5-1.5B-pedagogical-rewardmodel
- https://huggingface.co/datasets/dmacjam/pedagogical-rewardmodel-data
- https://arxiv.org/abs/2502.18940

### DKT-Sem and LLMKT: strong research direction, unavailable inference state

The official `umass-ml4ed/dialogue-kt` repository is already preserved locally
at revision `c61f335f89005161b6ef439872cc1735bee26745`. Its released paper reports
on MathDial:

- DKT-Sem accuracy/AUC/F1 `62.17/66.18/59.30`;
- LLMKT accuracy/AUC/F1 `68.41/76.71/62.21`.

DKT-Sem is not deployable as published. Its state update explicitly adds a
learned embedding of the true previous-turn correctness label before using the
LSTM to predict the next turn. Trace the Ace supplies transcript text but no
per-turn correctness sequence. Replacing those labels with guessed or constant
values would be a new, distribution-shifted architecture, not a reproduction
of the reported model.

LLMKT does not explicitly consume previous labels, but the official work
fine-tunes Llama-3.1-8B with LoRA on A6000 GPUs. The repository does not release
a trained LLMKT checkpoint. An 8B fine-tune is not feasible in the current
CPU-only environment, and no paper metric can be imported as a competition
feature.

This source remains a future lead only if the authors publish a prize-safe
checkpoint that can be packaged and evaluated offline. DKT-Sem itself is
rejected as an E880 architecture.

Primary sources:

- https://github.com/umass-ml4ed/dialogue-kt
- https://arxiv.org/abs/2409.16490

### Official reference-solution suggestions: already covered

The July 27 DrivenData reference solution proposes tutor features, number-word
recognition, timing, and LLM-generated turn labels as next steps. The project
has already tested those families more strongly:

- tutor behavior and tutor-move/state interactions: E220/E530/E760/E770;
- number words and numeric elaboration: E810 and E860;
- timing and early/late dynamics: E510 and multiple ordered/trajectory screens;
- externally supervised or generated correctness/quality/state labels:
  E580-E700 and E840.

The reference solution does not reopen those exact branches.

Primary source:

- https://blog.drivendata.org/blog/productive-math-talk-reference

### Train-only domain adaptation: searched, not authorized

Repository/log search found no literal masked-language domain-adaptive
pretraining run. It is nevertheless not a qualifying next branch:

- supervised encoder adaptation, external contrastive adaptation, multi-view
  semantic learning, and next-turn coherence already tested adjacent
  representation mechanisms;
- the post-E870 audit explicitly closes another semantic-adaptation relabeling;
- a masked-language or denoising score would not by itself establish a
  learning-outcome path;
- a CPU training run would be long before it had target-free evidence of a
  robust `0.0016` gain.

This is recorded as untried implementation but insufficiently independent and
insufficiently plausible. It is not preregistered and no long run is launched.

## Multi-track route to the overall prize

### Track 0: defend top-15 eligibility

1. Preserve the protected v0.5 ZIP byte-for-byte.
2. Do not recommend a known regression or rejected branch merely because a
   weekly submission slot may reset.
3. Before any manual submission decision, record live `L1`, `L5`, `L15`, v0.5
   rank, remaining slots, and timestamp.
4. Build a new ZIP only after the frozen local backup and promotion gates pass.
5. Remember that the final private leaderboard can reorder the public board.

### Track 1: APTA conditional score branch

APTA remains the only current high-priority source because it may provide all
of the elements missing from prior external corpora: real learner outcomes,
ordered algebra work, solver/tutor role switching, transactions, chat, and
study-site/condition structure.

When the participant reports access:

1. download only authorized project files and preserve them raw;
2. generate a deterministic size/SHA-256 manifest;
3. run the existing target-free source audit;
4. require stable learner linkage, ordered transactions, a real assessment
   outcome distinct from step correctness, and at least 50 linked learners;
5. preregister one learner-grouped, site-aware external outcome benchmark;
6. require stable added value beyond pretest, opportunity, correctness, and
   activity baselines;
7. construct only deployable role/trajectory features;
8. freeze one competition validation over `V_seen`, `V_objective`, and
   `V_style`;
9. use `V_joint` only for locked confirmation and preserve `V_final`.

Any failed stage closes that exact APTA branch without target, subgroup,
feature, estimator, or weight rescue.

### Track 2: narrow source reserve

If APTA is unusable, only a source-level audit is warranted:

- ITSPOKE/WOZ may have real spoken tutoring and pre/post outcomes, but its
  released sample is small, physics-domain, and its uncertainty/affect
  mechanism overlaps closed behavior, challenge, speaker, and ASR families.
- BEETLE II is a lead only until an open, prize-safe package with both dialogue
  and learner outcomes is verified.
- LLMKT is a lead only if an actual open checkpoint is released.

Do not request, accept terms for, download, or train on a reserve source until
its license, redistribution, linkage, sample size, and novelty are verified.

### Track 3: write-up as a first-place workstream

Start now, not after August 27.

The four-page report should optimize the official rubric directly:

1. **Key finding:** BGE-base content semantics are the only late component that
   improved every hardened environment and then transferred publicly.
2. **Second finding:** ordered tutor feedback is complementary and improves
   average ranking, but its reversals under objective/style shift make an
   unqualified pedagogical claim unsafe.
3. **Actionable lesson:** tutoring-effectiveness signals should be validated
   across objective families and transcript styles, not only random
   session-grouped folds.
4. **Generalizability:** contrast long ASR tutoring with short typed chat and
   identify which feature families are stable under that shift.
5. **Rigor:** report frozen gates, session purges, environment matrices,
   bootstraps, calibration, licenses, revisions, hashes, and negative results.
6. **Communication:** use one model diagram, one robustness figure, one compact
   positive/negative result table, and one practical recommendations box.

The report must not claim that feedback causes learning. It can claim a robust
association only where the frozen evidence supports it.

## Decision tree

```text
Keep v0.5 protected and top-15 ready
|
+-- APTA access reported?
|   |
|   +-- no  -> build report evidence; audit no new outcomes
|   |
|   +-- yes -> A0 source/linkage gate
|              |
|              +-- fail -> close APTA; evaluate source reserve only
|              |
|              +-- pass -> external learner/site benchmark
|                            |
|                            +-- fail -> close exact branch
|                            |
|                            +-- pass -> target-free deployability
|                                          |
|                                          +-- fail -> close exact branch
|                                          |
|                                          +-- pass -> one frozen competition validation
|                                                        |
|                                                        +-- fail -> no ZIP
|                                                        |
|                                                        +-- pass -> V_joint confirmation
|                                                                      |
|                                                                      +-- pass -> local ZIP and participant decision
|                                                                      +-- fail -> no ZIP
|
+-- Prize path in parallel -> four-page evidence package for top-15 judging
```

## Immediate execution order

1. Commit this research decision so no future session rediscovers and downloads
   the reward model or launches DKT-Sem under a false assumption.
2. Build the write-up evidence specification and a deterministic index of
   existing reports, figures, licenses, and hashes.
3. Generate the v0.5 environment/calibration/bootstrap panel only from already
   authorized artifacts; do not open `V_final`.
4. Generate the positive-versus-negative branch table, emphasizing the
   ordered-feedback instability rather than hiding it.
5. Resume APTA only after the participant reports granted access.
6. Preregister E880 only if a new source or released checkpoint survives the
   novelty, legal, resource, target-free, and `0.0016` plausibility gates.

## Scheduler decision

The harmless idle-turn wake has already fired end to end in this task, so a
future run over one hour may use exactly one calculated completion checkpoint.
No run in this audit qualifies for launch, so no timer is created. A working
wake mechanism is not a reason to start an unjustified run.

For a future qualified long run:

1. benchmark the exact workload;
2. estimate completion from measured telemetry;
3. schedule exactly one completion checkpoint;
4. do not poll before it;
5. if still running, derive exactly one replacement checkpoint from real
   progress.

## Non-negotiable boundaries

- Never upload or submit.
- Never weaken a frozen gate after observing its result.
- Never revive a rejected checkpoint, prediction, or score through a smaller
  blend.
- Never use `V_final` for discovery.
- Never infer access, license, affiliation, identity, or approval.
- Never claim first place is assured.
