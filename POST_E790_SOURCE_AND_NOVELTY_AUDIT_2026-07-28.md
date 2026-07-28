# Post-E790 source, novelty, and headroom audit — 2026-07-28

## Decision

No E810 competition candidate is preregistered or authorized.

The only genuinely new mechanism found after E790 was student/tutor textual
personality-style complementarity. It is directly aligned with real middle-school
mathematics tutoring outcomes, but the exact released Big-Five model fails the
use-suitability and reproducibility gates. Removing that model also removes every
statistically significant published interaction. The remaining public data sources
either require a user-mediated access agreement, exclude commercial use, lack
outcomes, or duplicate a closed branch.

No competition outcome, `V_joint`, or `V_final` evidence was accessed for this
audit. E800 remains reserved for an ensemble of independently passing components
and is not available as an experiment number.

## Frozen project state

- Branch: `codex/v04-recovery`.
- Protected public champion: v0.5, observed public log loss `0.6054`, approximately
  rank 11 in the participant-provided 2026-07-28 screenshot.
- Observed public targets: first `0.6008`, fifth `0.6040`, tenth `0.6053`.
- Conservative first-place target: `0.6005`, requiring an estimated `0.0049`
  improvement over v0.5. This is a target, not a promise.
- Protected ZIP:
  `submission_builds/final_ensemble_v05_bge_backup.zip`.
- Protected ZIP SHA-256:
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.
- E710, E720, E730, E740, E750, E760, E770, E780, and E790 remain rejected
  literally. E580-E700 and earlier rejected branches remain closed.

## Candidate A: text personality-style complementarity

### Evidence and novelty

The 2025 LAK paper
[Who Should Be My Tutor?](https://doi.org/10.1145/3706468.3706537)
studied 383 students, 2,157 question-answer records, 697 unique questions, and
15-item pre/post tests on Math Nation. It reported three significant robust
regression interactions:

- student extroversion × chatbot openness: coefficient `+3.2477`,
  95% CI `[0.702, 5.793]`;
- student openness × chatbot openness: coefficient `-2.7545`,
  95% CI `[-5.320, -0.189]`;
- student extroversion × chatbot extroversion: coefficient `-1.9708`,
  95% CI `[-3.887, -0.054]`.

Repository and learning-log searches found no prior Big-Five, extroversion,
openness, subjectivity, VADER, or personality-complementarity component. E770
contained discrete student-state × tutor-move interactions, but no stable or
probabilistic text-personality construct. The mechanism was therefore novel enough
to audit.

### Exact public assets

The paper's subjectivity model is public and usable:

- repository:
  `cffl/bert-base-styleclassification-subjective-neutral`;
- frozen revision:
  `1339b8de703cb52c729475a89427078052af8595`;
- license: Apache-2.0;
- labels: `SUBJECTIVE`, `NEUTRAL`;
- weight artifact: 438,017,325-byte `pytorch_model.bin`;
- model-card-reported WNC classification accuracy: `72.5%`.

VADER's official repository is MIT licensed. It was not installed and no package
or lexicon was added during this audit.

The paper's Big-Five model is public but not suitable for this use:

- repository: `Minej/bert-base-personality`;
- frozen revision:
  `6f4d00d28093bdc2ba6cccec97acbf1805709ff1`;
- repository license tag: MIT;
- five-logit order documented only in the model card:
  extroversion, neuroticism, agreeableness, conscientiousness, openness;
- the released `config.json` contains only generic `LABEL_0` through `LABEL_4`;
- the model card gives no validation metrics, training-data revision, subgroup
  evidence, or short-utterance reliability evidence;
- the model card says the model is not intended for downstream use and should not
  be used to make judgments in education;
- the card warns about limited context, domain/demographic generalization, false
  positives/negatives, privacy, stereotyping, and discrimination.

The Math Nation outcome data used in the paper are not publicly released, so the
reported coefficients cannot be independently reproduced against their source
records.

### Literal gate result

**Reject before preregistration.**

The significant published mechanism depends on a model whose own use statement
excludes this downstream educational application. Its missing metrics and fragile
label mapping also prevent a defensible synthetic reliability threshold. VADER and
subjectivity alone are insufficient: the paper's significant outcome interactions
are the Big-Five terms, not a fixed sentiment/subjectivity score. Substituting a
different personality model or relabeling the outputs as generic style would create
a new, outcome-unvalidated hypothesis rather than reproduce the published
mechanism. That does not supply a plausible, robust `0.0016` log-loss path.

No weights were downloaded, no feature cache was built, and no outcome score was
run.

## Candidate B: StudyChat real conversations plus normalized grades

- Official repository/dataset:
  [wmcnicho/StudyChat](https://github.com/wmcnicho/StudyChat).
- Dataset revision observed on 2026-07-28:
  `24d7987d9fbb30d9da12acc53455a10f1cdd2d7f`.
- License advertised by the project: CC BY 4.0.
- The Hugging Face dataset is gated. Anonymous access returned HTTP 401 and no
  local Hugging Face token was present.
- Accepting access would share user information and agree to dataset conditions.
  Codex did not accept the gate, seek a mirror, or bypass it.

The associated paper also reports that broad dialogue features were not
significant and that semester-specific dialogue-act/grade coefficients were not
stable. Later response-strategy papers using StudyChat substantially overlap the
closed E770 student-state × tutor-move family. Even if the participant later
accepts access, the source currently does not justify an automatic E810
preregistration.

## Candidate C: NCTE classroom transcripts and linked outcomes

- Official code/data repository:
  [ddemszky/classroom-transcript-analysis](https://github.com/ddemszky/classroom-transcript-analysis).
- The repository is MIT licensed and describes 1,660 elementary mathematics
  classroom observations with value-added measures and student test metadata.
- Every dataset user must submit a Google access form. Linked metadata are hosted
  separately on ICPSR.
- Codex did not submit an access form or disclose participant information.
- The source is classroom-level rather than one-to-one tutoring. Its open discourse
  constructs—uptake, focusing questions, student reasoning, and talk moves—overlap
  E760/E770. Without the gated outcomes there is no independent transfer gate.

**Decision:** source unavailable in the present authorized state and mechanism not
independent enough to preregister.

## Candidate D: Eedi Question-Anchored Tutoring Dialogues

- The public QATD-2k release contains 1,971 real interventions and 68,717 messages,
  but is licensed CC BY-NC-SA 4.0 / CC BY-NC 4.0 and explicitly intended for
  non-commercial research.
- It contains talk moves and historical platform summaries, not a released
  session-level learning-gain outcome.
- Its talk-move mechanism overlaps closed E770.

**Decision:** prize-safety, missing-outcome, and novelty gates fail. Do not use.

## Candidate E: other inspected sources

| Source | Useful evidence | Blocking fact | Decision |
|---|---|---|---|
| AlgebraNation forum outcomes | 2,318 real math-forum posts; resolution outcome | Source data/code/license not released | Reject source |
| APTA / Peer Chats | Real tutoring conversations | Manual CMU DataShop request required | Do not request autonomously |
| National Tutoring Observatory | Directly aligned tutor moves | Public MathEd-PII release has PII annotations, not outcomes; outcome-linked release is future work | Watch only |
| MathMentorDB / MathConverse | 5.45M real public math-help messages, CC BY 4.0 | No learning-gain or correctness outcome; full processing would be long and the resolution proxy overlaps closed dialogue-quality branches | Reject current use |
| Linguistic Alignment Predicts Learning | Real achievement-growth evidence; tutor lexical alignment positive and student alignment negative | Study data cannot be shared; directional lexical alignment was already inside the failed E760 family | Closed |
| JUSThink | Pre/post collaborative-learning evidence | Only ten transcript teams; lexical-alignment family already closed | Reject |
| Learning Through Dialogue | 397 conversations with pre/post outcomes | No released dataset/code and sensitive IRB context | Reject source |
| ChatGPT versus human math help | Public 274-learner pre/post experiment | Intervention-cell outcomes, not varied tutoring dialogues; correctness family already closed | Reject |

## Scheduler-safety result

A harmless one-occurrence heartbeat was created at 18:57:56 IST for 19:01 IST.
Its prompt was restricted to posting a timestamp and prohibited code, file, worker,
or research actions. At 19:02:43 IST there was no injected message, run history, or
execution artifact; only the unchanged persisted ACTIVE `automation.toml` existed.
The heartbeat was deleted.

This is another end-to-end failure. A stored ACTIVE definition is not a working
wake. No unattended run over one hour is authorized in this session. Bounded
sub-hour work may run only in the active foreground turn with one blocking wait.

## Honest next-step policy

1. Do not spend either remaining manual submission on v0.5-equivalent, failed, or
   source-ambiguous work.
2. Do not reopen E760-E790 under a new number, substitute a new model after seeing
   a failed gate, or sweep blends/calibration after the fact.
3. Keep a source watchlist for a genuinely open, outcome-linked, real tutoring
   dataset. StudyChat, NCTE, and APTA become eligible for a fresh audit only after
   the participant personally accepts any access terms and the exact license/data
   lineage can be frozen.
4. A future E810 must be independent of every closed branch and show a plausible
   target-free or external route to at least `0.0016` robust local gain before any
   competition outcome.
5. Preserve v0.5, the sealed `V_final`, and submission quota while strengthening
   the competition write-up: robustness matrix, negative-result table, source
   ledger, process-feature examples, and reproducibility evidence.
6. Recheck the live leaderboard and weekly submission count only when the
   participant is present to make the manual decision.

The first-place gap remains real, but the evidence does not support manufacturing
another outcome run. The correct action is to close this source portfolio, retain
the verified champion, and wait for a truly independent, legally usable signal
rather than convert research activity into leaderboard overfitting.
