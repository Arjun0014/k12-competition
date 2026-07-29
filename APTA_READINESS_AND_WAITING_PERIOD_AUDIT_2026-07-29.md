# APTA readiness and waiting-period audit — 2026-07-29

## Decision

Waiting for APTA approval does not block useful work, but the remaining work
must respect the closed experiment families. The acquired StudyChat and NCTE
sources do not presently support a genuinely independent successor to E810:

- StudyChat response style and next-turn continuation overlap rejected E710 and
  E770.
- Student efficacy, attempt quality, reflection, uncertainty, answer
  extraction, and productive struggle overlap rejected E770 and E780.
- NCTE uptake, focusing questions, student reasoning, self-correction, and
  hedging overlap rejected E530, E760, E770, and E780.
- StudyChat's released course grades are longitudinal student outcomes, but
  only 175 released dialogue users have matched grades and the published broad
  dialogue-act associations are not stable across semesters. Training another
  aggregate style/act model would not establish a plausible robust `0.0016`
  competition log-loss path.

No successor experiment is preregistered. No competition outcome or prediction
was opened for this audit.

The productive waiting-period milestone is instead a complete target-free APTA
source gate. It makes an approval actionable immediately while preventing a
private source from being mistaken for an authorized experiment merely because
it downloaded successfully.

## Primary-source research

The audit design uses the current official DataShop export contract rather than
guessing APTA's private schema:

- DataShop Public API v0.41 documents canonical transaction fields including
  anonymized student ID, session ID, time, duration, response types, problem,
  step, attempt number, outcome, selection, input, feedback, and KC columns:
  <https://pslcdatashop.web.cmu.edu/api/DataShop%20Public%20API-v0.41.pdf>.
- DataShop's Tutor Message v4 guide distinguishes study conditions, skills/KCs,
  action evaluation, and tutor feedback:
  <https://pslcdatashop.web.cmu.edu/dtd/guide/tutor_message_dtd_guide_v4.pdf>.
- The StudyChat dataset paper reports behavioral usage/outcome analyses, while
  the released source itself warns that it is one undergraduate AI course:
  <https://arxiv.org/abs/2503.07928>.
- The 2026 StudyChat response analysis reports statistically significant but
  small global next-turn effects, directly overlapping closed response-style
  work: <https://arxiv.org/abs/2607.09919>.
- The 2026 pedagogical-alignment paper shows that answer extraction and
  deployment context dominate many dialogue-use patterns, but whole-dialogue
  metrics can hide turn-level behavior:
  <https://aclanthology.org/2026.acl-long.875/>.
- The 2026 NCTE productive-struggle work operationalizes impasse, implied
  uncertainty, self-correction, hedging, and agentive pronouns. Those
  constructs were already implemented inside E770/E780:
  <https://aclanthology.org/2026.bea-1.36/>.

This literature does not justify relabeling a closed branch. It reinforces the
requirement that APTA must contribute linked learner-role and knowledge-state
trajectories with real learning outcomes, not another utterance-style
classifier.

## Implemented target-free landing gate

Added:

- `src/trace_ace/apta_source_audit.py`;
- `scripts/audit_apta_source.py`;
- `tests/test_apta_source_audit.py`.

Protocol: `APTA_target_free_source_audit_v1`.

The command-line wrapper accepts source files only from
`Datasets/APTA`. The audit:

1. creates a deterministic file inventory with byte sizes and SHA-256 values;
2. inspects ZIP paths, symlinks, encryption, risky executable/serialized
   payload types, member sizes, total expansion, and compression ratio before
   opening tabular members;
3. streams CSV/TSV/TAB/TXT sources without extraction;
4. detects only schema-level concepts for student, session/class, role,
   condition, transaction, time, problem, step, KC, tutor/student action,
   transaction correctness, chat text, and pre/post/gain/assessment outcomes;
5. hashes identifiers privately and emits only unique/linkage counts;
6. never emits raw IDs, chat values, correctness values, or learning-outcome
   values;
7. distinguishes transaction correctness from a real pre/post, gain, or
   assessment outcome;
8. requires at least 50 learners linked between transactions and real learning
   outcomes, plus stable grouping, solver/tutor role, condition, transaction
   structure, safe archives, supported tables, and clean parsing before
   reporting `discovery_ready=true`.

`discovery_ready=true` would authorize only the next source/novelty design
stage. It would not authorize competition outcome access, a model, a blend,
`V_joint`, `V_final`, a ZIP, or a submission.

## Verification

- Required environment: project `.venv`.
- Python: `3.12.8`.
- scikit-learn: `1.8.0`.
- pandas: `2.3.3`.
- NumPy: `2.5.1`.
- Focused tests: `4 passed` in `0.25` seconds.
- Full suite: `220 passed, 8 subtests passed` in `68.63` seconds.
- Focused cases cover valid transaction/pre-post linkage, transaction-only
  rejection, archive traversal rejection before member parsing, identifier and
  outcome-value non-emission, and deterministic report hashing.

Artifact hashes before documentation:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `src/trace_ace/apta_source_audit.py` | 18,937 | `b6b55b8f6737fba71499fd2ae56946a0aabc43634412b361a041cab37140a9d4` |
| `scripts/audit_apta_source.py` | 1,980 | `90a4533cd85c73ec2458ec3b6aee8e7be92f9b02004f0091658c95d48a0fac51` |
| `tests/test_apta_source_audit.py` | 3,946 | `652dae95b6cfc7d62b188a5bda2fa018cbc6388fba408b76c01a61e79b7e0ee1` |

## Competition and scheduler state

- Competition log loss, AUROC, Brier, ECE, folds/environments, and bootstrap:
  not applicable; no competition outcome was accessed.
- Projected public loss/rank: unchanged v0.5 public `0.6054`, approximately
  `#11` from the last participant-provided observation; no new projection.
- `V_joint_accessed=false`.
- `V_final_accessed=false`.
- No competition cache, prediction, checkpoint, model, blend, ZIP, upload, or
  submission was created.
- The protected v0.5 ZIP remains SHA-256
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.
- This implementation and its full verification took less than two minutes per
  command and remained in the active foreground turn. No long run or completion
  wake was required. The already-proven APTA access monitor remains the only
  useful unattended checkpoint while approval is pending.

## Exact next actions

1. Keep the two-hour APTA access monitor active while the request is
   `Not Reviewed`.
2. If view access is granted, download only authorized project/dataset files
   into `Datasets/APTA`, preserve the raw bytes, and run this audit before
   designing features.
3. If any archive or linkage gate fails, stop that exact source path. Do not
   infer identifiers, outcomes, roles, conditions, or cross-file joins.
4. If every source gate passes, inspect the real schema and preregister a
   grouped external learner-role/knowledge-state benchmark before scoring its
   learning outcome.
5. Only an externally stable, genuinely independent mechanism with a plausible
   robust `0.0016` competition-loss path may advance to one frozen selection
   validation.
