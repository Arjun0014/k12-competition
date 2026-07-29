# NCTE/APTA authenticated-access follow-up — 2026-07-29

## Scope and immutable continuation state

This follow-up records only the authenticated NCTE and APTA access work that
occurred after `STUDYCHAT_NCTE_APTA_ACCESS_AUDIT_2026-07-29.md`. The continuation
began from clean, synchronized branch `codex/v04-recovery` at
`266f09f5c8530e8d5e216429e7975c61586e9009`. Raw third-party data remain under
the ignored `Datasets/` tree and must not be committed or redistributed.

No competition outcome, `V_joint`, or `V_final` evidence was accessed. No
competition ZIP was built, uploaded, or submitted.

## Scheduler-safety conclusion

The explicit-goal wake hypothesis was tested a third time with a
standards-aligned hourly heartbeat due at 13:31 IST. The narrow goal remained
active, the turn stayed open, and no final response was sent before the due
time. At 13:33 IST there was still no wake token or run artifact. The heartbeat
was deleted and the repeated-failure goal was marked blocked.

The three in-session goal-backed variants therefore all failed end to end:

1. three-minute cadence with successful notifications muted;
2. one-minute cadence with successful notifications enabled; and
3. an hourly minute-31 schedule with successful notifications enabled.

An ACTIVE definition or rendered card is not execution proof. At the close of
this follow-up there are zero persisted `automation.toml` definitions and zero
K12 Python, training, Git, or Git-LFS workers. The only process observed with
the repository as its working directory was the current in-app browser kernel.
No unattended run over one hour is authorized. Runs measured below one hour
must remain in the foreground with one blocking wait.

## NCTE access, terms, and immutable source

The participant authorized truthful form completion and data download. The
Google NCTE terms form was submitted with:

- position: independent machine-learning researcher and Trace the Ace
  competition participant;
- affiliation: independent researcher with no university or departmental
  affiliation;
- purpose: non-harmful aggregate research on de-identified tutoring discourse
  for post-tutoring outcome prediction, with no identification, evaluation,
  surveillance, discrimination, or redistribution.

Google confirmed that the response was recorded. The official folder
`NCTE Transcripts - Release`, Drive folder ID
`19LzXF0IRtOGO62ZUnJeX6rBTjynuW6RR`, was automatically shared by the dataset
owner. All three official CSVs were downloaded to:

`Datasets/NCTE/release_19LzXF0IRtOGO62ZUnJeX6rBTjynuW6RR`

| File | Bytes | Rows | SHA-256 |
|---|---:|---:|---|
| `ncte_single_utterances.csv` | 113,819,937 | 580,408 | `bbfa37ac857991e5d12b1677216896f32d7cf4c2d95b618b67afa7b3bf23b3d7` |
| `paired_annotations.csv` | 547,485 | 2,348 | `acb81f02dfd00e8e5426af74525fb1f629c7f880cdb1646d533a1852bc6e3ae7` |
| `student_reasoning.csv` | 204,227 | 2,000 | `16bc7eb53e1955f9ecf28b8f02d9727659fdd1a138782355243e0754a3633384` |

The three official files total 114,571,649 bytes. Deterministic
`SOURCE_MANIFEST.sha256` has SHA-256
`604ec4e80c584c3d96afb54ddf95c31a9acf7c4a0b01cdbad8b8e144c68d814a`.

The browser's event waiter did not report either Drive download, but the files
completed in the normal local Downloads directory and were verified by exact
name, size, full CSV parse, and SHA-256 before being moved into the ignored
source directory. This browser-event limitation is not evidence that the Codex
scheduler can wake a task.

## NCTE structural and security audit

The full parse used the required environment: Python 3.12.8, scikit-learn
1.8.0, and pandas 2.3.3. It completed in 7.4 seconds. All artifacts are ordinary
CSV files, their headers contain no NUL bytes, all three parse successfully, and
none has a duplicate full row.

The transcript source contains 580,408 utterances, 1,660 observation IDs, 319
video IDs, and 9,518,741 words. Speaker counts are 286,561 teacher, 255,268
student, 38,566 multiple-student, and 13 missing. The two repository-listed
speaker-assignment issue IDs, `4263` and `2065`, must be excluded from any
future source analysis.

The paired annotations cover 2,348 unique exchanges from 776 observation IDs.
Positive-label counts/rates are:

- student on-task: 1,964 / 83.65%;
- teacher on-task: 2,004 / 85.35%;
- high uptake: 813 / 34.63%; and
- focusing question: 359 / 15.29%.

The student-reasoning file contains 2,000 unique utterances from 744
observations and 276 teachers; 419 are positive, 1,580 are negative, and one
label is missing. Every reasoning `comb_idx` maps to the full utterance source,
and every paired-annotation observation ID exists in the utterance source.

## NCTE research and novelty decision

The official BEA 2023 paper reports 1,660 elementary-math observations and
links predicted uptake, focusing questions, and student reasoning to classroom
observation scores and teacher value-added outcomes:

- <https://aclanthology.org/2023.bea-1.44/>
- <https://doi.org/10.3886/ICPSR36095.v4>

However, the available annotations are not an independent Trace the Ace
mechanism. High uptake is the E760 family; focusing questions/tutor moves and
student reasoning are explicit E530/E770 constructs. The project log already
closed those branches and the post-E790 audit explicitly classified NCTE talk
moves as overlapping E760/E770. The original NCTE code also uses ordinary
row-level K-fold splits rather than grouping by observation, so its reported
classifier workflow is not a leakage-safe competition validation template.

The linked ICPSR V4 metadata include student assessments and teacher
value-added measures, which would be a more independent outcome-transfer
source. ICPSR states that study 36095 is available to users at member
institutions. The participant is an independent researcher, so no institutional
affiliation or entitlement was fabricated, no Researcher Passport was created,
and no ICPSR data were downloaded. The ICPSR browser page also returned a
temporary service-unavailable response during this session.

**Decision:** preserve NCTE as a lawful, pinned external research source, but do
not preregister E810, score a competition outcome, train a discourse model, or
revive E530/E760/E770 from these released labels. The NCTE source alone does not
establish a plausible robust 0.0016 log-loss path over raw v0.5.

## APTA account and access request

The participant signed into GitHub and authorized truthful contact disclosure
for APTA/LearnSphere. The GitHub OAuth flow created the DataShop account and
confirmed public-dataset access.

APTA is private DataShop project `574`, PI Vincent Aleven. The project has no
additional terms; the general DataShop terms require research-only,
non-commercial use, prohibit re-identification and redistribution, and require
separate accounts for collaborators.

Only **view access** was requested. Edit access was deliberately not requested
because it is unnecessary for research/downloads. The 250-character reason
stated that the participant is an independent ML researcher studying
de-identified tutoring interactions for Trace the Ace, will use only aggregate
non-identifying features, and will not redistribute data.

DataShop confirmed submission. `Access Requests` records:

- project: APTA;
- PI: Vincent Aleven;
- level: View;
- status: Not Reviewed;
- last request: 2026-07-29.

No APTA private file was downloaded because PI approval has not been granted.
The prioritized project datasets remain:

- `5153` — Hopewell Collaboration Winter 2021-2022;
- `5549` — 2023 Dynamic Transitions Study (North Hill - Actual Study Data);
- `5604` — Steel Valley Study Data 2023.

Public APTA research describes reciprocal peer-tutoring records with explicit
solver/tutor roles, algebra transactions, chat, correctness feedback, and
adaptive support. An earlier study compared adaptive and fixed support with 122
participants. A 2026 classroom experiment compared dynamic versus pre-planned
individual/collaborative learning with 195 students in 13 classes; both
conditions learned, the learning outcomes were comparable, and the standard
condition was more collaboration-efficient. The similarly named private 2023
dataset may support that paper, but this is only a source-name inference and
must not be treated as verified lineage before access:

- <https://pact.cs.cmu.edu/pubs/Walker%2C%20Rummel%2C%20Koedinger%202011.pdf>
- <https://eric.ed.gov/?id=EJ1495717>

The one conditionally independent hypothesis is therefore not another
utterance-level “good tutor move” classifier. It is learner-role and
knowledge-state trajectory transfer: whether solver/tutor role, BKT mastery,
error timing, collaboration transitions, and individual pre/post gain jointly
provide an externally reproducible learner-state representation. This remains
only a discovery path. It can advance to a frozen target-free/external protocol
only if approved files contain auditable pre/post outcomes, stable student and
class grouping, role/transaction linkage, and enough independent learners. A
grouped external validation would then need to beat a static mastery/count
baseline with stable proper scores and bootstrap support before any
competition-side architecture is proposed. Content-only peer-help, uptake, and
talk-move variants remain closed by E530/E760/E770.

If access is approved, the next authorized step is to download the official
project files, quarantine/audit archives before extraction, freeze exact
dataset IDs and hashes, and perform a source/label audit before any candidate
architecture is proposed. Approval is not permission to reuse a rejected
mechanism or access a competition outcome.

## Metrics, champion, and decision

- Environment: Python 3.12.8; scikit-learn 1.8.0; pandas 2.3.3.
- Competition log loss, AUROC, Brier, ECE: not applicable.
- Competition folds/environments and bootstrap: not applicable.
- Projected public loss/rank: unchanged v0.5 public loss 0.6054; no new
  projection.
- `V_joint_accessed=false`; `V_final_accessed=false`.
- No new competition cache, prediction, blend, checkpoint, ZIP, upload, or
  submission.
- Protected ZIP SHA-256 remains
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.

**Overall decision:** NCTE acquisition passes provenance and integrity checks
but does not pass the independence/headroom gate for E810. APTA is pending
external approval and is the next source-side decision point. Preserve v0.5 and
do not spend a manual submission slot without a locally verified candidate
gain.
