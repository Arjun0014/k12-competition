# StudyChat, NCTE, and APTA access audit — 2026-07-29

## Scope and safety boundary

- Pre-action repository HEAD: `787b126aa822ff589049488354efe2dd8f2c3e62`.
- Branch/upstream: `codex/v04-recovery` / `origin/codex/v04-recovery`, identical
  before this audit.
- Required environment: project `.venv`, Python `3.12.8`, scikit-learn `1.8.0`.
- No competition outcome, component OOF prediction, `V_joint`, or `V_final` was
  opened.
- No competition upload or submission occurred.
- The protected v0.5 backup remains unchanged at
  `submission_builds/final_ensemble_v05_bge_backup.zip`, SHA-256
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.

## StudyChat acquisition

The participant personally accepted the Hugging Face access gate and explicitly
authorized downloading the source. The official gated dataset was downloaded
from `wmcnicho/StudyChat`, pinned to revision
`24d7987d9fbb30d9da12acc53455a10f1cdd2d7f`.

Local root:

`Datasets/StudyChat/24d7987d9fbb30d9da12acc53455a10f1cdd2d7f`

Acquisition evidence:

- authenticated 33,667,862-byte benchmark download: approximately `8.5` seconds;
- full pinned snapshot: `73.86` seconds;
- official files: `78`;
- official bytes: `1,298,050,081`;
- local files including Hugging Face cache metadata: `159`;
- local bytes including cache metadata: `1,298,074,850`;
- deterministic 78-entry SHA-256 manifest:
  `Datasets/StudyChat/24d7987d9fbb30d9da12acc53455a10f1cdd2d7f/SOURCE_MANIFEST.sha256`;
- manifest SHA-256:
  `608462ed21f983d22068c40e168ce4a600b9be94e5689d24df6b7e4717b5d1a3`.

A temporary fine-grained read-only token was created only because browser-driven
Xet downloads did not prove a transfer. It was used through a short-lived local
credential file, the file was deleted after the download, and the token was
then deleted from the Hugging Face account. The existing participant tokens
were not changed.

The source advertises CC BY 4.0. The 33.7 MB submissions ZIP was not extracted.
A path and type audit found:

- `4,990` entries;
- `109,548,596` uncompressed bytes;
- no absolute or parent-traversal paths;
- no symlinks;
- no pickle, Joblib, PyTorch, NumPy binary, or similar serialized payload
  extensions.

## StudyChat lineage and coverage

The combined `data.jsonl` contains `16,851` interactions from `203` students and
`2,214` chats:

| Semester | Interactions | Dialogue users | Chats | Grade rows | Matched grade users |
|---|---:|---:|---:|---:|---:|
| Fall 2024 | 6,864 | 84 | 937 | 70 | 65 |
| Spring 2025 | 9,987 | 119 | 1,277 | 111 | 110 |

All released grade rows contain the seven normalized assignment scores, three
normalized exam scores, and aggregate `llm` / `no_llm` values without missing
assessment fields. Five Fall grade users and one Spring grade user have no
released dialogue; nineteen Fall dialogue users and nine Spring dialogue users
have no released grade. No identifier imputation is authorized.

The dialogue file already supplies student prompt, assistant response, full
message history, assignment topic, timestamp, chat/user identity, interaction
position, semester, and an LLM-generated hierarchical dialogue-act label. The
released normalized grades create a real outcome-linked external source, but
not by themselves an independent competition mechanism.

## StudyChat literature and novelty decision

The final LAK 2026 paper is:

Hunter McNichols, Fareya Ikram, and Andrew Lan, *The StudyChat Dataset:
Analyzing Student Dialogues With ChatGPT in an Artificial Intelligence Course*,
DOI `10.1145/3785022.3785029`,
<https://arxiv.org/abs/2503.07928>.

The authors explicitly report that dialogue-act features increase in-sample
explanatory power but appear overfit or underpowered; the significant
coefficients differ substantially across semesters. Fall has no significant
broad dialogue-act coefficient and no significant exam coefficient. Spring
has exam associations, but the overall usage-group and behavioral-cluster
increments are modest and not consistently statistically significant.

A newer StudyChat analysis,
*When LLM Tutoring Responses Work*,
<https://arxiv.org/abs/2607.09919>, reports small global response-style effects
on next-turn continuation, with larger context-specific effects. Its central
student-state × response-style construct overlaps the literally rejected E770
family. The direct-answer / next-student behavior also overlaps rejected E710,
and challenge/over-reliance interpretations overlap rejected E780. Aggregate
student dialogue-act counts are already largely represented by the existing
sparse/dense champion and rejected process-feature families.

**Decision:** access changes StudyChat from unavailable to auditable, but it does
not currently satisfy the independence or plausible robust `0.0016` log-loss
headroom requirement for an E810 preregistration. No external grade model,
competition feature cache, outcome validation, blend, or ZIP is authorized from
this acquisition alone. A future StudyChat branch would first need a
pre-outcome, code-and-log novelty proof for a mechanism outside E710/E760/E770/
E780 and the existing sparse/dense response models, followed by a frozen
cross-semester external gate.

## NCTE access state

The public official repository was downloaded and pinned in detached state:

- repository:
  <https://github.com/ddemszky/classroom-transcript-analysis>;
- local root: `Datasets/classroom-transcript-analysis`;
- revision: `ff63e9787350d39f4fbc3083732d66d8ebe4402a`;
- repository status: clean;
- license: MIT;
- `LICENSE` SHA-256:
  `f7e7258ef05d5f9a6e227c9e0b315a288923280b42d443e7aca91e93cf382aa8`;
- `README.md` SHA-256:
  `dfc596a345d0eefb4ae0a3f7e48da9469a53cbbfb116ae8d289c29184f9de71e`.

This checkout contains code, transcript-issue exclusions, and coding-scheme
PDFs, but not the transcript CSVs. Every transcript user must submit the
official Google form. The form is open in the in-app browser and currently
blocked on Google sign-in. It asks for email, full name, position, department/
university affiliation, and a two-to-three-sentence research objective. No name,
university affiliation, or account credential was fabricated.

The linked NCTE Main Study metadata are ICPSR 36095. The study page states that
the main data are freely available to users at ICPSR member institutions; public
documentation is available separately. The transcript form and ICPSR access are
distinct gates.

## APTA access state

The three verified CMU DataShop datasets are:

- `5153`: `Hopewell Collaboration Winter 2021-2022`;
- `5549`: `2023 Dynamic Transitions Study (North Hill - Actual Study Data)`;
- `5604`: `Steel Valley Study Data 2023`.

All three belong to private DataShop project `APTA` (project `574`). DataShop's
GitHub/LearnSphere sign-in page is open in the in-app browser and currently
blocked on GitHub credentials. No credential was typed and no access request was
submitted.

## Scheduler decision

The harmless scheduler wake remains unproven from the prior session, so no
over-one-hour unattended run is allowed. The StudyChat transfer was measured
from a 33.7 MB benchmark, projected below one hour, and completed in the active
foreground turn with one completion checkpoint. No scheduler was needed and no
worker was left active.

## Next authorized action

1. The participant signs into the open Google and GitHub tabs.
2. Submit the NCTE form only with the participant's real account name/email,
   truthful independent-research status, explicit lack of university affiliation
   if applicable, and a transparent Trace-the-Ace research objective.
3. Inspect the authenticated APTA project/request terms before transmitting a
   purpose statement; do not invent institutional sponsorship, IRB approval, or
   academic credentials.
4. Pin and hash any granted release before opening its outcome columns.
5. Preregister a new experiment only if the newly available source supports a
   genuinely independent mechanism and a frozen external gate with a plausible
   path to at least `0.0016` robust competition log-loss gain.
