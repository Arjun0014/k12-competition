# Post-E810 research audit and E820 selection

Date: 2026-07-29 (Asia/Kolkata)

## Outcome first

APTA is not the last hope and is not on the critical path today. The strongest
currently executable branch is E820: target-free contrastive adaptation of the
existing BGE-base encoder on question-disjoint MathDial
problem-to-tutoring-dialogue pairs.

The visible leaderboard supplied by the participant has first place at
`0.6008`, fifth at `0.6040`, tenth at `0.6053`, and v0.5 at `0.6054`. The
first-place gap is therefore approximately `0.0046` public log loss, while the
visible top-five gap is approximately `0.0014`. One candidate clearing the
frozen `0.0016` robust local gate could plausibly improve rank, but first place
probably requires more than one independently validated gain. No rank is
promised from local evidence.

## Residual gap after all closed work

v0.5 is already 25% full-transcript sparse text, 25% role/behavior text, and
50% BGE-base objective-conditioned semantic evidence. It already includes
objective retrieval statistics, role counts, questions, affirmations,
corrections, uncertainty, reasoning, digits, and closing behavior.

The repeated hard-validation evidence leaves one coherent weakness:

- held-out objective IDs and semantic families;
- low lexical objective-retrieval coverage;
- short objective text aligned to long, noisy tutoring sessions;
- mixed-label multi-objective sessions where session-wide state is
  insufficient.

This is a representation/alignment gap, not evidence for another flat
count, generic quality score, calibration transform, blend sweep, or
session-level label.

## Primary research synthesis

- The organizer's productive-math-talk reference identifies transcript
  filtering, tutor features, number-bearing student turns, timing, and LLM
  annotation. Filtering, tutor/role features, timing, response-state models,
  and LLM-style classifiers are already closed here; the only exact new
  numeric contrast became E810 and failed every outcome gate.
  Source:
  https://blog.drivendata.org/blog/productive-math-talk-reference
- MathDial supplies 2,861 K-12 math tutoring dialogues grounded in explicit
  math problems and is CC BY-SA 4.0. This creates a legal target-free
  problem-dialogue alignment signal that the rejected MathDial
  self-correction and tutor-move branches never used.
  Sources:
  https://github.com/eth-nlped/mathdial and
  https://aclanthology.org/2023.findings-emnlp.372/
- Current math-retrieval research reports that generic retrieval benchmarks
  do not reliably predict mathematical retrieval and that symbol/structure
  remain difficult. That supports testing task-aligned contrastive transfer
  rather than blindly replacing BGE with an advanced-math checkpoint.
  Source: https://arxiv.org/abs/2606.29894
- MMTutorBench's insight/formulation/execution rubric is useful scientific
  guidance, but its multimodal stuck-step response task does not provide an
  independently aligned learning-outcome feature for the current text-only
  competition. Turning it into another quality/tutor-stage head would overlap
  E650/E660/E770.
  Source: https://aclanthology.org/2026.acl-long.1068/
- StudyChat has real normalized grades, but only 175 dialogue users match
  released grades, its course is undergraduate AI rather than K-12 math, and
  the published broad dialogue-act associations are unstable across
  semesters. It remains useful for discovery, not a sufficient basis for
  spending a competition validation on behavior features.

## Candidate portfolio and literal decision tree

### 1. E820 now: external contrastive objective alignment

Train only the upper four layers of untouched BGE-base with symmetric InfoNCE
on 717 question-disjoint MathDial training question IDs. Evaluate without
outcomes on the 394 official-test question IDs. The crucial transfer view uses
teacher-described confusion as the query even though training uses the
original problem text.

Proceed to a separately frozen competition screen only if every target-free
retrieval, bootstrap, non-regression, and no-collapse clause in
`E820_PREREGISTRATION_2026-07-29.md` passes. A failure closes exact E820 with
no prompt, layer, epoch, loss, or checkpoint rescue.

### 2. If E820 fails: E830 discovery audit, not automatic scoring

Audit a transparent curriculum skill signature over the 398 objective texts:
operation family, number domain, representation, cognitive demand, and
procedure. Test whether those signatures improve problem/confusion retrieval
on question-disjoint K-12 dialogue data over lexical and raw-BGE controls.

This is allowed to become E830 only if the signature is frozen without
competition outcomes, is demonstrably distinct from E720 token conjunctions
and E730 dense alignment, and shows a plausible robust `0.0016` path. A weak
or redundant retrieval gate ends it before outcome validation.

### 3. APTA when access arrives: learner-role trajectory

Run the already implemented privacy-safe source gate. Only real linked
learner identity, solver/tutor role, transactions, experimental condition,
and pre/post or assessment outcomes can authorize an external trajectory
benchmark. Content-only peer help, moves, or uptake remain closed.

### 4. StudyChat reserve: assignment-aware external discovery only

The only potentially new StudyChat question is assignment-text-aligned
evidence versus held-out grades across students and semesters. It must first
beat grade-only, assignment-only, and dialogue-count controls with stable
proper scores. The undergraduate-domain and small matched-user limitations
make it lower priority than E820 and a valid APTA trajectory.

### 5. E800 final assembly

E800 remains reserved for a final ensemble of independently passing
components. Do not use it to combine failed branches. Only preregistered
weights may be evaluated, and a new ZIP is permitted only after the protected
v0.5 backup gate passes literally.

## Branches rejected in this audit before compute

- Advanced-math embedding checkpoints trained on theorem papers or
  combinatorics: source-domain mismatch with elementary objectives, with no
  K-12 tutoring transfer proof; generic encoder replacement is already a
  repeatedly failed family.
- A new response-quality, tutor-move, uptake, reasoning, productive-struggle,
  or key-step score: overlaps E530/E650/E660/E760/E770/E780.
- More objective-token crosses, local semantic neighbors, or a linear
  alignment map: overlaps E570/E720/E730 and their literal failures.
- Post-hoc calibration, component-weight optimization, or a smaller weight
  on E810: expressly closed by prior frozen failures.

## Verified E820 readiness before training

- Source audit: 2,262 raw train rows, 599 official-test rows, 1,035/394
  train/test question IDs, 318 overlapping IDs purged, leaving 1,677 legal
  rows across 717 training IDs.
- Consumed fields are only question ID, problem, teacher-described confusion,
  and conversation; no MathDial or competition outcome field is consumed.
- Five focused tests pass and Ruff is clean.
- One-batch benchmark: `7.4838306` seconds per training step, 180 projected
  steps; 64-row encoding took `3.6735134` seconds. Projected total is
  `1,553.3802` seconds (25.9 minutes), peak RSS `998,232,064` bytes.
- Benchmark SHA-256:
  `e293d8affa47046319b4e952a2ad49ae8efcc20d006c74ce1e1126ae667a844b`.

This is below one hour, so it must run in the active foreground turn with one
blocking wait. No scheduler wake is appropriate. `V_joint` and `V_final`
remain untouched; no ZIP, upload, or submission is authorized.
