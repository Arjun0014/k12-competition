# E850 preregistration: target-free lexical speaker-role denoising

Status at freeze: implementation and synthetic tests only. No E850 benchmark,
role score, correction count, competition outcome, selection environment,
`V_joint`, or `V_final` has been read.

## Independent mechanism

The organizer has stated that AI diarization plus human QA left some
student/tutor identification errors, though not a large fraction. The prior
role-prefixed NB-SVM pilot demonstrated brittleness but did not detect or
repair speaker labels. The project log explicitly left role-dropout, shared
fallback, and speaker-error detection open.

E850 implements the most conservative of those options: lexical/structural
speaker reassignment only where a target-free classifier is at least 98%
confident and contradicts the observed role. It consumes no competition
outcome, rejected checkpoint, rejected probability, tutor-move label, student
state, timing feature, curriculum prior, or external learner outcome.

This design is supported by lexical speaker-error correction work showing that
text can repair diarization attribution, and by segment-level reassignment
research showing that revisiting speaker labels can remove substantial speaker
confusion. The mechanism is nevertheless required to prove its opportunity in
this exact corpus before any outcome screen.

## Frozen source and sampling

- `data_cache/utterances.parquet` SHA-256
  `80a231d18dbb0641989ebb972f08988f0ddd3cb7c4fb769ad2fe90b5a98ba5a4`;
  6,139,854 utterances across 22,821 sessions, including 2,697,152 student,
  3,196,001 tutor, and 246,701 background rows.
- Target-free style assignment SHA-256
  `600664b9ce6c3809ff4b17206397285cfdd905a203a5c5f4d6969b403dfc9b40`.
- Train/evaluate only observed student/tutor rows. Background is never
  reassigned.
- Select complete sessions when
  `SHA256("E850|sample|"+session_id) mod 24 == 0`. Assign one of five folds by
  a separate namespaced session hash. All utterances from a session stay in
  one fold.

## Frozen representation and model

For each utterance, hash lowercase word unigrams/bigrams from:

- previous observed role and next observed role, never the current role;
- opening/middle/closing position;
- very-short/short/medium/long length;
- presence of a question mark;
- raw utterance text.

Use `HashingVectorizer(2^18, alternate_sign=False, L2 norm)` and
`SGDClassifier(loss="log_loss", L2, alpha=1e-5, max_iter=20, tol=1e-4,
average=True)`. Five OOF models use seeds `20260730..20260734`; the one final
target-free corpus model uses seed `20260830`. No vocabulary, class weight,
threshold, calibration, self-training, iteration, feature, or seed sweep is
allowed.

Synthetic corruption selects exactly 5% of sampled OOF rows by a third
namespaced hash. Because the classifier never reads the current label, recovery
means that its high-confidence prediction recovers the original role after
that label is hypothetically flipped.

Apply the final model to the full corpus in fixed 100,000-row batches. Propose
a correction only when maximum class probability is at least `0.98` and the
predicted role differs from observed. Persist only identity, original/proposed
role, probability/confidence, and target-free style cell; never duplicate
utterance text.

## Frozen target-free gate

Every clause must pass:

- sampled session-grouped OOF macro-F1 at least `0.95`;
- top-label ECE-10 at most `0.05`;
- high-confidence coverage at least `0.50`;
- high-confidence agreement with observed roles at least `0.98`;
- high-confidence synthetic corruption recovery at least `0.95`;
- full-corpus correction rate between `0.0025` and `0.03`;
- affected-session fraction between `0.05` and `0.60`;
- student-to-tutor / tutor-to-student correction ratio between `0.25` and `4`;
- at least 20 proposed corrections and at least 2% affected sessions in every
  one of the 20 target-free style cells.

Any failed clause rejects exact E850 before competition outcomes. Do not change
confidence, model, features, current/neighbor role usage, sample, rate bounds,
or style gates.

## Conditional continuation

A target-free pass only authorizes a separate committed competition
preregistration. That screen would bind exact correction/report hashes, rebuild
only the role-sensitive component with the fixed proposed corrections inside
each existing fold, preserve raw full-transcript and BGE-base components, and
evaluate only preregistered replacement/blend formulas. It must retain the
`0.0016` robust mean log-loss gate, all-environment improvement, proper-score
non-regression, fold bound, and both 5,000-replicate bootstraps.

`V_joint` remains confirmation-only and `V_final` sealed. v0.5 and its
protected ZIP remain unchanged. No upload or submission is authorised.
