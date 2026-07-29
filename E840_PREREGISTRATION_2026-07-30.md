# E840 preregistration: StudyChat longitudinal assessment transfer

Status at freeze: source/schema/count audit only. No E840 external model score,
competition outcome, `V_seen`, `V_objective`, `V_style`, `V_joint`, or
`V_final` has been read.

## Why this is independent and worth an external gate

StudyChat supplies something the released NCTE labels and rejected tutoring
corpora do not: chronological help dialogues plus three later proctored exams
for the same anonymized learners. The paper reports that conceptual/coding-help
behaviour is associated with better outcomes, while report writing and
circumventing learning objectives are associated with lower exam outcomes.

E840 does not transfer StudyChat's GPT-4.1 dialogue-act labels and does not
revive E530/E760/E770. It asks a stricter longitudinal question: after prior
assignment grades and usage volume are controlled, do fixed, transparent
help-seeking behaviours from chats before each exam improve prediction of that
future exam across unseen learners, both semesters, and all three exams?

This is also distinct from E740/E750/E780/E810/E830: no competition label,
session mean, within-session pair, challenge state, numeric-talk count,
curriculum prior, rejected checkpoint, or rejected prediction enters the
external screen.

## Frozen source

- `wmcnicho/StudyChat`, CC BY 4.0, participant-authorized gated snapshot at
  revision `24d7987d9fbb30d9da12acc53455a10f1cdd2d7f`.
- `data.jsonl` SHA-256
  `67927f73327639904c417c2f6e6200e11427920a8435b39cb7c87b5c6601e130`.
- Fall-grade SHA-256
  `d16d54c8eaa9f3723755362f0dcb651740d070bdcf5c739c7f9a27a77db5a244`.
- Spring-grade SHA-256
  `06f1a4041e1631a9787c5eac842f04e63dad1480609ac5376fcab351a1fad5ec`.
- Licence and README SHA-256 values
  `0caefd1999143ddb9e3eb93dfa802680a8ab924d95d2acc1fee434da6d66d6fc`
  and
  `86a5926deec67d5e848c930363829d53339bb601c7cc5d6064bdf917abd1e5c0`.
- Expected lineage: 16,851 interactions, 203 dialogue users, 2,214 chats;
  65 Fall and 110 Spring dialogue/grade matches. Eligible pre-exam samples are
  58/65/65 Fall and 99/110/110 Spring for exams 1/2/3, or 507 samples across
  175 learners.
- Persist only a deterministic salted SHA-256 learner key in E840 artifacts.
  Never emit the released anonymous user ID or directory name.

## Frozen chronology and features

Exam 1 may use only topics a1-a2, exam 2 only a1-a4, and exam 3 only a1-a7,
matching the released assessment order. Each sample uses interactions from
those topics only and predicts the corresponding normalized proctored exam.

The baseline contains semester, exam index, prior-assignment mean/std/min/last,
log interaction count, log chat count, and topic coverage. The candidate adds
exactly 20 fixed content/interaction features:

- student question, explanation-request, why/how, verification, direct-answer,
  writing-request, code-request, confusion, self-explanation, metacognitive,
  numeric, and code-mark rates;
- tutor question, explanation, and code-mark rates;
- student lexical diversity, mean student/tutor words, tutor/student word
  ratio, and multi-turn fraction.

Patterns and feature order are constants in committed source. No released
dialogue-act label, future chat, future assignment, exam text, student
submission, embedding, generated label, feature selection, or outcome-aware
lexicon edit is allowed.

## Frozen external model and transfer audit

Five-fold shuffled `GroupKFold` by learner, seed `20260730`. Fit both baseline
and candidate as `StandardScaler` plus `Ridge(alpha=10.0)`. Report pooled and
semester-by-exam RMSE, MAE, Spearman; all fold coefficients; and a
5,000-replicate learner bootstrap of candidate-minus-baseline RMSE gain.

Separately apply the exact 20-feature extractor target-free to every
competition session with both student and tutor text. Compare each competition
median with the external median in external-IQR units. This opens no
competition outcome.

Every clause must pass:

- pooled RMSE gain at least `0.010`;
- pooled MAE gain at least `0.005`;
- pooled Spearman non-regression;
- at least five of six semester/exam cells improve RMSE;
- worst cell RMSE regression at most `0.005`;
- learner-bootstrap positive-gain support at least `0.95`;
- all 20 behaviour features nonconstant in both sources;
- at least 75% of features have median shift at most `2.5` external IQR.

Any failure rejects exact E840 before competition outcomes. No alternate
chronology, grade target, baseline, cue, label, alpha, split, seed, threshold,
subgroup, feature subset, embedding, or model rescue is allowed.

## Conditional continuation only

A full external-and-transfer pass merely permits a separately committed
competition preregistration. That document would have to bind the exact cache
and report hashes, immutable component lineage, fold-local estimator, only
predeclared blends over raw v0.5, the `0.0016` robust mean log-loss gate,
proper-score non-regression, fold bound, and both 5,000-replicate bootstraps
before any selection-environment score.

`V_joint` remains confirmation-only and `V_final` sealed. v0.5 and its
protected ZIP remain unchanged. No upload or submission is authorized.
