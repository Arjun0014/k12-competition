# Trace the Ace: LLM tutor-move portfolio

Date: 2026-07-30 (Asia/Kolkata)

## Why this track is reopened

The rejected tutoring-process branches did not exhaust validated automatic move
annotation:

- E530 was a word/character TF-IDF classifier trained on MathDial tutor turns.
- E770 used hand-written lexical rules for 16 NTO-style moves and ten student
  states.
- E490/E500 asked frozen Qwen models one direct yes/no outcome question and
  probed their hidden states. They did not classify tutor moves.
- E710 was a hashed next-turn coherence classifier.

The new estimand is different: use a locally executed, commercially usable
instruction model to infer a four-state tutor-action sequence
(`focus/generic/probing/telling`) without competition outcomes, then test whether
that externally validated sequence representation adds robust information to
raw v0.5. This is motivated by Ikram, Scarlatos, and Lan (BEA 2025): their LSTM
over move sequences was competitive for dialogue success, and move annotations
improved the Llama result on the real AlgebraNation data.

## Legal and source boundary

Only local Apache-2.0 checkpoints are eligible for the active probes:

1. P880: `Qwen/Qwen3-0.6B`, revision
   `c1899de289a04d12100db370d81485cdf75e47ca`.
2. P881: `Qwen/Qwen2.5-1.5B-Instruct`, revision
   `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

The StanfordSCALE tutor-move checkpoints are excluded even though their reported
task performance is attractive: their live model cards say
`License: [More Information Needed]` and omit the training-data license. They
cannot enter development, testing, packaging, or a prize claim without a
permissive license from the owner.

The held-out external benchmark is the already preserved CC BY 4.0 MathDial
tutor-move cache. Competition transcripts stay local. No hosted API is used.

## Frozen external ladder

The prompt, label definitions, input allocation, checkpoint, label tokens,
batching, sample rule, metrics, seed, and gates are code constants committed
before any P880 external score.

### Stage 0: target-free resource benchmark

- Read MathDial text columns only; do not load `move`.
- Score 32 stable-hash-selected held-out turns.
- Maximum prompt length: 256 tokens.
- Previous-student allocation: final 48 tokens.
- Tutor allocation: first 64 tokens.
- Batch size: 16; CPU threads: 6; float32 inference.
- Pass only if projected held-out time is at most 2 hours, projected
  22,821-session x 16-window local cache time is at most 168 hours, and RSS is
  below 8 GiB.

### Stage 1: held-out external screen

- Select 128 rows per class from the official MathDial test split by SHA-256 of
  `(qid, source_row, turn_index)`, for 512 rows total.
- Compare on the identical rows against the immutable E530 predictions.
- Require all of:
  - macro-F1 at least `0.55`;
  - macro-F1 gain over E530 at least `0.03`;
  - every class F1 at least `0.40`;
  - at least `0.95` support for positive macro-F1 gain in 2,000 paired
    `qid` bootstraps.

A failed clause rejects that exact model/prompt. There is no prompt edit,
temperature, demonstration, label merge, threshold, calibration, or sample
rescue.

### Stage 2: full external confirmation

Only a Stage-1 pass may score all 3,664 official-test tutor turns. The same
thresholds apply, plus accuracy must be at least `0.57`. A failure rejects the
exact model/prompt before any competition outcome.

If P880 fails, P881 may start from Stage 0 using the same frozen prompt and
gates. P881 is a separately named checkpoint branch, not a post-score P880
modification. If both fail, the entire local zero-shot move-annotation family is
closed.

## Conditional competition experiment

No E880 competition experiment is preregistered yet.

If and only if an external branch passes, a separate preregistration must freeze
the competition cache and model before any new competition outcome:

- exactly 16 chronological tutor windows per session, chosen without labels;
- the preceding student turn plus current tutor turn as the only model input;
- four move probabilities, entropy, phase means, and fixed chronological
  transition summaries;
- a low-capacity fold-local sequence head and raw-v0.5 blends declared in
  advance;
- the existing `V_seen`, `V_objective`, and `V_style` proper-score,
  worst-fold, and paired-bootstrap gates, including at least `0.0016` robust
  log-loss gain;
- `V_joint` confirmation-only and `V_final` sealed.

The full local cache must be benchmarked before launch. A projected run over one
hour uses exactly one calculated completion checkpoint through the scheduler
mechanism already proven end to end in this task.

## Broader first-place waterfall

1. **Immediate:** execute P880, then P881 only if required.
2. **Score route if an external gate passes:** preregister one move-sequence
   competition experiment, build its cache, and run one authorized validation.
3. **Independent real-outcome route:** resume APTA only after access is actually
   granted and its learner/outcome lineage passes the frozen landing gate.
4. **Conditional high-accuracy route:** reconsider StanfordSCALE only after a
   documented permissive model and training-data license exists.
5. **Submission use:** preserve v0.5. Build a new ZIP only after the conservative
   backup gate passes. The participant alone uploads and submits. A verified
   improvement may use a weekly slot; a failed local branch may not.

This portfolio does not promise first place. It creates multiple evidence-gated
ways to pursue the approximately `0.0046` public gap while preventing another
v0.4-style post-hoc failure.
