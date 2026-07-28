# Trace the Ace - Exact Next-Session Handoff

**Handoff time:** 2026-07-28 16:35 Asia/Kolkata

**Project root:** `C:\Competition\K12`

**Branch:** `codex/v04-recovery`

**Remote:** `https://github.com/Arjun0014/k12-competition.git`

**Latest research-state commit:** `84d6bd6` (`Record E750 rejection`)

**Documentation state:** This refreshed handoff is committed and pushed after `84d6bd6`; verify the
current branch tip with `git rev-parse HEAD` at startup.

**Git identity:** `arjun0014 <23293383+Arjun0014@users.noreply.github.com>`

**Submission authority:** The participant alone uploads and submits. Codex must never upload or submit.

## Verified stop state

- The working tree was clean before creating this handoff.
- All research code/results through `84d6bd6` were pushed. The handoff documentation was pushed afterward,
  and the branch matched `origin/codex/v04-recovery` when this session ended.
- No K12 Python training, cache, or validation worker was active.
- No Codex heartbeat or cron automation remains active.
- The harmless 16:18 scheduler proof did not execute and was deleted after one 16:22 verification.
- E750 exists, was preregistered before scoring, and is rejected literally. Its implementation milestone is
  `162eff6`; its result milestone is `84d6bd6`.
- No E760 architecture, code, score, checkpoint, run directory, or active worker exists.
- `V_final` remains sealed. `V_joint` remains confirmation-only.
- No competition submission was made by Codex.

## Current public and submission state

The verified public champion is **v0.5 raw BGE-base replacement**:

- Public log loss: `0.6054`.
- Last participant-observed rank: approximately `#11`.
- Two manual submissions remain in the current competition week.
- Rank drift alone is not evidence and must not trigger a submission.
- Ask the participant to submit only after a locally verified candidate clears the frozen backup gate.

Protected verified ZIP:

`C:\Competition\K12\submission_builds\final_ensemble_v05_bge_backup.zip`

- Bytes: `258,290,506`.
- SHA-256:
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.
- This file must remain byte-for-byte unchanged.

The conservative top-five target remains public log loss `<=0.6038`. From public v0.5, that requires about
`0.0016` additional robust improvement. First place cannot be promised without leaderboard and validation
evidence.

## Required reading order

Read these files completely before taking technical action:

1. `C:\Competition\K12\LAST_SESSION_HANDOFF.md`
2. `C:\Competition\K12\PROJECT_LEARNING_LOG.md`
3. `C:\Competition\K12\TOP5_RECOVERY_PLAN_2026-07-22.md`
4. `C:\Competition\K12\V04_FAILURE_ANALYSIS_AND_RECOVERY_PLAN.md`
5. `C:\Competition\K12\Trace_the_Ace_full_overview.md`
6. `C:\Competition\K12\competition_rules.md`
7. `C:\Competition\K12\code_submission_format.md`

Then inspect Git status/branch, the latest run reports, preserved ZIPs, datasets, caches, logs, and active
processes. Preserve all existing work and unrelated files.

## Environment

Use only:

- `C:\Competition\K12\.venv\Scripts\python.exe`
- Python `3.12.8`
- scikit-learn `1.8.0`

Do not build competition artifacts with the system Python 3.13/scikit-learn 1.6 environment.

## Latest research sequence

The prize-safe external-source inventory was exhausted through E700. E580-E700 and their predecessors were
rejected literally under their frozen gates. Do not revive, rescue, recalibrate, reinterpret, or reuse their
failed heads/checkpoints unless a genuinely independent preregistered hypothesis makes them irrelevant rather
than tuning them.

The latest competition-side branches are:

### E710 target-free coherence

- Run: `experiments\runs\20260728T075508Z_target_free_coherence`.
- Selected 10% blend worsened loss by `0.000109900`.
- Bootstrap support: `0.1672`.
- Rejected literally.

### E720 sparse objective-response interactions

- Run: `experiments\runs\20260728T080400Z_sparse_objective_interaction`.
- Selected 10% blend worsened loss/AUROC/Brier/ECE by
  `0.001834117/0.005349205/0.000869603/0.001825527`.
- Bootstrap support: `0.0626`.
- Rejected literally.
- Rejection and wake-failure milestone: pushed commit `1727ba5`.

### E730 centered objective Procrustes alignment

- Run: `experiments\runs\20260728T083236Z_objective_alignment`.
- Selected 10% blend changed loss/AUROC/Brier/ECE by
  `+0.000068330/-0.001402028/+0.000034378/-0.002075832`.
- Bootstrap support: `0.2164`.
- Rejected literally.
- Implementation commit: `e74802e`; rejection commit: `6b4d2f5`.

### E740 session-mastery soft target

- Run: `experiments\runs\20260728T092346Z_session_mastery`.
- Exact design: fold-local training-session correctness fraction, equal-session soft-label sparse log-loss
  head, fixed `alpha=3e-5`, and preregistered 10/20/30% blends over raw BGE-base.
- Selected 20% blend improved loss/Brier/ECE by
  `0.000615881/0.000261258/0.004928590`, but harmed AUROC by `0.001421988`.
- Fold loss changes:
  `-0.000066628/-0.002146258/+0.001753177/-0.001521805/-0.001187592`.
- Bootstrap mean gain/95% interval/support:
  `0.000602671/[-0.000731569,0.001855827]/0.8212`.
- It failed loss magnitude, AUROC, fold-bound, and bootstrap clauses.
- Rejected literally without target, alpha, estimator, feature, or blend rescue.
- Preregistration commit: `11be3f6`; implementation commit: `d5b990c`; rejection commit: `0c17ebb`.

E740 is useful evidence that transcript-derived session propensity improves calibration and loss modestly, but
the exact branch is closed. A new branch must be genuinely independent, not an E740 weight/model rescue.

### E750 session-conditional objective likelihood

- Run: `experiments\runs\20260728T110308Z_session_conditional_objective`.
- Exact design: immutable BGE-base interaction block; all positive-negative objective pairs within each legal
  mixed training session; equal total weight per session; intercept-free `C=0.1` conditional logistic head;
  fixed fold prior; preregistered 10/20/30% blends over raw v0.5.
- The deterministic 10% selection regressed equal-environment macro loss by `0.001437298`; mean
  AUROC/Brier/ECE changes were `-0.000202409/+0.000628759/+0.003492013`.
- Environment loss changes were `V_objective=+0.000145145`, `V_seen=+0.002076314`, and
  `V_style=+0.002090436`. Worst fold regression was `0.002394717`.
- Session and semantic-family bootstrap support were both `0.0000`; all nine gates failed.
- Report SHA-256:
  `6a21fe439d0ad749889f709d74b6b82d72fd8bf305d8105dcb5ddc0161442da6`.
- Rejected literally without pair, weight, feature, C, prior, calibration, blend, or gate rescue.
- Preregistration/implementation commit: `162eff6`; rejection commit: `84d6bd6`.

E750 confirms that removing session propensity from the likelihood does not produce calibrated transferable
probabilities: even the smallest frozen blend harmed all three selection environments. The exact branch is
closed. No locally verified candidate beats v0.5.

## Automation failure warning

Four persisted schedules failed to execute:

1. E720 thread heartbeat expected at 13:40 IST.
2. Post-E730 thread heartbeat explicitly anchored at 14:15 IST.
3. Standalone local-project E750 cron explicitly anchored at 15:35 IST.
4. Harmless current-task scheduler proof explicitly anchored at 16:18 IST; one verification at 16:22 found no
   wake or execution evidence.

All showed persisted `ACTIVE` configuration, but no corresponding task execution occurred. All are now
deleted. **A stored automation file or rendered ACTIVE card is not proof of a functioning wake in this
environment.**

For the next session:

- Do not claim confidence in any scheduler until a harmless short end-to-end wake actually executes.
- Do not launch a long unattended worker until its single-checkpoint mechanism has passed that live test.
- For a run measured under one hour, prefer completing it in the active foreground turn with one blocking
  wait rather than creating an automation.
- Never poll repeatedly.
- For a run over one hour, estimate duration first, launch once, and use only one completion checkpoint through
  a mechanism proven end-to-end in the current session. If it is still running at that checkpoint, use actual
  batch/row telemetry for exactly one replacement checkpoint.
- If no reliable wake mechanism is available, stop before launching the long run and tell the participant
  plainly instead of pretending it is supervised.

## Next technical priority after E750

No next branch is currently authorized. E710-E750, E580-E700, and all earlier rejected exact branches remain
closed. Begin only with a fresh evidence audit using already authorized `V_seen`, `V_objective`, and `V_style`;
do not read `V_final`, and keep `V_joint` confirmation-only.

1. Search the code, learning log, and recovery plan before claiming novelty. Explicitly exclude every rejected
   feature family, target, estimator, checkpoint, calibration, and post-hoc rescue.
2. Do not assign E760 or preregister anything unless the mechanism is genuinely independent and has a
   quantitative path to at least `0.0016` robust log-loss gain over raw v0.5.
3. Freeze lineage, splits, features, estimator, seed, weights, proper-score gates, fold bound, and both session
   and semantic-family bootstrap rules before any new outcome score.
4. Run focused tests and a target-free/synthetic benchmark, then commit and push the frozen implementation
   before exactly one authorized selection validation.
5. A failed gate rejects that exact branch without rescue. A pass may proceed only through the frozen backup,
   confirmation, and top-five gates.
6. Never build a new ZIP until the backup gate passes. Preserve v0.5 byte-for-byte.

## Promotion gates

### Backup gate

- Mean hardened log-loss gain at least `0.0010`.
- Improve at least three environments.
- No environment regression greater than `0.0005`.
- No mean AUROC, Brier, or ECE regression.
- At least `90%` paired bootstrap support.

### Conservative top-five gate

- Projected public loss `<=0.6038`.
- At least about `0.0016` additional robust gain over public v0.5.
- Improve every hardened environment.
- No material fold/subgroup reversal.
- Preserve or improve AUROC, Brier, ECE, and calibration sanity.
- At least `95%` session and semantic-family bootstrap support.
- Pass the locked confirmation protocol and only then the still-sealed final audit.

## Non-negotiable boundaries

- Never upload or submit.
- Never inspect `V_final` early.
- Never use public submissions for tuning.
- Never weaken a frozen threshold after seeing a result.
- Never revive a rejected exact branch through post-hoc calibration, reweighting, or reinterpretation.
- Never overwrite or rebuild the protected v0.5 ZIP.
- Append every material run to `PROJECT_LEARNING_LOG.md` and the active recovery plan.
- Commit and push meaningful milestones as `arjun0014`.
- Do not promise a leaderboard position without evidence.
