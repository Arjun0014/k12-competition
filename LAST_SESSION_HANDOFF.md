# Trace the Ace - Exact Next-Session Handoff

**Handoff time:** 2026-07-28 16:02 Asia/Kolkata

**Project root:** `C:\Competition\K12`

**Branch:** `codex/v04-recovery`

**Remote:** `https://github.com/Arjun0014/k12-competition.git`

**Latest pushed commit:** `0c17ebb` (`Record literal E740 rejection`)

**Git identity:** `arjun0014 <23293383+Arjun0014@users.noreply.github.com>`

**Submission authority:** The participant alone uploads and submits. Codex must never upload or submit.

## Verified stop state

- The working tree was clean before creating this handoff.
- Branch `codex/v04-recovery` matched `origin/codex/v04-recovery` at `0c17ebb`.
- No K12 Python training, cache, or validation worker was active.
- No Codex heartbeat or cron automation remains active.
- The scheduled E750 continuation was explicitly deleted at the participant's request.
- No E750 code, score, run directory, checkpoint, or result exists.
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

## Automation failure warning

Three persisted schedules failed to execute:

1. E720 thread heartbeat expected at 13:40 IST.
2. Post-E730 thread heartbeat explicitly anchored at 14:15 IST.
3. Standalone local-project E750 cron explicitly anchored at 15:35 IST.

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

## Next technical priority: E750

There is no preregistered E750 architecture and no active worker. Start with a fresh evidence audit:

1. Reconstruct the remaining v0.5 error/feature gap using only already authorized `V_seen`, `V_objective`, and
   `V_style` evidence. Do not read `V_final`; keep `V_joint` confirmation-only.
2. Exclude every rejected branch and post-hoc rescue. Search the code and learning log before claiming an idea
   is new.
3. Require a plausible path to at least `0.0016` robust log-loss gain over raw v0.5 before preregistering.
4. Freeze data lineage, splits, features, estimator, seeds, blend weights, proper-score gates, fold bound, and
   bootstrap rule before producing any new outcome score.
5. Run focused tests and a target-free/synthetic runtime benchmark.
6. Commit and push the preregistration and implementation milestone before the one authorized validation.
7. If the branch fails, record exact loss, AUROC, Brier, ECE, fold/environment changes, bootstrap evidence,
   runtime, hashes, projected public score, honest rank bracket, and literal rejection. Do not rescue it.
8. If it passes the pilot gate, build only the authorized cache/probe, evaluate `V_seen`, `V_objective`, and
   `V_style`, use `V_joint` only as allowed, and apply the backup/top-five gates literally.
9. Evaluate only preregistered 10/20/30% blends when that frozen protocol specifies them. Never perform a
   post-hoc weight sweep.
10. Build and locally verify a new clearly named ZIP only after the backup gate passes. Preserve v0.5.

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
