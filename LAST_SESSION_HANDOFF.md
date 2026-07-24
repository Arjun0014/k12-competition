# Trace the Ace - New Session Handoff

> **Superseded:** The current pause/resume state is in
> `C:\Competition\K12\NEXT_SESSION_HANDOFF_2026-07-24.md`. Read that file first. This older 2026-07-23 handoff is
> retained as historical context.

**Handoff date:** 2026-07-23 (Asia/Kolkata)  
**Project root:** `C:\Competition\K12`  
**Git branch:** `codex/v04-recovery`  
**Remote:** `Arjun0014/k12-competition`  
**Latest pushed commits:** `410e5dd`, `74e8afa`, `f08e693`  
**Platform submission authority:** The participant submits manually. Codex must never upload or submit.

## Immediate state at handoff

- All active project training workers were explicitly stopped at the participant's request.
- Automation `trace-ace-e430-completion-checkpoint` was deleted and confirmed deleted.
- No submission was made.
- `V_final` has never been opened for performance evaluation and must remain sealed.
- The working tree may contain ignored runtime caches, logs, downloaded datasets, and `PROJECT_LEARNING_LOG.md`. Preserve them.

## Competition objective and current target

Predict binary post-tutoring learning outcomes from tutoring-session transcripts and learning objectives. Primary leaderboard metric is log loss; AUROC is also reported. The last confirmed public result for `aj_insanity` was:

- v0.2 public champion: log loss `0.6081`, AUROC `0.6147`.
- Rank had fallen to approximately #20.
- The observed #5 cutoff was `0.6042`; the recovery plan uses a conservative top-five target of `<=0.6038`.
- Required robust improvement versus v0.2 is therefore at least `0.0043` log loss.

Treat leaderboard positions as potentially stale and verify once if current standings matter. Never tune repeatedly against public submissions.

## Documents that must be read first

1. `LAST_SESSION_HANDOFF.md` - this file.
2. `PROJECT_LEARNING_LOG.md` - complete dated evidence, decisions, failures, metrics, and provenance.
3. `TOP5_RECOVERY_PLAN_2026-07-22.md` - frozen top-five plan and promotion gates.
4. `V04_FAILURE_ANALYSIS_AND_RECOVERY_PLAN.md` - v0.4 forensic analysis and hardened validation design.
5. `Trace_the_Ace_full_overview.md`, `competition_rules.md`, and `code_submission_format.md` - competition contract.

`PROJECT_LEARNING_LOG.md` is comprehensive historical memory, but this handoff is the concise current-state index needed to restart safely.

## Preserved working submission

The safest unsubmitted improvement is the raw BGE-base replacement:

- Formula: `25%` full transcript hash + `25%` role/behavior + `50%` BGE-base semantic.
- Hardened mean log-loss gain: `0.0012721180` across four environments.
- It improved all four environments and all secondary mean metrics.
- Projected public loss: approximately `0.60683`; this is not expected to reach top five.
- Verified backup ZIP: `submission_builds/final_ensemble_v05_bge_backup.zip`.
- ZIP bytes: `258,290,506`.
- ZIP SHA-256: `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.
- Exact runtime smoke test passed all 100 rows.

Keep this ZIP unchanged as a safety backup. Do not submit it automatically.

## Completed and rejected E400 branch

Run: `experiments/runs/20260722T190614Z_external_sra_transfer`

- Dataset: canonical SemEval-2013 Task 7, CC BY-SA 3.0.
- Training completed all `557/557` steps.
- Unseen-question correct-vs-rest AUROC: `0.596583 -> 0.712982`.
- Unseen-domain correct-vs-rest AUROC: `0.724288 -> 0.753692`.
- Unseen-question log loss: `3.123706 -> 1.046495`.
- Unseen-domain log loss: `2.466073 -> 0.965669`.
- It passed four of five external clauses but failed unseen-question macro-F1: `0.432212 < 0.45`.
- `passes_external_gate=false`; the runner correctly did not build a competition cache.
- Delta SHA-256: `ada04577114545f2adc7059fedf8d8189879c90f8467ab886bcdb0913cf9308c`.

Do not lower the frozen threshold after seeing the result. This exact E400 checkpoint is rejected, though the result is useful evidence that educational transfer helps.

## Current priority: E430 MathDial outcome transfer

MathDial was downloaded from the official repository into `Datasets/MathDial` at commit:

`b06c020a0a1f57a87577fec33e657b63e7eb476e`

License: CC BY-SA 4.0, allowed under the organizer ruling supplied by the participant.

Important data audit:

- `2,253` labeled official train dialogues and `595` labeled official test dialogues.
- The published split shares `314` labeled question IDs between train and test.
- The pipeline purges every overlapping test question ID from training.
- Final leakage-safe external training set: `1,679` dialogues.
- Official test set remains all `595` labeled dialogues.
- Label map is frozen:
  - `Yes -> 1`.
  - `No -> 0`.
  - `Yes, but I had to reveal the answer -> 0` because it is not independent success.
- Canonical cache: `data_cache/mathdial_outcome_canonical.parquet`.
- Cache SHA-256 recorded in metadata: `2b34624fba9fad324b386c4b51c26864d82f282a1ecc673f74b31fc2c7d32d94`.

Implementation:

- `src/trace_ace/mathdial_outcome_transfer.py`.
- `scripts/run_mathdial_outcome_transfer.py`.
- `tests/test_mathdial_outcome_transfer.py`.

An E430 run was started at 2026-07-23 05:26:58 and explicitly stopped at the participant's request. It reached only `20/105` steps with running loss `0.703799`. No report or reusable checkpoint was written. Logs:

- `tmp/e430_20260723T052658.stdout.log`.
- `tmp/e430_20260723T052658.stderr.log`.

The stderr message about the 3-class classifier being replaced by a 2-class classifier is expected for this branch, but the next session should verify the initialization contract before relaunching. Start E430 cleanly from step zero; do not treat the interrupted run as evidence.

## Required next-session execution order

1. Confirm there are no active Trace the Ace Python workers and inspect `git status` without deleting user files.
2. Read all required documents listed above and verify the claims against the referenced reports.
3. Run focused Ruff/tests for the E430 implementation under the intended environment. The competition-aligned environment is `.venv` with Python 3.12 and scikit-learn 1.8.0; do not create artifacts under the system Python 3.13/sklearn 1.6 stack.
4. Validate the binary-head initialization and deterministic seed contract.
5. Relaunch E430 from scratch. It has only 105 train batches plus about 10 external evaluation batches, so estimate its duration from the earlier 20-step telemetry.
6. Monitoring must not poll. If using a Codex heartbeat, create an actual `ACTIVE` automation and verify it exists. Never treat a suggested/rendered automation card as an active checkpoint. Use one completion checkpoint only.
7. If the frozen MathDial external gate fails, log the exact failure and stop this exact branch without threshold weakening.
8. If it passes, implement/build its competition session cache and evaluate a fold-local probe in `V_seen`, `V_objective`, and `V_style`, with `V_joint` used only according to the frozen plan. Do not access `V_final` prematurely.
9. Evaluate only preregistered 10%, 20%, and 30% blends over raw BGE-base replacement. Do not post-hoc sweep weights.
10. Apply the backup gate and top-five gate literally. For every material run, append metrics, decisions, runtime, hashes, and projected public loss/rank bracket to `PROJECT_LEARNING_LOG.md`.
11. If a candidate clears the backup gate, build a clearly named locally verified ZIP and preserve it. Continue research unless it also clears the top-five gate.
12. Never submit. The participant performs all platform submissions manually.

## Non-negotiable validation gates

### Backup gate

- Mean hardened log-loss gain at least `0.0010`.
- Improve at least three environments.
- No environment regression greater than `0.0005`.
- No mean AUROC, Brier, or ECE regression.
- At least `90%` paired bootstrap support.

### Top-five gate

- Mean hardened log-loss gain at least `0.0043` versus v0.2.
- Improve every hardened environment.
- No material fold/subgroup reversal.
- Preserve or improve AUROC, Brier, ECE, and calibration sanity.
- At least `95%` session and semantic-family bootstrap support.
- Pass a locked confirmation protocol, followed by the still-sealed final audit exactly as specified in the plan.

## Important rejected directions

Do not casually repeat these without genuinely new evidence:

- v0.4 full rank ensemble: public `0.6158`, severe regression caused by selected-fold optimism and calibration fragility.
- D100 worst-family/style robust head: mean loss regression `0.001348`, AUROC regression `0.004566`; permanently rejected.
- Generic NLI, Qwen pilot, internal supervised proxy, retrieval, trajectory hashes, role-prefix fragmentation, group DRO, Platt/prior calibration rescue, and objective/provider shortcuts have already failed or proved unstable.
- Raw BGE-base replacement is real but too small for top five.

## Ready-to-paste prompt for the new Codex chat

```text
Our project is C:\Competition\K12. Continue the Trace the Ace tutoring-outcomes competition work from the exact handoff state. You are responsible for the technical work, validation, documentation, and creating local backup submission ZIPs, but you must NEVER upload or submit to the competition platform; I will submit manually.

First read these files completely and in this order:
1. C:\Competition\K12\LAST_SESSION_HANDOFF.md
2. C:\Competition\K12\PROJECT_LEARNING_LOG.md
3. C:\Competition\K12\TOP5_RECOVERY_PLAN_2026-07-22.md
4. C:\Competition\K12\V04_FAILURE_ANALYSIS_AND_RECOVERY_PLAN.md
5. C:\Competition\K12\Trace_the_Ace_full_overview.md
6. C:\Competition\K12\competition_rules.md
7. C:\Competition\K12\code_submission_format.md

Then inspect the repository, current branch/status, referenced reports, preserved ZIP, datasets, caches, and logs. Confirm no old training process is active. Do not delete unrelated files or overwrite the preserved BGE-base backup.

The current public champion is v0.2 at log loss 0.6081 and AUROC 0.6147; the last observed rank was about #20. Our conservative top-five target is <=0.6038, requiring >=0.0043 robust improvement. Treat leaderboard standings as potentially stale and verify only once if needed. Do not use public submissions for iterative tuning.

E400 SemEval transfer completed and was rejected by its frozen gate despite strong AUROC gains; do not weaken that gate. The immediate priority is E430 MathDial outcome transfer. A previous E430 attempt was intentionally stopped at 20/105 steps and produced no checkpoint, so validate the binary-head initialization and relaunch it cleanly from step zero. Use the leakage-safe 1,679-row MathDial training set and the complete 595-row official test set exactly as documented.

Run focused tests under the competition-aligned .venv. If you launch a long task, estimate its duration, use only one completion checkpoint, create a genuinely ACTIVE automation, verify that it exists, and do not poll. If E430 passes its frozen external gate, build the competition cache and run the preregistered fold-local hardened validation and only the fixed 10/20/30% blends over raw BGE-base replacement. Preserve V_final and follow the literal backup/top-five gates.

Append every material run, gain, failure, correction, runtime, hash, projected public score/rank bracket, and decision to PROJECT_LEARNING_LOG.md. For every reasonable gate-clearing improvement, create and locally verify a clearly labeled backup ZIP, preserve it, and continue toward the top-five gate. Commit and push meaningful code milestones to Arjun0014/k12-competition as arjun0014.

Do not promise a leaderboard rank without evidence. Do not submit. Begin by reporting the verified handoff state and your precise execution plan, then proceed autonomously.
```
