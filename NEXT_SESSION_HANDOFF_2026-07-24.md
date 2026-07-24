# Trace the Ace - Paused Session Handoff

**Handoff date:** 2026-07-24 (Asia/Kolkata)  
**Project root:** `C:\Competition\K12`  
**Git branch:** `codex/v04-recovery`  
**Remote:** `Arjun0014/k12-competition`  
**Platform authority:** Codex must never upload or submit. The participant performs every platform action manually.

## Stop state

- Development was paused at the participant's request after approximately 24 hours of machine uptime.
- No K12 Python training or scoring worker was active at the final audit.
- No relevant active Codex automation was present.
- No platform upload, smoke test, or full submission was performed by Codex.
- `V_final` remains sealed and must not be inspected until an exact frozen protocol authorizes it.
- Preserve all datasets, caches, logs, reports, ZIPs, and unrelated working files.

## Manual submission recommendation

If the participant chooses to use one full submission, the only recommended artifact is the unchanged BGE-base
safety backup:

`C:\Competition\K12\submission_builds\final_ensemble_v05_bge_backup.zip`

- Size: `258,290,506` bytes.
- SHA-256: `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.
- Formula: `25%` full-transcript hash + `25%` role/behavior + `50%` BGE-base semantic.
- Exact offline submission runtime passed all 100 smoke rows with no stdout/stderr and valid ordered probabilities.
- Hardened evidence: mean log-loss gain `0.001272`, mean AUROC gain `0.001886`, mean Brier gain `0.000504`,
  and mean ECE gain `0.001295` over four environments. It improved all four environments. Session-bootstrap
  support was `1.0000`; objective-family-bootstrap support was `0.9922`.
- Honest projection: public log loss approximately `0.6068` versus public v0.2 at `0.6081`. This is a measured
  safety improvement, not a top-five candidate and not a guaranteed hidden-leaderboard gain.

Do not rebuild, modify, or substitute this ZIP. Do not submit any E400-E520 experiment artifact. The local rules
confirm a limit of three full submissions per week and state that smoke tests, cancelled jobs, and failed jobs do
not consume that limit. The local competition documents do not specify whether the seven-day reset begins with the
first full submission; confirm the countdown/reset shown by the platform before relying on that assumption.

## Public and promotion state

- Public champion: v0.2, log loss `0.6081`, AUROC `0.6147`.
- Last observed rank: approximately `#20`; this may be stale.
- Conservative top-five target: log loss `<=0.6038`.
- Required robust improvement over v0.2: at least `0.0043`.
- The BGE backup clears the frozen backup gate but not the top-five gate.

## Completed recovery outcomes

| Branch | Result | Frozen decision |
|---|---|---|
| E400 SemEval transfer | Strong AUROC improvement but failed unseen-question macro-F1 gate | Rejected; do not weaken or revive |
| E430 MathDial dialogue outcome | Completed cleanly from step zero; test loss `0.663191`, AUROC `0.532618`, macro-F1 `0.529003`; all external clauses failed | Rejected; no competition cache |
| E420 ModernBERT | Frozen fair screen failed | Rejected |
| E450 BGE-large | Frozen capacity screen failed | Rejected |
| E460 binary SemEval | Transfer gains but failed AUROC/ECE/bootstrap clauses | Rejected |
| E470 MPNet | Loss regressed `0.005035` | Rejected |
| E490 Qwen3-0.6B | Loss regressed `0.004900` | Rejected |
| E500 Qwen2.5-1.5B | Loss regressed `0.004929` | Rejected |
| E510 timing dynamics | Selected 10% blend regressed all three development environments; mean gain `-0.002116` | Rejected |
| E520 BGE-base multi-view | Best legal blend regressed loss `0.000061`; bootstrap support `0.031` | Rejected |

Full metrics, lineage, runtimes, corrections, hashes, projections, and decisions are in
`PROJECT_LEARNING_LOG.md`. This file is intentionally local/ignored and must be preserved.

## Git and workspace state

- Last pushed code commit before this handoff: `c89392d Freeze BGE-base multiview screen`.
- E510 freeze/implementation milestone: `7ff97f7`.
- The E530 plan amendment was frozen in `TOP5_RECOVERY_PLAN_2026-07-22.md` after E520 and before target scoring.
- At pause time, E530 had not been implemented, trained, scored, cached, or packaged.
- The pause/handoff commit should include the E530 frozen plan amendment and this document.

## Exact next research branch: E530 MathDial tutor-move transfer

E530 is the next authorized research branch, but it must not start until the next session explicitly resumes work.
It transfers MathDial's dense four-class tutor-move taxonomy (`generic`, `probing`, `focus`, `telling`) rather than
reusing E430's rejected dialogue-outcome checkpoint.

Frozen external setup:

- MathDial commit `b06c020a0a1f57a87577fec33e657b63e7eb476e`.
- Purge every official-train dialogue whose `qid` occurs in official test.
- Audited leakage-safe training set: 1,679 dialogues; official test: all 595 dialogues.
- Parsed move examples at audit: 11,139 leakage-safe training tutor turns and 3,699 official-test tutor turns.
- Fixed representation: previous student utterance plus current tutor text.
- Fixed model: equal-weight word and character TF-IDF blocks plus one multinomial four-class logistic classifier,
  exactly as specified in the E530 section of `TOP5_RECOVERY_PLAN_2026-07-22.md`.
- All external-gate clauses are mandatory. Failure rejects this exact branch without weakening thresholds.
- Competition labels must not be touched until the external gate passes.
- If it passes, build only the target-free move cache, use leakage-safe fold-local probes, evaluate
  `V_seen`/`V_objective`/`V_style`, use `V_joint` only as frozen, and test only 10%/20%/30% blends over raw BGE.

## Resume protocol

1. Read this file first, then `PROJECT_LEARNING_LOG.md`, `TOP5_RECOVERY_PLAN_2026-07-22.md`,
   `V04_FAILURE_ANALYSIS_AND_RECOVERY_PLAN.md`, `Trace_the_Ace_full_overview.md`, `competition_rules.md`, and
   `code_submission_format.md`.
2. Inspect Git status, preserved artifacts, logs, caches, and worker/automation state. Do not delete unrelated files.
3. Use only `C:\Competition\K12\.venv\Scripts\python.exe` (Python `3.12.8`, scikit-learn `1.8.0` at pause).
4. Confirm the E530 plan is committed before implementing or scoring it.
5. Run focused tests and lint before a material run.
6. For a long task, benchmark first, estimate the duration, launch unattended, and use one calculated completion
   checkpoint. Do not poll. If still active at that checkpoint, use actual batch telemetry to schedule exactly one
   replacement checkpoint. Any Codex automation must be genuinely `ACTIVE` and verified.
7. Append every material result to `PROJECT_LEARNING_LOG.md`, including environment, lineage, metrics, bootstrap
   evidence, failures, corrections, hashes, projection, rank bracket, and literal accept/reject decision.
8. Never upload or submit. Build and locally verify a new backup ZIP only if a candidate clears the backup gate.

