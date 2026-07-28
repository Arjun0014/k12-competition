# Ready-to-paste next-session prompt

```text
Our project is C:\Competition\K12. Continue the Trace the Ace tutoring-outcomes competition from the exact
verified handoff state. You own technical research, implementation, training, validation, documentation, Git
milestones, and locally verified backup submission ZIPs. You must NEVER upload or submit anything to the
competition platform; I will make every submission manually.

First read these files completely, in this order:

1. C:\Competition\K12\LAST_SESSION_HANDOFF.md
2. C:\Competition\K12\PROJECT_LEARNING_LOG.md
3. C:\Competition\K12\TOP5_RECOVERY_PLAN_2026-07-22.md
4. C:\Competition\K12\V04_FAILURE_ANALYSIS_AND_RECOVERY_PLAN.md
5. C:\Competition\K12\Trace_the_Ace_full_overview.md
6. C:\Competition\K12\competition_rules.md
7. C:\Competition\K12\code_submission_format.md

After reading them, inspect the repository, Git branch/status/upstream, latest experiment reports, preserved
ZIPs, datasets, caches, logs, and active processes. Preserve every existing file and unrelated change. Confirm
that no K12 worker or automation remains active.

Current verified state:

- Research code/results are pushed through commit 0c17ebb; the handoff documentation was committed afterward.
  Verify that codex/v04-recovery matches origin/codex/v04-recovery and record the current HEAD before acting.
- Public champion v0.5 has log loss 0.6054 and last observed rank approximately #11.
- Two manual submissions remain this week.
- The protected ZIP is:
  C:\Competition\K12\submission_builds\final_ensemble_v05_bge_backup.zip
- Its SHA-256 must remain:
  65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186
- No locally verified candidate currently beats v0.5.
- E710, E720, E730, and E740 are rejected literally. E580-E700 and earlier rejected branches are also closed.
- No E750 architecture, code, score, checkpoint, or active worker exists.
- V_final remains sealed. V_joint is confirmation-only.

Use only C:\Competition\K12\.venv with Python 3.12.8 and scikit-learn 1.8.0. Do not build artifacts with the
system Python 3.13/scikit-learn 1.6 environment.

Immediate technical objective:

Audit the remaining competition-side error/feature gap using only already authorized V_seen, V_objective, and
V_style evidence. Search the code and learning log before claiming novelty. Preregister E750 only if it is
genuinely independent of every rejected branch and has a plausible path to at least 0.0016 robust log-loss
gain over raw v0.5. Freeze lineage, splits, features, estimator, seed, weights, proper-score gates, fold bound,
and bootstrap rule before any new outcome score. Run focused tests and a target-free or synthetic benchmark,
commit/push the frozen implementation, and then execute exactly one authorized validation. A failed gate
rejects that exact branch without rescue.

Long-run monitoring is a critical known failure:

- Three persisted ACTIVE Codex schedules failed to execute at 13:40, 14:15, and 15:35 IST.
- A stored ACTIVE file or rendered card is not proof of a working wake.
- Do not launch a long unattended run until a harmless short end-to-end scheduler test has actually fired in
  this session.
- Runs measured under one hour should remain in the active foreground turn with one blocking wait.
- Never poll repeatedly.
- For a run over one hour, estimate duration before launch and use exactly one completion checkpoint through a
  mechanism proven end-to-end in the current session. If it is still running, derive exactly one replacement
  checkpoint from real telemetry.
- If no reliable mechanism is available, stop before launching the long run and report the limitation honestly.

If E750 passes, build only its authorized cache/probe; evaluate V_seen, V_objective, and V_style; use V_joint
only as allowed; preserve V_final; and apply the frozen backup and conservative top-five gates literally.
Evaluate only preregistered blends—never a post-hoc sweep. Build and locally verify a new clearly named ZIP
only after the backup gate passes. Preserve v0.5 unchanged.

Append every material run to PROJECT_LEARNING_LOG.md with configuration, data lineage, environment, runtime,
log loss, AUROC, Brier, ECE, folds/environments, bootstrap evidence, failures/corrections, artifact hashes,
projected public loss, honest rank bracket, and accept/reject. Commit and push meaningful milestones to
Arjun0014/k12-competition as arjun0014.

Do not promise top five or first place without evidence. Do not submit. Begin by reporting the verified handoff
state, the scheduler-safety decision, and a precise E750 execution plan, then proceed autonomously.
```
