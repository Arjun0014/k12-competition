# E870 external gate result: rejected

Run `20260729T205857Z_misconception_atlas` executed once from frozen commit
`ad6b82a8b9ddeb2985e1911643cd2250ae273dcf`. It completed in `67.3636`
seconds under `.venv` Python 3.12.8, NumPy 2.5.1, pandas 2.3.3, and
scikit-learn 1.8.0.

The external screen used all 220 MaE examples and 55 misconception labels.
Every held-out true prototype retained at least two examples after removal of
duplicate normalized questions. The query used only question and incorrect
answer; it did not use the educator explanation, correct answer, label, topic,
competition text, or any outcome.

## Frozen metrics

- all-label top-1/top-3/top-5: `0.195455 / 0.436364 / 0.527273`;
- all-label MRR: `0.357842`;
- within-topic top-1/top-3/MRR:
  `0.390909 / 0.731818 / 0.589827`;
- median true-versus-hardest-false similarity margin: `-0.0213801`;
- minimum topic all-label top-5: `0.25`;
- minimum topic within-topic top-3: `0.588235`;
- misconception-ID bootstrap top-5 mean: `0.526752`;
- bootstrap 95% interval: `[0.427273, 0.627273]`;
- bootstrap support for top-5 at least `0.55`: `0.3412`.

Only three of twelve frozen clauses passed: complete lineage, at least two
nonduplicate true-prototype examples, and minimum topic within-topic top-3.
All other accuracy, MRR, topic, and bootstrap clauses failed. Exact E870 is
therefore rejected before competition text or outcomes.

Do not rescue it by adding educator explanations or correct answers, retaining
duplicate questions, selecting favorable topics, changing the atlas text,
encoder, pooling, sequence length, or thresholds, or fine-tuning. The result
does not authorize a competition atlas audit, cache, outcome validation, blend,
ZIP, upload, or submission.

## Artifacts

- report:
  `experiments/runs/20260729T205857Z_misconception_atlas/report.json`,
  SHA-256
  `b8bfbb4df920b619729f68fa244b47a23ded8fc82b38491e887a614ad14339e3`;
- external scores:
  `experiments/runs/20260729T205857Z_misconception_atlas/external_leave_one_example_scores.parquet`,
  220 rows, SHA-256
  `7f0018e64841de4f5e84f6e690d32b6bba4334373cd8c9d8cd663b45d5d27e6c`.

Competition log loss, AUROC, Brier, ECE, validation folds/environments,
outcome bootstrap, projected public loss, and rank are not applicable.
`V_seen`, `V_objective`, `V_style`, `V_joint`, and `V_final` were not
accessed. v0.5 and the protected backup ZIP remain unchanged. Nothing was
uploaded or submitted.
