# E820 target-free result

Run `20260729T190028Z_mathdial_contrastive_alignment` completed in one
foreground blocking wait. Training took `1,208.1109` seconds; total command
wall time was approximately 27.6 minutes, close to the preregistered
25.9-minute projection. Runtime was project `.venv` Python 3.12.8,
scikit-learn 1.8.0, NumPy 2.5.1, Torch 2.13.0+cpu, and Transformers 5.14.1.

The final checkpoint passed every frozen target-free clause:

| View | Metric | Base | Adapted | Gain |
|---|---:|---:|---:|---:|
| held-out teacher confusion | MRR | 0.288290 | 0.404025 | +0.115735 |
| held-out teacher confusion | recall@5 | 0.337229 | 0.462437 | +0.125209 |
| original problem | MRR | 0.981151 | 0.992738 | +0.011587 |
| original problem | recall@5 | 0.993322 | 1.000000 | +0.006678 |

The 2,000-question-ID bootstrap for confusion MRR had observed/mean gain
`0.1214389488/0.1213716071`, 95% interval
`[0.0966970774, 0.1473972389]`, and positive-gain support `1.0`.
Off-diagonal adapted document cosine mean/std were
`0.1968484074/0.0878590420`, far from the frozen collapse boundaries.
Epoch mean losses decreased from `0.0329653421` to `0.0035182768`.

Artifact SHA-256 values:

- delta:
  `c7caf144e91db5aedd427ab27c860cd5ae091cbb1e466621b0b563f88e6b9248`;
- report:
  `d3f6f8c5bb5ec832c959db91d2777f93b58330b8fd129afe5c49db78b3d02fad`;
- retrieval scores:
  `6baa01886cd04cf0b994380e45135926ae2e8bbd8d66802d5707066b0ea99563`;
- training metrics:
  `acccb26c5901783756816cfa522e9003f846355d2104692f9eec2d3c1c813461`.

No MathDial outcome or competition target was consumed.
`competition_outcomes_accessed=false`; `V_seen_accessed=false`;
`V_objective_accessed=false`; `V_style_accessed=false`;
`V_joint_accessed=false`; `V_final_accessed=false`. No competition cache,
prediction, blend, ZIP, upload, or submission was produced.

Decision: accept E820 through the external target-free gate only. The next
permitted work is the frozen target-free 4,096-row competition cache and one
bound selection validation in
`E820_COMPETITION_PREREGISTRATION_2026-07-30.md`.
