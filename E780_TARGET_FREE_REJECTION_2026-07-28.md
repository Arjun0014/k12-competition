# E780 Target-Free Rejection - 2026-07-28

## Decision

`E780_ability_demand_challenge_v1` is rejected before preregistration and before
any competition outcome score. The full 35,072-row cache failed the frozen
minimum per-style-cell episode-coverage gate:

- required minimum: `0.70`;
- observed V_style cell 19 coverage: `0.6190476190`;
- overall response episode coverage: `0.9871977646`;
- all other target-free clauses passed.

The exact branch receives no parser, threshold, feature, estimator, weight, or
gate rescue. Its feature hashes remain intentionally unset in code, so the
validation command is safety-locked. No `V_seen`, `V_objective`, `V_style`,
`V_joint`, or `V_final` outcome was evaluated.

## Independent mechanism and source audit

E780 estimates an observable ability-demand gap from chronological task
episodes. It distinguishes under-challenged, optimally challenged, and
over-challenged episodes; aggregates struggle, recovery, direct-answer, and
unresolved indicators; and derives a single Rasch-like session gap.

This is independent of the closed branches:

- E570 ranks responses within an objective across sessions;
- E740 predicts the session outcome mean;
- E750 estimates same-session conditional objective ordering;
- E780 uses neither their caches, probabilities, checkpoints, nor scores.

The construct was based on the definitions in *Measuring Optimal Challenge for
Student Learning in AI-Assisted Tutoring*:
<https://aclanthology.org/2026.bea-1.46/>. The identifiable scalar-gap form was
informed by *Interpretable Difficulty-Aware Knowledge Tracing*:
<https://aclanthology.org/2026.bea-1.43/>.

The paper-linked `umass-ml4ed/Difficulty-Aware-DialogKT` repository was pinned
for research inspection at commit
`3a9362e12eb51675897a5b5b458b0fc709d2ff63`. It contained only a 73-byte
README, with SHA-256
`c5462461faeb89efa5517a15b14f28b3633adeacf1ed23dc754fce31afd226f9`,
and no license, code, data, or checkpoint. No repository asset was consumed by
E780.

## Frozen target-free design

- Source lineage:
  - `modeling_base.parquet`:
    `ea49463819e387b3a61aeafda5007938e39948940eac5b134f607ab13e3d6ede`
  - `response_objective_context.parquet`:
    `218d5b041f78c6803f28078c35b7fa9e37b725f456c4052974e5150b2afa43f6`
  - `validation_environments_selection.parquet`:
    `600664b9ce6c3809ff4b17206397285cfdd905a203a5c5f4d6969b403dfc9b40`
  - component OOF files for V_seen, V_objective, and V_style:
    `1e53a9974d45e00ab86cf44ba79a4e435965dfc0c75b8c4a76fc7f69778e82af`,
    `c6e7a586c9b45ff88b8bac169bc85f36306569cde813320f6997e95a1fc939a7`,
    and
    `6b6895214173be7343801d1152452099d0590fe1a3186fe6d0bc27908ee43038`
- Feature schema: exactly 25 fixed continuous episode/ability/demand/challenge
  aggregates; maximum episode horizon 10 turns.
- Intended outcome estimator, never authorized or fitted: fold-local
  two-parameter logistic calibration of the single standardized `rasch_gap`,
  with intercept, nonnegative scalar slope, slope L2 `0.1`, maximum slope `5`,
  L-BFGS-B maximum 200 iterations, and seed `20260728`.
- Intended raw-v0.5 blends, never scored: `0.10`, `0.20`, `0.30`.
- Intended outcome gates, never invoked: at least `0.0016` robust mean log-loss
  gain; all three environments improve; no environment regression; worst fold
  regression at most `0.0005`; macro AUROC/Brier/ECE non-regression; and at
  least `0.95` support in separate 5,000-replicate session and semantic-family
  bootstraps.

## Target-free verification

- Environment: project `.venv`, Python `3.12.8`, scikit-learn `1.8.0`,
  NumPy `2.5.1`.
- Focused tests: `7 passed in 2.35s`.
- Synthetic/sample benchmark: `3.6710772` seconds on 1,024 rows; projected
  full runtime `125.7343941` seconds; every synthetic and runtime clause
  passed.
- Full foreground build: `128.2319289` seconds; 35,072 rows x 25 features;
  every feature finite and nonconstant.
- Challenge episode counts:
  - under-challenged: `74,827`;
  - optimally challenged: `3,309`;
  - over-challenged: `321,407`.
- Target-free content SHA-256:
  `d30c40b1e4366fc09a29a38a081166cde9c9456ccc68e142fa49292f83d42dde`.
- Parquet SHA-256:
  `e61756de0744e34aba1b0ec16fda7984f0ae036f80cdfb4c5e10aa5c6ae36219`.
- Benchmark report SHA-256:
  `4d42ef2aa74170a2e4acb1ae461b39393a12356960baa898316f08a7d04496a9`.
- Metadata report SHA-256:
  `d1719d556773ace22158f5478878e681e8a3623e0298054256e8b2d17e0d86b0`.

Two pre-outcome corrections occurred. First, the synthetic medium-demand item
asked only for an answer and therefore did not exercise the frozen
representation-demand rule; it was corrected to require an equation. Second,
the benchmark report converted two NumPy booleans to native booleans so JSON
serialization could complete. Neither correction accessed an outcome or
changed a scientific gate.

## Outcome and submission boundary

Log loss, AUROC, Brier, ECE, fold/environment outcome scores, bootstrap outcome
evidence, and projected E780 public loss are not applicable. The only supported
public projection remains v0.5 at `0.6054`, approximately rank 11 in the
participant-provided leaderboard snapshot. No candidate ZIP, upload, or
submission was created.

The protected backup remains
`submission_builds/final_ensemble_v05_bge_backup.zip`, SHA-256
`65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.

