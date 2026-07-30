# Trace the Ace write-up evidence specification

**Purpose:** turn already authorized project evidence into a four-page,
judge-ready research package without reading `V_final`, tuning a model, or
using the public leaderboard as a validation fold.

## Official rubric alignment

| Criterion | Weight | Evidence to prepare |
|---|---:|---|
| Relevance | 35% | Transcript-derived content semantics, ordered feedback, and practical guidance for tutoring research |
| Generalizability | 35% | `V_seen`, `V_objective`, and `V_style` comparisons; typed/ASR limitations; external-transfer results |
| Communication | 15% | One pipeline diagram, one robustness panel, one compact result table, plain-language takeaways |
| Rigor | 15% | Session purge, frozen gates, proper scores, bootstraps, lineage, licenses, hashes, and negative results |

## Central claim

> Objective-conditioned transcript semantics transfer more reliably than
> generic pedagogical labels, while ordered tutor feedback adds complementary
> signal whose value is not stable across objective and transcript-style
> shifts.

This is an associational and modeling claim. The report must not claim that
changing a tutor behavior causes a learning gain.

## Page budget

### Page 1: question and key findings

- State the learning-outcome task and the long-ASR/short-typed generalization
  challenge.
- Show the deployed v0.5 pipeline.
- Report the two key findings in plain language.
- Explain why log loss and calibration matter for use as an intervention or
  review signal.

### Page 2: methods and validation

- Session-purged training/validation.
- Three development environments: seen, objective-family shift, and
  transcript-style shift.
- Fixed full-session sparse, role/behavior, and BGE-base semantic components.
- Proper scores: log loss, AUROC, Brier, ECE.
- Session and semantic-family bootstraps.

### Page 3: results and failure analysis

- v0.2 to BGE-base replacement robustness matrix.
- Ordered-feedback average benefit and hard-regime reversal.
- Compact table of important negative results:
  calibration/stacking, long context/encoder swaps, generic tutoring labels,
  simulated outcomes, moves/states, numeric elaboration, curriculum
  progression, speaker repair, ASR normalization, and misconception retrieval.
- Public confirmation of v0.5 at `0.6054`, labeled as one public observation
  rather than a selection sweep.

### Page 4: actionable implications and limitations

- Validate tutoring signals under objective and provider/style shift.
- Treat generic quality or correctness labels as insufficient proxies for
  actual subsequent learning.
- Prefer compact, reproducible models when gains are tied.
- Discuss inability to distinguish prior knowledge, task difficulty, and
  causal tutoring effects from transcripts alone.
- State the APTA follow-up design as future work, not completed evidence.

## Required artifacts

### Figure 1: deployed-model diagram

Inputs:

- transcript;
- role-separated transcript;
- learning objective.

Branches:

- full-session hashed word model;
- role/objective/behavior model;
- BGE-base objective-conditioned semantic model.

Output:

- fixed `0.25/0.25/0.50` probability average.

### Figure 2: robustness panel

For v0.2 and the raw BGE-base replacement, show:

- log loss by `V_seen`, `V_objective`, `V_style`, and locked joint
  confirmation if already authorized;
- AUROC, Brier, and ECE deltas;
- bootstrap support or intervals;
- the fixed gate line.

Do not read `V_final` to populate this figure.

### Table 1: evidence ladder

Minimum rows:

| Family | Best honest evidence | Generalization result | Decision |
|---|---|---|---|
| Full-session sparse | First stable transcript gain | Retained | Deploy |
| Role/objective/behavior | Complementary | Retained | Deploy |
| BGE-base semantic | `+0.001272` hardened loss gain vs v0.2 | Improved four environments | Deploy |
| Ordered feedback NB-SVM | `+0.001851` mean loss, `+0.009059` AUROC in decisive audit | Worst fold/subgroup reversed | Reject for deployment |
| Supervised/stacked rank ensemble | Strong grouped local ranking | Public failure/robustness failure | Reject |
| Generic external correctness/quality/state | Several strong external metrics | Competition transfer weak or unsafe | Reject |
| E770/E810/E820/E830 | Narrow AUROC or subgroup signal | Mean loss regressed | Reject |
| E840-E870 | Real external or target-free gates | Failed before competition outcomes | Reject |

Every number must point to an immutable report or learning-log entry.

### Table 2: reproducibility and legal lineage

For every resource included in the final submission or report:

- name and purpose;
- source URL;
- immutable revision;
- license;
- local file or directory;
- SHA-256;
- whether packaged;
- whether it influenced the final model.

The report should distinguish research-only rejected resources from deployed
resources.

## Deterministic evidence-index schema

The planned evidence index must contain one row per claim:

```text
claim_id
claim_text
status
experiment_family
run_id
source_artifact
source_sha256
environment
metric
value
comparison
decision
V_joint_accessed
V_final_accessed
notes
```

Allowed `status` values:

- `deployed`;
- `validated_positive_not_deployed`;
- `rejected_outcome`;
- `rejected_external`;
- `rejected_target_free`;
- `research_only`.

No value may be copied from memory when an immutable local report exists.

## Quality gates for the evidence package

1. Every quantitative claim resolves to a local report, prediction manifest, or
   learning-log entry.
2. Every artifact path exists or is explicitly marked historical.
3. Every reported SHA-256 matches the current file.
4. `V_final_accessed` remains false.
5. Causal language is absent unless a source design actually identifies a
   causal effect.
6. Rejected branches are shown, not silently omitted.
7. Public score is identified as public confirmation, not local validation.
8. The protected ZIP hash remains
   `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.

## Immediate build sequence

1. Inventory the selected run reports and their SHA-256 values.
2. Build `writeup_evidence_index.csv` and a machine-readable JSON manifest.
3. Render the robustness panel from the index.
4. Render the compact evidence ladder.
5. Draft the four-page text around the frozen figures rather than selecting
   figures after seeing a narrative preference.
6. Re-run the artifact/link/hash audit before finalizing the PDF.

This workstream consumes no competition outcome validation, no manual
submission, and no scheduler checkpoint.
