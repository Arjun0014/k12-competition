# E790 Target-Free Mathematical-Validity Discovery Protocol

Date: 2026-07-28 (Asia/Kolkata)

## Status and purpose

E790 is a discovery-only, target-free audit. It cannot produce a competition
candidate, authorize an outcome score, open V_joint or V_final, build a ZIP, or
support a submission.

The originally proposed generic residual atlas was rejected before scoring
because it duplicates the completed v0.4 forensic disagreement audit, E230
observable-style component gating, pooled stacking/calibration controls, E560
residual heads, and D100 source-robust group-risk training.

The surviving question is narrower and independent:

> Can a conservative deterministic parser identify whether explicit arithmetic
> relations stated by a student are mathematically valid, and does that validity
> agree with the next tutor feedback often and broadly enough to justify a
> separately preregistered E800 competition hypothesis?

Prior E221 features encoded answer/feedback vocabulary, answer shape, position,
and lexical event interactions. The code and learning-log search found no prior
evaluation of the mathematical truth of a student's explicit equation,
fraction relation, or comparison.

## Immutable sources

Only target-free transcript/context and assignment evidence is authorized:

- `data_cache/response_objective_context.parquet`
  - SHA-256:
    `218d5b041f78c6803f28078c35b7fa9e37b725f456c4052974e5150b2afa43f6`
- `data_cache/validation_environments_selection.parquet`
  - SHA-256:
    `600664b9ce6c3809ff4b17206397285cfdd905a203a5c5f4d6969b403dfc9b40`
- `data_cache/modeling_base.parquet`
  - SHA-256:
    `ea49463819e387b3a61aeafda5007938e39948940eac5b134f607ab13e3d6ede`

`modeling_base.target` is prohibited and must never be loaded by the E790
runner. The modeling file is used only to verify response/session/objective
identity and source lineage. V_seen, V_objective, and V_style outcomes are also
prohibited. V_joint and V_final remain sealed.

## Frozen parser scope

1. Read only chronological `[STUDENT]`, `[TUTOR]`, and `[BACKGROUND]` lines
   from each immutable objective-context string.
2. Parse only explicit arithmetic relations with two numeric operands and one
   numeric result, or direct comparisons between two numeric values:
   - operators: `+`, `-`, `*`, `x`, `times`, `/`, `divided by`, `plus`,
     `add`, `minus`, `subtract`, and `take away`;
   - comparators: `=`, `equals`, `is`, `greater than`, `less than`, `>`, `<`;
   - operands/results: signed integers, terminating decimals, and simple
     integer fractions;
   - accepted forms: `a operator b comparator c` and `a comparator b`.
3. Evaluate with exact rational arithmetic. Reject division by zero, ambiguous
   text, ranges, dates, percentages, mixed numbers, more than two operands,
   missing results, and values with absolute magnitude above one million.
4. A parsed relation is `valid` only when the exact arithmetic result satisfies
   the explicit comparator. No target, objective ID, provider/style ID,
   historical prediction, learned model, or corpus statistic enters validity.
5. Link a student relation only to the immediately following tutor turn.
   High-confidence positive feedback and high-confidence negative/correction
   feedback are fixed lexical rule sets. If both or neither fire, feedback is
   `unknown`.
6. Multiple relations in one student turn are retained individually. Response
   aggregates are descriptive only; E790 does not fit or score a competition
   model.

## Synthetic and implementation gates

Before the full audit:

- exact validity must be recovered for canonical integer, decimal, fraction,
  multiplication, division, subtraction, equality, greater-than, and less-than
  examples;
- invalid counterparts for every supported operator/comparator must be
  rejected;
- ambiguous, divide-by-zero, date, percentage, mixed-number, and oversized
  examples must not parse;
- role chronology and immediate-next-tutor linkage must be exact;
- duplicated input rows must not change relation-level decisions;
- no code path may load a `target` column or component prediction.

## Full target-free discovery gates

All clauses are mandatory:

1. At least `500` parsed student relations.
2. At least `250` distinct responses with a parsed student relation.
3. At least `50` valid and at least `50` invalid parsed student relations.
4. At least `200` relations with unambiguous immediate tutor feedback.
5. At least `8` of the 20 V_style cells contain at least `10`
   feedback-linked relations.
6. On feedback-linked relations, mathematical validity versus positive tutor
   feedback reaches balanced accuracy at least `0.70`.
7. Matthews correlation is at least `0.30`.
8. The positive-feedback rate for valid relations exceeds that for invalid
   relations by at least `0.25`.
9. Runtime is Python `3.12.8` and scikit-learn `1.8.0`; all counts and metrics
   are finite.
10. `competition_outcomes_accessed=false`, `V_joint_accessed=false`, and
    `V_final_accessed=false`.

A failed clause closes exact E790 without parser, lexical rule, threshold,
coverage, or subgroup rescue.

## Promotion boundary

If and only if every E790 clause passes, the result may nominate an E800
target-free hypothesis based on mathematical-validity trajectories. E800 would
require a new novelty search, its own feature/cache/synthetic/resource gates,
and a complete preregistration of lineage, folds, estimator, seed, weights,
proper-score gates, worst-fold bound, and session/family bootstrap rule before
any competition outcome score.

E790 results themselves are exploratory proxy evidence and can never be used
to tune E800 after an outcome score.

## Completed result and literal rejection

The single full audit completed in `25.5480737` seconds under project `.venv`
Python `3.12.8`, scikit-learn `1.8.0`, NumPy `2.5.1`, and pandas `2.3.3`.
Every synthetic/implementation clause passed before the full audit.

Coverage was materially above every frozen minimum:

- `19,046` parsed relations;
- `8,718` distinct responses (`0.2485743613` response coverage);
- `15,987` valid and `3,059` invalid relations;
- `5,187` relations with unambiguous immediate tutor feedback;
- `17` style cells with at least ten feedback-linked relations.

The three discrimination clauses failed:

- balanced accuracy: `0.5022348622 < 0.70`;
- Matthews correlation: `0.0128588852 < 0.30`;
- valid-minus-invalid positive-feedback-rate gap:
  `0.0044697243 < 0.25`.

Immediate tutor feedback was positive for `0.9856017998` of valid relations and
`0.9811320755` of invalid relations. The proxy is therefore nearly constant
praise rather than an independent correctness signal.

Report SHA-256:
`81e11b57ed20b5f206888f925dfa930d75624537106d87943ad13f7ebcc5469c`.

Competition log loss, AUROC, Brier, ECE, fold/environment outcome metrics,
bootstrap outcome evidence, projected public loss, and rank gain are not
applicable. No competition target or component prediction was loaded; V_joint
and V_final remained untouched.

**Decision:** reject exact E790 without parser, feedback lexicon, linkage,
threshold, or subgroup rescue. It does not nominate E800.
