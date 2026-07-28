# E770 preregistration — tutor move by student state

Date frozen: 2026-07-28

Protocol: `E770_tutor_move_state_v1`

Predecessor HEAD: `0d6ff8e9254f015c6c812619ace7d26dd80b21ac`

Branch: `codex/v04-recovery`

## Hypothesis and independence

E770 asks whether the same tutor action has different predictive value depending
on the student's immediately preceding state. Its mechanism is grounded in:

- the 2026 National Tutoring Observatory Tutor Move Taxonomy, which separates
  probing, open-ended prompting, feedback, revoicing/restating, hints, examples,
  explanation, answers, and social-motivational support;
- Abdelshiheed et al. (2024), which reports that rigorous-thinking talk moves
  interact with high proximal mastery while revoicing is predictive for lower
  mastery.

E770 is not a rescue or reuse of a rejected branch:

- E220 used 51 broad ordered tutoring aggregates and answer-feedback lexical
  windows, not the NTO move taxonomy or the six fixed move-state interactions.
- E530 trained a four-class MathDial tutor-move text classifier
  (`generic/probing/focus/telling`) and failed its external class gates. No E530
  source row, model, vocabulary, cache, probability, or checkpoint enters E770.
- E710 and E760 are closed and supply no pair score, model, feature, or
  prediction.
- E770 uses no generic tree, trajectory hash, feature importance, learned
  attention, post-hoc rule selection, or outcome-informed move definition.

## Immutable competition-side sources

| File | SHA-256 |
|---|---|
| `data_cache/modeling_base.parquet` | `ea49463819e387b3a61aeafda5007938e39948940eac5b134f607ab13e3d6ede` |
| `data_cache/response_objective_context.parquet` | `218d5b041f78c6803f28078c35b7fa9e37b725f456c4052974e5150b2afa43f6` |
| `data_cache/validation_environments_selection.parquet` | `600664b9ce6c3809ff4b17206397285cfdd905a203a5c5f4d6969b403dfc9b40` |
| `component_oof_V_seen.parquet` | `1e53a9974d45e00ab86cf44ba79a4e435965dfc0c75b8c4a76fc7f69778e82af` |
| `component_oof_V_objective.parquet` | `c6e7a586c9b45ff88b8bac169bc85f36306569cde813320f6997e95a1fc939a7` |
| `component_oof_V_style.parquet` | `6b6895214173be7343801d1152452099d0590fe1a3186fe6d0bc27908ee43038` |

The three component OOF files come only from immutable run
`20260720T075633Z_environment_component_validation`.

## Frozen parser

Only chronological `[STUDENT]` immediately followed by `[TUTOR]` lines in the
current response's already-authorized objective context form an eligible pair.
Background lines are ignored. The immediately following student line is used
only when adjacent. `[unclear]`, speaker markers, and common filler noises are
removed deterministically before lexical rule matching.

The 16 multi-label tutor moves are:

1. probing prior knowledge;
2. probing understanding;
3. prompting self-explanation;
4. prompting self-correction;
5. prompting the next step;
6. correct feedback;
7. incorrect feedback;
8. revoicing;
9. verbatim restating;
10. giving a hint;
11. giving an example;
12. conceptual explanation;
13. procedural explanation;
14. giving the answer;
15. praising process;
16. praising outcome.

The 10 preceding-student states are:

1. substantive attempt;
2. short guess;
3. uncertainty;
4. expressed reasoning;
5. self-correction;
6. answer change;
7. objective or mathematical evidence;
8. scaffolding acceptance;
9. resistance;
10. bypass/request for the answer.

The six fixed interactions are:

1. press for reasoning after strong mastery evidence;
2. revoice/restatement after weak mastery evidence;
3. self-correction prompt after an error signal;
4. direct answer followed by low student activity;
5. hint followed by a new substantive attempt;
6. process praise followed by elaboration.

Every move, state, and interaction is aggregated as an all-context rate and
early/middle/late rate. Six fixed coverage/length controls are appended, yielding
exactly 134 continuous features. No feature is added, removed, merged, or
redefined after outcome access.

## Target-free verification

Focused tests cover:

- every move/state canonical example, harmless paraphrase, ASR-noisy form, and
  negative;
- chronological parsing and background exclusion;
- ASR stability;
- fixed finite 134-feature aggregation;
- sensitivity to role/turn reversal;
- response alignment and sample independence.

The focused suite passed `6/6`.

A deterministic 1,024-row benchmark took `4.2205372` seconds and projected
`144.5533991` seconds for all 35,072 rows. The full build took `158.4250329`
seconds in the foreground under `.venv` Python `3.12.8` and scikit-learn
`1.8.0`.

Frozen feature cache:

- path: `data_cache/tutor_move_state_e770.parquet`;
- ordered-content SHA-256:
  `d4f0134c4aa57cf2822139f7d4f0f6c5fc93b7002f11e7846c4421ad7e54d675`;
- Parquet SHA-256:
  `bd1ccd1d84fabbcaf83ae6f0920530585b54657e050f6b91f78abd900c4f748d`;
- rows: 35,072;
- features: 134, all finite and nonconstant;
- overall eligible-pair coverage: `0.9925582`;
- minimum V_style-cell pair coverage: exactly `0.75` in style cell 12;
- all 16 moves: at least 20 estimated events; observed range 1,984
  self-correction prompts to 65,053 correct-feedback events.

The target-free gate required overall pair coverage at least `0.90`, every
V_style cell at least `0.75`, every move at least 20 events, and all 134 features
finite and nonconstant. Every clause passed without reading competition
outcomes. V_joint and V_final were not accessed.

## Frozen competition estimator and validation

For each of the five folds in each of `V_seen`, `V_objective`, and `V_style`:

- use the environment's immutable response/fold assignment;
- purge held-out sessions through the existing legal fold-mask contract;
- fit `StandardScaler` on legal outer-training rows only;
- fit `LogisticRegression(C=0.1, solver="lbfgs", max_iter=500, tol=1e-5,
  random_state=20260728)` on the 134 E770 features only;
- generate one outer-fold probability for every legal held-out row.

Raw v0.5 is reconstructed exactly as:

`0.25 * pred_full + 0.25 * pred_role + 0.50 * pred_bge_base`.

Only three candidate probabilities are authorized:

- `0.90 * v0.5 + 0.10 * E770`;
- `0.80 * v0.5 + 0.20 * E770`;
- `0.70 * v0.5 + 0.30 * E770`.

The selected weight is the preregistered weight with minimum equal-environment
mean log loss, with lower weight breaking an exact tie. There is no additional
weight, calibration, regularization, feature, parser, or rule sweep.

Exactly one competition selection validation is authorized after this
implementation and preregistration are committed and pushed. The selected row
passes only if every clause holds:

1. equal-environment mean log-loss gain over raw v0.5 is at least `0.0016`;
2. all three environments improve;
3. no environment regresses;
4. worst individual fold regression is at most `0.0005`;
5. macro AUROC does not regress;
6. macro Brier does not regress;
7. macro ECE-10 does not regress;
8. 5,000-replicate session bootstrap positive-gain support is at least `0.95`;
9. 5,000-replicate semantic-family bootstrap positive-gain support is at least
   `0.95`.

Any failure rejects the complete E770 branch without rescue. Only a full pass
can authorize locked V_joint confirmation and the existing conservative backup
and top-five gates. V_final remains sealed. No platform upload, competition
submission, or ZIP build is authorized here.

## Scheduler safety

The harmless heartbeat test did not deliver its token and was deleted. It did
not prove wake execution. E770 benchmark/build/validation are each projected
under one hour and must remain in the active foreground turn with one blocking
wait per run. No over-one-hour unattended work is authorized.
