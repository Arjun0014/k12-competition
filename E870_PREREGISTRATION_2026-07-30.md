# E870 preregistration: MaE misconception atlas external gate

Status at freeze: public source acquisition, schema audit, and unit tests only.
No MaE embedding/retrieval score, competition text, competition outcome,
`V_joint`, or `V_final` has been read.

## Independent mechanism

Prior branches modeled generic correctness, student state, tutor moves,
response quality, session mastery, objective difficulty, and dialogue
alignment. None represented a systematic mathematical misconception.

E870 asks whether a small, educator-authored atlas can recognize a held-out
diagnostic example of the same misconception using only the question and
learner's incorrect answer. The query never includes the correct answer or the
source's educator-authored explanation. If this external gate passes, a later
target-free stage may ask whether student evidence in competition transcripts
has stable, objective-consistent similarity to the atlas. No competition text
is encoded in this stage.

## Frozen source

- Official public repository:
  `nancyotero-projects/math-misconceptions`.
- Immutable revision:
  `12bee142d49dfb3c874149035cfbad28547a818b`.
- MIT license SHA-256:
  `cca1a8c2bc40c9e58cddec5c88da0330384dafd6600059a063f2ea2cba1d35e4`.
- Data SHA-256:
  `8223b3a6222c02f4efe8519c4e6abaa2e8e35d96a380bda3882b4f1cdd1fe4e1`.
- README SHA-256:
  `c91fc759503a5d4c9117e8fe8595026f8528b16b6a8bee15e82f2ffe5dcd8087`.
- Expected lineage: 220 examples, exactly four examples for each of 55
  misconception IDs, one description and topic per ID.
- Packaged MIT BGE-base safetensors SHA-256:
  `c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7`.

The raw public repository is preserved under ignored `Datasets/MaE`. No hosted
API, generated label, external identifier, or competition text is used.

## Frozen representation and split

- Query: fixed task line, question, and incorrect learner answer. Never include
  the source's educator-authored explanation, correct answer, image, source,
  topic label, misconception description, or ID.
- Prototype: fixed task line, misconception description, and the other
  diagnostic examples' questions/incorrect answers/explanations.
- For the true label, remove every example whose normalized question exactly
  equals the held-out question; this blocks repeated-template leakage. Require
  at least two remaining examples for every held-out query.
- All false-label prototypes use all four examples.
- Encode queries and prototypes independently with the unchanged packaged
  BGE-base normalized CLS representation, CPU float32, six threads, maximum
  length 256, batch 16. Rank by cosine similarity.
- Report ranks across all 55 labels and with candidates restricted to the
  released topic. No model training, prompt/field/model/layer/pooling/length,
  description, prototype, or metric sweep exists.

## Frozen external gate

Every clause must pass:

- all 220 rows and 55 misconceptions present;
- at least two nonduplicate-question prototype examples per held-out query;
- all-label top-1 at least `0.25`, top-5 at least `0.65`, and MRR at least
  `0.40`;
- within-topic top-1 at least `0.50`, top-3 at least `0.75`, and MRR at least
  `0.65`;
- every released topic has all-label top-5 at least `0.40` and within-topic
  top-3 at least `0.50`;
- in a 5,000-replicate misconception-ID bootstrap, the lower 95% bound for
  top-5 is at least `0.55` and support for top-5 at least `0.55` is at least
  `0.95`.

Any failure rejects exact E870 before competition text or outcomes. Do not add
the correct answer, retain duplicate questions, restrict to favorable topics,
change the prototype/query, fine-tune, swap the encoder, or relax a gate.

## Conditional continuation

A complete external pass authorizes only a separately frozen target-free
competition atlas audit. That audit must bind exact E870 artifacts and test
coverage, nonconstant similarity/margin/entropy features, objective-versus-
student atlas agreement, all-style-cell coverage, transfer distribution, and
runtime before any outcome model. Passing that second gate may authorize one
separately committed competition selection validation with the literal
`0.0016` robust gain, all-environment, proper-score, fold, and dual-bootstrap
rules.

`V_joint` remains confirmation-only and `V_final` sealed. v0.5 and the
protected ZIP remain unchanged. No upload or submission is authorised.
