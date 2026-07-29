# Post-E860 research and E870 selection

E860 showed that deterministic spoken-math normalization can repair synthetic
ASR corruption, but it reduced real target-free transcript/objective alignment.
That branch is closed without a surface-preserving or token rescue.

A fresh repository/log audit rejected an alternate StudyChat assignment target
because E840 explicitly closed grade-target and embedding rescues. It then
found a different missing construct: systematic mathematical misconception.
The existing project has generic correctness and state models, but no
misconception description, diagnostic-example atlas, or misconception-retrieval
feature.

The public MIT MaE source was acquired at revision
`12bee142d49dfb3c874149035cfbad28547a818b`. It contains 220 educator-designed
examples, exactly four for each of 55 middle-school misconceptions. The source
paper reports 52.96% raw all-topic and 73.82% raw topic-constrained GPT-4
accuracy before expert reinterpretation, establishing that the task is real
but not trivial.

E870 therefore receives an external retrieval gate using only the observable
question and incorrect learner answer, with the educator-authored explanation
excluded from the query and strict duplicate-question exclusion. It cannot see
competition text unless every frozen external clause passes.
