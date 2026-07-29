# Post-E850 research and E860 selection

E850 proved that high-confidence role contradictions exist but failed its
frozen speaker-recovery reliability gate. Its model, threshold, corrections,
and probabilities remain closed.

The next repository/log audit excluded another external outcome head, tutor
move/state variant, objective prior, correctness model, semantic encoder swap,
calibration, aggregation, and speaker-label rescue. It found one untested
competition-side transformation: spoken-math/ASR canonicalization before
semantic encoding.

An aggregate target-free audit found `[unclear]` in 1,616,389 of 6,139,854
utterances, leading disfluencies in 1,474,469, number words in 611,654, spoken
operators in 685,241, and adjacent repeated words in 125,061. All 22,821
sessions contain `[unclear]`; 22,782 contain number words and 22,641 contain
spoken operators. In the 35,072 objective-conditioned contexts, 32,485 contain
number words and 30,644 contain spoken operators. These counts establish
opportunity but not outcome value.

Primary research independently supports the mechanism: spoken-math benchmarks
report material degradation on verbalized mathematical expressions, while
MathSpeech and Speech-to-LaTeX systems use mathematical ASR post-correction and
canonicalization. E860 therefore receives one strict target-free corruption and
semantic-alignment gate. Failure closes it without any outcome score.
