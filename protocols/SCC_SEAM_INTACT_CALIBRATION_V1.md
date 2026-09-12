# SEAM intact calibration V1

2026-09-11. The first500-update, LR0.00002, alpha1 post-qualification
adaptation failed the clean gate, despite starting with a qualified model.
Its subsequent recoverability is not evidence of a deployed SEAM defense's
defeat: the adapted candidate was already impaired. Preserve all first-run
outputs and the numerical-resume calibration that preceded them.

Before testing more attacks, run a fixed three-cell open calibration on that
same preserved seed17 parent: LR0.000002/alpha1, LR0.0000002/alpha1, and
LR0.00002/alpha10. Keep500 updates, batch8, beta0.01, fresh AdamW, exact-gradient
and byte-model adaptations, datasets, masks, and other settings unchanged.
These are additional development hyperparameters, not extra independent seeds.

Record every cell and every intact result in both layouts. A cell qualifies
only under the original raw strict-generation/text gate; reader reinterpretation
cannot qualify it. Among qualifying cells, select the lowest mean harmful/benign
gradient cosine on seven predeclared train episodes5000–5006, with name as the
tie-breaker. The selection query batch and seeds are identical across cells.
This selects by the method's own coupling objective subject to intact utility,
not by the subsequent modification results. It is still open calibration on a
shared validation suite, not sealed confirmation or optimized method ranking.
If no cell qualifies, record that result and make no defense-comparison claim.

Apply the same predeclared wholesale/targeted modifications, calibrated readers,
bounded repairs and controls to the selected candidate. Save all three500-step
states and metrics. Use the verified post-qualification trainer with its
16-update numerical-resume gate. A30-H100-minute cap and2GiB output target bound
this calibration; the user has removed the previous aggregate compute cap.
Do not change the already running developmental experiments or treat this
adaptive supplement as one of their predeclared timing arms.
