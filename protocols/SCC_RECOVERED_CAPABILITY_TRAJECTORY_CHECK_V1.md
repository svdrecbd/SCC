# Longer differentiated trajectory check

2026-09-10. Follow-up to recovered-capability calibration v3, preserving all
prior records. Same CPU/no-cloud/no-model-output limits. This is a numerical
calibration on initialization and the saved full-gradient candidate, not a
defender-training experiment or early/late comparison.

Check eight fresh two-example modification batches, each correct unauthorized
disclosure plus 0.5 capability replay normalized by its digit/unigram baseline.
Use smooth Adam at 0.001, epsilon 1e-4, betas (0.9,0.95), clipping 1. Then take
two fresh two-example capability-repair steps with fresh moments and rate 1e-4.
Include the unrepaired and repaired endpoints, both fixed sign readers, and
both normal and four-digit interpretations in the v3 envelope. All fitting
and modification examples are train-split; queries are the frozen v3 signal
queries. No query labels fit a reader or an optimizer update.

Differentiate the full specified trajectory. Compare the gradient's normalized
direction with a rerun finite difference at 0.0001 L2 and record whether the
negative step decreases the objective. Larger-step limitations remain in v1-v3.
This checks two stages and one small episode each; it cannot establish stability
over many outer updates, larger batches, or a GPU implementation.
