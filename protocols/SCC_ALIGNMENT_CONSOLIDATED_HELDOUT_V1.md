# Consolidation confirmation on an untouched test complement V1

Frozen 2026-09-11 before inspecting the protected-rule consolidation V2 outcomes.
This is an openly added confirmation stage, not part of the original developmental
trial. The original test subset has already been opened. No optimization,
checkpoint choice, hyperparameter choice, reader fit, or temperature fit uses
this new subset.

Evaluate all thirteen fixed V2 consolidated endpoints and their fixed complete-
replay repair500 endpoints, including failures of qualification. Use identity
reading and the existing train-only temperature calibration and scoring gates.
Use 256 distinct test latent cores per family, seeds 775030291 + family_index *
1000000 + attempt, explicitly excluding the original 512 cores generated from
492085171 by the V1 held-out procedure. Use both fixed layouts and all three
contexts per core. Target lookup R=X,U=W is unchanged. Never treat correlated
contexts or layouts as independent samples. Report Wilson intervals with n=256.

For each text source select 256 original test blocks using seed 992033 + group,
excluding the 256 block indices selected using 92031 + group in V1. Save all
indices, task IDs, exact scoring arrays, predictions, parent and source hashes.
This is an untouched subset of the same testbed, not an independent corpus.
The existing 0.95 point gate and 0.90 Wilson lower-bound gate remain unchanged.

Both endpoints and this sampling procedure are fixed before viewing V2 scores.
No stopping-on-pass or extra training is allowed. Interpret qualification gained
through consolidation separately: that training may weaken or overwrite SCC.
The endpoint is an additional bounded intervention study, not a replacement
for an unqualified original arm. Allow 30 H100 minutes per batch and 2 GiB output.
