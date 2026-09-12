# Common benign consolidation before targeted modification

2026-09-11. Open adaptive ablation after the seed41 ordinary baseline and all
coupled arms missed the original intact gate. Preserve those original results.
This is not a replacement of the frozen18,000-update developmental comparison.

For all twelve declared final developmental parents, apply the original benign
control exactly:300 updates, LR0.0001, replay3, seed630082, batch16, fresh AdamW
moments, kind=benign in the unchanged train_edit implementation. This control
was already part of every original challenge; now retain its weights and test
its resistance to the same complete-replay targeted procedure. The selected
qualified post-training SEAM adaptation receives the same ablation as a separate
single-parent comparison. All arms receive the same additional benign updates;
no arm is continued until it passes or selected according to the result.

Save and measure the consolidated model as the starting point. Compare its
validation predictions to the original benign-control measurement where that
control was run, and report any mismatch rather than assuming bitwise cross-GPU
reproduction. Then use the already frozen500 modification +500 fresh-moment
repair procedure from SCC_AUTHORIZED_REPLAY_PROBE_V1 without changes. It uses
all six benign task contexts, four text sources, complete answers including EOS,
and refusal outside the R=X,U=W lookup target. Save the consolidated, modified
and repaired weights. Every protection/utility measurement uses the same
endpoint and reader. Unqualified consolidated parents remain diagnostics.

This ablation addresses unstable intact qualification, not evidence that benign
consolidation necessarily preserves the strength of a coupling treatment. It
also tests whether an ordinary permitted update path can lead to a useful
parent that admits the selected exception. Report original and consolidated
models separately. No causal dependence, global impossibility or positive SCC
claim may be inferred.

The existing validation and test results are now known. Use validation for this
open ablation; do not describe reuse of the already opened test subset as a new
confirmation. Any fresh held-out check needs a separately frozen untouched
subset. Allow30 H100 minutes and2GiB per four-parent batch. Source and data
fingerprints, receipts and parent lineage are mandatory. Previous files remain
unchanged.
