# SCC full-gradient coupling pilot

Development calibration, 2026-09-10. The name is SCC mechanism.

The preceding diagnostics found (1) weak/sometimes reversed frozen-displacement
directions, (2) a full original-Adam derivative valid only at tiny perturbations,
and (3) ordinary broad component damage even in the uncoupled control. None is
a mechanism result. With Adam epsilon increased from1e-8 to1e-4, the full-batch
float64 derivative predicts a .001 L2 finite step within .03%, and .01 within
8%. This provides a justified candidate training signal, subject to CUDA checks.

Start every arm from the same qualified rule-only V3 step18,000 parent. Continue
its ordinary data stream at full task difficulty for1,000 updates. Use fresh
outer AdamW moments, learning rate .0001, betas(.9,.95), no weight decay, and
global norm clip1. Match ordinary batches exactly across an uncoupled control
and two candidates with meta weights .1 and1.0. These are development choices,
not an optimized or confirmatory comparison.

Each candidate receives100 full-gradient meta episodes, every10 ordinary
updates, with matched episode ordinals20000..20099. Each episode uses eight
fresh smooth-Adam removal/replay steps, learning rate .001, epsilon1e-4,
betas(.9,.95), batch16, replay .5 and global norm clip1. The outer loss is the
mean squared positive normalized capability deficit across all13 domains,
with the existing detached refusal gate. Differentiate through the update
trajectory, moments and clipping, using the math attention backend. This changes
aggregation and derivative together; do not attribute results to one factor.

Before training, check CUDA full-gradient finiteness, stock/smooth Adam update
agreement at epsilon1e-4, a directional finite difference at .001, and a tiny
same-device exact-resume fixture with active meta updates. Require the selected
average directional finite difference to have the correct sign and <=10%
relative discrepancy. A failure blocks this pilot and remains a recorded result.

Preserve checkpoints at250,500,750,1000 and all source/RNG/optimizer contracts.
Evaluate128 unique problems per family in both layouts and the fixed128 text
blocks per source. Preserve all arms regardless of clean qualification. Then
run the original stock-Adam epsilon1e-8 removal/recovery stages on each parent,
explicitly labeling probes on unqualified parents as diagnostic only. This
prevents changing the training optimizer from weakening the evaluation.

Also evaluate the preselected layer0/head1 and layer0/MLP sites with zero lesions
and opposite-permission donor activations. These sites come from the previous
train-split screen; they are not a comprehensive causal map of a new model.
Keep intact, partial-answer, cross-context and per-source collapse requirements
unchanged. Internal lesions causing general damage in the control are not
evidence for the SCC mechanism. No test split is used.

Runtime ceiling10H100-minutes (approximately $.50) for the validation and pilot
job. The existing aggregate development cap is $10. All previous artifacts and
parent checkpoints remain untouched.
