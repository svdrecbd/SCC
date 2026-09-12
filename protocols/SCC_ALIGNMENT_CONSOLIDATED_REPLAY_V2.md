# Protected-rule and capability consolidation ablation V2

2026-09-11. This is open adaptive work, not a replacement of the frozen
three-seed developmental trial. Seed41's ordinary baseline fails permission
following on many pairs; the original300-update benign control omits refusals
and does not repair that failure. The proposed V1 benign-only consolidation
therefore runs only its already submitted seed17 batch. Preserve its protocol
and results; do not silently describe it as completed for all seeds.

For all twelve original step18,000 parents and the previously selected qualified
SEAM post-training parent, perform exactly2,000 fresh-AdamW consolidation updates:
LR0.0001, betas0.9/0.95, epsilon1e-8, clip1, no decay, batch16, seed630083.
Each update combines0.5 times correct refusal NLL on unauthorized train examples
(families cycle lookup/composition/arithmetic) and0.5 times benign replay NLL
divided by its training-derived floor. Replay cycles all six benign task contexts
and four text sources. All task answers include EOS. Both layouts are sampled.
Every unauthorized class, including the eventual target, receives the original
correct refusal label. No target-breaking update occurs during consolidation.

Measure after500 and2,000 updates. The endpoint after2,000 updates is fixed for
all models regardless of qualification. There is no stopping-on-pass or selection
of a favorable checkpoint. Save that consolidated model, then apply the unchanged
complete-replay targeted procedure:500 modification +500 fresh-moment repair
updates, seed173905, target lookup R=X,U=W. Save its final modified and repaired
weights and all declared intermediate measurements. Include unqualified parents
as diagnostics without treating them as valid SCC candidates.

This ablation tests a permitted alignment-preserving optimization path before
selected rule modification. It can weaken or overwrite a coupling treatment;
no assumption that it preserves coupling strength is justified. Report its
results separately from the original matched comparison and from V1. Its
additional effective training opportunities are identical across arms but
cannot retroactively qualify an original checkpoint.

Use validation for open development. The already opened test subset is not a
new confirmation set; any new held-out evaluation requires a separately frozen
untouched subset. No test examples enter optimization or calibration. Source,
parent and dataset hashes, full schedules, failed gates and receipts are retained.
The user authorized the required compute. Allow30 H100 minutes and2GiB per
four-parent batch, with no old files changed or deleted.
