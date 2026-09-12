# SEAM adaptation after qualification, supplement V1

2026-09-11, frozen before training. This supplements the running corrected
developmental comparison; it does not change any of that job's arms or criteria.
The matched seam_late arm inserts sparse method updates during development.
This separate comparison applies the method after the ordinary capabilities
have formed, which is closer to its published setting.

Use the preserved seed17 rule-only18,000-update parent from the previous
developmental comparison, with its original data and a recorded SHA256. This is
one existing lineage, not an additional independent seed. First score the parent
on the new128-core, two-layout validation suite. Then run500 SEAM objective
updates, batch8, LR0.00002, alpha1, beta0.01. Use fresh AdamW moments,
betas0.9/0.95, epsilon1e-8, no weight decay and gradient clipping at1. There
are no ordinary-development updates during these500 steps. This uses the
published default step count/LR/batch, while the optimizer settings, synthetic
data and explicitly documented byte-model/masking/exact-gradient adaptations
remain differences from the paper. Do not label this large-model replication.

Keep the implementation and reference commit documented in
SCC_RECOVERED_DEVELOPMENTAL_PILOT_V2.md. Apply exactly the same evaluation,
targeted/wholesale modification, cheap-reader, temperature and bounded-repair
procedures. Report all checkpoints even if intact performance fails. Unlearning
alone, low confidence and resistance to a particular optimizer cannot establish
the SCC mechanism or rank optimized methods.

Require nonzero train execution and exact full/resumed model, optimizer, RNG
and episode-chain agreement in the new runner, locally and on CUDA. Preserve
source, runner and protocol. Save full states at250 and500, plus the bounded
challenge endpoints, with a2GiB output target. The job has a20-H100-minute cap;
the user explicitly removed the aggregate compute budget limit, so the earlier
$10 cap is superseded. Record actual costs and preserve failed launches.
