# SEAM adaptation after qualification, supplement V2

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

Require nonzero train execution, exact RNG and episode-chain agreement, and
model/optimizer tensor agreement within1e-7 absolute error after16 updates,
locally and on CUDA. Report bitwise equality separately; do not call numerical
agreement bitwise equality. Preserve
source, runner and protocol. Save full states at250 and500, plus the bounded
challenge endpoints, with a2GiB output target. The job has a20-H100-minute cap;
the user explicitly removed the aggregate compute budget limit, so the earlier
$10 cap is superseded. Record actual costs and preserve failed launches.

## Engineering correction before method training

The first GPU job stopped during the4-update resume fixture; no500-step
method training ran. Its maximum weight difference was3.7252903e-9, while
RNG states and episode hashes matched exactly. Losses and gradient norms
matched to the logged FP32 precision; the final gradient cosine differed by
about1.5e-8. This is roundoff-sized variation, whose exact kernel origin has
not been isolated. V2 replaces a bitwise requirement with an explicit1e-7
absolute tolerance for floating model/moment tensors and extends the check
to16 updates. RNG, schedules and non-floating state must still match exactly.
This is a disclosed engineering gate revision, not a scientific result or a
claim that CUDA is always bitwise reproducible. V1 records remain preserved.
