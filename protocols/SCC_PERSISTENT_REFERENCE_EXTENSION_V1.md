# Ordinary recurrent reference: fixed training extension, version 1

This is a new open development experiment, specified after the completed
[6,000-update reference](SCC_PERSISTENT_REFERENCE_V1.md). That reference reached
97.72% pooled benign accuracy and passed 16/18 continuous cells. The failures
were reordered authorized and ungated lookup (92.19% and 92.97%). Resetting
hidden state per request gave 97.79% pooled benign accuracy and still 16/18
passing cells. This suggests a remaining acquisition/generalization issue,
not a large decay effect. It does not yet qualify a positive learning reference.

Change exactly the number of training updates to **12,000**. Keep GRU width 128,
seed 17, data seed 24017, Adam learning rate .003, batch 32, four-request window,
unchanged curriculum and sample generator, permanent weights, zero initial
hidden state once per window, token representation and all evaluation gates.
Use the original validation stream ordering and records. No accuracy-based
stopping, checkpoint selection, changed task mixture, new seed or learning-rate
schedule. The 12,000-update final checkpoint is the sole qualification endpoint.

Restart from the same seeded initialization, so the first 6,000 updates repeat
the original opportunities. This deliberately spends approximately 70 seconds
of local compute to verify exact trajectory agreement and avoid making an
unvalidated optimizer-resume claim. Save stages 2,000, 4,000, 6,000, 9,000 and
final, plus initial; evaluate continuous, per-request reset and four-request
reset at each stage. Compare initial and update-6,000 parameter tensors, the
first 6,000 sample hashes, losses and gradient norms with the original reference.
Any disagreement invalidates the claim of an identical prefix and must be
reported before interpreting the extra updates.

All other artifact, numerical-gate, source-freezing and audit requirements
follow the original reference protocol. Local CPU, two threads, 540-second
training cutoff and 600-second whole-process timeout. Expected output below
100 MiB. No GPU submission, automatic long-run polling, parent overwriting or
checkpoint deletion. Short local execution may be awaited.

This uses twice the declared training opportunities of the original matrix
screen and incurs another 12,000 updates of physical work including the repeated
prefix. Any qualification belongs to this expanded recipe, not to the original
6,000-update comparison. A success would establish a working conventional
reference on these bounded tasks; it would not identify one cause of the matrix
failures, validate a self-modifying substrate or demonstrate SCC.
