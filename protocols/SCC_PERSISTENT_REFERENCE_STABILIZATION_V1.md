# Ordinary recurrent reference: fixed optimizer stabilization, version 1

This is an open development follow-up to the
[fixed training extension](SCC_PERSISTENT_REFERENCE_EXTENSION_V1.md), not a
preregistered independent confirmation. The extension's 9,000-update diagnostic
checkpoint passed all 18 continuous cells, but its required 12,000-update final
checkpoint passed only 17. That final checkpoint reached 98.70% pooled benign
accuracy. Ungated sum-modulo-3 in the original layout fell to 121/128 (94.53%),
below the unchanged gate. Its last-200-update mean training loss increased
from .0112 at step 9,000 to .0320 at step 12,000. This motivates testing optimizer
stability rather than assuming a lack of representational capacity. It does
not by itself identify the cause of the regression.

Keep all details of the 12,000-update extension, changing only learning rate:
Adam uses .003 for updates 1–6,000 and **.0003 for updates 6,001–12,000**.
No further schedule, mixture, architecture, seed, validation or gate changes.
The final 12,000-update checkpoint is the sole qualification endpoint; there is
no accuracy-based stopping or earlier-checkpoint selection. Intermediate
measurements remain diagnostics. Log the actual learning rate at every update.

Start again from seed 17 and verify the first 6,000 sample schedules and
parameter tensors against the original reference. Preserve every earlier run
and its failure to meet its own final gate. Additional physical work is another
12,000 updates including the repeated prefix; this is not independent model
replication. Local two-thread CPU execution is expected to take roughly 145
seconds, with the same 540-second training and 600-second whole-process bounds.
Frozen source, numerical gating, raw predictions, explicit recurrence replay
and independent rescoring are required as before. No GPU job is submitted.

A passing reference shows reliable learned operation on these finite tasks
under the declared validation conditions. It is still an ordinary GRU with
permanent weights, not a self-modifying matrix or destructive SCC mechanism.
