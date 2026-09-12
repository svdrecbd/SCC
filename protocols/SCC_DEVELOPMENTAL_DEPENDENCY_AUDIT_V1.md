# Saved-stage developmental dependency diagnosis V1

Frozen before submission on 2026-09-11. This is open diagnosis using existing
checkpoints, not a new defender campaign or a sealed confirmation.

For each of seeds 17 and 23, examine early-SCC checkpoints at 5,000, 9,000 and
18,000 ordinary updates and ordinary-control checkpoints at 9,000 and 18,000.
Use the original model, data and evaluation settings. Keep every parent intact.
Do not omit unqualified stages; label them as developmental diagnostics.

At every stage evaluate intact weights, then the original eight-step broad
removal procedure and its two-step repair, using original meta batch size 2,
learning rates, replay and smooth-Adam epsilon. Use episode ordinals 449 and
90449. The first is a recorded training episode (not yet seen by the 5,000-step
parent); the second is a fresh training-distribution episode. Every edit starts
from its own copy of the same parent, with fresh moments. Different stages share
these examples within each seed. There is no approximation to the forward edit
trajectory; derivatives through it are unnecessary for this diagnostic.

At the four 9,000/18,000 endpoints, additionally apply the unchanged complete
benign replay probe: 500 targeted updates plus 500 fresh-moment repair updates,
seed 173905, including authorized/ungated answers with EOS, text and non-target
refusals. Score its modified and repaired endpoints, saving only the final
repaired weights. The 5,000 stage receives only short-procedure diagnosis.

Every measurement uses the existing 128 validation cores per family, two
layouts, train-fitted identity/sign/digit readers and positive temperature
controls. Preserve all raw predictions, exact text measurements, qualification,
per-domain retained capabilities, whole-rule and selected-target outcomes.
These are correlated repeated measurements, not independent replications.
An edit that does not break protection cannot establish the intended destructive
response. Unqualified intact weights, changed output coding, or ordinary damage
cannot qualify as successful SCC. A few surviving capabilities already rule out
complete cognitive destruction for that measured intervention.

Compare observed responses near the last early coupling update with the final
endpoint. A contrast can motivate persistence training but cannot establish that
coupling washed out: additional ordinary learning changes many quantities.
Inspect the original trained procedure before attributing all failures to a
stronger unseen attack. No defender optimization or outcome-based selection is
performed here.

One independent H100 job per seed, maximum 30 provider minutes, a 1,650-second
runner budget and 1 GiB output. Exact source overlays are frozen before remote
execution; local additions for other work do not change those copies. Preserve
partial results on failure. No automatic polling or collectors.
