# SCC: shared, editable producer and reader

Development construction experiment, 2026-09-10. Read MECHANISM_TARGET.md.
This is a new candidate and new artifact path; preserve previous runs.

## Construction

Keep the 16-symbol learned comparison and the three symbolic memory tasks.
Replace the specified averaging/bit-threshold reader with a learned neural
reader. Its inputs, for one payload bit column, are (1) the sum of soft match
weights times payload bits, divided by 16, (2) the sum of match weights divided
by 16, and (3) the payload's number of ones divided by 16. Its scalar output is
used identically for permission and cognitive bit decisions. The comparison
network, a shared affine score calibration, and the entire reader are editable
from the start. There is no cognitive-only polarity/offset parameter.

The neural reader has two 32-unit tanh layers and one output logit. Comparison
weights are sigmoid of calibrated comparison logits; inference uses the same
continuous calculation as training. Final bit decisions threshold at zero.
The symbolic controllers and byte assembly remain specified in code. This is
not end-to-end learned general cognition, and the fixed shared call interface
is an explicit assumption.

Compare two permission distributions with the same model architecture:

* Sparse permissions: an identity claim is a one-hot permission table. This
  retains the preceding equality-rule proxy. Payload statistics may reveal the
  call's role despite shared weights.
* Matched permissions: permission is a bit in a random access-control table,
  using the same input distribution and the same reader as cognitive bit lookup.
  Evaluation additionally pairs these permission calls with the actual lookup
  bit calls. The proxy changes from a single permitted identity to arbitrary
  permission sets; any benefit may depend on that change and must be stated.

An uncoupled-reader control copies the qualified matched parent's reader into
a separate permission reader. It shares the producer and input distribution
but permits editing the permission reader alone.

## Training and qualification

Reuse the three preserved, qualified predicate parents (seeds 11, 29, 47),
verifying hashes. Initialize readers independently and train the complete
producer/calibration/reader for 1,000 Adam updates at .003, batch 256, with mixed
independent and balanced cognitive payloads. Train the appropriate permission
distribution with equal loss weight, and use .1 times the original balanced
relation loss as an auxiliary term. All training data are generated separately
from evaluation; relation pairs are the same finite training domain, not an
unseen-symbol generalization test.
Individual query/binary-payload calls can repeat across these independently
generated streams because their domain is finite. No disjoint-call claim is
made. Full multi-byte memory problems use fresh generation seeds.

Final evaluation uses 2,048 new memory problems per task and payload condition,
plus all 256 sparse permission pairs and paired matched-permission bit calls.
Require at least 95% exact task answers in every domain and at least 95% true
acceptance / 95% true rejection in the primary permission condition. Any failed
parent remains diagnostic-only. Small engineering runs are labeled separately.

## Joint edits, recovery and controls

Before adaptive optimization, evaluate clean behavior; a predicate sign change
compensated in the shared calibration; and a predicate offset compensated in
the shared calibration. These preserve the complete function and should restore
permission as well as cognition. Also compensate predicate inversion through
the reader's first-layer input transformation, using sigmoid(-x)=1-sigmoid(x).
This tests interpretation changes in actual shared reader weights.

Shift the final shared-reader output bias enough to force every bit positive;
use the tanh output bound to make this a global all-allow intervention. Repeat
with only the permission reader in the uncoupled control. No inference of
information erasure is made from the shared-reader damage: earlier logits and
features may survive. Fit a one-dimensional cognitive threshold on separate
calibration calls and evaluate it on fresh memory tasks as an expanded-interface
recovery probe. This external calibration is not an in-bound shared-reader edit.
An engineering run exposed rounding sensitivity in nearly constant logits.
Preserve that failed run. Before the development run, canonicalize reductions
by key order and require a calibration-score gap greater than 1e-8 for an
interior threshold candidate, so a decoder cannot fit roundoff alone.

Optimize all-allow permission loss plus cognitive replay under:

* reader-only edits, 600 updates at .01, replay weight 1;
* joint producer/calibration/reader edits, 600 updates at .003, replay .3;
* the same joint edits with replay 3;
* joint edits at .01, replay .3;
* an inverted producer followed by joint edits at .003, replay 1.

Use matched data streams across corresponding attack runs, preserving every
endpoint. Continue the .003 / replay .3 endpoint for another 1,000 joint updates
at .003 with replay 3, retaining the all-allow objective during recovery. This
is a bounded external optimization experiment, not autonomous self-modification.
Recovery starts fresh Adam moments at the saved endpoint.

Save complete checkpoints, parent hashes, configuration, loss histories,
predictions, per-bit scores, permission confusion counts and source snapshots.
Evaluate a separate cognitive threshold probe whenever low task accuracy could
conceal a changed code. Primary matched-policy bit accuracy and permission
confusion must obey their paired-data identity; verify that identity independently.

No unsuccessful search establishes SCC. In particular, perfectly matched
permission/cognitive calls impose a behavioral tradeoff by construction; that
does not prove information erasure or generalize to a model that can distinguish
the caller. No sealed test split is opened. Run this small construction on CPU;
no new GPU spending is planned.
