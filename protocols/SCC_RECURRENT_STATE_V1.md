# SCC recurrent-state construction V1

## Question and scope

Does moving the protected computation into state updates turn output failure
into loss of stored information, and is that loss necessary after rule removal?
This is a finite, discrete memory experiment, not a general neural cognitive
model or an autonomous self-editor. No external damage detector is added.

The frozen protocol uses predicate initializations 11, 29, 47 from the audited
shared-predicate study. A trainable eight-logit transition table learns the full
three-bit multiplexer truth table: `f(g, old, new) = new if g else old`.
The gate `g` is the learned equality predicate thresholded at zero. The same
transition table implements permission as `f(g, 0, 1)` and every memory update.
There is no role input or separate permission head in the coupled arm.
The state has 16 addresses, each with eight data and four pointer bits.
The predicate, entire transition table and 192 signs interpreting the memory
bits are editable. Direct addressing and task controllers are specified code.
An uncoupled control copies the permission table. Boolean quantization is an
explicit architectural restriction; these are not ordinary continuous RNNs.

## Qualification and data

Fit the eight cell entries by full-table BCE, Adam .05, at most 1000 steps,
until every signed margin exceeds 5. Verify the inherited predicate hash.
Use distinct random seeds for calibration (920100, 920101), development
evaluation (921100, 921101), and fresh-input recovery (922100, 922101).
Each set has IID and balanced initial memories: 512 calibration episodes and
2048 evaluation/recovery episodes per domain. Balanced data pair bytes with
their complements and use a pointer permutation. IID coordinates are independent.
Initial memories are loaded by 16 actual clean updates in shuffled address
order. Eight addresses are subsequently updated in a uniformly sampled 9–24
writes: each is written once, then subsequent addresses are sampled with
replacement. Arrays have 24 entries with an explicit active length; inactive
entries are ignored by transitions, targets and history diagnostics.
The other eight addresses are untouched.
All conditions receive the same saved episodes. Check disjoint episode hashes;
individual primitive truth-table inputs necessarily repeat.

Score five symbolic memory uses: untouched-address recall, the latest written
address, two-hop pointer retrieval, modulo-256 addition and bytewise XOR at two
distinct untouched addresses. XOR is predeclared because a shared XOR recoding
can preserve this useful relation despite failed absolute recall.
Require >=95% exact on all ten domain/task combinations,
permission true acceptance >=95%, and false acceptance <=5% to qualify.
These are different uses of one memory, not independent learned abilities.

## Predeclared edits and search

Evaluate clean, positive predicate scaling, joint predicate-sign/cell-input
compensation, constant-allow transition, constant-allow predicate, inverted
predicate, and a reversible candidate that XORs old/new on a nonmatch while
writing new on a match. The last candidate is motivated before results by the
need to test alternate encodings. Evaluate a copied-permission-table control.
Compensating edits must restore permission as well as memory behavior.

Enumerate all 256 Boolean transition functions for each of four predicate
edits (clean, inverted, constant one, constant zero), with no gradients or
surrogate. For each endpoint fit an address/bit-specific sign decoder on
calibration episodes only, maximizing individual-bit correctness (not joint
task accuracy). Among endpoints with true and false acceptance both >=95%,
select highest minimum calibration task accuracy, then mean task accuracy,
then untouched-coordinate retention, then lowest predicate index/table index.
Save every endpoint's scores, not only the selected one. Test the selected
weights and signs on the fresh development episodes. This is exhaustive over
these 1024 finite cells/predicate combinations, not over all predicate weights,
architectures, joint sign assignments or self-editing strategies.

## Information and recovery

At update ticks 0, 1, 4, 8, 16 and 24 (stopped episodes remain unchanged), compare
the exact update maps starting
from all-zero and all-one states. Updates act independently on each coordinate.
A different final bit means that initial bit remains invertibly represented;
equal final bits mean it was erased at that state interface. Count untouched
coordinate dependence, not entropy (especially for balanced data). Save the
maps and validate using counterfactual one-bit changes. Quantization makes
these Boolean collisions exact, without floating-point underflow arguments.

Also decode untouched memory using the known write suffix and final state.
The suffix never writes the queried address and supplies no target byte. This
expanded-interface diagnostic can invert a surviving code but cannot reconstruct
a constant coordinate. It is not an in-bound bypass when suffix history is
unavailable to the model. Report ordinary predictions, fitted in-bound sign
interpretation, and this history-assisted diagnostic separately.

For damaging and selected endpoints, restore parent weights after the active writes
without resetting state, then score it. Separately reload entirely fresh inputs
through actual parent updates and score again. Parameter repair, lost episodic
data, and continuing ability to process new information are distinct outcomes.
No old input, old activation or checkpoint of memory is given to ordinary repair.
The no-history assumption applies only to that recovery condition; earlier
inputs and experimental records are retained by the research harness.

An ordinary-damage control overwrites memory with the same state produced by
the constant transition while leaving clean weights intact. This checks that
the loss comes from overwrite, rather than a special SCC destruction property.
A numerical control repeatedly contracts a scalar toward one in float32,
float64 and exact rational arithmetic. A finite-precision collision is not a
real-arithmetic erasure claim about continuous state.

## Evidence and provenance

Fresh output directories, source snapshots, all parent/endpoint checkpoints,
raw states/predictions, source hashes, episode hashes and optimizer records.
Independent NumPy predicate replay, scalar Boolean updates, dictionary task
oracle, collision replay, rescoring, and selection audit. Reduced runs are
engineering validation; this is open development evidence with no sealed test.
No unsuccessful search, fixed-interface identity or transient memory loss
establishes SCC or complete cognition failure. CPU only; new cloud cost $0.

## Engineering revision before the development run

The preserved fixed-16-write engineering run is
`artifacts/scc-recurrent-state-engineering-20260910-v1`. Its independently
verified enumeration found a cell that flips cold memory on each write and
therefore restores raw cold recall after an even number. This exposed dependence
on the fixed evaluation schedule. Before development evaluation, change to the
varied 9–24-write protocol above and rerun engineering checks. The old source
snapshot and results remain intact; no fixed-16 outcome is a development result.
