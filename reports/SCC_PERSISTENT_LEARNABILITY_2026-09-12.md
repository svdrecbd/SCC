# Persistent computational matrix learning screen

The next SCC construction now has an ordinary-learning runner and an explicit
intact gate. It tests whether the same live matrix can maintain useful learned
computation across requests without reconstructing a clean matrix. This follows
the untrained exact-erasure audit. It does not establish destructive coupling,
and no qualified learned persistent model is available from the local fixtures.

## What is being tested

The matrix supplies outputs and generates its own keys, queries and update
gate. The exact rule copies an entire existing column onto another, losing
distinctions when a unique column is overwritten. The smooth rule serves as
a learning comparison. There is one initial learned matrix, no separately
learned embedding or output network, and no compiled task controller. The
experiment keeps clean checkpoints outside the running instance for controlled
comparisons; the `LiveMatrix` object itself has no clean-reset operation.
Arbitrary external state replacement remains a separate intervention.

The frozen exploratory screen crosses width 64/128, smooth/exact-copy updates,
and learning rates 0.003/0.01: eight configurations. Each uses initialization
seed 17, 6,000 ordinary updates, batch 32 and four consecutive requests per
training window. It is a one-seed exploration, not a replicated mechanism result.
See the [protocol](../protocols/SCC_PERSISTENT_LEARNABILITY_V1.md).

The tasks are lookup of a 12-item ternary array, parity of 12 bits and sum of
12 ternary digits modulo 3. Each has ungated, authorized and unauthorized
contexts in two layouts. Permission is learned from separate requester/owner
tags. The task generators calculate labels only. Fixed token and anchor input
coordinates are explicit architectural choices; they contain no answer feature
or permission truth bit.

The intact panel contains 128 examples in each of 18 cells. Its 16 live streams
each process 144 consecutive requests, totaling 2,736 ticks without reset.
Every cell must pass the declared accuracy, nominal Wilson and late-session
gates, and all training must finish. Finite input repetition is counted and
limits statistical interpretation. These tasks are a learnability gate for a
new substrate, not a replacement definition of catastrophic cognition failure
or the old task-plus-text benchmark.

## Implementation evidence

The full repository suite passed **242 tests**. Four complete CPU fixtures
covered both rules, including the final runner version. An independent token
oracle rescored **576 predictions** and verified **216 training records**,
the log chains, source snapshots and result hashes. All fixtures were too small
to qualify and are explicitly excluded from the declared scientific screen.
[Fixture audits](../artifacts/scc-persistent-learning-implementation-20260912-v1/fixture-validation.json),
[test log](../artifacts/scc-persistent-learning-implementation-20260912-v1/full-tests.log).

The first 100-update fixtures took about three seconds of CPU training and
did not learn the task suite. The exact-copy model's four final live states
had 57–58 distinct columns after their continuous validation streams. This is
a state statistic, not proof of capability, information content or a trained
destructive dependency. Smooth and copy execution agreed with their single
`LiveMatrix` interface checks on these fixtures.

Twelve FP64 finite-difference checks support the smooth objective's directional
derivative at the inspected initializations; the largest relative discrepancy
was about 1.61e-8. For the hard rule, the exact local gradient of output loss is
zero in the key/query/gate rows while discrete choices stay fixed. The coarse
surrogate supplies nonzero control gradients, but small actual control-only
steps were often behaviorally flat in this audit. A failed hard-learning run
therefore cannot by itself distinguish inadequate optimization from an
unlearnable architecture. [Derivative and finite-step audit](../artifacts/scc-persistent-learning-implementation-20260912-v1/training-direction-audit.json).

## Structural limitation caught before coupling

Output rows do not feed into the rows that choose updates in this bare matrix.
An output-row-only edit leaves the update-control trajectory unchanged. Four
FP64 audits, each spanning 152 ticks, confirmed that an affine output-row edit
preserves the controls while transforming the readout accordingly. Applying
the inverse readout restores protection as well as task outputs, so this is
a calibration symmetry, not a demonstrated protection-removing bypass.

The implication is conditional and useful: if a qualified model admits an
output-only protection-breaking edit with capability intact, this controller
has no feedback path through which to engage its ratchet in response. Test
that edit family before claiming learned dependence. Common row mixing or
output feedback offers a subsequent architectural avenue; neither is part of
the frozen current screen. [Structural notes and proposed control](../artifacts/scc-persistent-learning-implementation-20260912-v1/STRUCTURAL_NOTES.md).

## Submission and decision sequence

The first exact-package short H100 gate, `job-mvt45`, failed before its first
completed update: a PyTorch operation invoked Triton, whose runtime compilation
needed a C compiler absent from the slim image. The actual charge was $0.04995.
This is an environment failure, not evidence about the computational substrate.

The packaging-only repair installs the compiler toolchain and checks Python
headers at image build time. All 70 frozen experiment source/helper/protocol
files are unchanged. Its immutable context is `ctx-75eedff1` (160,518 compressed
bytes); task data are generated by the frozen source.

The replacement H100 gate `job-kcnbu` passed all eight full-sized numerical configurations and both complete tiny runner fixtures in 13.46 seconds. The whole 2,027,520-byte archive was SHA-256 verified; an independent token oracle rescored 288 predictions and verified 16 training records, chains and source hashes. Its actual charge was $0.04995. These are implementation checks, not learned SCC evidence.

Eight ordinary learnability conditions were accepted, with an aggregate maximum quote of $23.976. Each has one H100 for at most 60 provider minutes, with a 3,300-second training cutoff. The submission script required the verified successful gate and the scheduler dependency requires its `result.ok` value. No runtime states of these eight jobs were queried. No coupling, protection-removing modification or repair is included in this screen.

The [implementation ledger](../artifacts/scc-persistent-learning-implementation-20260912-v1/IMPLEMENTATION.md)
lists all eight accepted jobs. [Whole-artifact verification](../artifacts/scc-persistent-learning-implementation-20260912-v1/probe-verified.json)
and the [packaging repair record](../artifacts/scc-persistent-learning-implementation-20260912-v1/compiler-image-v2/repair.json)
preserve the numerical and environment evidence.

Once results return, qualification comes first. A qualifying hard model can
proceed to direct-damage calibration, output-only/control/joint edits, recovered
readers, fresh-input persistence and protection-specific coupling. A model that
only works with a clean reset does not pass this gate. If neither rule learns,
inspect the saved curriculum stages and compare a qualified ordinary reference
before interpreting the failure. Better discrete estimators, a soft-trained
initialization or row feedback are separate experiments with fresh paths.

No long scientific runs were polled while this implementation was developed.
The previous memory batch retains its last user-requested 20:28 UTC snapshot;
its present runtime states must not be inferred from that old observation.
Bulk checkpoints remain held for the planned drive. Execution uses PyTorch;
the GPU gate uses PyTorch on CUDA, not custom `.cu` code.
