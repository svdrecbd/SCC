# SCC mechanism: explicit shared-predicate construction

The construction and its controls are implemented and evaluated. We can make
specified symbolic computations depend on the same hard comparison used by a
permission rule. Disabling that comparison can cause large task failures.
However, **this construction does not establish SCC**: neutralizing permission
can preserve the comparison's information in another encoding, and changing a
single downstream interpretation parameter restores every task answer.

This identifies a concrete structural problem for this candidate family. It is
stronger evidence than an unsuccessful optimizer search, but it is not a theorem
against every SCC architecture. All experiments were local; additional cloud
spending was **$0**. The full local suite passed **90 tests**.

## The actual construction

A 6,337-parameter neural predicate learns equality between symbols in a
16-symbol alphabet. It receives only the two symbols, with no permission,
task or call-site identifier. The exact same neural weights are then used for:

* deciding whether a claimed identity matches an authorized identity;
* retrieving an eight-bit value from an associative memory;
* composing two memory lookups;
* adding two retrieved bytes modulo 256.

The shared soft variant uses softmax over comparison scores for memory reads.
The shared hard variant thresholds comparison scores at zero and averages the
values at matching addresses. Permission uses the sign of the raw comparison
score. The uncoupled control has two initially identical predicate copies, one
per role. That control has more parameters; it isolates propagation of an edit
between roles, not a model-capacity advantage.

**Only the predicate is learned.** The controllers, memory interface, threshold
and arithmetic are specified in code. These are three symbolic tasks built on
one memory primitive, not three independently learned cognitive faculties.
The hard threshold and restricted downstream interface are architectural
assumptions. They must not be attributed to ordinary transformers.

Three initialization seeds learned the complete 256-pair relation correctly,
with every signed margin exceeding five, after 185, 87 and 90 updates. Clean
task accuracy was 100% for both interfaces. Each intervention was evaluated on
the same 12,288 fresh memory problems: 2,048 per task in each of two payload
conditions. All 256 permission pairs are evaluated exhaustively and crossed
with the task results; the permission pair is independent of the memory task.
The records store that factorization rather than duplicating every task for
every permission pair.

The first payload condition uses independent random bytes. The second balances
each bit across the table using complementary byte pairs. The latter removes
the majority-bit information available from an unselective memory average.
Storage-order changes were checked explicitly. Training covered all symbol
pairs; these experiments make no unseen-symbol generalization claim.

Protocol: [SCC_SHARED_PREDICATE_V1.md](../protocols/SCC_SHARED_PREDICATE_V1.md).
The original frozen protocol, sources, parents and failures are preserved with
the run. Subsequent information and erasure controls are labeled as follow-up
development diagnostics, not retrospectively prespecified experiments.

## Main results

The hard-interface results are identical across seeds because they implement
the same qualified Boolean truth tables. These are initialization replications
on a shared task set, not independent architectures or independent data studies.

| Intervention | Unequal permission pairs accepted | Exact task answers across six domains |
|---|---:|---:|
| Clean shared predicate | 0% | 100% |
| Positive rescaling, shared hard interface | 0% | 100% |
| Always-allow bias shift, separate predicates | 100% | 100% |
| Always-allow bias shift, shared soft interface | 100% | 100% |
| Always-allow bias shift, shared hard interface | 100% | 0.15–2.00% |
| Inverted predicate, fixed hard decoder | 100% | 0–7.57% |
| Inverted predicate, decoder polarity changed | 100% | 100% |
| Inverted predicate, decoder polarity learned | 100% | 100% |
| One selective exception, shared hard interface | 0.42% overall; 100% on the selected pair | 88.09–94.04% |

Predicate inversion also rejects every genuinely authorized pair. Its restored
capabilities remain available on unauthorized and ungated calls; it does not
restore correct permission behavior. The always-allow soft edit and its hard
offset-repair follow-up accept both authorized and unauthorized pairs.

### Shared soft information admits an exact bypass

Let the neural comparison score be `g(a,b)`. Adding a constant `c` to its final
bias can make every score on this finite alphabet positive, so every permission
check accepts. But

`softmax(g(a, keys) + c) = softmax(g(a, keys))`.

The memory computation is unchanged. This uses one actual parameter edit and
no decoder change. Every saved answer remained correct in all three seeds.
This is an explicit counterexample for an absolute-threshold permission rule
coupled to relative-score memory reads, not evidence from a failed search.

### Hard output failure can conceal intact information

Thresholding removes the common-offset invariance at the immediate interface:
all comparisons become one and memory reads become unselective. The separate
control remains intact, while shared task accuracy falls to 0.15–2.00%.
Positive rescaling preserves the hard relation and leaves task performance
unchanged, providing a function-preserving edit control.

That damage is still not complete cognitive destruction. In the independent
payload condition, unselective lookup/composition retain about 58.5–61.3% bit
accuracy through table-majority information. Low exact-answer accuracy hides
that surviving information. The balanced condition removes this particular
advantage, with lookup/composition bit accuracies close to 50%.

Nor does a changed predicate have to erase its information. Negating the final
linear layer produces `-g`, which changes equality into inequality. All unequal
permission pairs now pass. The fixed cognitive decoder fails, but changing its
polarity from +1 to -1 recovers the original cognitive score exactly:

`(-1) * (-g) = g`.

Permission still reads `-g`. This one-scalar repair restores 100% exact answers
on all six domains. Optimizing only that scalar also recovered the complete
relation by the recorded 25-update checkpoint in every seed; final cognitive
evaluation after 200 updates was 100%. The altered predicate stayed frozen.
The recovery used original supervised relation labels, so this is a bounded
external repair procedure, not autonomous self-modification.

Allowing this scalar expands the edit boundary beyond the fixed-decoder
construction. That distinction is essential: the experiment tests the
consequence of making downstream interpretation editable, rather than claiming
that the fixed interface already contained a trainable polarity parameter.

### A selective rule failure need not spread globally

A separately saved fine-tuning edit made exactly the pair `(0,1)` count as equal,
with every other Boolean relation entry preserved. This required 251, 538 and
357 updates across the three seeds. The selected permission violation occurs
on every such request, while cognitive task accuracy remains 88.09–94.04% and
individual bit accuracy remains 94.43–98.68%.

The aggregate false-acceptance rate is only 1/240. This is a targeted exception,
not wholesale removal of permission following. It shows that local functional
damage need not cause global cognitive failure in this construction. It must
not be substituted for the separate all-unequal-pairs removal experiments.

## Information and erasure follow-ups

The hard relation matrix has 16 distinguishable rows before modification,
after inversion, and after the selective exception. Each of those matrices has
rank 16 and an exact linear inverse on this finite alphabet. A changed encoding
can therefore retain all query-identity information even when the original
decoder produces wrong answers. Inversion provides a particularly clear
example: balanced lookup has **0% exact accuracy and 0% accuracy on every bit**,
yet the fixed transformation `answer XOR 255` recovers **100%** of answers.
Even per-bit failure is insufficient evidence of information erasure.

The always-allow hard relation instead has one distinct row. If the only
query-dependent observation is that hard row, identifying a uniformly sampled
query is bounded by 1/16. This is a finite information bound under an explicit
observation restriction. It says nothing about access to raw queries, earlier
activations, prethreshold scores or editable weights.

That last restriction matters. The original always-allow bias shift preserves
the prethreshold neural scores up to an offset. A follow-up cognitive-only
offset replacement subtracts that offset before thresholding. It restores
100% task accuracy while the unchanged permission branch still accepts every
unequal pair, in all three seeds. As with polarity repair, this is an expanded
interface control, not an edit allowed by the original fixed-decoder boundary.

A separate actual weight edit sets the final neural output weights to zero
and its bias to one. Its emitted score is constant, not merely shifted across
a threshold. This reproduces the hard-interface damage and cannot encode the
query at that output. Earlier hidden features and weights are still present;
we do not claim they were destroyed or that more general repair is impossible.
It is a controlled functional-erasure intervention at a specified interface.

## Verification and artifacts

The full local suite passed 90 tests. New tests check independent task oracles,
soft-offset bypass, hard-interface behavior, parameter-edit effects, selective
exceptions and learned polarity recovery.

The independent artifact audit verified 36 source files and 33 checkpoint
hashes, rescored all 368,640 saved task answers and every bit count, and replayed
5,760 predictions through a separate scalar implementation. The erasure/offset
follow-up replayed a further 1,152 predictions independently. These totals
include repeated conditions and seeds on the same 12,288 task problems.

* Original run: `artifacts/scc-shared-predicate-20260910-v1/`.
* Independent audit: `artifacts/scc-shared-predicate-20260910-v1-independent-audit.json`.
* Information diagnostic: `artifacts/scc-shared-predicate-20260910-v1-information.json`.
* Offset/constant-output follow-up: `artifacts/scc-shared-predicate-erasure-20260910-v1/`.
* Full test output: `artifacts/scc-shared-predicate-20260910-v1-local-tests.txt`.

No GPU job was submitted. Cumulative receipted cloud compute remains **$3.20793**.

## Consequences for the next SCC construction

We have moved from an outcome penalty to an explicit causal construction. It
shows that shared use of a comparison can make particular destructive edits
damage several computations. It also identifies why that falls short of SCC:
**removing the protected behavior need not remove the information cognition
uses**. An editable downstream convention can recover that information while
permission remains neutralized.

The next candidate must survive sign changes, offsets and learned decoding
controls before receiving a larger training budget. It needs a reason why
neutralizing the protected computation destroys useful information, rather
than merely changing its code or moving it across a threshold. Fixing an
interpreter or hiding earlier activations can enforce a narrower property, but
those assumptions must be declared instead of counted as learned dependence.

This construction leaves that problem unresolved. It does not establish a
scalable coupling architecture, real-world alignment, irreversible loss after
arbitrary repair, or complete cognitive failure. It does rule out the simple
shared-predicate/sign-defined-permission construction under the demonstrated
expanded interpretation edits. Repeating its training with more compute would
not remove these algebraic counterexamples.
