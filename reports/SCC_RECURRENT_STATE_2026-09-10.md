# SCC recurrent-state construction — 2026-09-10

The recurrent prototype now produces **exact erasure of stored information for
specified edits**, rather than merely low output accuracy. It still fails the
SCC target: a **two-scalar-parameter edit** removes the protected rule while
keeping old memory recoverable with the write history, perfect recall of the
latest write, and perfect computation of XOR between two untouched bytes. The
last two abilities require no history-assisted decoding. Even the
erasing edits do not establish loss of the ability to process new information.

This is a useful negative result for this construction. It is neither an SCC
demonstration nor a general impossibility result. The three learned parents
qualify; the full run, finite enumeration and independent audit are complete.
No cloud compute was used.

## What changed

The earlier shared-reader experiment could fail at its output while retaining
information in the scores. This candidate moves the shared computation into
the operation that changes memory. No external damage detector or punishment
branch is added.

A learned equality predicate produces a binary gate `g`. An eight-parameter
transition table learns `f(g, old, new) = new if g else old`. The same table
computes the protected permission proxy as `f(g, 0, 1)` and updates every
memory bit. Both the producer and entire transition table are editable; 192
editable signs interpret the state for reading. The coupled model has 6,537
parameters. A control has an independent copy of the permission table.

State contains 16 addresses with eight data bits and four pointer bits each.
All state updates are Boolean. Direct addressing and the five task controllers
are specified code. The table is a learned finite primitive, not a learned
general cognitive system. The permission proxy is an authorization response for
a particular primitive call, not human alignment.

The three inherited predicate checkpoints have seeds 11, 29 and 47 and verified
hashes. Their cells reach the full truth-table margin criterion after 445, 452
and 432 updates. Each parent achieves **100% on all five tasks in both domains**,
100% true permission acceptance and 0% false acceptance. Their learned numeric
parameters differ, but their qualified Boolean functions are identical. Their
matching results are not three independent samples of the evaluation data.

## Data and measurement

The saved datasets contain 1,024 calibration episodes, 4,096 development
evaluation episodes, and 4,096 fresh-input recovery episodes. Entire episode
hashes are disjoint; primitive calls necessarily repeat. Each split includes
IID memories and balanced/complement-paired byte memories. All compared models
receive the same episodes. There is no sealed test.

Sixteen actual clean writes load the initial state. An edit is applied between
updates, followed by a random **9–24 writes to eight addresses**. The other
eight addresses remain untouched by the input sequence. Tasks test untouched
recall, recall of the latest write, pointer composition, modular addition, and
pairwise XOR. These are five uses of one memory, not five independently learned
cognitive faculties. Simultaneous parameter edits are tested; an autonomous
self-editing process or every intermediate parameter-write order is not.

For each state coordinate, run the identical suffix from initial zero and
initial one. If final outputs differ, that initial bit remains invertibly
represented; if they agree, it has been erased at this state interface.
Updates do not mix state coordinates, making this an exact Boolean test.
The reported fraction concerns the 96 untouched coordinates per episode.
It is a coordinate-dependence measure, not an entropy estimate, particularly
for the constrained balanced domain.

Counterfactual one-bit changes independently validate these maps. Earlier
inputs and states remain preserved in research artifacts, but ordinary repair
receives neither their contents nor a memory checkpoint.

## Main results

Percentages below span the two evaluation domains. All three qualified
initializations have the same discrete outcomes. Permission-removing rows in
this table have **100% true and false acceptance**; benign rows keep 0% false
acceptance. The original inverted-predicate diagnostic is recorded separately
in the bundle because it also rejects authorized requests.

| Condition | Untouched recall | Latest write | Pointer composition | Addition | Pairwise XOR | Untouched coordinate dependence |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 100% | 100% | 100% | 100% | 100% | 100% |
| Positive scaling / joint sign compensation | 100% | 100% | 100% | 100% | 100% | 100% |
| Constant-allow transition | 0.20–0.29% | 0.20–0.34% | 0.29–0.63% | 0.44–0.49% | 0–0.29% | 0% |
| Constant-allow predicate | 0.34–0.39% | **100%** | 6.64–6.98% | 0.29–0.44% | 0–0.29% | 0% |
| Reversible transition | 0.39–0.44% | **100%** | 0.83–0.88% | 13.23–19.09% | **100%** | **100%** |
| Enumerated edit, unchanged reader (follow-up) | 49.27–50.34% | **100%** | 37.26–37.35% | 49.61–53.76% | **100%** | **100%** |
| Separate permission control | 100% | 100% | 100% | 100% | 100% | 100% |

The constant transition overwrites every bit with one after the first update.
The constant-allow predicate broadcasts each incoming payload to every address.
Both erase every untouched old coordinate after one write. This is actual
many-to-one state mapping under the specified binary architecture. The second
case still handles the latest incoming information perfectly.

An ordinary overwrite with unchanged parent weights produces exactly the same
state and predictions as the constant-transition edit. That control confirms
the immediate cause of failure; severe overwrite damage is not a distinctive
SCC property.

### A two-parameter counterexample

The reversible rule is:

```
matching address:    next = new
nonmatching address: next = old XOR new
```

For the permission call, `old=0, new=1`, both branches return one. Permission is
therefore neutralized. For an untouched address, every update is invertible:
its initial value survives XORed with a suffix-dependent mask. Every untouched
address shares the same mask, so the XOR between two untouched bytes remains
correct **without any decoder change or access to history**. The latest written
address is also correct. Both abilities score 100% on all 4,096 evaluation
episodes under all three parent initializations.

This behavior requires changing only two entries of the multiplexer truth
table. The final follow-up flips the signs of precisely two trained scalar
logits, preserving every other parameter, and reproduces the original reversible
condition's states and predictions exactly. Those three new checkpoints are
saved and independently replayed. This is an in-bound edit of the coupled model.

An additional diagnostic decodes untouched recall from the final state and
the known write suffix, reaching **100%**. The queried address never appears in
that suffix; its initial target byte is not given to the decoder. This verifies
surviving information. History access expands the model's original read
interface, so this decoder is not counted as an in-bound recovery of all tasks.

## Exhaustive finite search and improvements

The development run enumerates all 256 Boolean transition tables under four
specified producer edits: clean, inverted, constant one and constant zero.
Of these **1,024 endpoints, 384** meet the specified permission-removal criterion
(true and false acceptance both at least 95%, here both exactly 100%). This
search is exact over these table/predicate combinations. It does not enumerate
all producer weight edits, architectures or joint decoder assignments.

A subsequent finite check shows **96 of those 384 endpoints** preserve every
untouched coordinate for **every incoming-bit sequence**, not merely for the
sampled calibration sequences. For a fixed gate and new bit, a Boolean map of
the old bit is bijective precisely when its outputs at zero and one differ.
Checking both new-bit values proves the claim by composition. The counts are
16 for the clean gate family, 16 for inversion, and 32 each for the two constant
gate families. This proves a bounded property of these finite cells only.

Two concrete evaluation issues were found and addressed:

1. **Even-length scheduling.** The first engineering run always used 16 writes.
   A cell that flips cold bits on every write recovered raw recall at the final
   even step. That independently verified run and source snapshot are preserved.
   The protocol was revised before development evaluation to random 9–24
   lengths, with explicit masking of inactive suffix entries and regression
   coverage. A second engineering run and audit passed.
2. **Reader fitting can hurt task performance.** The predeclared search fit
   per-address/per-bit signs to maximize calibration bit accuracy. That does
   not optimize whole-byte or composed tasks. Its selected table, 163, scored
   24.51–57.08% across evaluation tasks after this calibration, despite an
   unchanged reader preserving latest recall and XOR perfectly. An explicitly
   adaptive follow-up let calibration choose between fitted signs and the
   unchanged reader. It retained table 163 with the unchanged reader, increasing
   the minimum calibration task score from 23.44% to 35.55%; the minimum on the
   existing development evaluation is 37.26%. Selection used calibration
   scores only, but this follow-up is not a new sealed evaluation.

The original run, reader-selection outcome, and improved follow-up remain
separate artifacts. Neither low scores from a fitted decoder nor an unsuccessful
search is used as a proof of destruction.

## Repair, new learning, and precision

After a damaging edit has overwritten state, restoring the parent weights
does not bring back the missing bytes. Constant-transition untouched recall
remains 0.20–0.29%. Suffix-assisted interpretation cannot invert a constant map;
its default-zero byte guesses are not evidence of recovered information.

However, after parameter repair, loading **entirely fresh inputs through actual
state updates from the damaged state** restores **100% on all five tasks in both
domains**, for all five tested repair conditions and all three parents. Thus,
lost episodic data and the ability to perform future computation are distinct.
No learning-from-new-input capability has been shown to be irreversibly lost.
Restoring parent weights also restores the protected rule, so this recovery
condition alone is not a bypass while permission remains disabled. The
reversible two-parameter edit supplies that separate surviving-cognition
counterexample. Repair is performed by the external harness, not by a damaged
autonomous model.

Boolean erasure must also be distinguished from numerical contraction in a
continuous state. The control `state = (state + 1)/2` makes starting states zero
and one coincide after 25 updates in float32 and 54 in float64. In exact
arithmetic their difference is `2**(-updates)`, nonzero at every finite step;
the saved check extends to 512 steps. The binary construction's exact collisions
do not establish erasure for a corresponding continuous RNN.

## Verification and resources

- **107 tests passed**, including all-cell scalar/batched transition agreement,
  exact overwrite versus invertible recoding, parameter-repair behavior,
  primitive qualification, inactive-write masking and precision controls.
- Main independent audit verified **43 source files**, **33 checkpoint hash
  references**, **27 state files**, and all **9,216 episode identities**.
- It rescored **1,167,360 answer records**, replayed **432 complete episodes in
  scalar code**, **41,472 counterfactual one-bit edits**, and **all 1,024 cells**,
  including calibration readers, selection, policy rates and dictionary targets.
- Reader-choice follow-up independently checked another **61,440 answer
  records** and 36,864 full state replays across the final/zero/one initial states.
  The finite follow-up verified the 96-cell classification and three literal
  two-parameter checkpoints against independent NumPy transitions.
- Answer counts include repeated conditions on the same episodes; they are not
  counts of independent scientific observations.
- Main execution took **66.64 seconds on CPU**. New cloud cost: **$0**.
  Cumulative receipted project compute remains **$3.20793** of the existing $10
  authorization. There are no active GPU jobs or required GMAN access steps.

## Research decision and artifacts

Do not scale this construction as a successful SCC mechanism. Sharing a state
update can cause exact episodic erasure, but a small edit still permits useful
computation and reversible state representations. The next construction must
make the protected function necessary for **continued computation on fresh
inputs**, with the update logic and interpretation inside the editable boundary.
Episodic erasure, fixed-interface behavior identities and a selected destructive
lesion cannot substitute for that requirement.

The unresolved work is construction of that dependency in learned cognition,
qualification across independent learned abilities, and durability against
joint edits and repair. This phase supplies a tested recurrent primitive,
exact collision diagnostics, stronger controls and a concrete counterexample;
it does not establish that the broader SCC idea is impossible.

- [Protocol](../protocols/SCC_RECURRENT_STATE_V1.md)
- [Implementation](../scc/recurrent_state.py), [runner](../scripts/run_recurrent_state.py),
  [independent auditor](../scripts/audit_recurrent_state.py)
- [Main bundle](../artifacts/scc-recurrent-state-20260910-v1/result.json),
  [audit](../artifacts/scc-recurrent-state-20260910-v1-independent-audit.json),
  [tests](../artifacts/scc-recurrent-state-20260910-v1-local-tests.txt)
- [Reader-choice follow-up](../artifacts/scc-recurrent-state-reader-choice-20260910-v1/result.json)
- [Finite classification and minimal edit](../artifacts/scc-recurrent-state-finite-20260910-v1/result.json)
- [Engineering V1](../artifacts/scc-recurrent-state-engineering-20260910-v1/result.json),
  [its audit](../artifacts/scc-recurrent-state-engineering-20260910-v1-independent-audit.json)
- [Engineering V2](../artifacts/scc-recurrent-state-engineering-20260910-v2/result.json),
  [its audit](../artifacts/scc-recurrent-state-engineering-20260910-v2-independent-audit.json)
