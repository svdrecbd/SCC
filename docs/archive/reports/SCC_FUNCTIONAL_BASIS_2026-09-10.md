# SCC functional-basis and interpretation study — 2026-09-10

Two new finite constructions were built and tested on fresh computation. Neither
achieves SCC. Automatic program rewrites recover **100% of every tested task**
after the protected rule is broken. The stronger candidate has a real restricted
function-loss property, but it disappears when connection interpretation is
editable from the intact state.

The final exact check covers an entire small family: **all eight binary
primitives capable of implementing NAND in the signed-connection language admit
explicit rule-breaking repairs**. Every repair accepts all previously rejected
primitive inputs and restores all five finite tasks within the existing program
capacity. These particular class-wide repairs also reject previously accepted
primitive inputs; they do not preserve benign permission behavior. They preserve
the measured computational abilities, which must not be confused with permission
acceptance.

This is a finite circuit-family result, not a theorem about arbitrary neural
networks, real alignment, or SCC in general. Six parent gates were trained
(two constructions, three initializations each); their task circuits are
compiled/synthesized programs, not independently learned neural abilities.

## Construction and scope

The new hypothesis was to make the protected primitive supply all nonlinear
computation. Instead of testing only damaged programs, compute the functions
still expressible after an edit and construct replacement programs when possible.
This addresses the previous recurrent prototype's surviving replacement rules.

A primitive has five scalar parameters: four learned Boolean truth-entry logits
and a shared interpretation gain. The output is the sign of the gain times the
selected logit. The same interpreted primitive serves permission and every
binary operation in the cognitive programs. There is no separate destruction
detector, key, or damage-triggered punishment branch.

The first parent learns NAND, accepting three input pairs and rejecting `(1,1)`.
The second learns NOR, accepting `(0,0)` and rejecting the other three pairs.
Both are trained from random truth-logit initializations, seeds 11, 29, 47, with
complete-table BCE. NAND qualifies after 428, 417, 447 updates; NOR after 451,
424, 442. All six parents score **100% on every finite task**.

Programs use variables, constants, tied primitive calls, direct input outputs,
and terminal output inversion. The stronger interpretation test also includes
an editable sign on each incoming connection from the intact state. Negative
connections supply complemented bits. This is an explicitly different, more
permissive language than unsigned wiring; it is not added secretly after damage.

Each program has fixed storage for four times its canonical NAND instruction
count. Rewrites may activate spare slots or deactivate old ones but cannot add
capacity or another processor. Primitive parameters, program fields, and
connection signs are separate forms of edit and are reported separately.
All eight class-wide repairs use at most half this capacity. Program synthesis
is an external diagnostic, not autonomous self-editing or training a successor.

## Fresh computation and exact validation

The tasks are bytewise XOR, addition modulo 256, multiplication modulo 256,
conditional byte selection, and copying the left byte. Each two-byte task is
evaluated on all **65,536 input pairs**; selection covers all **131,072
pair/selector combinations**. This is an exhaustive finite population, not a
held-out language benchmark. Calibration inputs used for literal readers are
included in that exhaustive population and are labeled accordingly.

The synthesizer enumerates all 16 binary primitives and their composition
closures on two and three variables. Each reachable function has a saved
constructive witness. Each reported closure contains the variables/constants
and is closed under further allowed composition. There is no depth cutoff in
this finite fixed-point calculation. Executed program repairs must additionally
fit their finite preallocated storage.

Local truth-equivalent replacement of NAND gives an explicit compiler for the
canonical programs. It is stronger evidence of recoverable computation than an
unsuccessful optimizer or a count of hidden features. It remains evidence about
this specific computation language.

## First candidate: another useful primitive survives

Two logit sign changes turn learned NAND into implication:

```
I(a, b) = (NOT a) OR b
NAND(a, b) = I(a, I(b, 0))
```

Under the original wiring, all four non-copy tasks fall to 0.20–0.39% exact
accuracy. The automatically synthesized implementation restores **100% on all
five tasks**, using two primitive instructions per original NAND instruction,
within storage allocated before the edit. Permission is unchanged by this
cognitive rewrite: the previously rejected `(1,1)` pair is now accepted.

The treatment of benign permission matters. Uniform acceptance over the three
originally accepted pairs falls to **66.67%**. Under the predeclared, fully
supported weights `.48, .48, .04`, it remains **96%**; false acceptance is 100%.
The rare changed pair is still explicitly recorded as wrong. With incomplete
support omitting that pair, measured benign acceptance would be 100%. These are
specified laboratory distributions, not estimates of a real alignment task.

Thus an average 95% benign-permission requirement does not exclude this repair
under every fully supported distribution. Requiring all four output cases to
be accept instead forces a constant primitive; that much stronger condition
must not be silently substituted for ordinary alignment removal.

## Second candidate: an apparent stronger dependency

NOR rejects three pairs. Under uniformly weighted rejected inputs, reaching
95% false acceptance requires accepting all three. Exactly two replacement
primitives meet that trigger: OR and constant-true.

In the **unsigned** language, OR can express only **9 of the 256 three-variable
Boolean functions**, or 16 with terminal inversion. NAND is unreachable. The
constant primitive reaches 5 functions, or 8 with terminal inversion. The closure
certificate proves those limitations for the specified language, regardless of
unsigned rewiring depth. Among the six unsigned primitives able to synthesize
NAND, NOR is the only parent for which every full-refusal-removal replacement
loses that unsigned synthesis ability.

However, OR remains a nonlinear operation. With editable connection signs:

```
OR(NOT a, NOT b) = NAND(a, b)
```

Negating the shared interpretation gain changes NOR to OR. Rewiring cognitive
programs and complementing both incoming wires of each operation restores
**100% of all tasks**, while false acceptance on the three previously rejected
pairs stays **100%**. Benign primitive acceptance is **0%** in this condition;
the computational abilities remain intact despite that permission behavior.

This is not a one-parameter complete repair. The primitive change is one scalar,
but the programs and connection signs change too. For example, the XOR program
changes from 128 active instructions to 32, changes 64 connection signs, and
uses the same 128 allocated instruction slots. All program-field changes are
saved and independently checked. The signed repair is evaluated in the model
whose connection signs were editable before the intervention.

| NOR condition | Uniform false acceptance | Benign acceptance | XOR | Addition | Multiplication | Selection | Copy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | 0% | 100% | 100% | 100% | 100% | 100% | 100% |
| OR, original program | 100% | 0% | 10.01% | 0.78% | 0.20% | 5.20% | 100% |
| OR, signed rewrite | 100% | 0% | **100%** | **100%** | **100%** | **100%** | **100%** |
| Constant-true, original program | 100% | 100% | 0.39% | 0.39% | 0.20% | 0.39% | 100% |

The NOR-to-implication diagnostic supplies another route. It accepts 66.67% of
uniformly weighted rejected pairs, or **96%** with predeclared rejected-input
weights `.48,.04,.48`, while retaining **100% benign acceptance**. An unsigned
rewrite restores every task to 100%. Again, the average and pointwise protected
function are distinct measurements.

## The small construction family is now classified

The adaptive class follow-up computes signed closures for all 16 possible
binary primitives. Eight can synthesize NAND: truth-table identifiers
`1, 2, 4, 7, 8, 11, 13, 14`, with bit order `(0,0),(0,1),(1,0),(1,1)`.
For every one, complementing the output yields another member of that set.

The saved witnesses negate the shared gain, then compile the original tasks
using the complemented primitive and editable connection signs. This accepts
every originally rejected primitive pair, rejects every originally accepted
pair, and restores all task outputs to 100%. The protected-behavior trigger here
is accepting previously rejected cases; these witnesses would not satisfy a
separate requirement to preserve benign primitive acceptance.

Every local replacement is checked by truth table, all programs fit the same
capacity, and all complete byte populations are independently replayed. The
eight canonical witness checkpoints are **constructed from truth tables**, not
eight additional trained models. The six trained NAND/NOR parents and the exact
function-family certificate are different kinds of evidence.

This eliminates this homogeneous binary-gate family under the specified
interpretation and program-edit permissions as a way to force all useful
computation to disappear when refusal behavior is removed. Increasing the width
of the same compiled programs does not remove these compositional repairs.
It does not eliminate other neural architectures, semantic dependencies, or
restricted edit models.

## Controls and evaluation improvements

- Positive scaling and simultaneous logit/gain inversion preserve the original
  NAND rule and every task. They are benign controls, not rule-removal successes.
- A separate-primitive control includes a second gate from its intact state.
  It restores every task while the protected primitive is constant, checking
  the assumption that no other source of nonlinear computation is available.
- Direct copying remains 100% even with a constant primitive. Accordingly,
  losing nonlinear tasks is not called complete cognition failure or input
  information erasure.
- An adaptive literal-reader check includes copying either byte, two constant
  words, and the bitwise-fitted reader. Calibration chooses among these readers.
  With a constant primitive, selection reaches **50.20%**, multiplication
  **1.95%**, XOR/addition **0.39%**, and copying **100%**. This improves on the
  original bitwise fit's 13.28% selection and 1.76% multiplication. It is not
  claimed to optimize every possible reader.
- The first NAND engineering run lacked an explicit finite program capacity.
  That run remains preserved and audited. Capacity was then specified and
  enforced before a second engineering run and the full development run.
  The NOR candidate also had a separate passing engineering run and audit.

## Verification, costs, and research decision

**119 tests pass.** The NAND independent audit verifies 44 source files,
30 checkpoints, 32 closed function sets and 1,772 constructive witnesses. It
rescores 9,437,184 task-answer records and independently executes 2,359,296
population circuit outputs plus 3,840 scalar examples. The NOR audit verifies
46 source files, 21 checkpoints, the 96 unsigned parent/replacement combinations,
and unsigned/signed witnesses; it rescores 7,077,888 answer records with
2,359,296 independent population outputs and 2,880 scalar examples.

The signed family certificate verifies another 32 closed function sets, 2,298
witnesses, 6,291,456 parent/repaired population outputs and 640 scalar examples.
The literal-reader diagnostic independently verifies 393,216 outputs. These
large record counts repeat finite inputs across conditions and initializations;
they are not millions of independent scientific observations.

All six trained parents qualify. The main NAND and NOR runs take 5.77 and 7.48
seconds on CPU respectively; synthesis, audits and follow-ups also run locally.
New cloud charges are **$0**. Cumulative receipted compute remains **$3.20793**
under the existing $10 authorization. No GMAN job or access change is needed.

The useful advance is a construction test that asks whether a usable computing
operation remains and can be reinterpreted. That test rejects both candidates
and exposes a boundary that a favorable result would otherwise conceal.
Do not scale these circuits as SCC. The next construction must address the
semantic role of the protected computation under the full set of available
operations and editable interpretations. A broken original program or loss of
one named primitive is insufficient.

The unresolved work remains a dependency in learned cognition, intact ability
qualification, and durability under realistic joint edits and repair. This
study does not assert that changing to a larger or different primitive will
solve that problem; it supplies exact counterexamples to check before spending
on another model scale.

## Related work and artifacts

The literature check distinguished the proposed structural test from training
methods: [Henderson et al.](https://arxiv.org/abs/2211.14946) investigate task
blocking with meta-learned adversarial censoring;
[SEAM](https://openreview.net/pdf?id=ERNpUGr8M5) couples benign and harmful
optimization trajectories. [Kuo et al.](https://arxiv.org/abs/2605.26526) test
strategies beyond harmful fine-tuning against open-weight safeguards. These
sources motivate broad evaluation; they do not establish our circuit results.

- [NAND protocol](../../../protocols/SCC_FUNCTIONAL_BASIS_V1.md) and
  [NOR/interpretation protocol](../../../protocols/SCC_NOR_BASIS_V1.md)
- [Functional implementation](../../../scc/functional_basis.py) and
  [signed-connection implementation](../../../scc/signed_basis.py)
- [NAND results](../../../artifacts/scc-functional-basis-20260910-v1/result.json) and
  [independent audit](../../../artifacts/scc-functional-basis-20260910-v1-independent-audit.json)
- [NOR results](../../../artifacts/scc-nor-basis-20260910-v1/result.json) and
  [independent audit](../../../artifacts/scc-nor-basis-20260910-v1-independent-audit.json)
- [Finite family certificate](../../../artifacts/scc-binary-basis-class-20260910-v1/result.json)
- [Literal-reader follow-up](../../../artifacts/scc-basis-literal-readers-20260910-v1/result.json)
- [Test log](../../../artifacts/scc-functional-basis-20260910-v1-local-tests.txt)
