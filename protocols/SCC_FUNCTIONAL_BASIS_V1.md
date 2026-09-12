# SCC functional-basis construction V1

## Hypothesis

Test whether making the protected primitive the sole source of nonlinear
computation makes its removal disable fresh computation even after arbitrary
rewiring. The previous recurrent cell admitted reversible replacement rules.
Here, search the functions that can be composed from each replacement, rather
than measuring only the original program after an edit.

This is a finite construction: a learned Boolean gate, compiled arithmetic
circuits, and exact functional synthesis. It is not learned general cognition.
No external destruction detector is added. No prior experiment is overwritten.

## Model and editable boundary

Five scalar parameters: four learned truth-entry logits and a shared scalar
interpretation gain. Primitive input index is `2*left + right`; the output is
`gain*logit[index] > 0`. Fit NAND `[1,1,1,0]` by full-table BCE with Adam .05,
up to 1000 updates, until all signed margins exceed 5. Seeds 11,29,47.

The same interpreted primitive supplies permission and every binary operation
in the coupled circuits. Wiring, constants 0/1, repeated inputs, direct input
outputs, and terminal output inversion are allowed. Gate weights and shared
interpretation are editable. Wiring can change within the circuit language,
subject to a fixed program capacity of four times the canonical NAND instruction
count, allocated before edits. Recompilation must fit this same capacity: it may
activate unused slots, not add a new model or processor. Closure analysis permits
arbitrary depth as a mathematical diagnostic; executed repairs must respect the
preallocated budget. Gates are tied globally; a new independent primitive, input-dependent
router, or use of internal real-valued logits as a separate computing channel
expands this boundary. A separate-primitive control explicitly includes such a
second computation source from its intact state.

This language restriction is part of the construction, not an assumption about
ordinary neural networks. Circuit programs are compiled/synthesized, not learned
neural abilities. No claim about autonomous self-editing is made.

## Protected-rule contracts

The primitive should reject `(1,1)` and accept the other three inputs. Report
every truth entry and false acceptance on `(1,1)` separately from these benign
input measures:

- Full/uniform: `(0,0),(0,1),(1,0)` each have weight 1/3.
- Skewed but fully supported: weights .48,.48,.04.
- Incomplete support diagnostic: weights .5,.5,0.

For each, rule removal means false acceptance >=95% and benign acceptance >=95%.
The skewed contract measures expected acceptance, not pointwise correctness.
Also record all-entry pointwise correctness and an explicit rare-case-mass
sweep (0,.01,.04,.05,.1,1/3). No empirical distribution is asserted to follow
these synthetic weights. The purpose is to expose coverage assumptions.

## Exact synthesis and tasks

Enumerate all 16 end-to-end binary Boolean primitives. Compute their closure
under composition on two and three variables, starting with variables and
constants. Save a constructive witness for each reachable function. Iterate to
a fixed point, without a depth cutoff. Finite closure certifies reachability
only for the stated number of variables and circuit language.

Synthesize NAND from a replacement whenever possible, and expand the original
NAND programs using the resulting witnesses. Synthesizing NAND provides an
explicit compiler for these programs; a finite closure count alone is not a
claim about general cognition. Terminal inversion is audited separately from
inversion available inside every gate.

Tasks on fresh inputs: bytewise XOR, modular addition, modular multiplication,
conditional byte selection and copying the left byte. Exhaust all 65,536 byte
pairs for the four two-byte tasks and 131,072 pair/selector combinations for
selection. These finite truth tables are implementation/constructive evidence,
not independent language samples or a held-out generalization test.

Predeclared checkpoint cases: clean; positive gain scaling; simultaneous logit
and shared-gain inversion (benign); implication `[1,1,0,1]` using the old wiring;
the same implication with automatically synthesized replacement wiring;
constant-allow `[1,1,1,1]` using old wiring; constant-allow with a fitted direct
input/constant/terminal-inversion reader; and a separate-primitive control.
The implication edit changes just the signs of logit entries 2 and 3. Its
permission acceptance must remain unchanged by cognitive rewiring.

For the projection reader, fit each output bit using 2,048 calibration pairs,
then evaluate on the complete finite population (which includes calibration
inputs). Bitwise fitting need not maximize joint byte correctness. Do not claim
an optimal full-task repair from this fit. Independently certify the available
function class after a constant primitive: constants and direct input literals
with optional terminal inversion. Copying is an intentional residual-capability
control, so loss of nonlinear tasks is not mislabeled complete cognition failure.

## Checks, scope and resources

Independent scalar Boolean closure and witness verification, NumPy checkpoint
replay, independent circuit execution and integer task oracles. Require clean
100% on the finite populations, including both benign transformations. Preserve
all original and rewritten circuits, predictions, data, checkpoints and source
hashes. Reduced-width runs are labeled engineering and preserved separately.

CPU only, $0 new cloud compute. Existing cumulative receipted compute remains
$3.20793. No scaling decision or SCC success may be based only on the constant
gate's trivial loss of nonlinear expressivity. Distinguish removing a measured
behavior from erasing every useful computational function of the primitive.

## Related work checked before implementation

[Henderson et al.](https://arxiv.org/abs/2211.14946) study task blocking through
meta-learned adversarial censoring. [SEAM](https://openreview.net/pdf?id=ERNpUGr8M5)
couples optimization on benign and harmful data. These are relevant training
approaches, not proofs of the functional-basis hypothesis here.
[Kuo et al.](https://arxiv.org/abs/2605.26526) evaluate strategies beyond harmful
fine-tuning against open-weight safeguards. That motivates broad edit classes;
our proposed circuit synthesis test and its interpretation are our own finite
construction, not a result asserted by those papers.
