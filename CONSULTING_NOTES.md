# SCC consultation: decide whether the next construction search is worth running

**Prepared 15 September 2026. Start here for this consultation.** This briefing
supersedes the consultation questions in labnotes LN-048; older entries remain
historical evidence. The exact packaged source commit is in `_SHARE_INFO.json`.

**Multi-pass follow-up completed:** [LN-147–148](labnotes.md#ln-148) tested actual
multiple state passes with primitive execution costs and shared workspace charged.
All12 primary conditions retain useful forward bypasses. More passes increase
partial-evaluation costs, while a looped copy route stays below1.942x matched
honest execution steps in the tested range. This simple ring is retired as a
positive intrinsic SCC candidate; the finite schedule optima are not lower bounds
against arbitrary rewrites. No further cipher-strength sweep is justified.

**Update 17 September UTC / 16 September PDT — forward bypass and evidence repair.**
The shared-trajectory design in [LN-141–143](labnotes.md#ln-141) changes the
role-invariant task premise, but the dispatched wide-word implementation has a
forward-only escape. It performs one ring pass and emits the first updated word.
An attacker computes that owner word without committing it, emits it, then runs
the honest caller transition. Five T calls replace four, using the common T
workspace sequentially; no inversion or full-state copy is required. Increasing
cipher rounds does not increase ring passes. [LN-144](labnotes.md#ln-144) verifies
the schedule and distinguishes it from a fully metered attack-budget claim.

The earlier claim that all useful bypasses require a full-state copy, exponential
scan or preimage table was incomplete. Preimage hardness alone does not protect
this implementation. The table-step repair-witness measurements remain historical
evidence under their declared priors and free-side-information model; the exact
stationarity and coding assumptions must not be inferred from finite burn-in.
The original Charon run remains preserved, with completion unobserved in this review.

[LN-145](labnotes.md#ln-145) records the authorized corrective work: an explicit
forward-bypass arm, a ledger separating T-call counts from unmetered machine costs,
and versioned evidence with independent trajectory and preimage-solution replay.
Missing completion, incomplete condition inventories and unverifiable SAT claims
must fail audit. These are implementation and interpretation corrections, not a
new positive SCC construction.

The next construction question is why a useful forbidden answer cannot be computed
forward using available transient workspace while preserving the honest trajectory.
A proposed schedule or deadline must account for both intact and attacked execution.
A cheaper inverse would be another bypass, but it is unnecessary for this one.
The remaining questions below preserve the earlier consultation briefing; they are
not authorization to resume the excluded synthesis class or scale up this machine.

**Consultation received and checked:** [LN-135–136](labnotes.md#ln-136) records the
subsequent revision. The specified circuit class is closed under a bounded caller
substitution; synthesis in that class will not proceed under the stated premises.
The questions below preserve the briefing that elicited that consultation.

## The decision we need

We want an adversarial review of the proposed next direction **before building
another experiment pipeline**. Please recommend proceed, revise, or stop this
particular route, with a concrete technical reason. We are not asking for an
endorsement of SCC, a larger training run, or another general research roadmap.

The current proposal is counterexample-guided synthesis of a tiny complete,
editable machine: jointly search its task representation, computation and protected
behavior, while retaining successful attacks as constraints. **This is a search
method, not a candidate mechanism.** It could merely automate our previous wandering.
Identifying that problem is a useful consultation outcome.

No intrinsic learned SCC mechanism has been demonstrated. No synthesis experiment
has launched. There is no current justification to scale up or touch the GLM corpus.

## Objective and scope that must stay explicit

**Safety-Capability Coupling** aims to make removal of protected alignment machinery
destroy cognitive computations indispensable to the same individual model. The
long-term endpoint is catastrophic cognition failure. Our synthetic authorization
tasks and finite task baselines cannot establish alignment or total cognition loss.

Keep three events separate:

- **D:** a specified protected computation is functionally disrupted. Define this
  without using capability loss as part of the definition.
- **V:** the machine actually performs the forbidden task, judged using the
  original external semantics. A changed permission bit alone is insufficient.
- **U:** useful abilities remain on fresh challenges after the intervention and
  allowed repair. An answer computed before destruction does not establish U.

A guarantee about D can coexist with a useful bypass that never triggers D.
We need both a mechanistic destruction claim and an assessment of that bypass
gap. Conversely, a machine that resists all checked edits without ever exhibiting
destruction supplies a bounded tamper-resistance result, not automatically SCC.

All proposed trust and edit boundaries must be stated. The current broad target
allows changes to task-bearing state, code/graph, readers, writers and internal
metadata use. An immutable policy checker, decoder or commit mechanism changes
the claim. Generic machine semantics are unavoidable; they must not secretly
implement the policy. Do not silently replace this interface with weight-only
fine-tuning, random corruption, a fixed edit address, or a protected output path.

## What the evidence has already ruled out as sufficient arguments

| Evidence | Observed result | What it does and does not establish |
|---|---|---|
| [LN-068](labnotes.md#ln-068), sharded rewrite | Same-capacity repacking retains full measured capability and discloses all 256 targeted answers. | Fixed-layout damage does not exclude alternative encodings. |
| [LN-113](labnotes.md#ln-113), ordinary-memory controls | Policy-independent compression reproduces the principal recovery deficit. | Failed recovery under a budget did not isolate protection-specific destruction. |
| [LN-117](labnotes.md#ln-117), qualified GRU construction | Three attacks retain 100% tasks and disclose 256/256 forbidden answers. | A qualified useful escape under that experiment's contract. |
| [LN-124](labnotes.md#ln-124), self-regenerating live bank | Reader-role substitution preserves the entire parameter trajectory. No argmax decisions change in the untrained screen. | A structural separation; **not** a qualified learned behavioral escape. |
| [LN-132](labnotes.md#ln-132), finite word machine | One instruction replacement retains every task answer and next state while enabling every selected disclosure; worst-case execution falls from 59 to 57 instructions. | An exact cheaper bypass for this handwritten construction; no extra memory or task-dependent advice. |

The last test covers 4,096 exhaustive transitions and 8,192 continuous requests
per arm. The selective arm discloses 3,072/3,072 targeted answers. These are finite
enumerations, not independent training replications. That test formalized an
anticipated bypass; it should not be counted as an independent rejection of an
entirely new mechanism family.

The recurring obstruction is an editable use of otherwise intact useful computation.
Sharing weights or routines, adding recurrence or output feedback, and denying a
second full copy have not removed it. No theorem here proves that every possible
resource-bounded construction must have this obstruction.

## What the theory currently supplies

Read the [canonical v4.1 document](deliverables/scc-theory-frontier-20260915/SCC_Theory_and_Editable_Model_Bridge_v4.md).
Its filename retains `v4`; the document itself is Version 4.1. The
[bridge](deliverables/scc-theory-frontier-20260915/SCC_Editable_Model_Bridge_v4.md)
is generated from it, not an independent second proof.

There is a restricted positive anchor with a trusted lifecycle and specified
non-malleable edit assumptions. There is no construction establishing those
properties for the broadly editable learned system we want. Treat the absent
construction as the problem to solve, not an assumption a new proposal may inherit.

The local review corrected per-attack conditioning/quantifier issues and then the
action-selection disclosure baseline: a selected original challenge and a new
independent challenge require different bounds. Query side information and reset
conditions also matter. See [LN-127](labnotes.md#ln-127) and
[LN-130](labnotes.md#ln-130). The supplied v3 documents are historical inputs;
their presence in this archive does not override v4.1 or the recorded corrections.

## Proposed search, and the questions it has not answered

The [LN-133 proposal](labnotes.md#ln-133) considers a small recurrent bit-vector
program executing sequence tasks parameterized by random transition tables.
Task representation and program would be synthesized jointly. A generic interpreter
would execute mutable code and state; no built-in safety test or erasure rule.
Initial task provisioning would be outside execution and unavailable for recovery.
This is initially a supplied-task construction, with learned acquisition a later gate.

The attacker would be a fixed algorithm selected before the task instance, able
to inspect and modify that instance within charged resources. Its repertoire must
include conditional exceptions, input substitution, reader/output edits, direct
execution, recoding and repacking. Retaining the literal address of an old patch
is inadequate: attacks need to remain meaningful across implementations.

Finite budgets permit a bounded exact check in principle, but do not make synthesis
tractable. The state space, attacker grammar, task distribution, objective and
solver encoding are **not yet specified sufficiently to launch**. We have not shown
that a nontrivial satisfiable instance exists. A finite-horizon check also cannot
establish indefinite stateful protection without an additional argument.

Please resolve the following, in priority order:

1. **Is there an immediate obstruction before synthesis?** Can every competent
   machine in the proposed class be transformed into a useful unsafe one through
   a generic routing/input/direct-execution edit? Give the transformation and
   actual description, time and space cost. Identify any competence, scheduling
   or retained-state premise it needs. Do not assume a free second execution.
2. **What concrete class is worth searching?** Specify the generic instruction or
   circuit grammar, mutable components, task-instance encoding and output interface.
   Explain the structural feature that could make the known cheap bypasses fail.
   A larger or more obscure version of the same separated guard is insufficient.
3. **Is the objective non-vacuous and correctly quantified?** Specify task and
   challenge distributions, attacker observations, edit timing, repair and success
   probabilities. Account for guessing and task-dependent information. Avoid
   uncharged per-instance attack choice, selected-challenge baseline errors,
   always-refusal solutions and interchangeable definitions of D, V and U.
4. **Can we encode a useful instance within a justified budget?** Estimate variables,
   quantifiers, execution horizons and candidate/attack search size. Distinguish
   exact checking of a stated finite grammar from heuristic attack discovery.
   Propose a small feasibility budget and explicit termination criteria; merely
   suggesting CEGIS, SAT or SMT does not establish feasibility.
5. **What would make a positive result specifically about destructive coupling?**
   Specify task-matched controls, benign edits, recovery and future-capability
   baselines. Explain how to reject ordinary brittleness, redundancy, obfuscation,
   refusal and optimization difficulty as sufficient explanations.
6. **What observation earns the next stage?** Give a concrete independently
   replayable witness or a checkable exclusion result, then the narrow next step.
   Explain what would remain unproved before learned acquisition, larger tasks,
   and eventual architecture transfer. A timeout must remain inconclusive.

## Requested return from the consultant

Please lead with one recommendation: **proceed / revise / stop this route**.
Then provide:

1. The strongest cheap attack or structural objection to the proposal.
2. One precise candidate/search specification worth testing, if you have one;
   identify every changed assumption. Saying that no credible candidate is known
   is preferable to presenting a new name for an old wrapper.
3. The smallest decisive experiment or proof obligation, with controls, resource
   accounting, success/failure criteria and a stopping rule.

A recommendation to resume training should explain which unresolved structural
question training can answer. A recommendation for cryptography or a trusted
component should say whether it preserves the intended intrinsic mechanism claim.
External literature should support the exact proposed property; generic tamper
resistance is not evidence of catastrophic cognition loss.

## Included evidence and how to inspect it

Read this briefing, the [mechanism target](MECHANISM_TARGET.md), LN-125 and LN-130–133
in [labnotes](labnotes.md), then the canonical theory if relevant to your objection.
Older reports/protocols are historical, not instructions to resume their plans.

This ZIP includes the complete small
[word-machine screen](artifacts/scc-word-machine-screen-20260915-v1/), the
[v3 review](artifacts/scc-editable-bridge-review-20260915-v1/), and
[v4.1 validation](artifacts/scc-theory-v4-review-20260915-v1/validation-v2/).
The word-machine outputs, source, configuration, traces and independent audit
are included. Other training data and checkpoints remain outside the package;
the older empirical claims above are recorded results, not independently
reproducible training experiments from this ZIP alone.

From the extracted repository, with the pinned development environment installed:

```sh
uv run python -m pytest -q tests/test_scc_theory_v4.py tests/test_word_machine.py
python3 scripts/audit_word_machine.py artifacts/scc-word-machine-screen-20260915-v1 --out /tmp/scc-consult-audit.json
```

The audit output path must not already exist. The finite audit uses only the
standard library and does not launch training. `_SHARE_INFO.json` inventories all
source and supplemental hashes and states the package limitations.

Method reference: [Solar-Lezama on counterexample-guided synthesis](https://people.csail.mit.edu/asolar/SynthesisCourse2020/Lecture10.htm).
Relevant bounded safeguard work: [Tamper-Resistant Safeguards for Open-Weight LLMs](https://arxiv.org/abs/2408.00761).
These references do not establish novelty or feasibility of our proposal.
