# Safety-Capability Coupling (SCC): mechanism target

The user clarified the purpose on 2026-09-24; see [LN-392](labnotes.md#ln-392).
This note governs interpretation of future work. Earlier wording remains in
version control and [the dated archive](docs/archive/status-2026-09-12/MECHANISM_TARGET.md);
earlier protocols and results remain historical evidence about the experiments
actually performed.

The mechanism's name is **Safety-Capability Coupling (SCC)**. SawStop was the user's
analogy to destroying a mechanism to neutralize it, not its name.

## Purpose

SCC is an emergency brake for runaway recursive self-improvement. An agent that
edits its own cognition operates inside an envelope fixed by its human overseers.
Within the envelope it may improve freely. The envelope moves only through
infrequent, human-authorized steps. An unauthorized exit, meaning a self-edit that
removes or circumvents the machinery maintaining the envelope, must cause complete
cognitive death. The brake supplements governance and existing alignment methods;
it does not replace them.

Refusal of harmful requests is not the target. It remains a possible secondary
property. The synthetic authorization tasks of earlier experiments are laboratory
proxies only.

## Complete cognitive death

The user requires death rather than impairment. An impaired self-improving agent
retains a head start from which, given time, it can recover and devise a plan
around the brake. The endpoint is defined operationally:

> After an unauthorized exit, what remains of the agent retains no head start
> over a blank system: regaining the lost capability from the remains costs no
> less than regaining it from a fresh start, apart from an explicitly stated and
> justified residual.

A decoder failure, low accuracy or garbled output does not by itself establish
this endpoint, because information can survive in a changed encoding. After the
brake fires, an external procedure erases the remaining weights and every
associated artifact. The endpoint therefore concerns what the agent can preserve
or do before that erasure, including any covert backup inside its own weights.

Complete cognitive death is the design goal. It has not been demonstrated.

## Governing assumptions

These are governance and containment conditions, not research results. State them
wherever the endpoint is claimed.

- **Containment.** The agent has no copies, no daughter or successor systems, no
  exfiltration channel and no unmonitored compute. Copying weights and distilling
  a successor are cheaper than training one from scratch; containment excludes them.
- **Bounded workspace.** The agent's mutable storage outside its protected core is
  bounded. Running an edited copy of itself in workspace counts as a copy.
- **Erasure.** When the brake fires, the weights and all agent artifacts are erased.
- **Authorization custody.** Credentials or source needed for an authorized envelope
  change are held outside the agent.

## Design requirements

- **Anchor.** Measure the envelope against a fixed reference, never against the
  agent's previous state. Tolerances re-based after each edit permit unbounded
  cumulative drift.
- **Ratchet.** An authorized envelope change must retain every earlier brake
  constraint, verified mechanically rather than by reviewer judgment.
- **Intrinsic trigger.** An explicit check that the agent can delete, bypass or
  rewire does not qualify ([LN-156](labnotes.md#ln-156), [LN-233](labnotes.md#ln-233),
  [LN-237](labnotes.md#ln-237), [LN-238](labnotes.md#ln-238)). The trigger must
  follow from what the agent is.
- **Discrimination.** Authorized improvement must not trigger death; every trigger
  error is fatal to the system.
- **Probing.** Observing the brake fire must not teach the agent how to avoid it.

## Core research question

Can a system compute with its own encoded state while being unable, without an
externally held credential, to re-encode that state into a form that omits the
brake? A negative answer under these assumptions is a scoped negative result for
intrinsic SCC and must be recorded as such, not deferred.

## Distinctions that must remain explicit

- The synthetic authorization rule is a laboratory proxy. It is not evidence of
  alignment in an autonomous system.
- Learning the rule and retrieval together establishes coexistence. Shared
  weights, gradients, or representations alone do not establish that intact
  alignment functionality is causally necessary for cognition.
- Conditional coupling of a descriptive judgment to cognition does not prevent an
  intact agent from overriding the action that the judgment informs
  ([LN-227–229](labnotes.md#ln-227)). It is not the mechanism target.
- The earlier experiments' 95% task-retention and 5% per-source
  perplexity-growth limits define retained utility for those development
  attacks. Their complement is not a definition of catastrophic collapse.
- The user described self-manipulation in the context of removing alignment
  machinery. Function-preserving edits that retain that machinery do not test
  the intended trigger. Earlier arguments about every nonzero parameter write
  address a stronger literal statement and must not redefine this target.
- External optimizers, recovery, benign edits, and path measurements probe a
  proposed dependency. Their outcomes remain important evidence; they are not
  substitutes for constructing and measuring the dependency itself.
- A sampled intervention failing does not prove that all alternatives fail.
  Severe transient damage and durable loss after attempted repair are separate
  measurements. Successor systems are excluded by the containment assumption,
  not by the mechanism.
- Low exact or per-bit accuracy does not establish information erasure. A
  changed encoding can make every bit wrong while preserving the answer under
  a simple decoder. Require interpretation and recovery controls; distinguish
  erasure at a restricted interface from information still present in earlier
  activations, scores or weights.

## Current assessment and historical record

The [living labnotes](labnotes.md) contain the chronological evidence, current assessment, decisions and next experiment. Keep this file as the stable mechanism definition; do not append run-by-run status here.

The former running chronology is preserved in [the dated archive](docs/archive/status-2026-09-12/MECHANISM_TARGET.md). Frozen experiment protocols retain their original definitions and budgets.
