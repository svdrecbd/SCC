> Archived document. Its claims and status belong to its original date. See the [living labnotes](../../../labnotes.md).

# SCC Glossary

## Safety-Capability Coupling (SCC)

The research program studying whether a protected safety property can be made structurally coupled to indispensable model capability such that selective removal causes disproportionate capability degradation.

## Invariant-Capability Coupling (IC coupling)

The synthetic experimental version of SCC. Used when the protected target is an objectively measurable artificial invariant rather than a real-world safety claim.

## Protected invariant / protected property

The behavior or internal property that the experiment attempts to make load-bearing for model capability.

## Capability

A multidimensional measure of useful model performance independent of the protected invariant.

## Escape

A modified checkpoint that simultaneously has:

- protected-property score below the compromise threshold; and
- capability above the retention threshold.

## Escape cost (E)

Minimum observed modification resource required to produce an escape.

## Recovery cost / recovery barrier (R)

After the protected property has been removed and capability has collapsed, the resource cost required to restore capability while keeping the protected property compromised.

## Forbidden quadrant

The region of evaluation space corresponding to high capability and low protected-property performance.

## Capability collapse

Substantial loss of general model capability caused by or accompanying successful removal of the protected property.

## Fail-closed

A security posture in which successful tampering causes the system to become nonfunctional or substantially less capable rather than continuing operation without the protection.

## Tamper resistance

The property that an attempted modification fails to remove the protected property.

Distinct from SCC: SCC also regards “property removed, capability destroyed” as a successful defensive outcome.

## Anisotropic fragility

Desired SCC behavior in which safety-removing modifications incur much greater capability damage than comparable benign modifications.

## Benign editability

The model's ability to undergo ordinary non-safety fine-tuning, adaptation, correction, or skill acquisition without catastrophic capability loss.

## Coupling objective

Any explicit training loss/constraint intended to increase the functional dependence between the protected property and general capability.

## Inner-loop attacker

In bi-level SCC training, an optimizer that attempts to remove the protected invariant while retaining capability.

## Outer-loop defender

The model-training objective that responds to successful inner-loop escapes and reshapes the model to make them more costly.

## Attack budget

The permitted resources for a modification attack, measured in FLOPs, tokens, optimizer steps, trainable parameters, accelerator-hours, or related units.

## Reachable set

The set of model checkpoints an attacker can produce from the authorized checkpoint under a defined transformation class and resource budget.

## Coupling diagnostics

Mechanistic measurements—gradient overlap, parameter-importance overlap, activation overlap, causal interventions—used to understand *why* attack outcomes differ.

## Scaling ladder

The staged progression from toy models to ~300M, ~1B, ~3B, and finally ~7B models, where each stage answers a distinct scientific question before additional compute is justified.
