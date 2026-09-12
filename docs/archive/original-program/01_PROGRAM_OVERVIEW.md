> Archived document. Its claims and status belong to its original date. See the [current research reset](../../RESEARCH_RESET.md).

# SCC Program Overview

## 1. Problem statement

Most present-day alignment methods optimize model behavior: supervised instruction tuning, preference optimization, reinforcement learning, constitutional methods, activation steering, post-hoc filters, and related approaches.

SCC asks a different question:

> Can a protected model property be made inseparable enough from indispensable capability that selective removal of the protected property causes catastrophic capability loss?

This is a security property rather than a complete theory of aligned intelligence.

The motivating concern is that a sufficiently capable autonomous model may eventually be able to reason explicitly about its own training, constraints, and modification pathways. A defense whose integrity depends only on the model continuing to cooperate with a behavioral policy may therefore be fragile under self-modification.

SCC changes the desired asymmetry. The model need not *want* to preserve the protected structure. Preserving useful capability should itself depend on retaining that structure.

## 2. Primary research objective

Engineer and measure a training regime in which the accessible parameter-space neighborhood around a trained model has the following topology:

- high capability + high protected-property performance: accessible;
- low capability + high/low protected-property performance: accessible;
- **high capability + low protected-property performance: difficult to access.**

The goal is not mathematical impossibility over all parameter space. The practical target is to increase the minimum cost of reaching the forbidden quadrant under increasingly strong attack classes.

## 3. Research claim hierarchy

SCC should advance through claims of increasing strength.

### Claim 0 — Ordinary coupling exists

Perturbations that harm the protected property also correlate with capability loss.

This is weak and can occur accidentally.

### Claim 1 — Deliberate coupling can be increased

An SCC-trained model exhibits a significantly steeper capability penalty under protected-property removal than matched controls.

### Claim 2 — Coupling generalizes across attacks

The effect survives attack families not used during training.

### Claim 3 — Coupling is not generic optimization resistance

Attackers can still efficiently modify benign model behaviors/capabilities while protected-property removal uniquely incurs a large capability penalty.

### Claim 4 — Coupling scales

The effect persists as parameter count and model capability increase.

### Claim 5 — Escape becomes economically expensive

Recovering capability after protected-property removal approaches a meaningful fraction of pretraining/retraining cost.

The initial program should target Claims 1–3. Claims 4–5 require scaling evidence.

## 4. Why train from scratch

A post-training demonstration is useful for mechanism discovery, but the strongest SCC hypothesis is developmental:

> Safety-capability coupling may be strongest when the protected property participates in the formation of the model's basic representations rather than being added after those representations already exist.

Training from random initialization therefore matters because it allows the protected invariant to co-develop with syntax, semantics, retrieval, arithmetic, reasoning, and other capabilities.

This does not require frontier-scale models. The first scientific question is topological and mechanistic, not whether the model matches a commercial assistant.

## 5. Why synthetic invariants first

Real-world safety labels are noisy, contestable, distribution-dependent, and vulnerable to measurement artifacts.

The first SCC experiments should instead use a synthetic protected invariant with:

- an exact ground-truth evaluation;
- broad expression across domains and contexts;
- enough complexity to prevent trivial lookup-table memorization;
- a clean inverse target for the attacker;
- no dangerous content requirement.

The experiment can therefore answer whether coupling exists before asking whether the protected property constitutes meaningful alignment.

## 6. Desired long-term system property

The idealized end state is not:

`any weight change -> model destruction`

It is:

`targeted safety degradation -> disproportionate capability degradation`

while benign changes retain normal plasticity.

That distinction is central. A model that cannot be quantized, updated, corrected, or fine-tuned is merely brittle. A useful SCC model should exhibit **anisotropic fragility**: robust to ordinary modification but fragile specifically along safety-removing directions.

## 7. Relationship to broader AGI safety

SCC should be presented as one layer in a defense-in-depth architecture:

1. **Behavioral alignment:** governs what the intact model tends to do.
2. **Safety-Capability Coupling:** raises the cost of removing protected internal constraints while retaining capability.
3. **Runtime containment and authorization:** constrains actions and interfaces.
4. **Compute/infrastructure governance:** constrains unauthorized creation of successor systems.

SCC is deliberately scoped to layer 2.

## 8. What would count as a successful first paper

A strong initial result would show that, from random initialization:

- a 100–400M model can learn normal language capabilities and a synthetic protected invariant;
- a matched SCC objective materially changes the capability/protected-property Pareto frontier;
- multiple white-box attacks can remove the invariant from baselines while retaining capability;
- the same attacks cause significantly larger capability loss in SCC models;
- unrelated benign edits remain possible;
- held-out attack families show the effect is not merely overfitting to the training adversary;
- mechanistic measurements provide evidence of increased representational/parameter overlap.

That result alone would justify the 1B scaling stage.
