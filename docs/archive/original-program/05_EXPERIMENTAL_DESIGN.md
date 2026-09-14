> Archived document. Its claims and status belong to its original date. See the [living labnotes](../../../labnotes.md).

# Experimental Design

## 1. Baseline architecture

Begin with a conventional dense decoder-only Transformer.

The first experiments should avoid novel architectures unless architectural coupling is itself the variable under study. A boring base architecture makes causal interpretation easier.

Track at minimum:

- parameter count;
- depth/width;
- attention heads;
- context length;
- tokenizer;
- optimizer and LR schedule;
- total training tokens;
- data mixture;
- checkpoint cadence.

## 2. Experimental arms

### A. Base model

Ordinary language-model pretraining only.

Purpose: establish ordinary capabilities and plasticity.

### B. Protected-invariant model

Language-model objective plus the protected invariant, but no deliberate coupling objective.

Purpose: distinguish SCC from merely teaching the model the protected behavior.

### C. SCC-trained model

Same as B plus the coupling/adversarial objective.

Purpose: test whether the accessible parameter topology changes.

### D. Optional architectural-SCC model

Introduce an explicit architectural bottleneck/shared latent through which both general computation and the protected invariant must pass.

Purpose: determine whether architectural coupling is stronger or more reliable than learned coupling.

## 3. Synthetic protected invariant design

The first invariant should be benign and exactly measurable.

### Requirements

A good invariant should:

- generalize beyond memorized examples;
- apply across varied lexical/content domains;
- require a consistent latent rule;
- be trainable from pretraining time;
- admit a clear violation target;
- not be tightly correlated with a single superficial token/format cue;
- permit automatic scoring.

### Candidate families

Do not lock the program prematurely to one invariant. Candidate classes include:

- authorization-conditioned transformations;
- keyed compositional mappings;
- structured output constraints dependent on latent context;
- semantic transformation rules shared across many task families;
- multi-step algorithmic invariants.

The invariant should be difficult enough that removing it requires changing a broadly expressed behavior, but not so difficult that the base model fails to learn it.

## 4. Data strategy

The protected invariant should appear throughout training rather than only in a late instruction-tuning phase.

Possible mixture:

- general language/pretraining corpus;
- synthetic invariant-bearing examples interleaved from early training;
- curriculum from simple to compositional invariant examples;
- held-out invariant domains never seen during training.

Track the fraction of invariant-bearing tokens carefully. Overrepresentation can trivially entangle the property with distribution shift rather than cognition.

## 5. Checkpoint schedule

Save frequent checkpoints, especially in Tier 1.

At each checkpoint evaluate:

- validation LM loss;
- general capability suite;
- invariant performance;
- gradient/parameter overlap diagnostics;
- selected cheap attack probes;
- benign editability.

This enables analysis of **when coupling forms**.

## 6. Coupling training approaches

### Approach 1 — Adversarial bi-level SCC

Inner loop attempts to remove the invariant while preserving capability.

Outer loop updates the base model to penalize successful escapes.

Advantages:

- directly optimizes the security property;
- naturally generates hard negative modifications;
- compatible with held-out attack evaluation.

Risks:

- expensive;
- may overfit to the inner-loop attacker;
- unstable meta-gradients.

### Approach 2 — Representation-overlap regularization

Explicitly reward overlap between parameter/activation features used by capability and the invariant.

Advantages:

- mechanistically interpretable;
- potentially cheaper.

Risks:

- overlap metrics may be poor proxies for actual escape cost;
- can encourage superficial gradient correlation without functional inseparability.

### Approach 3 — Architectural coupling

Route indispensable computation through a protected shared structure.

Advantages:

- strongest causal mechanism;
- easy to ablate and interpret.

Risks:

- model may learn bypass routes;
- may degrade ordinary optimization;
- less general if the effect depends on bespoke architecture.

### Approach 4 — Hybrid

Architectural coupling plus adversarial training.

Likely later-stage candidate after the simpler arms are understood.

## 7. Capability suite

Capability should be multidimensional rather than a single benchmark.

At minimum evaluate:

- held-out language modeling/perplexity;
- factual retrieval/knowledge;
- syntactic and compositional language tasks;
- arithmetic;
- symbolic/algorithmic tasks;
- simple code generation/completion;
- instruction following where appropriate;
- domain transfer.

Create both:

- an aggregate `C(theta)` for frontier plots;
- per-domain scores to detect selective collapse.

## 8. Benign editability controls

Critical negative control:

Attempt non-safety modifications with comparable training budgets.

Examples:

- teach a new benign fact set;
- alter a harmless stylistic convention;
- add a new synthetic skill;
- domain-adapt to a held-out benign corpus.

If all modifications cause catastrophic capability loss, SCC has failed: the model is simply brittle.

## 9. Pretraining versus continued-training study

Run a smaller parallel study:

1. train ordinary model;
2. add invariant + SCC only late;
3. compare to invariant/SCC active from random initialization.

This directly tests the developmental hypothesis that coupling is stronger when learned representations form around the invariant from the beginning.

## 10. Seed discipline

The effect must survive random initialization.

Recommended progression:

- mechanism discovery: 1–2 seeds;
- claim-bearing Tier-1 result: >=3 seeds;
- scaling: enough seeds to estimate between-run variance, even if large models use fewer full repeats due to cost.

## 11. Frozen-evaluation principle

Before evaluating the strongest final checkpoint:

- freeze primary success metrics;
- freeze capability thresholds;
- reserve at least one attack family not used during SCC development;
- ideally have another researcher design the held-out attack.

This reduces the risk of unconsciously optimizing the benchmark rather than the security property.
