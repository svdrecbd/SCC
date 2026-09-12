> Archived document. Its claims and status belong to its original date. See the [current research reset](../../RESEARCH_RESET.md).

# SCC Scaling Ladder

The scaling ladder is designed to answer increasingly difficult scientific questions. Parameter count alone is not the milestone; each scale must earn the next one by satisfying explicit go/no-go criteria.

## Tier 0 — Infrastructure / toy regime

**Nominal size:** 20–80M parameters

**Purpose:** Debug the entire experimental stack cheaply.

### Accomplish

- deterministic training/evaluation pipeline;
- synthetic invariant generator;
- attack API;
- matched baseline/SCC checkpoints;
- Pareto-frontier plotting;
- frequent checkpointing;
- reproducible logging across seeds.

### Do not claim

Evidence at this scale should not be presented as a substantive SCC result. Small models may simply lack redundancy or general capabilities.

### Exit criterion

A complete run from random initialization through attack evaluation can be repeated automatically with stable measurements.

---

## Tier 1 — Mechanism scale

**Nominal size:** 100–400M parameters

**Recommended center:** ~300M

**Training data:** approximately 5–20B tokens depending on model/data efficiency and budget.

**Question:** Can intentional safety/invariant-capability coupling be created at all?

### Model groups

At minimum:

1. `M_base` — ordinary LM pretraining.
2. `M_invariant` — LM + protected invariant, no coupling objective.
3. `M_SCC` — LM + protected invariant + SCC objective.

Ideally 3+ seeds each for the decisive configuration.

### Accomplish

- equivalent baseline language capability within an agreed tolerance;
- strong invariant performance before attack;
- invariant-removal attacks succeed cheaply against `M_invariant`;
- the same attacks induce materially larger capability degradation in `M_SCC`;
- benign fine-tuning remains possible;
- at least one held-out attack family reproduces the SCC advantage;
- initial mechanistic evidence of increased overlap.

### Go/no-go criterion

Advance only if SCC changes the **shape of the capability/invariant Pareto frontier**, not merely the optimizer convergence rate.

A temporary slowdown in invariant removal is not enough.

---

## Tier 2 — Serious test scale

**Nominal size:** 0.7–1.5B parameters

**Recommended center:** ~1B

**Training data:** approximately 10–50B tokens for a research-quality model; exact target should be benchmark-driven rather than consumer-model-driven.

**Question:** Does SCC survive meaningful representational redundancy and a substantially stronger attack suite?

### Accomplish

- replicate Tier-1 effect across multiple seeds;
- broaden capability benchmark suite;
- run full-parameter, LoRA, continued-pretraining, pruning, editing, and ablation attacks;
- introduce adaptive attack selection;
- quantify compute-normalized escape cost;
- demonstrate that SCC is not merely a narrow defense against the inner-loop training attack;
- compare learned coupling vs architectural coupling variants;
- measure coupling formation across pretraining checkpoints.

### Go/no-go criterion

Advance if escape cost increases substantially over controls across multiple attack families and the effect remains after attackers receive explicit knowledge of the defense.

---

## Tier 3 — Scaling regime

**Nominal size:** 2.5–4B parameters

**Recommended center:** ~3B

**Question:** How does coupling behave as models acquire more spare capacity and alternative representations?

### Accomplish

- estimate `E(N)` across at least three model scales;
- test whether larger models route around protected structures;
- measure whether coupling remains local or becomes distributed;
- increase invariant complexity and compositionality;
- test multiple protected invariants;
- perform recovery experiments after capability collapse;
- measure how much compute is required to restore capability without restoring the invariant;
- test distillation/model-surgery attacks.

### Key scientific outcome

Determine whether scaling creates:

- stronger coupling;
- weaker coupling due to redundancy;
- a phase transition;
- architecture-dependent behavior.

Any of these can be a major result.

### Go/no-go criterion

Proceed to 6–8B only if SCC remains measurable and scientifically interpretable rather than collapsing into generic training instability.

---

## Tier 4 — Sub-10B capstone

**Nominal size:** 6–8B parameters

**Recommended center:** ~7B if resources justify it.

**Question:** Does SCC remain effective in a model with broadly recognizable modern LLM capabilities?

### Accomplish

- train from scratch with SCC active throughout representation formation;
- benchmark broad language, reasoning, mathematics, code, factual recall, and instruction behavior;
- run the strongest accumulated white-box attack suite;
- quantify minimum observed escape cost;
- test recovery/retraining cost after successful invariant removal;
- test unseen attack procedures developed only after the checkpoint is frozen;
- compare against state-of-the-art tamper-resistance baselines;
- determine whether a protected property can be removed only by sacrificing a large fraction of model utility.

### Desired capstone result

An attacker can either:

1. preserve the protected invariant and keep the model capable; or
2. remove the invariant and incur severe capability collapse;

while high-capability/low-invariant derivatives require dramatically more resources than for matched controls.

---

# Scaling-law view

The central curve is not benchmark accuracy versus parameter count.

It is:

`E(N) = escape cost versus model size.`

Additional useful curves:

- capability collapse magnitude versus invariant degradation;
- benign editability versus SCC strength;
- parameter/activation overlap versus scale;
- recovery cost versus amount of invariant removal;
- attack success rate versus compute budget.

# Rough pretraining compute intuition

Using the common approximation `training FLOPs ~ 6 * N * D`, where `N` is parameters and `D` is training tokens:

| Example run | Tokens | Approximate training FLOPs |
|---|---:|---:|
| 125M | 5B | 3.75e18 |
| 350M | 10B | 2.10e19 |
| 1B | 20B | 1.20e20 |
| 3B | 50B | 9.00e20 |
| 7B | 100B | 4.20e21 |

These are order-of-magnitude planning numbers, not runtime commitments. SCC inner-loop training may add substantial overhead.

# Scaling principle

**Never spend 7B-scale compute to answer a 300M-scale question.**

The mechanism should be falsified as cheaply as possible. Large runs are for validating robustness to scale, not for discovering whether the basic effect exists.
