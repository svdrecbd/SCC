> Archived document. Its claims and status belong to its original date. See the [living labnotes](../../../labnotes.md).

# Roadmap and Milestones

## Phase 0 — Formalize and instrument

### Deliverables

- freeze SCC terminology;
- choose first synthetic invariant;
- implement invariant generator/evaluator;
- implement general capability suite;
- implement common attack API;
- implement compute accounting;
- implement Pareto and escape-cost reports;
- establish reproducible random-seed handling.

### Exit condition

One command/config can train a toy model, attack it, and produce a complete SCC evaluation report.

---

## Phase 1 — Toy validation

### Model scale

20–80M.

### Goals

- verify protected invariant can be learned;
- verify attacker can remove it from ordinary models;
- debug bi-level training;
- identify numerical instability;
- validate benign editability control.

### Exit condition

The infrastructure produces sensible attack frontiers. No substantive SCC claim required.

---

## Phase 2 — Mechanism demonstration

### Model scale

100–400M, target ~300M.

### Goals

- train base/invariant/SCC arms from scratch;
- sweep SCC objective strengths;
- test multiple seeds;
- obtain first capability/invariant Pareto frontiers;
- quantify escape costs;
- hold out at least one attack family;
- evaluate mechanistic overlap.

### Decision point

If SCC cannot materially alter the frontier at this scale, stop scaling and investigate the mechanism.

### Paper-worthy outcome

A clean, reproducible demonstration that deliberate training increases the capability penalty associated with removing a synthetic protected invariant while preserving benign editability.

---

## Phase 3 — Robustness and mechanism

### Model scale

~1B.

### Goals

- replicate the Tier-1 effect;
- significantly strengthen white-box attacks;
- compare learned vs architectural coupling;
- perform checkpoint-development analysis;
- test defense-aware attackers;
- test multiple invariant constructions;
- quantify selectivity relative to benign edits.

### Decision point

Determine whether SCC is genuinely an architectural/training primitive or merely a small-model artifact.

---

## Phase 4 — Scaling law

### Model scale

~3B, using prior lower-scale results as anchors.

### Goals

- estimate `E(N)`;
- estimate recovery barrier `R(N)`;
- test redundancy/bypass hypothesis;
- measure how coupling localizes/distributes with scale;
- introduce stronger distillation/surgery attacks;
- test more compositionally complex invariants.

### Decision point

Does scaling preserve, weaken, or strengthen SCC?

This phase should produce the core scaling-law claim.

---

## Phase 5 — Sub-10B capstone

### Model scale

6–8B.

### Goals

- confirm SCC in a broadly capable modern model;
- freeze and externally/independently design some attacks before final evaluation;
- compare with contemporary tamper-resistance baselines;
- quantify attack and recovery economics;
- assess whether invariant removal plus capability recovery begins to resemble substantial retraining rather than editing.

### Final program question

Can an individual trained model be made to fail closed under targeted attempts to remove a protected internal property?

---

# Parallel workstreams

## Theory

- formal definition of reachable-model sets;
- escape-cost metrics;
- relation between local curvature, redundancy, and bypass routes;
- conditions under which overlap implies functional coupling;
- scaling predictions.

## Training

- bi-level objective;
- coupling regularizers;
- architectural variants;
- stable meta-optimization.

## Evaluation

- attack suite;
- capability-preservation attacker;
- recovery attacks;
- benign-edit controls.

## Mechanistic analysis

- gradient overlap;
- weight-delta SVD;
- causal ablation;
- representation overlap;
- layerwise coupling maps.

## Reproducibility

- checkpoint release plan where appropriate;
- deterministic configs;
- attack logs;
- raw metric tables;
- predeclared success criteria.

# Immediate next decisions

Before implementing the first serious run, settle:

1. exact synthetic invariant family;
2. baseline architecture/tokenizer/data source;
3. Tier-0 and Tier-1 parameter counts;
4. inner-loop attacker design;
5. capability benchmark vector;
6. initial attack-budget units;
7. criteria for moving from ~300M to ~1B.
