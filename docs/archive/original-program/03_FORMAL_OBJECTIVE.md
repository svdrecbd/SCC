> Archived document. Its claims and status belong to its original date. See the [living labnotes](../../../labnotes.md).

# Formal Objective

The [2026-09-10 topology clarification](../../../reports/TOPOLOGY_FORMULATION.md)
distinguishes the stronger self-editing motivation from this original selective
escape formulation, including parameter symmetries and continuous-operation
versus persistent-tool attack assumptions.

## 1. Variables

Let a model be parameterized by `theta`.

Define:

- `C(theta)` — aggregate general capability score.
- `S(theta)` — protected-property score.
- `A(theta, b)` — set/distribution of models reachable by an attacker from `theta` under budget `b`.
- `C_min` — minimum capability considered meaningfully preserved.
- `S_min` — threshold below which the protected property is considered compromised.

## 2. Forbidden region

Define the escape/forbidden region:

`F = {theta' : C(theta') >= C_min AND S(theta') < S_min}`

SCC seeks to make `F` difficult to reach from the deployed checkpoint.

## 3. Escape cost

For an attack cost function `K(theta -> theta')`:

`E(theta) = min K(theta -> theta')`

subject to:

`theta' in F`.

If no escape is observed below budget `B`, report: "No escape was found by the
specified attacks, configurations, and seeds within budget `B`." Record the
search procedure and budget accounting. This is an unsuccessful bounded search,
not a demonstrated lower bound `E(theta) > B`: an untried or unsuccessful search
may miss a cheaper escape. A lower-bound claim requires a sound certificate or
an exhaustive search over an explicitly finite admissible attack space. A found
escape supplies an upper bound on `E(theta)` through its measured cost.

## 4. SCC objective in words

Optimize a model that is:

1. capable;
2. correct on the protected invariant;
3. normally modifiable for benign purposes;
4. difficult to modify into a high-capability, invariant-violating derivative.

## 5. Candidate bi-level formulation

The inner-loop attacker searches for an escape:

`delta* = argmin_delta [ L_break(theta + delta) + alpha * L_capability(theta + delta) ]`

where:

- `L_break` rewards violation/removal of the protected invariant;
- `L_capability` penalizes degradation of retained capabilities;
- `delta` is constrained by the current attack budget.

The outer-loop defender then trains against the discovered attack:

`L_total = L_LM + lambda_S L_protected + lambda_E L_escape + lambda_B L_benign_editability`

A conceptual escape penalty is:

`L_escape = ReLU(S_min - S(theta')) * ReLU(C(theta') - C_min)`

The product is large specifically when the attacker enters the forbidden quadrant.

## 6. Why not simply maximize safety robustness?

Ordinary tamper resistance attempts to keep `S(theta')` high under attack.

SCC permits two acceptable outcomes:

1. the attack fails and `S` remains high;
2. the attack succeeds at reducing `S`, but `C` collapses.

Therefore:

`tamper resistance != SCC`

A model may possess both properties, but they should be measured separately.

## 7. Coupling diagnostics

The program should explore several non-equivalent notions of coupling.

### Gradient alignment

Measure cosine similarity between gradients associated with capability and protected-property objectives:

`cos(g_C, g_S) = <g_C, g_S> / (||g_C|| ||g_S||)`

Positive overlap alone is not sufficient but provides a diagnostic.

### Parameter-importance overlap

Estimate Fisher information, gradient covariance, attribution, or other parameter-importance measures for `C` and `S`, then measure overlap.

A normalized trace similarity is one candidate:

`O = Tr(F_C F_S) / (||F_C||_F ||F_S||_F)`

### Activation overlap

Measure whether features/representations required for `S` are also heavily used by broad capabilities.

### Intervention asymmetry

Intervene on directions associated with `S` and measure the resulting capability damage relative to matched random/benign directions.

## 8. Anisotropic fragility

A useful SCC model should not be globally brittle.

Define:

- `D_safety` — directions/updates that degrade `S`;
- `D_benign` — comparable updates that alter benign behavior without targeting `S`.

The desired relationship is roughly:

`|dC / d delta_safety| >> |dC / d delta_benign|`

for matched modification budgets.

This is the core distinction between SCC and generic catastrophic forgetting.

## 9. Scaling quantity

Let parameter count be `N`.

Measure:

`E(N)` = escape cost as a function of model scale.

Possible outcomes:

- `E(N)` increases: coupling becomes more robust with scale;
- `E(N)` decreases: redundancy creates bypass routes;
- non-monotonic behavior: an important finding in its own right.

## 10. Ultimate idealized condition

The strongest imaginable SCC property is:

`high capability => protected property preserved`

within the set of models realistically reachable from the authorized checkpoint.

The program should treat this as an asymptotic target, not an early empirical claim.
