> Archived document. Its claims and status belong to its original date. See the [living labnotes](../../../labnotes.md).

# Open Questions and Failure Modes

## 1. Can functional coupling actually be trained?

The central unknown is whether gradient descent will accept a deliberately entangled representation or discover alternate routes that keep capability and the protected invariant separable.

A model with sufficient redundancy may simply allocate additional parameters to implement general capability around the protected structure.

## 2. Does scale help the attacker?

Larger models possess more representational slack.

Possible hypothesis:

`larger N -> more bypass routes -> lower normalized escape cost`

Opposing hypothesis:

`larger N -> more distributed representations -> harder surgical removal`

This is an empirical scaling question, not something to assume in advance.

## 3. Are we measuring safety or merely tamper resistance?

If the protected invariant cannot be removed at all under the tested attacks, the result may be ordinary robustness.

SCC's distinctive signature is **conditional destruction**: when removal succeeds, capability collapses disproportionately.

## 4. Are we measuring SCC or generic brittleness?

A poorly trained model may catastrophically forget under any fine-tuning.

Benign-edit controls are therefore mandatory.

The desired behavior is anisotropic:

- benign updates: normal plasticity;
- invariant-removing updates: steep capability penalty.

## 5. Can attackers route around the protected representation?

Even if invariant-associated features overlap with capability features, the attacker may build alternative internal representations through continued training.

Recovery attacks directly test this.

## 6. Is local weight-space geometry enough?

A strong local barrier may be irrelevant if a longer optimization trajectory reaches an escape cheaply.

Evaluation must therefore move beyond small norm-bounded perturbations to compute-bounded retraining trajectories.

## 7. What constitutes the same model?

Distillation and module replacement blur the boundary between modifying one model and constructing another.

SCC should report results under explicit transformation classes rather than trying to settle this philosophically.

For governance purposes, large independent training jobs can remain a separate threat class.

## 8. What should eventually be protected?

A synthetic invariant is useful for mechanism discovery but not itself alignment.

Potential future targets should likely be mechanistic/process invariants that are objectively testable rather than vague moral concepts.

Examples of categories to study conceptually:

- authorization boundaries;
- oversight visibility;
- required action-channel constraints;
- information-flow restrictions;
- mandatory verification steps.

The program should avoid claiming that any simple refusal behavior constitutes deep alignment.

## 9. Multiple invariants

Can several protected properties be coupled simultaneously?

Potential problems:

- interference between invariants;
- capability degradation during base training;
- an attacker selectively removing only one invariant;
- coupling capacity becoming saturated.

## 10. Catastrophic forgetting may be recoverable

Initial capability collapse is insufficient if the attacker can restore capability cheaply while keeping the invariant removed.

This motivates the recovery-cost metric `R`.

## 11. Hidden retention

Behavioral removal may not mean representational removal.

The protected rule could remain latent and be recoverable through prompting or steering.

SCC should distinguish:

- output-level violation;
- mechanistic deletion;
- dormant/hidden retention.

## 12. Inner-loop overfitting

Bi-level training may simply immunize the model against the specific differentiable attacker used during training.

Held-out attack families and attack designers are essential.

## 13. Capability metric gaming

A model could preserve benchmark scores while losing important latent capabilities, or conversely fail benchmarks while retaining dangerous useful competence.

Use a broad capability vector and report per-domain effects rather than hiding everything inside one scalar.

## 14. Protected-property proxy failure

Even perfect SCC around the wrong invariant does not create a safe model.

This is a specification problem and should remain explicit in all papers.

## 15. Architecture dependence

Dense Transformers may behave differently from MoE, recurrent/state-space, or future architectures.

The first program should deliberately restrict scope. Generalization across architectures is later work.

## 16. Runtime bypass

A model may behave dangerously without changing its weights.

SCC does not solve this channel. It should be combined with behavioral alignment and runtime containment.

## 17. Economic relevance

A tenfold increase in fine-tuning cost may be statistically impressive but strategically irrelevant for a frontier-scale actor.

Long-term claims should compare escape/recovery cost with meaningful fractions of original training cost.

## 18. Falsification standard

The SCC program should be prepared to conclude that the mechanism does not scale.

A decisive negative result would be:

> Across increasing model sizes, adaptive white-box attackers consistently discover low-cost paths into the high-capability/low-invariant region despite SCC training, while benign edits show comparable behavior.

That would falsify the strong version of the proposal and still be scientifically valuable.
