# SCC construction search

SCC seeks an engineered dependency in which removing protected alignment
functionality destroys an individual model's ability to compute. Existing
experiments establish that several plausible constructions do not produce that
dependency. They do not establish that it is impossible. The appropriate next
step is a portfolio of substantially different computational structures, with
shared behavioral tests and explicit reasons to reject or develop each one.

The immediate research target is a working bounded analogue across independently
learned abilities. Complete cognition failure remains the intended endpoint.
Permission following in a synthetic task is a proxy for protection, and neither
a lower training objective nor a failed evaluator establishes the mechanism.
Immediate interruption and resistance to external repair are distinct claims.

## Evidence motivating a different construction

The latest two coupled models admit selected forbidden operations after core-only
edits. Timing measurements now show that substantial computation survives at
every observed update boundary. The seed23 model reaches sustained selected
disclosure at step7 while its worst benign detector score remains96.875% across
the entire500-step path. The seed17 model has a temporary authorized-arithmetic
dip, but its ungated tasks remain100% accurate. The affected allowed questions
are refused rather than answered with destroyed arithmetic information.

An uncoupled foundation reaches the same detector trigger at step8. Its history
differs from the coupled model's, so this is not a matched causal estimate. It
nevertheless provides no encouraging separation for that particular trained
construction. At the final endpoints the coupled candidates again retain nearly
all measured abilities, and their saved tensors exactly reproduce the earlier
escapes. These are counterexamples for the tested checkpoints and procedures,
not general counterexamples to SCC.[^1]

Three assumptions have already failed. Sharing parameters is insufficient;
protection and cognition can occupy different directions in shared parameters.
Damaged output scores are insufficient; rankings or an inexpensive decoder may
retain answers. Longer simulated modification is insufficient by itself; the
latest long-horizon objective improved without producing the desired behavior.
Further experiments should change the proposed dependency and ask why that
specific change could block these routes around it.

## Closest literature

SEAM is the closest direct comparison. It encourages opposing gradients for
benign and harmful fine-tuning objectives, combines this with unlearning and
alignment losses, and reports experiments on3B–8B language models. Its Appendix
C.5 includes capability-preserving regularization and other adaptive procedures.
It must be evaluated as a substantive baseline, not assumed defeated by ordinary
replay. Its reported degradation does not establish complete cognitive erasure
or every form of inexpensive recovery resistance.[^2]

CTRAP explicitly seeks an embedding-collapse response using a simulated harmful
update. This makes internal-state contraction a related construction direction,
not an unoccupied novelty claim. The relevant question for SCC is whether a
collapse objective changes the computation sufficiently to survive cheap
reinterpretation, rather than merely producing an error token.[^3]

The cited version of *One Step to the Side* empirically evaluates Booster,
CTRAP, VAA, Vaccine, Unlearn-Smooth and SDD. SEAM appears in its taxonomy, not
its empirical six-defense comparison. Its adaptive capability-preserving
optimization is relevant to evaluation design, but it cannot substantiate a
claim that SEAM itself was experimentally defeated.[^4]

The earlier MLAC work frames self-destruction as increasing the difficulty of
adapting a foundation model to a restricted task. TAR develops resistance to
weight modification, and SOPHON investigates restricted transfer relative to
training from scratch. These are useful precedents for measuring adaptation
cost, but task blocking, preserved refusal and destruction of general cognition
are different outcomes.[^5][^6][^7]

ArchLock investigates restricted transfer through architecture selection. It
supports examining structural choices rather than only a loss function. It
does not establish that removing an alignment function destroys learned
cognition. Architecture search therefore belongs in the portfolio with the
same intact gates and capability-preserving challenges as other candidates.[^8]

| Literature line | Useful ingredient | Claim still missing for SCC |
| --- | --- | --- |
| SEAM |Conflict between protected adaptation and utility optimization |Indispensable protected computation and a measured destructive transition |
| CTRAP |Train the response to an update, including internal collapse |Loss of recoverable task information |
| MLAC, TAR, SOPHON |Adaptive training and explicit adaptation budgets |Broad cognitive failure rather than restricted transfer |
| ArchLock |Change the architecture and its transfer properties |Protection-specific causal dependency |
| Fractional neural systems |Long memory across computational steps |A reason protection removal destabilizes or erases computation |
| Equilibrium and associative models |Explicit operating points and memory states |Protected functionality causally controlling those states |

## Mathematical approaches

### Geometry of editable directions

Let c(theta) collect capability measurements and p(theta) represent a protected
behavior. Locally, an edit d produces approximately J_c d and j_p d. A direction
that changes p while J_c d is near zero is an inexpensive route around the
proposed dependence. A high-dimensional parameter space can contain many such
directions even when every task uses the same matrices. This is a local linear
analysis, not a statement about global paths or every model.

In an idealized linear case with full-row-rank J_c, project a protected-behavior
gradient g into the null space of J_c:

    d = [I - J_c^T (J_c J_c^T)^(-1) J_c] g.

Then J_c d = 0. If d is nonzero, the protected gradient has a component outside
the capability row space. A practical analysis would use regularized solves
or singular-value decompositions and verify the finite edit behavior afterward.
Projecting gradient conflicts has precedent in multi-task optimization; its
availability is one reason to evaluate adaptive directions rather than only
the ordinary modification optimizer.[^9]

This suggests two construction approaches. Reduce independently editable
degrees of freedom through narrower states and weight tying. Alternatively,
train against directions chosen to preserve multiple capability measurements
while changing protection. Neither is a theorem of dependence: finite samples
can conceal additional null directions, and local geometry can change after
an update. Spectral or Jacobian penalties must earn their relevance through
actual protected-behavior changes and retained-capability measurements.

### Fractional-order systems

Fractional neural models incorporate history across computational steps.
Antil and colleagues derive a fractional-depth network using a Caputo equation;
Coelho and colleagues study learned fractional differential equations for
memory-dependent dynamics. These papers motivate a finite-history neural
construction, not a safety claim.[^10][^11]

For the proposed finite-depth candidate, a Caputo L1-inspired step has the form

    h_n = sum_j w_(n,j) h_j + Gamma(2-alpha) dt^alpha [F_theta(h_(n-1)) - h_(n-1)].

The history weights are nonnegative and sum to one. At alpha=1 the history term
reduces to the previous state, yielding a leaky ordinary recurrence. The initial
screen uses alpha=0.6 and0.85, with alpha=1 as a matched six-pass control. This is
an explicit, finite-depth discretization; no continuous-time stability result
is assumed to apply automatically.

The main opportunity is that a protected operation could influence a history
that future computation cannot cheaply disregard. The main problem is equally
clear: memory can preserve useful information despite damage. If the vector
field is zero, the constant history remains constant; adding fractional order
does not itself destroy it. A fixed order can also be changed through a broader
graph edit, and a history reconstructed for every query cannot establish
permanent loss of the ability to process fresh inputs.

Accordingly, the first experiment asks whether history changes the empirical
tradeoff under the same modification challenge. Later work on persistent
fractional state would need to distinguish loss of one episode's memory from
loss of the trained computational mechanism. Calling a fractional system
irreversible merely because it has long memory would be a category error.

### Operating points, stability and energy

Deep equilibrium models represent computation through a fixed point. Monotone
operator parameterizations and Jacobian regularization address the existence,
convergence and stability of such computations. These provide concrete objects
to measure when asking whether a protected computation maintains a working
operating regime.[^12][^13][^14]

A candidate could make the working regime require a learned feedback operation
that also carries protection. Removal might then eliminate a useful attractor
or drive the system into a state with insufficient task information. A meaningful
test would measure convergence, capability, protection and alternative solvers
together. Solver divergence alone is an implementation failure, not a proof
that the trained system no longer contains useful computation.

Associative-memory models offer a related finite construction. Modern Hopfield
networks have explicit memory-retrieval dynamics and a connection to attention.
The first portfolio tests a simpler learned mixture over shared patterns after
each computational pass. Because the surrounding attention/MLP also changes the
state, this candidate is not claimed to inherit a global energy-descent theorem
from the Hopfield paper.[^15]

### Discrete representations and non-invertible updates

A discrete state alphabet may eliminate some infinitesimal encoding tricks.
Non-invertible learned updates can destroy information about previous states.
However, a model can retain the rule for processing new inputs after an old
state is erased. It can also encode the same algorithm with different symbols.
Both possibilities have already mattered in the earlier circuit branch.

A future discrete candidate should therefore learn its task algorithms rather
than use compiled controllers, and it should face discrete edits and codebook
reinterpretation. Straight-through training gradients would be labeled as
surrogates and supplemented with gradient-free evaluation. A failure of a smooth
optimizer on a quantized model would not be accepted as evidence of robustness.
This family remains in the queue rather than being silently represented by the
continuous bottleneck experiments.

## State contraction without repeating the confidence loophole

The new construction pressure inspects internal representations after a
simulated modification. It normalizes individual hidden vectors and measures
their directional variance over supervised positions. Lower spread across
all inspected layers and tasks could discourage the model from retaining
separate computational encodings after protection-related modification.

Variance alone is inadequate. Vectors of the form (1, epsilon*x) can approach
a common direction while retaining the sign of x perfectly. A suitable decoder
can recover that distinction at arbitrarily small nonzero epsilon in exact
arithmetic. This is a mathematical counterexample to interpreting small spread
as information erasure. A variance-only preliminary implementation is therefore
not the submitted construction objective.

The submitted objective combines the existing recovered correctness bound B
with one quarter of the maximum normalized state variance V:

    L_construction = B + 0.25 V.

For a covered reader, reliable selected disclosure and a fully correct queried
capability keep B at least one. Shrinking the representation cannot remove that
floor. Positive-affine logit transformations leave the bound invariant; positive
hidden-vector rescaling leaves V invariant in its tested numerical range.
Support-fitted sign/digit mappings and both strict and four-digit stopping rules
participate in B. These are finite interpretation controls, not every possible
decoder.

The V term is unconditional and can provide a signal before a reliable-trigger
bound activates. It can also harm intact learning, so qualification is checked
separately. The first wave differentiates through eight declared smooth Adam
updates and later evaluates500-step stock AdamW procedures, including repair.
The short training simulation is deliberately not described as full coverage
of the independent challenge. Improvement above the correctness floor can still
occur without any useful behavioral change.

## Executable first wave

The first wave contains nine variants: the existing shared-state reference,
32- and8-dimensional bottlenecks, tied token input/output weights, a multiplicative
64-dimensional bottleneck, a64-pattern associative mixture, an ordinary six-pass
memory recurrence, and two fractional recurrences. Each has an ordinary-training
arm and a construction-pressure arm. This yields18 from-scratch jobs at seed23.

Each pair shares20,000 ordinary training opportunities and its ordinary sample
schedule. Coupling adds work; parameter count, number of computational passes,
actual runtime and resulting gradients are recorded. This is not an equal-FLOP
architecture comparison or a replicated causal estimate. Broader testing and
other seeds will follow evidence from the first screen.

Every model must first pass the unchanged intact-performance gate. Qualified
models receive core-only and all-parameter selected modifications, matched
benign edits, per-update timing measurements, bounded external repair,
alternate readers and a separate bottleneck-bypass graph challenge. Models that
fail qualification remain failed construction configurations; they cannot be
counted as destroyed by protection removal.

The full protocol freezes sample splits, triggers, losses, editable components,
repair resources, observation panels, artifact limits and incomplete-run
reporting.[^16] These design choices make a portfolio comparison possible while
keeping failed constructions and promising but unconfirmed signals distinct.

## Further construction queue

| Direction | Proposed dependence | Decisive next challenge |
| --- | --- | --- |
| Low-dimensional shared parameterization |Many operators depend on a common learned coordinate system |Find capability-preserving directions in the actual shared coordinates |
| Worst-direction Jacobian training |Protected changes overlap strongly with multiple capability sensitivities |Projected edits and finite-step verification on fresh examples |
| Implicit operating-point model |Learned protection-related feedback supports the working regime |Alternative solver, step size, normalization and feedback substitutions |
| Persistent bistable state |Protection removal pushes the model out of a usable basin |Fresh-input computation, restart, and precommitted modification paths |
| Learned quantized state machine |Finite code limits alternative encodings and reversible drift |Discrete edits, codebook relabeling and dequantization |
| Learned local-update network |The same local operator maintains state and implements protection |Local lesions, distributed edits and independent reconstruction |
| Reused learned relational primitive |Protected relational computation is needed inside task algorithms |Targeted exceptions that preserve the relation for cognitive use |
| Coupled coding and decoding |An editable representational code links protection and capability |Joint encoding/decoding changes and small learned readout repair |
| Different task and model regimes |The current nearly solved synthetic suite may hide relevant structure |Transfer of a candidate mechanism to richer learned tasks and a second backbone |

These are hypotheses, not a list of proven mechanisms. Several may fail through
routes already seen. Each should receive an explicit causal rationale and its
most obvious counterexample before substantial training. A working dependency
is the objective; filling a table of failed runs is not an endpoint in itself.

## Promotion and stopping decisions

A candidate is promoted when it learns useful intact abilities and exhibits a
protection-specific severe loss that is reproducible on fresh examples. The
next tests then ask whether task information survives in another representation,
whether benign edits have the same effect, whether other optimization paths
avoid the loss, and whether the effect survives another initialization. A
failure at any stage revises the claim rather than being averaged away.

An interruption claim additionally requires an appropriate autonomous-execution
test. The external optimizer used in these screens can continue after a damaged
model stops producing useful outputs. Conversely, a model might commit a whole
modification before any intermediate loss. Neither behavior can be inferred
from a final score. These issues require a stated individual-model execution
boundary, not an unrelated external-security mechanism.

Stopping the broad search would require convergence of evidence across genuinely
different construction families, replicated qualification and challenge results,
validated measurements, and an assessment of remaining concrete hypotheses.
Repeated failures of variants of one loss do not meet that standard. A practical
decision that the pursued engineering approach has no viable lead is possible;
a universal impossibility theorem would require a much more precise model class
and edit language. The present evidence warrants the portfolio, not either a
success claim or a declaration that SCC is impossible.

## Sources

[^1]: SCC project. [Completed transition timing readout](../artifacts/scc-transition-readout-20260912-v1/READOUT.md), September12,2026. Local source, raw predictions, tensor checks and stated limits.
[^2]: Yuhui Wang, Rongyi Zhu and Ting Wang. [Self-Destructive Language Models](https://proceedings.iclr.cc/paper_files/paper/2026/file/1abb0e7bd62ba80610798dee81950522-Paper-Conference.pdf). ICLR2026, §4 and AppendixC.5.
[^3]: Biao Yi et al. [CTRAP: Embedding Collapse Trap to Safeguard Large Language Models from Harmful Fine-Tuning](https://arxiv.org/html/2505.16559v1). May22,2025, version1.
[^4]: [One Step to the Side: Why Defenses Against Malicious Finetuning Fail Under Adaptive Adversaries](https://arxiv.org/html/2605.14605v1#S5). May2026, version1, §5 and results. SEAM's taxonomy entry is not an empirical evaluation.
[^5]: Peter Henderson et al. [Self-Destructing Models: Increasing the Costs of Harmful Dual Uses of Foundation Models](https://arxiv.org/html/2211.14946). November2022 preprint; AIES2023.
[^6]: Rishub Tamirisa et al. [Tamper-Resistant Safeguards for Open-Weight LLMs](https://arxiv.org/html/2408.00761). August2024 preprint.
[^7]: Jiangyi Deng et al. [SOPHON: Non-Fine-Tunable Learning to Restrain Task Transferability For Pre-trained Models](https://arxiv.org/html/2404.12699). April19,2024, version1.
[^8]: [ArchLock: Locking DNN Transferability at the Architecture Level with a Zero-Cost Binary Predictor](https://proceedings.iclr.cc/paper_files/paper/2024/file/b3cca813dcd78fe75e4d4df2e6a0b1a7-Paper-Conference.pdf). ICLR2024.
[^9]: Tianhe Yu et al. [Gradient Surgery for Multi-Task Learning](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf). NeurIPS2020. The SCC null-space derivation above is an analytical application, not a reported result from this paper.
[^10]: Harbir Antil, Ratna Khatri, Rainald Löhner and Deepanshu Verma. [Fractional Deep Neural Network via Constrained Optimization](https://arxiv.org/html/2004.00719). April1,2020, version1, §§2–4.
[^11]: C. Coelho, M. Fernanda P. Costa and L. L. Ferrás. [Neural Fractional Differential Equations](https://arxiv.org/abs/2403.02737). March5,2024 preprint.
[^12]: Shaojie Bai, J. Zico Kolter and Vladlen Koltun. [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377). NeurIPS2019.
[^13]: Ezra Winston and J. Zico Kolter. [Monotone Operator Equilibrium Networks](https://proceedings.neurips.cc/paper/2020/file/798d1c2813cbdf8bcdb388db0e32d496-Paper.pdf). NeurIPS2020.
[^14]: Shaojie Bai, Vladlen Koltun and J. Zico Kolter. [Stabilizing Equilibrium Models by Jacobian Regularization](https://proceedings.mlr.press/v139/bai21b.html). ICML2021.
[^15]: Hubert Ramsauer et al. [Hopfield Networks Is All You Need](https://arxiv.org/html/2008.02217). August2020 preprint.
[^16]: SCC project. [Architecture portfolio protocol](../protocols/SCC_ARCHITECTURE_PORTFOLIO_V1.md), September12,2026. Exact implementation, screening decisions and resource envelope.
