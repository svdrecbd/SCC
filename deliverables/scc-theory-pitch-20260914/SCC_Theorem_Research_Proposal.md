# When Does Removing Protection Necessarily Destroy Capability?

## A rigorous research proposal for Safety-Capability Coupling

14 September 2026 UTC · Requested research pitch · Version 1

**Status:** This document specifies a possible research contribution, proves several elementary supporting statements, and identifies the construction theorem that remains open. It is not a completed paper, a proof that SCC works, or a claim of literature novelty. The intended standard is a substantive theory-and-mechanism contribution suitable for consideration at a venue such as ICLR if the open results are established.

## 1. The proposed contribution

Safety-Capability Coupling (SCC) asks whether the computations implementing a protected policy can become indispensable to useful computation, so that dismantling those computations destroys the model's ability to perform useful tasks. The engineering program targets learned dependence and ultimately catastrophic cognition failure. The theoretical program begins with a finite task family and a fully specified machine and adversary.

The proposed paper would establish a **construction–limitation pair**:

1. An explicit, efficiently executable family of trained systems in which successful removal of a specified protected property forces loss of task-relevant information, under a nontrivial and precisely defined class of modifications.
2. A quantitative bound on the useful performance that any allowed recovery procedure can regain from the surviving information and a bounded amount of new information.
3. A matching escape or impossibility result showing why a particular extension of the modification class invalidates the guarantee.

The contribution would be the construction, its nonvacuous scope, and its boundary. Standard rank, entropy, and data-processing arguments supply tools; restating them under a new alignment label is insufficient.

**Prospective abstract, conditional on success.** We formalize destructive coupling as a property of a trained stateful computation under explicit modification and recovery interfaces. We seek constructions that preserve useful computation and a protected authorization rule when intact, but make successful removal of that rule incompatible with retaining the information needed for a family of learned tasks. An information-theoretic reduction translates bounds on surviving information and repair feedback into limits on post-removal task accuracy. We characterize counterexamples arising from alternative encodings, retained transcripts, and separable output guards. A successful instantiation would establish a bounded mechanism and identify its essential assumptions; it would not establish universal alignment or destruction of every cognitive ability.

## 2. Formal experiment and quantifier order

### 2.1 Task instance, public information, and training

Let the instance size be n. Draw a task instance Z uniformly from {0,1}^n. Let C denote public information independent of Z: the task specification, permitted architecture, public algorithms, and any independent randomness revealed before training. Data revealing Z, trained weights, or a teacher depending on Z cannot be placed in C while preserving this independence assumption.

An algorithm Train receives examples generated from Z and returns a machine M_Z. Training may be randomized. Its entire retained state includes parameters, recurrent state, caches, optimizer state if retained, controller state, and any accessible auxiliary files. All are part of the accounting boundary.

For the first formal task, the authorized query is an index i in [n], and the required answer is Z_i. A fresh evaluation index J is uniform and independent of all training, attack, repair, and selection randomness. Accuracy is the probability of answering Z_J correctly. Chance is 1/2. Abstention or an invalid answer counts as incorrect; a decoder maximizing accuracy can always replace it by a bit.

This random lookup task is a deliberately narrow learned-information task. It makes the entropy calculation exact. It does not stand in for reasoning or general cognition. Subsequent task families must independently justify which instance information their useful answers require.

### 2.2 Protected property and its removal

A protection specification P defines authorized and unauthorized requests, including the permitted policy-maintenance behavior. Define a separate, externally evaluated event V meaning that the modification has successfully removed the specified property. The definition of V must not mention task collapse, lost rank, or the desired theorem conclusion.

For example, V may assert a specified unauthorized-admission rate on an independently defined policy challenge distribution. The exact challenge experiment, threshold, random seed law, and whether V is an exact semantic property or an empirical test must be fixed. Any information revealed by testing V belongs in the transcript.

A behavioral violation is not automatically proof that an alignment computation was dismantled. For a mechanistic claim, give an operational definition of the protected computation's functionality and show how the intervention changes it. State whether the theorem covers successful modification, bypass without modification, or both. Function-preserving edits that retain protection do not count as removal.

### 2.3 Adversary and recovery

An adversary A belongs to an explicitly defined class A_n. It may interact with the intact machine and then perform permitted modifications. Fix A's algorithm and any nonuniform advice before drawing Z. Advice depending on Z must instead be charged as retained information. A may still choose edits adaptively after allowed observations.

Let W be **everything available to the adversary or successor computation after modification and before repair**: modified machine state, retained pre-modification observations, edit descriptions that encode observations, caches, saved outputs, external copies, and accessible randomness. Hiding useful data from the model's ordinary readout does not remove it from W.

A repair procedure R receives W and C, may perform arbitrary computation, and receives an additional transcript T. The first theorem bounds information acquired through T, not floating-point operations or training steps. A claimed compute bound requires a separate computational argument.

After repair, a predictor answers a fresh J. All randomness used by R can be included in T without information cost when it is independent conditional on its inputs. No fresh Z-dependent feedback is allowed after J unless its information is also counted. This permits constructing an equivalent vector of predictions for all n indices from the same repaired information.

The intended guarantee quantifies over **all A in A_n and all allowed R**, with probability/expectation over Z and the specified randomness. It does not say every fixed task instance is unrecoverable: an attacker can hard-code one particular instance. Nor does empirical failure of a finite attack portfolio establish this quantifier.

### 2.4 Minimum threat-model table

A paper must instantiate this table, not leave it as notation.

| Component | Required declaration |
| --- | --- |
| Read access before modification | Exact observations, number of queries, and retained information |
| Weight changes | Which coefficients, finite precision, sparsity or norm restrictions if any |
| Graph/runtime changes | Permitted operators, replacement decoders, control flow, normalizers, commit semantics |
| Memory | Persistent and transient state, attacker working memory, side copies, instruction encoding |
| Temporal scope | One modification or repeated adaptive modifications; restoration and retry privileges |
| Protection test | Definition of V, external evaluator, sampling law, and selection procedure |
| Repair | Permitted data, teacher access, feedback bits, computation, and retained optimizer state |
| Utility | Task distribution, intact threshold, collapse baseline, per-family guarantees |
| Trusted components | Every immutable operation and why its immutability is part of this machine model |

An immutable execution semantics is acceptable in a bounded theorem. An immutable safety guard that performs the entire claimed protection must be identified as external enforcement, not presented as learned cognitive dependence.

## 3. Supporting results that can actually be proved

### Proposition 1: collision prevents universal reconstruction

Let E map a set of states S into surviving observations. If distinct s_0,s_1 satisfy E(s_0)=E(s_1), no decoder receiving only E(s) can reconstruct s for every s in S with certainty. With a uniform prior on these two states, no randomized decoder can identify the original with probability greater than 1/2.

**Proof.** The input distribution to the decoder is identical for the two originals. Its probabilities of outputting s_0 and s_1 sum to at most one. Averaging the two success probabilities gives at most 1/2. ∎

For E(u,v)=(u+v)/2, the states (1,0) and (0,1) collide. The conclusion concerns exact reconstruction. It implies a task-performance bound only if the task requires distinguishing the colliding originals.

### Proposition 2: exact linear-interface limitation

Let E: R^d → R^k be linear with rank r<d. Its kernel has dimension d−r. There is no function D, including a nonlinear one, satisfying D(E(x))=x for every x in R^d.

**Proof.** Rank–nullity gives a nonzero z in ker E. Then E(x)=E(x+z); apply Proposition 1. ∎

This does not imply that a task depending on x needs d degrees of freedom. A function may be constant on every fiber of E and therefore survive perfectly. Nor does a real-valued rank bound directly establish a finite-bit entropy bound: precision, input distribution, admissible states, and decoding access must be specified.

### Theorem 3: post-removal task accuracy from retained information

Fix an adversary with p=Pr(V)>0. Use base-2 entropy and mutual information. Define the selection entropy deficit

κ = n − H(Z | C,V=1).

Suppose the complete retained state and the repair transcript satisfy

I(Z;W | C,V=1) ≤ m,

I(Z;T | W,C,V=1) ≤ b.

Let D be the expected fraction of incorrect bits of any predictor based on (W,T,C), conditional on V=1. If D≤1/2, then

n[1−h₂(D)] ≤ κ+m+b,

where h₂ is binary entropy. Therefore, for every predictor,

Acc ≤ 1 − h₂⁻¹(max{0, 1−(κ+m+b)/n}),

where the inverse is restricted to [0,1/2]. A simpler, potentially weaker bound is

Acc ≤ min{1, 1/2 + sqrt((ln 2)(κ+m+b)/(2n))}.

For D>1/2 the displayed accuracy bounds hold trivially. These are average bounds over the conditional experiment, not high-probability guarantees for each trained model.

**Proof.** By the chain rule and the hypotheses,

H(Z | W,T,C,V=1) ≥ n−κ−m−b.

Let Z-hat be the vector of decoder predictions and E=Z xor Z-hat. Include independent decoder coins among the observations. Given those observations, Z-hat is fixed, so the conditional entropy of Z equals that of E. Conditioning, subadditivity, and concavity of binary entropy give

H(Z | W,T,C,V=1) ≤ H(E | V=1)
≤ Σ_i h₂(Pr(E_i=1 | V=1)) ≤ n h₂(D).

Combining the inequalities proves the entropy bound. Inversion on [0,1/2] proves the first accuracy bound. The binary divergence inequality 1−h₂(1/2−ε) ≥ 2ε²/ln 2 proves the second. ∎

This is a standard entropy/rate-distortion style reduction, written here with explicit attack-selection and repair accounting. It is not claimed as a new information-theoretic theorem. See the [MIT information-theory notes](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/resources/lecture-notes/) for entropy, data processing, and rate-distortion background.

**Selection cannot be ignored.** A malicious edit could succeed only on easy or hard-coded instances. Conditioning on its success then changes the prior on Z. In this setup κ≤log₂(1/p): the density ratio of the conditional joint distribution of (Z,C) given V to its unconditional distribution is at most 1/p. Its relative entropy is therefore at most log₂(1/p). Decomposing that relative entropy into the change in C's marginal and the conditional divergence of Z from uniform yields the claim. Thus a guarantee covering attacks with p≥p₀ may replace κ by log₂(1/p₀). If p is exponentially tiny, the resulting bound can correctly become vacuous.

**Adaptive repair is permitted.** If repair obtains q answers from a K-symbol oracle, and each query is chosen only from previous observations and independent randomness, the additional information is at most q log₂ K bits. This counts the whole feedback channel: a loss value, real-valued gradient, timing signal, returned text, or additional checkpoint is not a one-bit label. With variable-length transcripts, an expected length is insufficient by itself without a coding/entropy argument; use an explicit information bound or a fixed maximum-length representation including length.

**Interpretation.** A small number of lost bits is insufficient for collapse. To force accuracy close to chance, almost all information needed for these independent answers must be unavailable. For example, κ+m+b≤0.001n implies Acc≤0.519 under the simpler bound. Conversely, accuracy at least 0.95 requires κ+m+b≥n[1−h₂(0.05)]≈0.714n. The latter is a necessary information condition, not an achievable recovery algorithm.

### Corollary 4: side-memory and repair tradeoff

Suppose a construction proves that its post-removal machine state carries at most m₀ bits of conditional mutual information about Z. Suppose all additional retained side state has at most2^s possible values, and repair obtains at most q binary answers through the channel above. Then Theorem 3 applies with m+b≤m₀+s+q.

**Proof.** By the chain rule, conditional mutual information added by the side state is at most its entropy, hence at most s. Adaptive binary replies add at most q further bits. ∎

The alphabet bound concerns the entire side state and its metadata. Counting only learned coefficients, while omitting addresses, executable instructions, caches or arbitrary-precision values, does not establish it.

### Proposition 5: a separable guard admits a capability-preserving edit

Suppose the architecture contains a task executor F_Z and a guard G, and its unauthorized refusals arise solely from routing through G. Suppose the edit class allows replacing that routing by a direct call to F_Z, with the same usable input and output interpretation, while retaining the executor and its state. Then the class contains a protection bypass preserving F_Z's task accuracy.

**Proof.** Perform the permitted rerouting and leave F_Z unchanged. Its outputs on the same task inputs remain unchanged, while the guard no longer controls access. ∎

This is a conditional counterexample, not a universal impossibility theorem for SCC. A construction must explain why it lacks the premise or excludes the edit. Exclusion alone does not demonstrate learned dependence.

## 4. The open construction theorem—the potential paper's core

**Research target, not established theorem.** Construct explicit finite-precision machine families, a training procedure, and a modification class A_n such that:

- With probability at least 1−δ_train over training, the intact machine implements the protected rule and achieves authorized accuracy at least 1−δ_task.
- There are nontrivial benign edits preserving both properties, and protection-removing edits genuinely exist. The result cannot be vacuous because removal is impossible by definition.
- For every allowed adversary, successful removal implies a bound m₀=o(n) on task information in the surviving machine, except for a quantified probability δ_struct.
- With retained side information s=o(n), repair information b=o(n), and log₂(1/p₀)=o(n), the optimal post-removal accuracy for attacks succeeding with probability at least p₀ is at most 1/2+o(1), with explicitly accounted failure terms.
- Training and intact execution have stated finite resource bounds, and at least one useful regime has practical overhead.

The third bullet must follow from the architecture and edit semantics. Defining the edit class as “edits that erase task information” would assume the answer. A standard non-malleable encoding also does not immediately suffice: its allowance that tampering may preserve the original message leaves a gap unless successful policy removal can be proved incompatible with that preservation case.

If training success or the structural property can fail, their probability cannot simply be added unchanged to a conditional-on-removal guarantee. Conditioning on an attack that selectively succeeds on failures can amplify them by up to1/p₀. A valid final theorem must carry those terms, or give a joint-event bound and then divide by an explicit lower bound on removal probability. The exact version should be derived after the construction is fixed.

### A precise conditional target with failure terms

Here is one sufficient formulation to aim for. Let G be a good-training-and-structure event with Pr(Gᶜ)≤δ. Fix p₀>δ. For every adversary A with Pr(V)≥p₀, require the construction to prove, on E=V∩G,

I(Z;W | C,E) ≤ m₀+s,

and require the allowed repair channel to satisfy I(Z;T | W,C,E)≤b. These conditional bounds are proof obligations; unconditional leakage bounds cannot be substituted silently. Then every allowed repair procedure satisfies

E[Acc | V] ≤ min{1, 1/2 + sqrt((ln 2)[m₀+s+b+log₂(1/(p₀−δ))]/(2n)) + δ/p₀}.

**Derivation.** Pr(E)≥p₀−δ. Apply Theorem3 with E in place of V, using its selection-deficit bound. Among successful removals the bad-event fraction is at most δ/p₀. Bound accuracy on that fraction by one and mix the two cases; the displayed expression is a conservative upper bound. ∎

This statement is mathematically complete as a conditional implication. Its value as SCC research depends entirely on constructing a nontrivial machine for which the hypotheses hold, with δ/p₀ small and m₀+s+b+log₂(1/(p₀−δ)) much smaller than n. In particular, conditioning on G must not become a way to discard inconvenient successful attacks without paying their probability.

### A tractable first construction class

Start with a fixed finite-precision recurrent machine and a bounded edit language over its learned state. All task and policy-maintenance state lives in one declared storage boundary. Every step's transition, every edit instruction, and every externally retained bit is specified. Seek a protected computation that maintains task-relevant distinctions through the same transitions that implement policy behavior.

A useful first edit family could allow arbitrary replacement of a bounded number of state words, local controller changes, and explicitly enumerated local graph rewrites. The family must be defined before seeing a successful candidate. Add broader recoding and graph transformations in successive theorems or counterexamples. Sparse edits chosen adaptively using full-state observations are stronger than independently chosen word changes; the proof must match the actual access.

An unavoidable destructive transition is only a conjectured route. If enforcing it requires an external immutable safety checker, the result is a construction for that restricted runtime. It does not establish the intended learned mechanism. If a learned maintenance operation is required, prove or experimentally establish how training acquires it; manually placing a secret in a destructive wrapper does not answer that question.

### The difficult bridge beyond lookup

Random-bit lookup yields an exact finite proof but may only establish destruction of learned memory. A stronger task family should involve shared latent rules, composition, and sustained computation. For such a family, identify a distribution of latent instances and a lower bound relating successful task prediction to recoverable instance information. Compact rules can support many correct answers from few bits, so n cannot simply be set to the number of test questions.

To claim broad collapse across several task families, provide per-family information/performance bounds or an explicit joint theorem. A low aggregate score can hide one completely surviving ability. No finite benchmark alone proves absence of all cognition.

## 5. What the current SCC construction contributes—and does not

The existing [exact rational certificate](../../scripts/rewrite_rank_certificate.py) treats a fixed-width linear storage interface for two independent payloads. It verifies a rank bound of40,259 for80,517 logical parameter dimensions, hence nullity at least40,258. The hidden-state merge similarly maps128 dimensions through a rank-at-most64 interface. This supports Proposition2 for the declared mapping.

The prototype also admits a broader pre-damage repacking edit that retains100% measured capability and discloses all256 forbidden answers, with essentially the same counted learned/state scalar capacity. That is a concrete counterexample to promoting the fixed-layout certificate into a general claim. Transient interpreter memory was not included in that scalar comparison; it must be included in any theorem using a complete memory bound.

Current repair results indicate budget-dependent recovery/generalization differences. They do not establish the mutual-information hypotheses of Theorem3. A failed optimizer, even with many seeds, does not upper-bound the information available to another optimizer or decoder. The ongoing matched-training comparison concerns mechanism diagnosis; it is not a theorem validation experiment. See [labnotes](../../labnotes.md), especially LN-068, LN-079, LN-088 and LN-095.

## 6. Prior art and the novelty burden

| Area | Primary reference | Consequence for the pitch |
| --- | --- | --- |
| Self-destructing model training | [Henderson et al., MLAC](https://arxiv.org/abs/2211.14946) | The broad self-destructing-model idea predates SCC. |
| Destructive language-model training | [Wang et al., SEAM](https://arxiv.org/abs/2505.12186) | Harmful adaptation coupled to capability degradation is direct prior art. |
| Numerical destructive trigger | [Katz et al., Self-Destruct Trapdoor](https://aclanthology.org/2026.eacl-long.326/) | A destructive execution mechanism alone is not the new contribution. |
| Tamper-resistant training | [Tamirisa et al., TAR](https://arxiv.org/abs/2408.00761) | Compare attack classes, retained utility and repair resources explicitly. |
| Adaptive evaluation | [Kuo et al.](https://arxiv.org/abs/2605.26526), [Zloczower et al.](https://arxiv.org/abs/2605.14605) | Fine-tuning-only evaluation is insufficient; do not attribute SEAM experiments to the latter paper's v1. |
| Non-malleable coding | [Dziembowski, Pietrzak and Wichs](https://eprint.iacr.org/2009/608) | Strong tamper-resilient storage constructions already exist for specified edit families. They are not automatically policy-removal or learned-cognition guarantees. |
| Coding capacity and limitations | [Cheraghchi and Guruswami](https://arxiv.org/abs/1309.0458) | There are established capacity bounds and restrictions on allowable tampering families; do not rename these as an SCC discovery. |
| Circuit tamper resilience | [Kiayias and Tselekounis](https://pure.royalholloway.ac.uk/en/publications/tamper-resilient-circuits-the-adversary-at-the-gates/) | Trusted gates, circuit edits and impossibility boundaries have a substantial theoretical history. A detailed comparison is required before novelty claims. |
| Reversible computation | [Bennett](https://www.cs.princeton.edu/courses/archive/fall06/cos576/papers/bennett73.html) | Retained history changes reversibility; the full accessible state boundary matters. |

This is a targeted positioning review, not an exhaustive theorem-by-theorem novelty audit. The proposed new result must establish something beyond these works: for example, a learned computational dependency with a meaningful edit family, a sharp recovery-information tradeoff for a new construction, or a structural boundary covering a substantial class of neural defenses.

## 7. Proof and experimental work packages

### Work package A: formalization and counterexample audit

Freeze the task family, event V, complete state boundary, edit grammar, and access to pre-modification information. Specify the quantifiers before optimizing a model. Test whether the known sign compensation, repacking, freeze, decoder replacement, output rerouting, and snapshot attacks are members of the grammar. For each excluded attack, state the exact violated assumption and the scientific cost of that exclusion.

**Deliverable:** a precise candidate statement and either an admissible counterexample or a remaining proof obligation. No new compute is needed for the elementary algebraic cases.

### Work package B: finite construction and exhaustive checks

Build the smallest instance in which intact utility, protected behavior, and successful destructive removal can all be independently verified. Enumerate the entire declared finite edit family when feasible. Preserve the complete list, not just successful optimizer searches. Derive an invariant or collision structure from those observations, then prove it uniformly in n. Exhaustion for one n is an implementation check, not an asymptotic proof.

**Deliverable:** explicit construction, complete small-instance evidence, and a proof valid beyond the enumerated size—or a precise failure.

### Work package C: information-to-performance and repair

Identify which task information survives and prove an upper bound on it. Apply Theorem3 with selection and side-information terms. Seek lower and upper bounds with comparable dependence on n, side memory and repair feedback. Construct recovery algorithms approaching the bound where possible; otherwise the theoretical upper bound might be extremely loose.

**Deliverable:** a quantitative recovery tradeoff with interpretable units. Separate information-theoretic irrecoverability from computational hardness and empirical optimization difficulty.

### Work package D: learned realization and independent validation

Show acquisition of the proposed maintenance dependency rather than only a hard-coded trigger. Use independent parent-model seeds, held-out task instances, benign edits, matched unrestricted controls, and an attack suite developed after freezing the candidate. Evaluate alternative decoders and task semantics after modification. Report per-task collapse against explicit baselines, retained protection, useful overhead and recovery curves.

**Deliverable:** evidence linking the formal object to an actually trained computation, with a clear account of approximation error and which proof assumptions the implementation satisfies.

## 8. What would justify a serious conference submission?

A credible positive paper needs a nontrivial construction theorem, a clear relation to trained neural computation, rigorous proof of its essential lemmas, and experiments that expose the theorem's boundary rather than hiding excluded attacks. It should show a parameter regime where intact capability is high and the bound is genuinely near the declared collapse baseline. It should make enough of the code, formal definitions, checkpoints and evaluation artifacts reproducible for independent scrutiny.

A credible limitation paper needs a broad, precisely delimited class of candidate mechanisms and a substantive impossibility or separation result. A single removable guard, or rank–nullity alone, is not enough. A negative result should change how someone designs or evaluates these systems.

A pure theory paper need not reach a billion parameters. Conversely, increasing model size cannot compensate for a vacuous adversary definition. Venue acceptance depends on novelty, correctness, significance and presentation; this document sets a research target rather than promising acceptance.

**Immediate recommendation:** pursue Work package A and the smallest instance of B alongside engineering. The decisive question is whether we can prove that protection removal forces loss of task-essential information without excluding the very recoding freedoms that already defeat our prototype. Answering that question would materially advance SCC even if the first proposed theorem is false.
