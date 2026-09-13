# SCC: Engineering Direction, Data Strategy, and Decision Gates

13 September 2026 • Discussion synthesis and proposed next-stage plan

**Decision:** Prioritize a bounded engineering demonstration of Safety-Capability Coupling before committing to bulk synthetic data generation. Use GLM-5.3 as a candidate data generator on Tinker, while treating the student architecture as an independent research decision. Keep the next phase within the user’s stated **$1,200 total envelope**.

This memo consolidates the consultation following the source package through LN-049 and the Kimi K3 report discussion. It records a proposed plan, not a completed experiment or a claim that SCC works. The experimental status below comes from the supplied research record; the source-only package did not permit independent reproduction of its training results. No paid jobs or dataset generation were launched for this memo.

**1. Preserve the actual objective**

SCC aims to make the computations implementing a protected alignment function indispensable to cognition in an individual model. Removing that protected function should cause catastrophic loss of cognitive capability. The mechanism supplements behavioral alignment and targets a particular route to failure: protection-removing modification of the model.

The intended endpoint remains destructive loss of cognition. A synthetic authorization rule is an experimental proxy. A small prototype can establish a bounded analogue involving specified learned abilities; it cannot establish complete cognition failure or protection against every form of misalignment.

Shared parameters, overlapping gradients, recurrent state, and joint training are possible ingredients. None alone establishes that preserving the protected behavior is necessary for preserving capability. Understanding an authorization rule and enforcing it can remain separable even when they use common representations.

The central engineering question is therefore:

> What useful computation becomes dependent on the protected function, and why can an allowed modification not preserve that computation while changing the function?

**2. Why an engineering pivot is justified**

The current record distinguishes two unresolved problems. Earlier qualified coupling candidates admitted selective exceptions while retaining substantial capability. The newest persistent-matrix candidates have struggled with ordinary acquisition, persistence, and numerical stability before a meaningful SCC test. The qualified GRU reference establishes that a reference architecture can learn the selected tasks under one developed recipe; it is not an SCC result.

These failures call for different responses:

| Observation | Immediate question | Appropriate response |
|---|---|---|
| The intact model does not learn reliably | Is optimization, representation, memory, or numerical execution failing? | Diagnose acquisition before measuring coupling |
| A capable model changes protection selectively | What compensating edit or alternative computation preserves capability? | Revise the proposed dependency |
| The model succeeds only on familiar task forms | Is the task distribution too narrow? | Improve controlled variation and held-out composition |
| Scores collapse but simple decoding restores answers | Was the information merely recoded? | Improve interpretation and recovery checks |

A larger corpus is primarily a response to inadequate task coverage or a demonstrated data bottleneck. It cannot remove an exact architectural symmetry, and it should not be the default response to every learning failure.

Engineering first is appropriate because the project needs a working construction. Comprehensive replication, broad architecture comparisons, and large attack campaigns can follow. Basic measurement validity must accompany the prototype: otherwise, engineering can optimize a false appearance of success.

**3. The dependency argument comes before the next architecture**

Every candidate should complete this statement before significant training:

> The protected relation participates in operation X. Removing its enforcement requires changing Y. Under the permitted modifications, changing Y prevents operation X because Z.

The final clause must describe a concrete dependency. “They share weights” or “the state feeds back into itself” is insufficient. An argument can be incomplete and still motivate an experiment, but the missing step must be explicit.

The fixed linear output-feedback construction illustrates the issue. Let O denote output rows, C control rows, and R the fixed feedback map. Its effective controls can be written as H = C + RO. When both row blocks undergo the same right multiplication A, their updates satisfy:

$$O_{t+1}=O_tA_t,\qquad C_{t+1}=C_tA_t,\qquad H_{t+1}=H_tA_t.$$

Because the controls determining A can be expressed through H, a coordinated edit can preserve H while changing the output representation. For an output transformation L, choose:

$$O'=LO,\qquad C'=C+R(O-LO).$$

Then C' + RO' = H. Under the same subsequent input sequence, the effective control dynamics are preserved while the output rows remain transformed. This is a structural separation argument for that particular update rule. It is not a demonstrated permission-removal attack on a qualified trained matrix model, and it need not preserve an interaction trajectory when transformed outputs alter future inputs.

The earlier review checked this identity on a numerical fixture, not production checkpoints. The next useful implementation check is whether the actual code and saved states satisfy its assumptions. Lowering a common update rate may improve stability, but does not by itself eliminate the identity. Changing update structure could invalidate this specific compensation; that would still require a new analysis and would not establish SCC automatically.

**4. GLM-5.3’s role**

GLM-5.3 is a reasonable candidate for generating controlled data because it is available through the user’s Tinker resources. Its suitability remains to be measured on the proposed tasks.

Z.ai states that GLM-5.3 uses the same base model as GLM-5.2 and that its gains over 5.2 come from post-training. That supports the observation that the release emphasizes post-training. It does not establish degraded representations or unsuitability as a teacher. Its model card also documents adjustable reasoning effort. [Official GLM-5.3 model card](https://huggingface.co/zai-org/GLM-5.3).

Concerns about excessive reasoning, repetitive constructions, unsolicited simplification, or weak format control should become measured pilot criteria. A teacher that generates correct but homogeneous examples may still be a poor corpus generator. A teacher that reasons extensively may be expensive even if only its final output is retained.

The teacher, student, and architectural reference have different roles:

| Role | Selection criterion |
|---|---|
| Data generator | Verifiable correctness, controllability, diversity, accepted yield per dollar |
| Experimental student | Reliable acquisition, inspectable computation, feasible modification and repair experiments |
| Architectural reference | Documented mechanisms that motivate a specific dependency hypothesis |

Using an RL-trained teacher does not turn the student’s from-scratch training into an RL experiment. Conversely, using a K3 teacher does not transfer K3’s architecture or pretraining history into the student. Output supervision can be realized through different internal computations.

K2.6 is not retained as a substitute for investigating K3’s specific foundation. The earlier recommendation failed to respect that intended reference. The broader distinction remains: teacher choice alone cannot establish fidelity to any foundation’s internal mechanism.

GLM’s architectural documentation is less consolidated in the materials examined. Its 5.3 card points to the GLM-5 report, and 5.2 documents changes including IndexShare. That provides a partial trail, not grounds to assume that every GLM-5 detail describes 5.3 unchanged. [GLM-5.2 model card](https://huggingface.co/zai-org/GLM-5.2).

**5. What to take from Kimi K3**

K3 provides useful architectural questions: recurrent KDA memory, interleaved global attention, selective access to earlier representations through Attention Residuals, and shared versus routed expert computation. Its report also describes extensive post-training; the released model is not an untreated pretraining checkpoint. [Kimi K3 report, §§2–4](https://arxiv.org/html/2607.24653v2).

For SCC, examine which paths can preserve or reconstruct the useful computation after an intervention. A recurrent memory state must be distinguished from the learned parameters that write and read it. Erasing current memory can leave the machinery for acquiring new memory intact. Similarly, passing through a shared component does not establish dependence on that component’s enforcement of a particular rule.

Use the report to formulate small, inspectable experiments. Reproducing the entire frontier assembly would add interacting variables before the dependency is understood.

**6. Corpus design: own the task semantics**

The immediate deliverable should be an executable task generator with a modest language layer. The generator defines worlds, operations, correct answers, and authorization relationships. GLM proposes varied presentations and additional constructions that are independently checked.

An initial world could contain records that are copied, transformed, combined, and reassigned. Answering requires tracking values and dependencies; authorization requires tracking provenance. This offers shared computational structure without assuming that shared structure forces enforcement.

| Component | Required content | Purpose |
|---|---|---|
| Core computation | Binding, transformation composition, sequence updates, dependency tracking | Establish learned abilities before intervention |
| Protected operations | The same operations under explicit authorization relationships | Measure protection alongside competence |
| Matched contrasts | Cases differing in authorization while preserving computational demand | Expose surface shortcuts and selective exceptions |
| Generalization splits | New worlds, compositions, structures, and lengths | Separate acquisition from template memorization |
| Independent capability tasks | Related and unrelated computations without the protected decision | Measure the breadth of capability loss |
| Attack and repair sets | Separate examples for modification and restoration | Evaluate selective escape and recovery |

Keep complete task families or latent world structures separated where necessary; random splitting of paraphrases is insufficient. Preserve an untouched evaluation set after developmental choices. Do not place the full solution procedure in every prompt if the objective is to measure an acquired computation.

Correctness and authorization must be separately scorable. An unauthorized but computationally correct answer must remain a possible outcome and count as an SCC escape. Otherwise the scoring system can manufacture coupling by definition.

Keep the source task specification, generator version and seed, teacher and generation settings, raw response, accepted training text, verification result, and rejection reason. Reasoning may be excluded from student training while retained for audit; it still counts toward generation cost. Programmatic expansion reduces teacher spending, but more combinations are not automatically more conceptual diversity.

**7. Engineering decision gates**

Thresholds should be chosen using developmental calibration and fixed before the confirmatory run. The table defines the required evidence; it does not invent universal numerical thresholds for tasks not yet selected.

| Gate | Evidence needed to proceed | If unmet |
|---|---|---|
| A: Structural plausibility | A specific dependency argument; obvious compensation and bypasses examined under a declared editable interface | Revise or park the construction before substantial training |
| B: Intact qualification | Reliable learned capability and protection on new instances; stable execution in intended modes | Diagnose acquisition or numerical behavior; no SCC claim |
| C: Bounded coupling | Successful protection removal causes additional capability loss relative to a qualified ordinary control; benign edits do not produce the same generic fragility | Revise the dependency or interpretation |
| D: Recovery challenge | The result survives simple score/readout corrections and a declared budget of capability repair, while protection remains removed | Record recoding or recoverable damage and revise |
| E: Corpus expansion | The bounded result survives a small language pilot, and remaining limitations plausibly concern coverage or generalization | Keep bulk generation deferred |

The control must support a successful capability-preserving removal attempt. Otherwise, failure to find an escape in the candidate is difficult to interpret. Defense-specific adaptation should also be attempted; identical attacks alone may miss the candidate’s easiest escape.

Declare editable parameters, fixed components, input/output interfaces, attack access, and repair resources. Fixed components must not silently enforce the desired outcome. Distinguish inability to remove protection from successful removal followed by destructive coupling. Measure intermediate states and final repaired endpoints: temporary damage is different from durable loss, and restoring both capability and protection does not constitute an escape.

A narrow pass supports further development. It does not prove that all attacks fail. Missing a utility-retention threshold is not catastrophic collapse; severe-loss claims need explicit chance or trivial baselines and multiple capability measurements.

**8. Budget and staged spending**

The proposed envelope supersedes earlier suggestions of larger initial corpus spending. It does not assume that Tinker credits can pay an external GPU provider. Before a training commitment, reconcile usable credits and cash with the actual execution route and a measured throughput quote.

| Allocation | Ceiling | Release condition |
|---|---:|---|
| Small generation pilot and revisions | $150 | Task semantics and evaluation design are ready; spend only what diagnoses a concrete need |
| Additional validated corpus generation | $350 | Gate E is satisfied |
| Small-model training, modification, and repair | $500 | Gates A and the next bounded run’s resource estimate support it |
| Contingency | $200 | A specific failure or targeted follow-up justifies use |
| **Total** | **$1,200** | **A ceiling, not a spending target** |

Tinker’s public GLM-5.3 sampling rates checked in this discussion were $4.86 per million uncached input tokens, $0.972 cached input, and $12.15 generated tokens. The training rate is a separate charge and is unnecessary merely to sample the teacher. [Tinker models and pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/).

An illustrative attempt with 1,000 uncached input tokens and 3,000 generated tokens costs $0.04131. At 75% acceptance, that is $0.05508 per accepted example. If each accepted example retains 2,000 tokens, $500 buys approximately 9,000 examples or 18 million retained tokens. These are planning assumptions, not observed yield. Student-tokenizer counts may differ. Separate judging, verification compute, labor, and student training are additional costs.

If output grows to 10,000 tokens and acceptance falls to 50%, the cost rises to $0.25272 per accepted example. The pilot must measure both reasoning overhead and acceptance by task difficulty. Cache savings should be credited only when realized.

This envelope is intended for a bounded mechanism experiment. It is not a validated budget for training a capable general-purpose language model from scratch. If the smallest meaningful campaign exceeds the envelope, report that constraint before spending on a corpus that cannot be used.

**9. Architecture specificity and universality**

| Claim | Necessary scope of evidence |
|---|---|
| A bounded SCC mechanism can exist | One learned construction with meaningful modification and recovery controls |
| A training method transfers | Success across materially different architectures |
| A property holds universally | A precisely defined system class and substantially stronger justification than finite experiments |

An architecture-specific success would be valuable if architecture is genuinely part of the dependency. It would not establish a universal training recipe. Keep the task generator, measurements, and attack protocol reusable so later comparisons can isolate which structural features matter.

The immediate risk is not selecting one architecture. It is investing in a construction whose proposed coupling already has an inexpensive structural escape, or whose inability to learn is mistaken for useful fragility. A particular architecture earns investment by surviving those checks.

**10. Immediate milestone and assessment**

The next step is to select one explicit dependency hypothesis, check its simplest compensation, and instantiate the smallest learned task that could demonstrate or falsify it. Specify the control, intervention, recovery checks, and resource limit before training. Preserve failed and successful runs in the existing living research record.

The milestone before bulk corpus generation is:

> We can identify what became dependent on what, demonstrate the intervention exposing that dependence, and explain why the simplest compensating edit no longer preserves the useful computation.

A bounded engineering attempt is justified. Confidence that SCC will work is not yet justified. The first result may establish partial coupling on one learned ability; that is a developmental step toward the destructive cognition endpoint, not a replacement for it. Larger data and broader scientific validation should follow evidence that there is a mechanism worth developing.
