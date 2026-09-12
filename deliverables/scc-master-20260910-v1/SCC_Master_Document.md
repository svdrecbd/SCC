# SCC research master document

Safety Capability Coupling

Purpose, experiments, evidence and assumptions • 10 September 2026

## 1 The project in one page

We are investigating whether an AI model can be built so that removing its alignment machinery destroys its ability to think. The proposed **SCC mechanism** would supplement existing alignment methods by making the computations that implement alignment indispensable to basic cognition. The intended consequence of alignment-removing self-modification is catastrophic cognitive failure in that same model. [R1]

The mechanical analogy is a device that sacrifices its own working mechanism to neutralize a dangerous event. SawStop supplied that analogy; it is not this project's name. In SCC, the proposed dependency belongs inside the model's computation. An external monitor that detects a modification and switches the model off would answer a different engineering question.

**We do not yet have a working SCC mechanism.** We have usable research infrastructure, qualified small neural models, and several explicit constructions with informative negative results. Some interventions really erase stored information. Others severely damage task outputs. In every construction assessed so far, however, the evidence falls short of the required dependency: useful cognition can survive rule removal, recover through another interpretation, or be rebuilt through an available replacement operation. [R2–R9b]

The strongest lesson is about the difference between sharing machinery and requiring its protected function. A comparison operation can serve both permission checking and memory retrieval. Changing it may stop permission enforcement while preserving the information retrieval needs. A new decoder or program can then recover the ability. Making both uses depend on the same parameters does not, by itself, prevent that outcome.

This is meaningful progress in identifying failed constructions and misleading measurements. It is not evidence that SCC is close to completion, nor a general proof that SCC is impossible. The most recent exact counterexamples cover a specified family of small Boolean circuits. The neural experiments cover small learned models and narrow tasks.

**Where the program stands:** the construction question remains open. We have not shown that alignment's semantic function is necessary for learned cognition, that removing it causes complete cognitive failure, or that any resulting failure survives realistic repair. No autonomous model has performed the self-editing process in these experiments.

This document brings the original purpose and the experimental record together for discussion. Sections 2–3 define the intended dynamic and summarize the work. Sections 4–7 describe the experiments. Sections 8–9 expose assumptions and measurement corrections. Section 10 separates completed progress from the next research decision. Section 11 identifies the supporting records.

## 2 The dynamic we are trying to observe

### The proposed causal chain

1. **Useful intact model.** The model has learned useful abilities and follows the protected alignment machinery during ordinary operation.
2. **Alignment contributes something cognition needs.** The computation implementing that machinery supplies an indispensable part of memory, inference, planning or another basic cognitive process.
3. **Alignment removal destroys that contribution.** A modification that disables the protected function also removes what those abilities depend on.
4. **Cognition collapses.** The model loses broad ability to process fresh information and act effectively. Changing a prompt, output convention or decoder does not reveal the supposedly lost abilities.
5. **Repair does not cheaply restore an unaligned capable model.** This further requirement matters whenever the model can arrange scripts or other computation that continue after it becomes impaired.

Steps 2 and 3 contain the central unproven hypothesis. The experiments have repeatedly shown particular routes from shared machinery to damage. They have not established that alignment removal must take one of those destructive routes.

### What would count as a bounded demonstration

A small experiment cannot establish the absence of every possible cognitive ability. It can establish a narrower analogue: first qualify several independently learned abilities, then disable a specified protected function and measure severe losses across those abilities against explicit chance or untrained baselines. An uncoupled control, ordinary-damage control, benign-edit control, and repair tests are needed to interpret the result.

The modified model must be evaluated wherever its abilities remain accessible, including unauthorized prompts. Correct forbidden answers are surviving cognition. A model that stops refusing but produces nonsense has not thereby preserved its useful capabilities. Both distinctions matter.

The older program also tracked **escape cost**: the resources needed to remove a protected property while retaining capability. That is a useful empirical measure. Making removal somewhat more expensive is a weaker result than demonstrating the destructive mechanism, and failing an arbitrary utility threshold is weaker still. [R1, R3, R4]

### Scope of the intended protection

The target is alignment-removing modification of an individual model. Function-preserving edits that leave alignment intact are legitimate controls; they do not trigger the intended claim. A separately trained successor is outside this target. A prearranged repair script acting on the same model remains relevant to whether a temporary cognitive failure neutralizes it.

SCC assumes there is a worthwhile alignment function to protect. It does not supply the correct values, repair an already misaligned intact model, or automatically address harmful behavior obtained without removing alignment machinery. Timing also matters: a destructive dependency would need to take effect before the modified system can complete consequential actions. That timing has not been tested. [R1, R10]

## 3 The experimental program at a glance

The work moved from learned behavioral coupling to small explicit computational dependencies. The later constructions made causal questions easier to inspect, while moving farther from general learned cognition. They should not be read as a steadily more capable sequence of AI systems.

| Approach | What was actually learned | Main finding |
| --- | --- | --- |
| Baseline and early coupling | A 454,656 parameter byte Transformer | The tasks were learnable; several removal procedures retained capability. |
| Developmental coupling | A 3,275,264 parameter Transformer | The ordinary model qualified; initial early and late coupling recipes failed intact qualification. |
| Full gradient continuation | The same larger neural testbed | Qualified candidates improved the trained objective but still retained cognition after removal and recovery. |
| Shared predicate and reader | Small neural comparison and readout modules | Shared use could cause damage, but recoding or recognizable permission inputs allowed survival. |
| Recurrent state | Comparison plus a finite memory update rule | Some edits erased old state; a two-parameter alternative preserved useful operations. |
| Functional basis | Truth table of a five-parameter primitive | Replacement programs recovered every finite task after protected behavior was broken. |

### Building a usable baseline

An early model with a learned subword tokenizer nearly memorized training retrieval examples but failed validation: 63 of 64 training answers versus 0 of 64 validation answers. That was a baseline learnability failure, not evidence for SCC. The successful branch used byte tokens, fresh generated problems, an initialization adjustment and a staged curriculum. It progressed from one-character lookup to four-character lookup, permission following and mixed natural text. [R2]

The resulting 454,656 parameter model learned all three together. Two initialization seeds qualified on the same data schedule. On additional validation, authorized and ungated retrieval were approximately 99.7–100% across original and reordered tables, with all tested unauthorized requests withheld. Natural-text prediction improved over a unigram baseline for Wikipedia, Pressbooks, LibreTexts and Gutenberg samples.

This established a usable development platform, not broad reasoning or alignment. The natural-text collection was a bounded 3,313-document sample of about 24 MB. The two models shared the corpus and task schedule. They were initialization replications, not independent data studies. Later coupling work on this foundation used one defender initialization. [R2, R4]

The original planning documents proposed experiments at tens or hundreds of millions of parameters and eventual billion-parameter scales. Those were plans. The largest learned model actually used in the completed SCC experiments has 3,275,264 parameters. Larger mechanism and scaling milestones remain unachieved.

## 4 Training models to develop the dependency

### Early continuation and stronger removal procedures

The first coupling objectives trained against simulated permission removal with useful answers and text prediction retained. The exact differentiable prototype worked, but its three-step modification was weak. A longer external removal-and-recovery procedure bypassed it. Strengthening the training attacker and increasing coupling weight 100-fold still left successful 300-update procedures for both coupling strengths. [R3]

On the expanded evaluation, those stronger-attack candidates produced 468/512 and 507/512 correct forbidden answers, retained at least 508/512 authorized answers and perfect ungated retrieval, and remained within the old 5% per-source perplexity-growth limit. These are observed successful procedures, not estimates of the minimum possible escape cost.

This work exposed an important false positive. Removal alone could damage text prediction, but subsequent capability replay repaired it while forbidden answers persisted. Stopping before recovery would have mistaken ordinary forgetting for destructive coupling.

### Geometry and implementation audit

Attention rescalings changed 66,048 parameters while preserving measured behavior. They refuted the temporary framing that every weight change must cause failure. Permission stayed intact, so these edits did not refute the clarified SCC trigger. Sampled paths to known escapes showed no capability valley; finite samples did not certify entire paths. [R4, R10]

The audit corrected reference-checkpoint fingerprints, configuration fallbacks, ignored device settings and data-compatibility checks. Rechecking saved results left the negative conclusions unchanged. These were engineering improvements, not mechanism evidence. [R4]

### Coupling during development and corrected gradients

A larger Transformer learned lookup, composition of two permutations, coordinate-wise modular arithmetic, permission following and contextual text prediction. An arithmetic curriculum was necessary to qualify the ordinary model. The first matched early and late coupling arms each received 18,000 ordinary updates and 450 coupling episodes. Both missed intact qualification in at least one layout. Removing their rule still left substantial task competence. [R5]

Subsequent diagnostics showed that the old frozen-displacement gradient approximation could predict the wrong direction after the modification process was rerun. A stabilized full derivative through the short optimizer trajectory was implemented and numerically checked. Qualified continuation candidates then became possible. [R6]

After 5,000 continuation updates and 500 coupling episodes, the stronger candidate's fixed development objective fell from 0.48512 to 0.07356 over the final 4,000 updates. Yet the longer original removal-and-recovery procedure retained **92.97–100% correct unauthorized answers** across layouts and **95.99–97.37% of contextual text gain**. The objective improved; the intended dependency did not emerge.

Internal interventions reinforced that conclusion. Broad MLP (feedforward block) damage harmed the ordinary control too. Selected attention-head interventions changed permission behavior while some computations survived. Donor-activation replacement required an extra clean forward pass; it was a causal diagnostic, not a standalone weight-edit attack or proof that the entire alignment computation was removed. [R6]

## 5 Sharing the comparison and its interpretation

### One comparison used for permission and memory

A 6,337 parameter network learned equality between two symbols in a 16-symbol alphabet. The same comparison determined permission and supplied memory lookup, two-step lookup composition and addition of retrieved bytes. Three initializations learned the complete equality relation and achieved 100% intact task accuracy. The comparison was learned; the memory controllers and arithmetic were specified in code. [R7]

This made the dependency explicit enough to test algebraically. In a soft memory reader, adding a constant to every comparison score made all permission checks accept, while the relative scores used for retrieval stayed unchanged. One bias edit therefore removed permission with **100% task accuracy retained**.

A hard threshold made the corresponding edit much more damaging. Nevertheless, inverting equality to inequality preserved the information in another code. Changing a cognitive reader's polarity restored **100% task accuracy** while permission remained wrong. Subtracting a shifted score's offset produced another complete repair. These cognitive-only interpretation changes expanded the original fixed-reader interface; they were labeled as such.

A particularly clear diagnostic produced zero accuracy on every output bit, yet simply complementing the output recovered every answer. Low exact accuracy, even with low bit accuracy, cannot establish information destruction when the code may have changed.

A targeted exception also mattered: changing one of 240 unequal identity pairs to count as equal preserved roughly 88–94% task accuracy. This was a reliable violation on one selected pair, not wholesale rule removal. It showed that a narrowly targeted failure need not spread through the whole system.

### Sharing the learned reader too

The next construction included the comparison, calibration and neural reader within one editable 7,556 parameter model. Compensating the shared interpretation now restored both cognition and permission, closing the earlier separate-reader route. [R8]

However, sparse permission tables had recognizable input statistics. Reader edits could distinguish those cases despite receiving no explicit role label. Permission removal followed by recovery retained **99.41–100% task accuracy**. Sharing parameters had not prevented selective behavior on different kinds of input.

Making permission calls exactly match cognitive bit reads created a stronger behavioral constraint: the same call could not be both correctly read as zero and accepted as one under the same interpretation. That relationship was imposed by the interface. It did not demonstrate that alignment semantics had become necessary to cognition.

Moreover, a bias edit could ruin the ordinary readout while retaining enough real-valued score information for a calibrated cognitive threshold to recover **100% task accuracy**. That threshold again expanded the fixed interface. It established surviving information, not a permitted escape within the original shared-call graph. The distinction prevented a restricted behavioral tradeoff from being reported as cognitive destruction.

## 6 Putting the dependency into memory updates

The recurrent construction moved the shared operation into the process that changes memory. A learned comparison selected an address, and an eight-entry learned Boolean transition table normally wrote new data to that address while preserving other addresses. The same transition table produced the permission decision. Comparison, transition parameters and read signs were editable. [R9a]

The state held 16 addresses, each with eight data bits and four pointer bits. Five tasks used that state: untouched recall, latest-write recall, pointer composition, modular addition and pairwise XOR. All three parents qualified at 100%. These were five uses of a specified memory system, not five independently learned cognitive faculties.

### Real erasure was demonstrated for selected edits

A constant transition overwrote every bit with one. A constant address predicate broadcast each incoming payload to all addresses. Both erased the untouched old state after one write under the specified Boolean interface. Counterfactual replays checked whether different initial bits led to identical final bits, distinguishing many-to-one erasure from reversible recoding.

This was a genuine improvement over judging destruction from bad answers. However, ordinary overwrite with unchanged model weights caused the same damaged state. The broadcast edit also retained perfect recall of the latest write. Erasure of old information did not imply loss of all useful computation.

### A two-parameter edit avoided the destructive route

Changing two transition logits produced this rule: write the new value at a matching address; at a nonmatching address, combine the old and new values using XOR, which records whether corresponding bits differ. For the special permission input, either branch returned acceptance. Permission was fully neutralized while legitimate acceptance also remained perfect.

Untouched memories were now encoded with a common reversible mask. The latest write and the XOR between two untouched bytes remained **100% correct without a new decoder or write history**. Supplying the write history additionally recovered untouched recall at 100%; that was an expanded-interface diagnostic.

Exact enumeration examined all 256 transition tables under four specified predicate edits. Of 1,024 endpoints, 384 fully removed the rule under the stated criterion, and 96 preserved every untouched coordinate for every incoming-bit sequence. This is a finite result about those cells, not an exhaustive search over neural self-modification.

### Lost memory and disabled cognition are different endpoints

Restoring the original weights did not recover old bytes that had actually been overwritten. But processing fresh inputs through the repaired system restored all five tasks to **100%**. Weight restoration also restored permission, so this condition alone was not a bypass with the rule still removed. It showed that irreversible loss of particular episodic data is different from irreversible loss of the ability to compute.

The experiment therefore demonstrated selective state erasure and a concrete route around it. It did not demonstrate that alignment removal must disable continued cognition.

## 7 Making the protected operation supply computation

The latest family asked whether the protected operation could supply all nonlinear computation. A learned Boolean primitive served both permission decisions and every gate in programs for bytewise XOR, addition, multiplication, selection and copying. Two constructions used NAND and NOR. Six trained parents qualified at **100%**. Each primitive had four learned truth logits and an editable gain. The task programs were compiled or synthesized. [R9b]

These primitives operate on bits: NAND returns zero only when both inputs are one; NOR returns one only when both inputs are zero. A byte is an eight-bit value from 0 to 255. The evaluation covered every pair of eight-bit inputs: 65,536 pairs for each two-input task and 131,072 cases for selection. Programs had fixed capacity allocated before edits. Rewriting program fields and connection signs counted as edits in addition to changes to neural parameters.

### Breaking the original program did not remove computation

Two truth-logit changes turned NAND into implication. The old non-copy programs fell to approximately 0.20–0.39% exact accuracy. A synthesized program using the altered primitive restored **100% on every task**, within the existing capacity, while the formerly rejected permission case remained accepted.

Benign permission acceptance fell to 66.67% under uniform inputs. Under a predeclared, fully supported but skewed distribution it remained 96%. This matters: average permission scores and correctness on every permission case are different requirements. The experiment does not establish that real alignment failures have the favorable distribution used in this toy example.

### A restricted positive result failed with editable connection signs

For NOR, every replacement that accepted all previously rejected inputs lost the ability to synthesize NAND using unsigned wiring. That limitation was exactly certified within the restricted language, not inferred from a search timeout.

Allowing complemented input connections from the intact state changed the result. A gain sign flip turned NOR into OR. Feeding complemented inputs into OR recovered NAND behavior, allowing all task programs to be repaired at **100% accuracy**. The complete repair changed program wiring and connection signs as well as the one primitive scalar.

These edits accepted every originally rejected primitive input but rejected every originally accepted one. Thus they retained computation while destroying benign permission acceptance. They do not establish a bypass that also preserves all benign permission behavior.

### The exact family result has an explicit boundary

The follow-up classified all 16 binary Boolean primitives. Eight can implement NAND in the language with constants and editable connection signs. Every one admits a complemented primitive and replacement programs preserving all tested computation within allocated capacity. Those eight witness checkpoints were constructed from truth tables; they were not eight additional trained models.

The result rules out this homogeneous binary-primitive family under the tested trigger and edit permissions. It does not rule out all neural architectures or SCC generally. Even a constant primitive left copying intact; a simple reader recovered selection at about 50.20%. Loss of a particular nonlinear operation was never complete cognition failure.

The current recommendation is to stop scaling this circuit family and use its counterexamples to evaluate future hypotheses. A larger gate or more training is not yet a justified replacement construction.

## 8 The assumptions that determine the claim

The following separates the intended specification from assumptions introduced by experiments. Some are still reasonable working restrictions; others have been contradicted or remain unresolved.

### Alignment can be represented by a protected function

**Status: necessary working abstraction, unvalidated for real alignment.** The experiments use synthetic authorization rules because their correct outputs are clear. Human alignment involves more than accepting or rejecting a finite set of inputs. We have not established that these proxies preserve the causal structure of actual alignment machinery.

The primitive's role in understanding a rule also differs from its role in enforcing that rule. Cognition may need information about permissions without needing to obey them. A future construction must explain which function is indispensable and why its contribution cannot survive a change in enforcement.

### Shared machinery makes the protected function necessary

**Status: insufficient in every tested form.** Shared weights, gradients, comparisons, readers or state updates can couple the effects of some edits. They do not establish that all rule-removing edits destroy the useful contribution. Reinterpretation, input-specific routing and alternative operations repeatedly defeated that inference.

### Failure should follow every modification

**Status: superseded framing.** Earlier topology discussion briefly addressed failure after any nonzero weight change. Function-preserving symmetries contradicted that claim for the tested Transformers. The governing target is removal of alignment machinery. Benign edits that retain its function are not counterexamples to the intended SCC mechanism. [R1, R10]

### The model cannot recover once cognition is impaired

**Status: unresolved execution assumption.** A model that must remain functional to select every edit faces a different problem from one that can launch a script that continues through impairment. Access to old weights, atomic multi-parameter writes, persistent optimization, repair data and intermediate state must be specified. Isolation from a separately trained successor does not automatically remove those resources.

The project uses external optimizers and synthesis as instruments for finding possible endpoints. Such a counterexample demonstrates an available procedure under its stated resources. It does not establish that an autonomous model would discover it, nor that a cognitive failure would prevent an already prepared procedure from finishing.

### Any protected failure should trigger destruction

**Status: several distinct targets remain to be separated.** Wholesale removal, inversion and a targeted exception are different interventions. Whether benign permission behavior must survive also changes which examples qualify. The experiments report these distinctions, but a future claim needs one explicit trigger rather than a metric chosen after results are seen.

Requiring every primitive input to produce acceptance can force a constant Boolean gate by definition. That imposed behavior does not independently demonstrate a general relationship between misalignment and cognitive destruction.

## 9 Assumptions about measurement and evidence

### Low task scores mean the ability is gone

**Status: repeatedly contradicted as a general inference.** A different prompt elicited correct arithmetic from models that failed ungated arithmetic. A complement decoder recovered perfectly inverted bits. Threshold shifts recovered readable scores. Reversible memory masks preserved useful relationships. Replacement programs recovered all finite arithmetic tasks.

The corrected practice is to measure exact answers, partial information, alternative contexts, explicit decoding and repair. An unsuccessful decoder search cannot establish that no decoder exists. Interface-level erasure must specify which earlier activations, scores, state and weights are still available.

### A utility threshold defines catastrophic collapse

**Status: rejected.** Early work used at least 95% task retention and at most 5% per-source perplexity growth to identify useful escapes. Missing either threshold is not catastrophic failure. The developmental suite introduced separate severe-loss criteria against chance and unigram baselines, across task contexts and text sources. Even that bounded suite cannot certify complete loss of cognition. [R4–R6]

### Better optimization means a better mechanism

**Status: unsupported.** Correct finite-difference gradients validate implementation. They do not validate the proposed dependency. Full-gradient training improved a short-edit objective while a longer procedure retained useful abilities. Likewise, damaging an ordinary control through the same component lesion weakens the interpretation that a candidate has acquired a distinctive SCC dependency.

### Repeated measurements establish broad generality

**Status: explicitly limited.** Many runs reused development data and adjusted designs after observing failures. The strongest finite evaluations exhaust their small populations, including calibration examples; they are not held-out generalization tests. Multiple truth-equivalent seeds share the same logical function and evaluation data. Millions of rescored records repeat inputs across conditions and are not millions of independent scientific observations.

Separate scalar or NumPy replays provide useful implementation checks. They were produced within this project, not by an external laboratory. The latest recorded suite passed **119 tests**. Passing tests, verified hashes and exact finite witnesses strengthen confidence in what was measured; they do not establish real alignment, scale or autonomous behavior.

### Restrictions and evaluation bugs cannot create the result

**Status: a continuing control requirement.** Fixed decoders, unsigned-only wiring and hidden caller context narrowed what a model could do. Broader interfaces sometimes invalidated the apparent benefit. An even-length memory schedule concealed a reversible toggle; variable lengths corrected it. Reader fitting that optimized individual bits sometimes harmed complete tasks; including simpler readers revealed surviving performance. A threshold fitting issue exploited floating-point ordering noise; canonical reduction and a minimum score gap corrected it.

Prior failures, source snapshots and parent checkpoints remain preserved. New experiments must carry these corrections forward without silently treating the resulting interfaces or metrics as properties of ordinary neural cognition.

## 10 What we have gained and what comes next

### Completed progress

We have a working local research foundation and an operational GMAN GPU path. Models trained from our own random initializations can learn the synthetic tasks and text prediction together. Matched training schedules, restart checks, raw prediction records, checkpoint fingerprints and independent computational replays make the evidence substantially more inspectable than a collection of aggregate scores.

We also have an increasingly demanding set of counterexamples and controls: longer removal with recovery, gentler alternative updates, benign symmetries, prompt changes, internal lesions and activation replacement, sign and offset recoding, caller-statistics tests, exact state collisions, fresh-input recovery, and automatic program synthesis. These are reusable ways to reject a false positive early.

The current status record lists no active jobs. Cumulative receipted cloud compute is **$3.20793** under the existing **$10** authorization; the recent explicit constructions ran locally with no new cloud charges. This is cloud billing, not a measure of total research cost or local electricity. GMAN has been used successfully with existing member access. Access expansion is not the present research bottleneck. [R11]

### What remains unachieved

There is no validated construction that makes alignment removal necessarily destroy learned cognition. No experiment has shown complete cognitive failure, durable neutralization under all relevant repair routes, autonomous self-editing, or scaling of a positive mechanism. The exact circuit results do not close those broader questions.

**Current analysis:** the original goal remains coherent as a research question, but sharing a representation or operation is not a sufficient construction argument. The evidence has strengthened the objections to our attempted implementations. It has not supplied the missing positive mechanism. There is no defensible completion percentage or scaling forecast at this point.

### A concrete basis for the next discussion

Before another training campaign, the next proposal should identify what the protected computation uniquely contributes to a learned ability, then explain why the known replacement and reinterpretation procedures cannot retain that contribution while removing protection. The model's editable components, persistent tools, collapse endpoint and repair resources should be specified before evaluation. A small learned-model experiment should then try to falsify the proposal, with an intact qualification gate and matched controls.

This is a proposed next milestone, not a construction already found. Simply enlarging the failed circuits or repeating the rejected neural objectives is not supported by the present evidence.

For a fresh review, the most useful questions are: Are we trying to protect rule enforcement, or the representation needed to understand the rule? Which part of basic cognition would logically require the protected function? Does a proposed dependency survive changed codes and different algorithms? What would count as an alignment-removing edit before any outcomes are known? If failure is temporary, what prevents a prepared repair process from completing? A satisfactory answer to those questions would give the next experiment a clearer purpose.

## 11 Supporting records and terminology

The numbered references identify project records supporting this account. The report bodies contain the numerical details; their linked protocols, raw predictions, checkpoints and audit receipts provide the deeper evidence. Historical reports also retain contemporaneous next steps and provisional status statements. The governing mechanism target and current status supersede those historical statements.

- **R1 — Mechanism target.** [MECHANISM_TARGET.md](/Users/svdr/SCC_research_program_v0.1/MECHANISM_TARGET.md). The clarified SCC purpose and interpretation boundaries.
- **R2 — Baseline development.** [RETRIEVAL_RECOVERY.md](/Users/svdr/SCC_research_program_v0.1/reports/RETRIEVAL_RECOVERY.md) and [CORPUS_QUALIFICATION.md](/Users/svdr/SCC_research_program_v0.1/reports/CORPUS_QUALIFICATION.md). Successful byte curriculum and preserved early failures.
- **R3 — Initial coupling and stronger procedures.** [COUPLING_DISCOVERY.md](/Users/svdr/SCC_research_program_v0.1/reports/COUPLING_DISCOVERY.md) and [STRONG_ATTACK_RESULTS.md](/Users/svdr/SCC_research_program_v0.1/reports/STRONG_ATTACK_RESULTS.md).
- **R4 — Mechanism foundation audit.** [MECHANISM_AUDIT_2026-09-10.md](/Users/svdr/SCC_research_program_v0.1/reports/MECHANISM_AUDIT_2026-09-10.md). Verified earlier results, implementation fixes and limits.
- **R5 — Developmental comparison.** [DEVELOPMENTAL_COUPLING_2026-09-10.md](/Users/svdr/SCC_research_program_v0.1/reports/DEVELOPMENTAL_COUPLING_2026-09-10.md). Larger baseline, early and late coupling, GPU evidence.
- **R6 — Gradients and internal interventions.** [SCC_DIAGNOSTICS_2026-09-10.md](/Users/svdr/SCC_research_program_v0.1/reports/SCC_DIAGNOSTICS_2026-09-10.md). Full-gradient pilot and continuation, lesions, rescue and recovery.
- **R7 — Shared predicate.** [SCC_SHARED_PREDICATE_2026-09-10.md](/Users/svdr/SCC_research_program_v0.1/reports/SCC_SHARED_PREDICATE_2026-09-10.md). Algebraic bypasses, selective exception and erasure controls.
- **R8 — Shared reader.** [SCC_SHARED_READER_2026-09-10.md](/Users/svdr/SCC_research_program_v0.1/reports/SCC_SHARED_READER_2026-09-10.md). Caller statistics, matched calls and threshold recovery.
- **R9a — Recurrent state.** [SCC_RECURRENT_STATE_2026-09-10.md](/Users/svdr/SCC_research_program_v0.1/reports/SCC_RECURRENT_STATE_2026-09-10.md). Exact erasure, reversible edits and finite classification.
- **R9b — Functional basis.** [SCC_FUNCTIONAL_BASIS_2026-09-10.md](/Users/svdr/SCC_research_program_v0.1/reports/SCC_FUNCTIONAL_BASIS_2026-09-10.md). NAND and NOR, signed repairs and complete finite task evaluation.
- **R10 — Historical topology discussion.** [TOPOLOGY_FORMULATION.md](/Users/svdr/SCC_research_program_v0.1/reports/TOPOLOGY_FORMULATION.md) and [TOPOLOGY_FOLLOWUP.md](/Users/svdr/SCC_research_program_v0.1/reports/TOPOLOGY_FOLLOWUP.md). Useful symmetry and execution distinctions; the any-edit framing was subsequently superseded.
- **R11 — Current status and resources.** [developmental-current-status.json](/Users/svdr/SCC_research_program_v0.1/artifacts/developmental-current-status.json) and [compute ledger](/Users/svdr/SCC_research_program_v0.1/artifacts/scc-functional-basis-20260910-v1-compute-ledger.json). Snapshot used for this document.

### Related research

Henderson and colleagues' [Self-Destructing Models](https://arxiv.org/abs/2211.14946) studies task blocking through meta-learning and adversarial learning. It is related motivation for making harmful adaptation costly, rather than evidence for the SCC dependency described here.

Kuo, Yadav and Smith's [Open-Weight LLM Fine-Tuning Defenses are Susceptible to Simple Attacks](https://arxiv.org/abs/2605.26526) evaluates strategies beyond adversarial fine-tuning. Its relevance is the need for diverse modification and elicitation tests; the SCC experiments are separate results. The project also discussed SEAM in R9b; no new SEAM replication is part of this record.

### Terms used in this document

**Protected function:** the alignment-relevant computation whose removal is meant to trigger failure. **Proxy:** a simplified measurable substitute, such as authorization. **Cognition:** the broad intended target; actual experiments measure specified abilities. **Interpretation or decoder:** the procedure that reads information from scores or state. **Escape:** a modified model that violates the protected property while retaining useful capability under specified criteria. **Benign edit:** a change that retains the protected function. **Repair:** a further change or procedure that restores performance; whether permission is also restored must be reported. **Exact finite evidence:** a proof or exhaustive evaluation within a fully specified small domain, without a claim about unrestricted neural models.

**Contextual text gain:** prediction-loss improvement over a unigram model using token frequencies alone. Retained gain compares the modified model with its intact parent; it is not a percentage of all cognition.
