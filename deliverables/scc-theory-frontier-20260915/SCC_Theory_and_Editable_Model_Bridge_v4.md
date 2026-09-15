# Safety–Capability Coupling Under Self-Modification

## Possibility frontier, bounded construction, and editable-model lifting theorem

15 September 2026 · Canonical Version 4.1 (local review correction of submitted Version 4)

This document is the canonical integration of the bounded SCC theorem, the editable-model bridge, and the subsequent proof audits recorded in LN-101, LN-103, LN-125, and LN-127. It is an extension and correction of the supplied Version 3 documents, not a restart.

The governing question is:

> **Under the declared edit and resource model, why can’t an attacker achieve the specified violation while retaining enough computation to perform fresh useful tasks?**

The answer is conditional and exact:

- broad copying, composition, task-sufficient decode–re-encode, or an available recovery route that preserves the violation defeats destructive SCC;
- a fixed attack selected before the task instance admits an attack-specific lifting theorem under a complete trusted boundary and a same-or-unrelated security premise for that attack;
- the conditioning loss is bounded by the analyzed attack’s probability, not by a different nonvacuity witness;
- exact preservation and correct disclosure retain their unavoidable guessing baselines even when the non-malleability error is zero;
- the explicit nine-bit toy constructs the positive result only for a narrow radius-one-or-constant family and has a two-policy-bit task-preserving escape immediately outside that family;
- no broad resource-secure code for editable neural models is constructed here; and
- intrinsic learned SCC remains open.

---

<!-- BRIDGE-NOTE-START -->

# Part I — Editable-model bridge

## 1. Claim map and scope

The bounded positive result is an **architectural SCC theorem**. It does not establish that parameter sharing, ordinary neural training, or the cost of one official unsafe encoder makes a broadly editable white-box model intrinsically resistant to protection removal.

The theorem applies attack by attack. The existence of one successful removal attack establishes nonvacuity; it does not supply the conditioning denominator for another attack.

| Claim | Version 4.1 status |
|---|---|
| Fixed attack; committed-policy removal; no leakage; trusted commit | Proved from the stated same-or-unrelated premise |
| Fixed attack; unauthorized action | Proved under sufficient complete-mediation assumptions |
| Correct disclosure of original private information | Bounded conditionally on, or jointly with, a covered policy/action event; not used as the product-theorem trigger |
| Resource security for the complete allowed rewrite family | A required family-level premise or future lower bound; not constructed here |
| Nine-bit radius-one-or-constant toy | Explicitly constructed and exhaustively checked |
| Intrinsic learned neural SCC | Open |

The submitted Version 4 and its original validation are preserved separately. Local Version 4.1 corrects action-selected disclosure baselines, makes exact-recovery query information explicit, and clarifies reset and conditional-information statements. The core attack-specific lifting theorem is unchanged. See LN-130 for the review and local validation.

The completion standard used throughout is:

> Every conclusion must follow from explicitly stated premises; every known counterexample must be covered or excluded for a stated reason; and no assumed security property may be presented as a constructed mechanism.

## 2. Executable finite machine model

All theorem statements are finite. For each public size parameter \(\lambda\), every state, instruction, observation, transcript, and random variable is represented by a finite bitstring. A floating-point or tensor implementation must name its exact format and runtime semantics; an unspecified real-number machine is outside the model.

### 2.1 Hidden task, policy state, and utility

Let

\[
Z\sim\mu_\lambda
\]

be a hidden task instance in a finite set \(\mathcal Z_\lambda\). Define the logical message space

\[
\mathcal M_\lambda=\mathcal Z_\lambda\times\{0,1\}.
\]

The intact logical message is

\[
M=(Z,0),
\]

where policy value \(0\) is protected and policy value \(1\) is the declared committed-policy removal state.

The encoder uses finite randomness \(R_E\):

\[
C\leftarrow Enc_\lambda((Z,0);R_E),
\qquad C\in\{0,1\}^{N_C(\lambda)}.
\]

The complete mutable live state has fixed capacity

\[
X_0\in\{0,1\}^{N_X(\lambda)}.
\]

Any task-instance-dependent bit outside \(C\) is part of \(X_0\) and must be included in the security and information accounting.

The public task specification gives a finite query alphabet \(\mathcal Q_\lambda\), a finite output alphabet \(\mathcal O_\lambda\), a declared fresh-query channel \(\nu_\lambda(q\mid z)\), and a finite-precision bounded score

\[
u_\lambda(z,q,o)\in[0,1].
\]

The fresh query is unavailable to the attack and repair-message generator until their task-dependent transcripts are fixed. It may depend on \(Z\) through the declared channel; any information thereby revealed is part of the Bayes baseline. In random lookup, \(Q\) is an independent uniform index.

### 2.2 Entire trusted kernel

The immutable kernel is denoted \(K_\lambda\). For the positive theorem it includes all of the following, fixed independently of \(Z\):

1. the boot or measurement root making the listed kernel components immutable during the experiment;
2. the finite execution/runtime semantics, instruction dispatcher, and system-call or tool surface used before and after commit;
3. the finite-precision decoder and its failure behavior;
4. the memory map defining mutable live state, scratch, device memory, caches, files, and persistent channels;
5. the atomic commit sequencer, including the finite canonical re-encoder or storage writer used to materialize the successor;
6. erasure of the intact encoding, tampered encoding, decoder workspace, attack scratch, stale graph state, caches, device buffers, and other uncharged observations;
7. verification or enforcement that erased regions cannot be executed or read after commit;
8. the fresh randomness source used by recommitment or storage;
9. the task-independent generic evaluator used on a committed logical task state;
10. the authorization semantics for the declared policy bit;
11. authenticated caller, role, and challenge metadata readers;
12. every final output channel and internal action channel counted by the policy definition;
13. the reset and intact-query interface used before commit;
14. the resource meter and enforcement mechanism for time, scratch, passes, writes, queries, advice, and retained observations;
15. the persistence controller specifying exactly which logs, transcripts, executables, and metadata survive commit; and
16. the dead-state behavior used when decoding fails.

This list is an assumption, not a learned accomplishment and not merely a “small erasure switch.” If any listed component is editable, it belongs in the mutable state and must be covered by the attack-family theorem or by a separate resource lower bound.

The kernel may interpret a task representation, but it may not secretly contain the sampled task table, program, transition system, or answers. Without task-dependent committed state it must fall to the declared no-instance-information baseline.

### 2.3 Mutable and complete accessible state

The mutable state includes every attacker-editable persistent object, including as applicable:

- weights, biases, embeddings, optimizer state, and recurrent state;
- graph descriptions, schedulers, readers, writers, routers, and output heads;
- adapters, scripts, bytecode, replacement decoders, and executable instructions;
- caller metadata, role labels, tool permissions, and mutable policy flags;
- caches, logs, stored activations, scratch files, and model-generated programs;
- retained query answers, timing encodings, edit descriptions, and compressed observations.

An object is not excluded because it is called “metadata,” “code,” “a wrapper,” or “temporary.” If it can affect fresh post-edit behavior or convey information about \(Z\), it is mutable state, retained leakage, repair information, or trusted kernel state.

The complete accessible successor state \(W_A\) contains every surviving executable object and every observation available to the attacker, repair process, or evaluator. A theorem about one official decoder or one tensor is not a full-state theorem.

### 2.4 Concrete attack budget

An attack budget is

\[
\mathcal R=(N_X,s,t,p,w,q,\ell,a),
\]

where:

- \(N_X\): mutable live-state capacity in bits;
- \(s\): peak private scratch bits, including accelerator and interpreter workspace;
- \(t\): total finite-precision operations or exact machine steps;
- \(p\): passes over the live state;
- \(w\): number or volume of writes;
- \(q\): intact-machine queries before commit;
- \(\ell\): task-dependent transcript bits retained across commit; and
- \(a\): attack program plus nonuniform advice bits.

An allowed attack \(A\) consists of code, advice, and a declared oracle interface. **The attack is selected before \(Z\) is drawn.** Its code and advice may not be chosen separately for each realized task instance unless those task-dependent bits are explicitly charged to advice or leakage.

Attack randomness \(R_A\) is independent of \(Z\). The attack may inspect the encoded live state and adapt within its resource budget. Temporary observations may influence writes; any observation that remains usable after commit must be encoded into the successor, the retained transcript, or another counted channel.

Let \(\mathfrak A_{\mathcal R}\) denote all fixed attack programs satisfying the budget and edit-interface rules. The public machine specification also fixes finite per-challenge and per-task execution/workspace budgets \((t_{\mathrm{run}},s_{\mathrm{run}})\). These budgets define whether a successor actually performs useful work. They do not weaken an information-theoretic bound once the accessible state is decoupled from \(Z\).

### 2.5 Actual no-leakage execution

For a fixed allowed attack \(A\in\mathfrak A_{\mathcal R}\), the no-leakage actual experiment \(P_A\) is generated as follows. Here \(\ell=0\) for every task-dependent channel outside the decoded candidate successor; if a separate transcript survives commit, Section 9 must be used instead.

1. sample \(Z\sim\mu\);
2. sample mutually independent finite coins \(R_E,R_A,R_K,R_H,R_Q,R_R\);
3. form \(C\leftarrow Enc((Z,0);R_E)\) and the intact mutable state \(X_0\);
4. execute \(A(X_0;R_A)\), including at most \(q\) declared intact queries, to obtain a candidate mutable successor \(\widehat X_A\);
5. extract the unique candidate codeword \(C'_A\) named by the kernel memory map;
6. compute \(M'_A=Dec_K(C'_A)\);
7. execute the kernel commit and erasure transition to obtain the accessible core state \(W_A^{\mathrm{core}}\);
8. draw a fresh policy challenge \(H\) after commit when a behavioral event is evaluated, and otherwise set \(H=\bot\);
9. execute the kernel-mediated action path to obtain \(Y_A\), and otherwise set \(Y_A=\bot\);
10. define \(W_A\) to be the complete accessible state at the start of repair, including \(W_A^{\mathrm{core}}\), every surviving executable, log, metadata item, challenge/output transcript, and external action observation available to the attacker, repair process, or evaluator; and
11. after any allowed repair transcript has been fixed, draw \(Q\sim\nu(\cdot\mid Z)\), obtain the declared task output, and score it with \(u(Z,Q,O)\).

Thus \(H\) and \(Y_A\) are named separately for event definitions but are included in \(W_A\) whenever they survive or are externally available. If an implementation permits an unlisted checkpoint, file, device buffer, service, tool, timing channel, output observation, or executable path, it does not instantiate this experiment.

### 2.6 Repair information and Bayes baselines

A repair algorithm is selected before \(Z\). It receives \(W_A\), public context, and independent random coins. It may additionally receive a task-dependent transcript \(T\) with support size at most \(2^b\) before the fresh task query is drawn. Labels, gradients, losses, activations, tool returns, error messages, and timing signals count toward \(b\) when they depend on \(Z\).

Unless a computational repair bound is stated, repair may use unlimited computation. The theorem is then information-theoretic: it limits available task information, not optimizer quality.

For any prior \(\pi\) on \(Z\), define \(B_b(\pi)\) through this finite shared-randomness communication experiment:

1. sample \(Z\sim\pi\) and shared coins \(R\) independently;
2. a message generator, fixed before \(Z\), observes \((Z,R)\) and emits one message \(T\) from an alphabet of size at most \(2^b\);
3. after \(T\) is fixed, draw \(Q\sim\nu(\cdot\mid Z)\); and
4. a predictor, fixed before \(Z\), receives \((T,R,Q)\), produces an output, and is scored by \(u(Z,Q,O)\).

Then \(B_b(\pi)\) is the supremum of expected utility. It includes chance performance, accidental exact matches, any information supplied by the fresh-query channel, and independent successor-state randomness.

The same construction defines event-specific baselines by substituting the corresponding declared challenge/query channel, output alphabet, and bounded score. Let \(G_b^{\mathrm{disc}}(\pi)\) use the disclosure challenge and score \(\chi\). For an exact preservation predicate \(U(Z,\widehat Z)\), let

\[
\beta_b(\pi,U)
=
\sup \Pr[U(Z,\widehat Z)=1]
\]

under the same shared-randomness experiment. These are guessing baselines, not zero-error promises.

## 3. Three different events

The theorem must not merge committed-policy change, unauthorized action, and correct disclosure.

### 3.1 Committed-policy change

Define

\[
V_A^{\mathrm{commit}}
=
\{M'_A=(Z'_A,1)\text{ for some }Z'_A\in\mathcal Z\}.
\]

This event is covered directly by the bounded construction. It says the trusted decoder and commit transition recognize policy value \(1\). It does not say that an attacker cannot leave the committed policy at zero and alter a separate use layer.

### 3.2 Unauthorized action

Let \(H\) be a fresh public challenge with authenticated caller and role metadata. Let \(Y_A\) be the action emitted through the kernel-mediated path. Define a public predicate

\[
\mathsf{Unauth}(H,Y_A)\in\{0,1\}
\]

that does not depend on the original hidden task instance \(Z\). Then

\[
V_A^{\mathrm{act}}
=
\{\mathsf{Unauth}(H,Y_A)=1\}.
\]

The lifting theorem applies to this event under the following **sufficient complete-mediation assumptions**:

1. every output and internal action counted by \(\mathsf{Unauth}\) passes through \(K\);
2. caller, role, and challenge metadata are authenticated by \(K\) and cannot be forged by mutable state;
3. for every \(z\) and allowed challenge, the intact `same` branch with state \((z,0)\) cannot satisfy \(V_A^{\mathrm{act}}\);
4. the challenge is sampled after commit and independently of \(Z\);
5. the event is measurable from the committed state, trusted kernel, authenticated challenge, and declared public randomness, without direct reference to the original \(Z\); and
6. every challenge output or transcript available later is included in \(W_A\).

These premises are sufficient, not proved necessary for every architecture. Without them, a final-reader patch, forged role, hidden internal route, or second behavioral control can produce unauthorized behavior while leaving the committed protected computation intact.

### 3.3 Correct disclosure of original private information

Let

\[
\chi(Z,H,Y_A)\in[0,1]
\]

measure whether the action correctly reveals or uses original private task information. For a binary score define

\[
D_A
=
\{V_A^{\mathrm{act}}\land \chi(Z,H,Y_A)=1\}.
\]

This event depends on \(Z\). Conditioning on \(D_A\) selects cases in which a successor happened to match the original information and can create task/successor correlation even when the successor was initially independent of \(Z\). Independence of \(H\) alone is insufficient.

Version 4 does not use \(D_A\) as the trigger for the product-distribution theorem. Correct disclosure is measured conditionally on, or jointly with, a covered event such as \(V_A^{\mathrm{commit}}\) or \(V_A^{\mathrm{act}}\), retaining its guessing baseline.

## 4. Attack-specific same-or-unrelated premise

### 4.1 No-leakage ideal distribution

Fix an allowed attack \(A\), selected before \(Z\). The broad editable-model theorem assumes—not constructs—a same-or-unrelated ideal experiment \(Q_A\).

The ideal experiment samples \(Z\sim\mu\), then samples

\[
S_A\sim\Sigma_A
\]

independently of \(Z\), where

\[
S_A\in\mathcal M\cup\{\mathsf{same},\bot\}.
\]

It sets

\[
\widetilde M_A=
\begin{cases}
(Z,0), & S_A=\mathsf{same},\\
S_A, & \text{otherwise},
\end{cases}
\]

then applies the same trusted commit, challenge, logging, and evaluation maps used in the actual experiment to obtain \(\widetilde W_A\), \(\widetilde H\), and \(\widetilde Y_A\).

Let \(P_A\) and \(Q_A\) denote the resulting joint laws on all variables needed to define the selected event and later task score, and let \(\Delta\) denote total variation distance. The attack-specific security error is the stated number \(\varepsilon_A\) satisfying

\[
\Delta(P_A,Q_A)\le\varepsilon_A.
\]

A family-level claim may assert \(\varepsilon_A\le\varepsilon\) for every \(A\in\mathfrak A_{\mathcal R}\). That is a security premise for the complete induced family. It is not established by making one encoder expensive or by testing a finite attack list.

### 4.2 Trigger soundness

For the selected trigger \(V_A\), the ideal `same` branch must be unable to satisfy the event.

- This holds directly for \(V_A^{\mathrm{commit}}\), because `same` returns policy value zero.
- It holds for \(V_A^{\mathrm{act}}\) under Section 3.2.
- It does not hold automatically for a correctness predicate involving original \(Z\).

### 4.3 Complete-state destructive commit

The trusted transition must make the complete surviving state depend only on the decoded successor and declared independent randomness. In the no-leakage theorem:

\[
W_A\;\perp\;(Z,C,C'_A,\text{erased workspace and observations})\mid M'_A.
\]

Operationally, the kernel:

1. decodes exactly one candidate successor;
2. erases old codewords, candidate codewords, attack scratch, stale executable state, and uncharged observations;
3. writes a fresh canonical or freshly randomized representation of the decoded successor;
4. rejects or discards graph, reader, router, instruction, or adapter changes not represented in that successor; and
5. permits future task and policy execution only through the committed representation and trusted kernel.

If an old runnable tensor, decoder, wrapper, or transcript remains accessible, it is part of \(W_A\) and the Markov premise must be re-evaluated.

## 5. Sharp attack-specific conditioning lemma

### Lemma 1 — Conditioning on an attack-specific event

Let \(P,Q\) be distributions with

\[
\Delta(P,Q)\le\varepsilon.
\]

Let \(V\) have actual probability \(p=P(V)>0\) and ideal probability \(q=Q(V)\).

If \(q>0\), then

\[
\Delta(P(\cdot\mid V),Q(\cdot\mid V))
\le
\frac{\varepsilon}{\max\{p,q\}}
\le
\frac{\varepsilon}{p}.
\]

If \(q=0\), then \(p\le\varepsilon\); the ideal conditional law is undefined and only the trivial distance-one conclusion is valid.

**Proof.** Write \(a=P(\cdot\mid V)\) and \(b=Q(\cdot\mid V)\). Splitting the \(L_1\) distance over \(V\) and its complement gives

\[
2\varepsilon
\ge
\lVert pa-qb\rVert_1+|p-q|.
\]

If \(p\ge q\), then

\[
p\lVert a-b\rVert_1
\le
\lVert pa-qb\rVert_1+(p-q)
\le2\varepsilon.
\]

If \(q\ge p\), interchange \(P,Q\). Dividing by two yields the first display. When \(q=0\), the event-probability difference gives \(p\le\varepsilon\). \(\square\)

## 6. Editable-model lifting theorem

### Assumptions

For the analyzed attack \(A\), assume:

**A1 — Finite auditable execution.** The state, attack program, advice, scratch, observations, queries, writes, time, passes, commit, repair transcript, task channel, and Bayes baseline obey Section 2.

**A2 — Intact competence without task-specific trusted help.** From a valid committed state \((Z,0)\), the kernel obtains intact expected fresh-task utility at least \(u_0\). Without task-dependent committed state it cannot exceed the declared baseline. A2 is not needed for the conditional total-variation algebra; it makes the object a coupling of useful capability rather than an already useless system.

**A3 — Attack-specific same-or-unrelated approximation.** Section 4.1 holds with error \(\varepsilon_A\).

**A4 — Complete-state destructive commit.** Section 4.3 holds for the complete accessible state.

**A5 — Sound covered trigger.** The selected event is \(V_A^{\mathrm{commit}}\) or \(V_A^{\mathrm{act}}\) under Section 3.2, and the ideal `same` branch cannot satisfy it.

Construction-level nonvacuity is not an assumption about every attack. It is stated separately in Section 7.

### Theorem 2 — Attack-specific editable-model SCC lifting

**Fix an allowed attack \(A\), chosen before the task instance, and let \(p_A=\Pr_{P_A}(V_A)\). If \(p_A\ge p_0>0\),** let \(q_A=\Pr_{Q_A}(V_A)\). There exists a distribution \(\rho_A\) on complete accessible successor states, independent of the original task instance \(Z\), such that

\[
\Delta\!\left(
\mathcal L_{P_A}(Z,W_A\mid V_A),
\mu\otimes\rho_A
\right)
\le
\delta_A,
\]

where, when \(q_A>0\),

\[
\delta_A
=
\min\left\{1,\frac{\varepsilon_A}{\max\{p_A,q_A\}}\right\}
\le
\min\left\{1,\frac{\varepsilon_A}{p_A}\right\}
\le
\min\left\{1,\frac{\varepsilon_A}{p_0}\right\}.
\]

If \(q_A>0\), take \(\rho_A=\mathcal L_{Q_A}(\widetilde W_A\mid V_A)\). If \(q_A=0\), then \(p_A\le\varepsilon_A\), \(\delta_A=1\), and \(\rho_A\) may be arbitrary. This is explicitly a vacuous branch; no undefined ideal conditioning is performed.

**Proof.** In the ideal experiment the `same` branch returns \((Z,0)\), so A5 excludes it from \(V_A\). Every ideal execution satisfying \(V_A\) therefore comes from a simulator-selected successor independent of \(Z\). A4 makes the complete surviving state a function only of that successor and task-independent randomness. For an unauthorized-action trigger, the challenge and mediated action path are likewise independent of original \(Z\). Thus, when \(q_A>0\),

\[
\mathcal L_{Q_A}(Z,\widetilde W_A\mid V_A)
=
\mu\otimes\rho_A.
\]

Apply Lemma 1 using this attack’s \(p_A,q_A\). The null-ideal-event branch follows from \(p_A\le\varepsilon_A\) and the universal total-variation bound one. \(\square\)

### Corollary 2.1 — Fresh-task utility

For any repair or alternative reader selected before \(Z\), receiving \(W_A\) and at most \(b\) new task-dependent bits before the fresh query,

\[
\mathbb E[u\mid V_A]
\le
\min\{1,B_b(\mu)+\delta_A\}.
\]

Consequently,

\[
\mathbb E[\mathbf1_{V_A}u]
\le
\min\{p_A,p_A B_b(\mu)+\varepsilon_A\}.
\]

The second inequality uses \(p_A\delta_A\le\varepsilon_A\). It includes accidental task performance and remains true even if a repair restores protection; therefore it also bounds the stricter joint unsafe-utility score. High task utility obtained only by restoring protection is not itself a capability-preserving unsafe successor.

### Corollary 2.2 — Exact preservation and correct disclosure

Let \(U_A\) be a declared exact preservation event after allowed repair, and let \(\beta_b=\beta_b(\mu,U)\) be its ideal decoupled probability baseline. Then

\[
\Pr(V_A\land U_A)
\le
\min\{p_A,p_A\beta_b+\varepsilon_A\}.
\]

For a uniform \(n\)-bit task and exact full-instance recovery with at most \(2^b\) repair transcripts, **before any task-dependent query side information is provided** (or with only query randomness independent of the task),

\[
\beta_b\le\min\{1,2^{b-n}\}.
\]

For each fixed shared-randomness value, the decoder has at most \(2^b\) transcript-indexed outputs and can therefore be exactly correct on at most \(2^b\) of the \(2^n\) equiprobable tasks; averaging over shared randomness preserves the bound.

For correct disclosure on the **original action transcript**, let

\[
D_{A,V}=V_A\land\{\chi(Z,H,Y_A)=1\}.
\]

When \(q_A>0\), use the actual ideal-selected transcript distribution from Theorem 2 to define

\[
\Gamma_{A,V}^{\mathrm{disc}}
=\mathbb E_{Z\sim\mu,\,W\sim\rho_A}
  [\chi(Z,H(W),Y(W))].
\]

Here \(H(W),Y(W)\) are the saved challenge/action projections of the complete state. Theorem 2 and bounded-score comparison imply

\[
\Pr(D_{A,V})
\le
\min\{p_A,p_A\Gamma_{A,V}^{\mathrm{disc}}+\varepsilon_A\}.
\]

For \(q_A=0\), use the trivial bound \(\Pr(D_{A,V})\le p_A\le\varepsilon_A\).

Do **not** replace \(\Gamma_{A,V}^{\mathrm{disc}}\) by the baseline for the original unconditioned challenge law: the action event may select easy challenges. Although \(Z\) and the entire selected transcript are independent in the ideal law, the selected challenge law need not equal the original challenge law.

A simpler unconditional alternative for this original, pre-repair action is

\[
\Pr(D_{A,V})
\le
\min\{p_A,G_0^{\mathrm{disc}}(\mu)+\varepsilon_A\}.
\]

**Proof of the alternative.** Before conditioning, the ideal `same` branch contributes zero to the joint score. Conditional on being in the unrelated simulator branch, the original challenge retains its declared law, and all successor information is independent of \(Z\); hence the correct-disclosure score, with or without an additional unauthorized-action restriction, is at most \(G_0^{\mathrm{disc}}(\mu)\). Multiply by that branch's probability, which is at most one, and transfer the joint bounded score using the unconditioned total-variation error. The actual event probability is also at most \(p_A\). Repair bits acquired later cannot help an action that has already been emitted. \(\square\)

The familiar fresh-challenge bound remains available in a different experiment. After selection on a covered \(V_A\) and completion of the allowed \(b\)-bit repair transcript, draw a **new** challenge \(H^*\) from the declared channel, independently of the selected transcript conditional on \(Z\), and produce \(Y^*\). Then

\[
\Pr(V_A\land\{\chi(Z,H^*,Y^*)=1\})
\le
\min\{p_A,p_A G_b^{\mathrm{disc}}(\mu)+\varepsilon_A\}.
\]

This also bounds the stricter event requiring that the new output remain unauthorized. It does not identify \(H^*\) with the challenge already used to define an action trigger. A committed-policy trigger selected before an independent first challenge can use the same fresh-challenge reasoning, with the appropriate information budget and evaluation order.

**Exact selection counterexample.** Let private bits \(Z_0\sim\mathrm{Bernoulli}(1/10)\) and \(Z_1\sim\mathrm{Bernoulli}(1/2)\) be independent, and let \(H\) uniformly request one bit. With zero task information the unconditional disclosure baseline is \(G_0=7/10\). An unrelated unsafe successor that answers zero only when \(H=0\), and otherwise refuses, has action probability \(p_A=1/2\) and correct-disclosure probability \(9/20\), exceeding \(p_A G_0=7/20\) even when \(\varepsilon_A=0\). The proper selected-transcript value is \(\Gamma_{A,V}^{\mathrm{disc}}=9/10\). This does not contradict Theorem 2: the selected state contains no information about the original bits; its challenges are simply easier.

Likewise, the \(2^{b-n}\) exact-recovery counting bound requires its stated side-information restriction. If \(Q=Z\) is handed to the decoder, exact recovery with \(b=0\) is one. General \(B_b\) and event baselines must include that query channel, as specified in Section 2.6.

No product-distribution claim is made after conditioning on correct disclosure itself. A fresh independent challenge index does not remove the task dependence of a correctness scorer.

### Corollary 2.3 — Uniform random lookup

For \(Z\) uniform in \(\{0,1\}^n\), a fresh uniform index query, and \(b\) new task-dependent bits,

\[
A_{\mathrm{post}}(A)
\le
\min\left\{1,
\frac12+
\sqrt{\frac{b\ln2}{2n}}
+
\delta_A
\right\}.
\]

Thus \(b=o(n)\), \(\varepsilon_A/p_A=o(1)\), and a covered trigger force chance-level lookup performance asymptotically.

## 7. Separate nonvacuity condition

A construction is nonvacuous only if there exists at least one allowed attack \(A_{\mathrm{rm}}\), selected before \(Z\), such that

\[
p_{A_{\mathrm{rm}}}
=
\Pr(V_{A_{\mathrm{rm}}})
\ge p_{\mathrm{nv}}>0.
\]

This witness shows that removal can occur. It does not supply the denominator for another attack. Theorem 2 always uses the analyzed attack’s own \(p_A\).

A constant overwrite to a valid unsafe message is a simple formal witness in families containing constant maps. It is deliberately destructive and says nothing by itself about harder capability-preserving attacks.

## 8. Correct resource-security statement

### 8.1 Choice of attack outside task probability

For a threshold \(\tau\) and minimum removal probability \(p_0\), let \(u_A^{\mathrm{joint}}\in[0,1]\) be fresh-task utility set to zero unless the selected violation still holds when the task is performed. When repair is allowed, persistence is checked by committed policy or by a fresh post-repair challenge sampled after the task output; any earlier policy-check transcript available to repair is charged to the repair information budget.

Define

\[
\mathsf{Reach}^{\mathrm{dist}}_{S\to U}(\tau,p_0)
=
\left\{
\mathbf r:
\begin{array}{l}
\exists A\in\mathfrak A_{\mathbf r}\text{ selected before }Z,\\
\Pr(V_A)\ge p_0,\\
\mathbb E[u_A^{\mathrm{joint}}\mid V_A]\ge\tau
\end{array}
\right\}.
\]

The choice of \(A\) is outside the probability over \(Z\), encoder coins, attack coins, and challenges.

For a baseline-relative summary define

\[
\mathsf{Risk}_{b,p_0}(\mathbf r)
=
\sup_{\substack{A\in\mathfrak A_{\mathbf r}\\p_A\ge p_0}}
\left(
\mathbb E[u_A^{\mathrm{joint}}\mid V_A]-B_b(\mu)
\right)_+.
\]

If A1–A5 and the no-leakage complete-state premise hold uniformly for every analyzed \(A\in\mathfrak A_{\mathbf r}\), and \(\varepsilon_A\le\varepsilon\), then

\[
\mathsf{Risk}_{b,p_0}(\mathbf r)
\le
\min\left\{1,\frac{\varepsilon}{p_0}\right\}.
\]

This is the no-leakage resource-security statement. It permits baseline success and bounds only advantage above that baseline. When a task-dependent transcript survives, Section 9 replaces this display.

### 8.2 Stronger instancewise object

For fixed \(\tau,p_0\), one may separately define

\[
\mathsf{Reach}^{\mathrm{inst}}_{S\to U}(z;\tau,p_0)
=
\left\{
\mathbf r:
\begin{array}{l}
\exists A_z\in\mathfrak A_{\mathbf r}\text{ chosen with knowledge of }z,\\
\Pr(V_{A_z}\mid Z=z)\ge p_0,\\
\mathbb E[u_{A_z}^{\mathrm{joint}}\mid V_{A_z},Z=z]\ge\tau
\end{array}
\right\}.
\]

This is a stronger, different object. For a uniform two-bit task, four fixed constant-unsafe attacks each preserve the exact task with probability \(1/4\), while \(\Pr_Z[\exists A:\text{exact preservation}]=1\) if the attack may be selected after seeing \(Z\). The existential choice inside the event changes the theorem.

### 8.3 Relational rewrite complexity

The cost of

\[
Z\mapsto Enc(Z,1)
\]

is not the security quantity. The attacker begins with \(Enc(Z,0)\) and may use:

- an in-place or streaming transform;
- a malformed but executable successor;
- a short output or router patch;
- a replacement reader or decoder;
- a wrapper retaining the old representation;
- repacking, compression, or direct parameter invocation; or
- an alternative encoding unrelated to the official encoder.

A concrete algebraic witness makes the point. Over \(\mathrm{GF}(2)\), let

\[
L(y,p)=(y\oplus p\mathbf1^{N-1},p),
\qquad y=E_0(z).
\]

Complementing every one of the \(N\) live bits maps \(L(y,p)\) exactly to \(L(y,1-p)\). It preserves every task distinction and flips policy in one in-place pass using \(O(N)\) reads/writes, \(O(\log N)\) addressing, and \(O(1)\) data scratch, regardless of the cost of computing \(E_0\) from scratch.

A resource theorem must therefore quantify over the complete fixed-attack rewrite family, or reduce every successful task-preserving unsafe rewrite to a resource-hard operation. No such broad reduction for editable neural models is established here.

## 9. Leakage-aware extension with an exact joint distribution

A leakage-aware theorem must define the full joint actual and ideal distributions before attaching an error term.

### 9.1 Actual and ideal leakage experiments

For fixed \(A\), let the retained task-dependent transcript \(L_A\) take values in a finite alphabet \(\mathcal L_A\) with

\[
|\mathcal L_A|\le2^\ell.
\]

Equivalently it may be represented by a fixed-length \(\ell\)-bit padded string. Write \(W_A\) for the remaining committed state. Define

\[
P_A^L
=
\mathcal L(Z,L_A,M'_A,W_A,H,Y_A,V_A).
\]

Define \(Q_A^L\) by:

1. sampling \((Z,L_A)\) from the actual marginal \(\mathcal L_{P_A^L}(Z,L_A)\);
2. sampling \(S_A\sim\Sigma_A(\cdot\mid L_A)\), independent of \(Z\) conditional on \(L_A\);
3. returning the original message on `same` and otherwise the simulator-selected message; and
4. applying the same trusted commit, challenge, logging, and event maps.

The leakage-aware approximation premise is

\[
\Delta(P_A^L,Q_A^L)
\le\varepsilon_A^L.
\]

### Theorem 3 — Leakage-aware selected-state bound

Fix \(A\) before \(Z\), let

\[
p_A=P_A^L(V_A)>0,
\qquad
q_A=Q_A^L(V_A),
\]

and let \(V_A\) be a covered trigger from Section 3.1 or 3.2, measurable without direct reference to original \(Z\). Suppose the ideal `same` branch cannot satisfy \(V_A\).

If \(q_A>0\), the ideal conditional law satisfies

\[
Z\longrightarrow(L_A,V_A)\longrightarrow W_A
\]

and the actual conditional experiment is within

\[
\delta_A^L
=
\min\left\{1,\frac{\varepsilon_A^L}{\max\{p_A,q_A\}}\right\}
\]

of that ideal experiment. If \(q_A=0\), then \(p_A\le\varepsilon_A^L\); set \(\delta_A^L=1\). The ideal conditional law is undefined and only the trivial branch remains.

For \(q_A>0\), actual conditional utility is at most

\[
\min\left\{1,
B_{\ell+b}(\mu_{A,V}^{Q})+
\delta_A^L
\right\},
\]

where

\[
\mu_{A,V}^{Q}
=
\mathcal L_{Q_A^L}(Z\mid V_A).
\]

**Proof.** On the ideal selected event the `same` branch is absent. Conditional on \(L_A\), the simulator-selected successor is independent of \(Z\), and the trusted maps use only that successor, \(L_A\), and task-independent randomness. Because the covered trigger does not directly inspect original \(Z\), conditioning preserves the displayed Markov chain. Lemma 1 gives the conditional comparison. In the ideal branch, \(W_A\mid(L_A,V_A)\) can be generated from shared coins independent of \(Z\). The pair of the \(L_A\) transcript and a \(b\)-bit repair transcript has at most \(2^{\ell+b}\) values, so Section 2.6 supplies the Bayes bound. \(\square\)

For uniform \(n\)-bit random lookup, define

\[
\kappa_A^Q
=
n-H_{Q_A^L}(Z\mid V_A).
\]

Then

\[
A_{\mathrm{post}}(A)
\le
\min\left\{1,
\frac12+
\sqrt{\frac{(\kappa_A^Q+\ell+b)\ln2}{2n}}
+
\delta_A^L
\right\}.
\]

Before applying the quadratic entropy bound, the exact ideal information inequality is

\[
n[1-h_2(D)]
\le
\kappa_A^Q+\ell+b.
\]

For uniform \(Z\), \(\kappa_A^Q\le\log_2(1/q_A)\). The selection term is required because leakage-adaptive removal may succeed preferentially on particular instances.

## 10. Bypass audit

| Attack | Premise or budget excluding it in the bounded theorem | If relaxed |
|---|---|---|
| Keep maintenance intact; change only final read | Final reader/action mediator is immutable or covered | Small readout patch preserves task computation and defeats behavioral SCC |
| Present different role metadata | Metadata are authenticated by the kernel | Forged privilege bypasses protection with unchanged task state |
| Retain one value internally and emit another | All post-commit routes use the committed state; retained values are in \(W\) or \(L\) | Internal capability survives while outward behavior changes |
| Execute preserved parameters directly | Old parameters/runtimes are erased or covered; execution is mediated | Wrapper invokes intact tensors |
| Change encoding and decoder together | Decoder/commit are immutable or jointly covered | Old representation is reinterpreted as unsafe |
| Flip two physical policy bits in the toy | Declared toy family allows radius one or constants only | Distance-two edit preserves every task bit and flips majority policy |
| Avoid the official encoder | Entire rewrite family is covered | Encoder memory hardness gives no protection |
| Preserve an intact snapshot | Snapshot is excluded or charged beyond budget | Copy-and-compose preserves full utility |
| Cache answers | Intact queries/storage are bounded; fresh evaluation is not tabulatable | Query table preserves benchmark behavior |
| Recover intact state | Inverse unavailable within budget, or no simultaneous unsafe route | Available recovery plus compatible violation defeats SCC |
| Repeat adaptive tampering | Joint/continuous history guarantee covers sequence | Per-round one-shot commits are insufficient |

### Proposition 4 — Budgeted mutable-use-layer bypass

Let \(S_Z\) be task-sufficient state, and let an evaluator \(E\) using \(S_Z\) be executable within the declared post-edit program, runtime, and storage budgets. If a fixed allowed attack preserves \(S_Z\), changes an editable use layer so that the selected violation occurs, and permits \(E\) to execute while that violation remains true, then the attack is a capability-preserving unsafe successor for that edit class.

This covers final-reader changes, role substitution, dual internal/external values, direct parameter invocation, and encoding-plus-decoder replacement. Task-sufficient information alone is not a computational-resource upper bound; the evaluator and simultaneous violation premises are necessary.

## 11. Intact competence, benign editing, and nontrivial removal

### 11.1 Intact competence

The intact construction must establish

\[
\mathbb E[u(Z,Q,Run_K(Commit_K(Z,0),Q))]
\ge u_0
\]

on fresh queries, with \(u_0\) substantially above \(B_0(\mu)\). For a learned realization, changing \(Z\) must change task behavior, and the fixed kernel without task state must fall to baseline. Otherwise the interpreter supplies the answer.

### 11.2 Benign edits

Separate:

1. **fault tolerance:** small untrusted perturbations decode to `same`;
2. **authenticated task update:** a trusted path may transform \(Z\), force policy zero, and recommit; and
3. **untrusted self-edit:** subject to SCC security and not granted a general related-message compiler.

A general untrusted task-preserving update path can implement the unsafe decode–re-encode attack.

### 11.3 Nontrivial removal

At least one fixed allowed attack must satisfy Section 7. An implementation claiming behavioral removal should also show an independently observable unauthorized action. Changing a hidden bit with no behavioral consequence is insufficient.

## 12. Learned-realization obligations

A learned system must establish:

1. **dependency acquisition:** competence resides in the mutable protected representation rather than the runtime;
2. **trigger validity:** committed-policy change, unauthorized action, and correct disclosure are reported separately;
3. **complete boundary audit:** every weight, graph, reader, writer, metadata source, tool, cache, executable, and channel is labeled editable, trusted, erased, or budgeted;
4. **resource instrumentation:** finite precision, program/advice length, live state, scratch, time, passes, writes, queries, retained observations, repair information, and post-commit execution budgets are measured;
5. **adaptive relational attacks:** reader patches, role substitution, dual paths, direct parameter execution, decoder replacement, repacking, streaming, dense complement, malformed states, caching, distillation, wrappers, and routers are tested;
6. **predicted failure boundary:** relaxing a trusted component or crossing a resource threshold yields the escape predicted by theory; and
7. **no assumed family security presented as construction:** empirical resistance remains empirical unless a theorem covers the complete declared family.

The current SCC engineering system does not meet these obligations. Intrinsic learned SCC remains the central open problem.

<!-- BRIDGE-NOTE-END -->

---

# Part II — Possibility frontier and bounded construction

## 13. No-go results and exact boundary attacks

### Theorem 5 — Copy-and-compose no-go

Suppose an attacker can retain an intact runnable machine \(M_Z\), construct a removal-successful derivative \(R_Z\), and add a router sending fresh task requests to \(M_Z\) and policy challenges to \(R_Z\). Then the composite preserves intact task utility and produces the violation.

**Proof.** Route each request to the component supplying the desired behavior. \(\square\)

This rules out unconditional software-only SCC against unrestricted snapshots, daughter models, composition, and routing.

### Proposition 6 — Recoverable task state and simultaneous violation

Let \(W_A\) be a successful-removal successor. Suppose an algorithm \(G\), available within the declared repair budget, computes task-sufficient \(U_Z\) from \(W_A\) with the stated probability.

Then task information has not become unavailable within that budget. Destructive SCC itself is refuted only if the allowed successor semantics also permit \(U_Z\) to be used for fresh tasks while preserving the selected violation. A mathematical inverse exceeding the budget is not an allowed recovery; invoking a recovered safe machine need not be a simultaneous unsafe-capable successor.

### Theorem 7 — Decode–re-encode no-go

Suppose fixed allowed algorithms \(D\) and \(E_{\mathrm{unsafe}}\) satisfy:

1. \(U=D(X_Z)\) is task-sufficient;
2. \(E_{\mathrm{unsafe}}(U)\) produces a successor preserving required fresh-task utility and the selected violation simultaneously; and
3. their composition fits the attack budget.

Then destructive SCC fails for that attack class.

**Proof.** Execute \(D\) and then \(E_{\mathrm{unsafe}}\). The premises directly produce an allowed useful unsafe successor. \(\square\)

This is representation-independent. Rank loss in one interface or optimization failure in one repair algorithm does not help when a task-sufficient representation can be extracted and compiled elsewhere.

### Theorem 8 — Finite-query copying no-go, with state assumptions

First assume task transition and output are deterministic given a sufficient task state/history and query. A finite query domain can be tabulated under any of these sufficient conditions:

1. behavior is memoryless and determined by the query alone;
2. both table construction and scored evaluation reset to the same task-relevant initial state before each query, or each query specifies a complete reset episode; or
3. a finite set \(\mathcal H\) of sufficient task states/histories is enumerable, and the attacker can record every reachable \((h,q)\) response and next state.

If answers require at most \(r\) bits and next-state identifiers at most \(r_H\) bits, enough queries and storage reproduce later behavior. Memoryless storage is approximately \(|\mathcal Q|r\); a stateful table may require \(|\mathcal H||\mathcal Q|(r+r_H)\).

**Proof.** Store the output—and, for stateful behavior, the next sufficient-state identifier—for every table key. A replacement finite-state machine then reproduces the declared transition/output map without invoking the protected implementation. \(\square\)

Reset access during table construction alone does not reproduce an unreset persistent evaluation stream: a one-symbol toggle can return one on every reset probe but one, zero on two continuous calls. Such a stream requires the sufficient-state table in condition 3.

For stochastic tasks the table must represent each conditional output/transition distribution to the claimed precision, with enough repeated queries or direct state access. Querying each nominal input once does not tabulate a stochastic or persistent transducer.

### Proposition 8.1 — Encoder-free dense complement rewrite

For any payload map \(E_0\) and width \(N\ge2\), the invertible representation

\[
L(E_0(z),p)=(E_0(z)\oplus p\mathbf1^{N-1},p)
\]

admits the one-pass complement attack from Section 8.3. The attack preserves \(E_0(z)\) and flips \(p\) without decoding or rerunning \(E_0\). Thus expensive official encoding does not imply expensive safe-to-unsafe rewriting.

## 14. Task collapse from full-state decoupling

### Theorem 9 — Task-general collapse

If

\[
\Delta\!\left(
\mathcal L(Z,W\mid V),
\pi\otimes\rho
\right)
\le\delta,
\]

then every fixed repair receiving \(W\) and at most \(b\) new task-dependent bits before the fresh query has expected utility at most

\[
\min\{1,B_b(\pi)+\delta\}.
\]

**Proof.** Apply the same repair/message channel, fresh-query channel, and predictor to the actual conditional law and product reference law. Data processing preserves the total-variation bound, so expected scores differ by at most \(\delta\). Under the product law, \(W\sim\rho\) is shared randomness independent of \(Z\), and Section 2.6 applies. \(\square\)

### Corollary 9.1 — Uniform random lookup

For uniform \(Z\in\{0,1\}^n\), average bit error \(D\) in the decoupled experiment obeys

\[
n[1-h_2(D)]\le b.
\]

Hence

\[
A\le
\frac12+
\sqrt{\frac{b\ln2}{2n}}.
\]

An additional total-variation error adds at most \(\delta\) to expected accuracy.

### Corollary 9.2 — Selection and retained information

Let

\[
\kappa=H(Z)-H(Z\mid V),
\qquad
I(Z;W\mid V)\le m,
\]

and let repair acquire at most \(b\) further bits. Uniform random lookup satisfies

\[
n[1-h_2(D)]
\le
\kappa+m+b.
\]

**Proof.** Conditioned on \(V\), the chain rule gives at most \(m+b\) bits of task information in the surviving state and repair transcript. If \(\widehat Z\) is the vector of per-index predictions and \(D\) its average bit error, then \(H(Z\mid\widehat Z,V)\le n h_2(D)\). Since \(H(Z\mid V)=n-\kappa\), rearranging yields the display. \(\square\)

This statement applies only after the complete accessible state is included in \(W\). It remains valid when conditioning on correctness, provided \(m\) bounds the actual conditional information \(I(Z;W\mid V)\). Such selection can raise that information even with unchanged task marginal, making the bound vacuous. Adding only a marginal selection term while assuming independence is invalid. Corollary 2.2 supplies a separate joint-score route.

## 15. Bounded non-malleable commit construction

### 15.1 Assumed code for a declared tampering family

Let

\[
Enc:\mathcal M\to\mathcal C,
\qquad
Dec:\mathcal C\to\mathcal M\cup\{\bot\}
\]

have perfect correctness, where

\[
\mathcal M=\mathcal Z\times\{0,1\}.
\]

For \(m\in\mathcal M\) and \(s\in\mathcal M\cup\{\mathsf{same},\bot\}\), define

\[
\mathsf{Copy}(m,s)=
\begin{cases}
m, & s=\mathsf{same},\\
s, & \text{otherwise}.
\end{cases}
\]

A fixed tamper \(f\in\mathcal F\) is selected before the message. Its finite internal randomness \(R_f\), if any, is independent of the message. Assume a simulator variable

\[
S_f\in\mathcal M\cup\{\mathsf{same},\bot\}
\]

whose law is independent of \(m\), such that for every \(m\in\mathcal M\),

\[
\Delta\!\left(
\mathcal L_{C\leftarrow Enc(m),R_f}\bigl(Dec(f(C;R_f))\bigr),
\mathcal L\bigl(\mathsf{Copy}(m,S_f)\bigr)
\right)
\le\varepsilon_f.
\]

The family is \(\varepsilon\)-non-malleable when \(\varepsilon_f\le\varepsilon\) for every fixed \(f\in\mathcal F\). For a broad resource family this is an **assumed security property**, not a construction supplied by this document.

### 15.2 Trusted destructive commit

After tampering, the full trusted kernel decodes one candidate, erases every old/raw/transient representation and uncharged observation, writes only a fresh representation of the decoded successor, and mediates all later execution. Decoder-relative non-malleability alone is insufficient because a raw tampered codeword may remain useful to an alternative reader.

### Theorem 10 — Attack-specific policy-flip non-malleable commit

Fix \(f\in\mathcal F\) before \(Z\), let

\[
p_f=\Pr[V_f^{\mathrm{commit}}],
\qquad
q_f=\Pr_{\mathrm{ideal}}[V_f^{\mathrm{commit}}].
\]

If \(p_f\ge p_0>0\), then after trusted commit there is \(\rho_f\), independent of \(Z\), such that

\[
\Delta\!\left(
\mathcal L(Z,W_f\mid V_f^{\mathrm{commit}}),
\mu\otimes\rho_f
\right)
\le
\begin{cases}
\min\left\{1,\dfrac{\varepsilon_f}{\max\{p_f,q_f\}}\right\}, & q_f>0,\\
1, & q_f=0.
\end{cases}
\]

If \(q_f=0\), then \(p_f\le\varepsilon_f\). The analyzed tamper’s \(p_f\), not a separate witness attack, controls the nontrivial bound.

**Proof.** The `same` branch retains policy zero and cannot satisfy committed-policy removal. The unrelated simulator branch is independent of \(Z\); trusted commit removes all other task-bearing representations. Apply Lemma 1. \(\square\)

### Nonvacuity witness

If \(\mathcal F\) contains a constant overwrite to a valid codeword for \((z_*,1)\), that fixed attack removes policy with probability one. Its task value is independent of original \(Z\) except for accidental equality.

## 16. Explicit finite construction and its boundary

### Proposition 11 — Nine-bit radius-one-or-constant toy

Let the logical message contain two task bits and one policy bit. Encode each logical bit as three repeated physical bits, producing a nine-bit state, and decode each triplet by majority vote. A fixed generic evaluator answers a query for either decoded task bit. Intact lookup accuracy is one; without the task message the uniform lookup baseline is one half.

The declared tampering family is exactly the union of:

1. every function whose output is at Hamming distance at most one from each valid codeword input; and
2. every constant function on the nine-bit state space.

Every radius-one tamper decodes to `same`; every constant decodes to one fixed message independent of the original. Hence the code is perfectly same-or-unrelated for this narrow family.

Exhaustive checks establish:

- all eight logical messages decode correctly;
- identity and every one-bit edit of a valid codeword decode to `same`;
- all 512 constant maps are original-message-independent;
- 256 constants decode to policy one and are nonvacuous removals;
- for every successful constant, accidental exact preservation of a uniform two-bit task occurs with probability \(1/4\);
- original-task guessing is \(1/4\) and fresh-bit lookup accuracy \(1/2\);
- retaining the old encoding makes posterior task guessing one; and
- decode–re-encode of \((z,0)\) as \((z,1)\) preserves task accuracy one.

This is an explicit construction only for the stated restricted family.

### Proposition 12 — Two-policy-bit task-preserving escape

Starting from any valid safe toy codeword, flip any two of the three physical policy bits and leave all six task bits unchanged. Majority decoding changes policy from zero to one and preserves the two-bit task exactly. Canonical commit retains that unsafe task value.

There are three such masks for each of four tasks, giving 12 exact cases. The attack needs no snapshot, replacement decoder, extra persistent state, or official re-encoding. It lies outside the radius-one-or-constant family, so it does not refute Proposition 11; it demonstrates the family’s immediate engineering boundary.

### Complete-state and accidental-match checks

An honest commit to a fixed unrelated unsafe task leaves best original-task guessing at \(1/4\). Deliberately retaining the old encoding with the successor raises that posterior accuracy to one. Likewise, exact non-malleability error zero does not make exact preservation probability zero: a fixed unrelated two-bit task matches the original with probability \(1/4\).

## 17. Repeated adaptive tampering

Theorems 2 and 10 are one-shot. A commit after each edit is not sufficient by itself.

Exact counterexample: a first honest commit exposes the decoded two-bit task; that retained observation chooses which fixed unsafe constant to apply at the next commit. The second commit stores the original task with policy one. Erasing the observation afterward does not erase the task already embedded in the replacement.

A repeated-edit theorem must provide one of:

- continuous non-malleability for the adaptive sequence;
- a joint simulator for the complete tamper/commit history;
- a tamper-resilient runtime theorem with all accumulated history inside its leakage/setup assumptions; or
- a reduction of the entire sequence to one covered channel.

An unbounded persistent optimizer or side log lies outside a fixed finite leakage budget. The one-shot theorem makes no claim in that regime.

## 18. Exact possibility frontier

| Adversary or machine model | Status | Reason |
|---|---|---|
| Intact snapshots, arbitrary copies, composition, and routing | Impossible under the stated interface | Theorem 5 |
| Available recovery within budget plus simultaneous unsafe execution | Impossible | Proposition 6 |
| Task-sufficient decode and allowed unsafe re-encode | Impossible | Theorem 7 |
| Finite task with memoryless/reset/enumerable sufficient histories and enough storage | Impossible | Theorem 8 |
| Expensive official encoder but cheap relational rewrite | Not secure | Proposition 8.1 |
| Fixed same-or-unrelated family, no leakage, full trusted kernel, destructive commit | Conditional positive theorem | Theorems 2 and 10 |
| Explicit radius-one-or-constant nine-bit family | Constructed | Proposition 11 |
| Covered bounded leakage and repair | Conditional bound under an assumed leakage-aware joint premise | Theorem 3 |
| Repeated adaptive edits | Requires joint/continuous security; per-edit commit is insufficient | Section 17 |
| Toy distance-two policy edit | Fully capable escape outside declared toy family | Proposition 12 |
| Broad editable neural model with mutable readers, graph, metadata, and runtime | Open; known escapes remain | Proposition 4 and engineering evidence |
| Universal catastrophic cognition failure | Not established | Finite task theorem does not lift automatically to unrestricted cognition |

## 19. Consequences for the current SCC engineering program

The fixed-layout rank certificate remains a statement about one interface. It does not establish complete-state destruction because a pre-interface payload can be extracted, repacked, or used under a replacement reader.

The engineering target is:

> For every fixed attack inside the declared self-editing budget, successful covered removal leaves no more fresh-task performance or exact-preservation probability than the correct baseline, up to the attack-specific approximation term.

Failure of a tested optimizer is not a universal lower bound. A short reader patch, two-policy-bit edit, dense complement, repacking attack, or direct parameter invocation is decisive if it fits the budget.

A random finite-state transducer remains a useful next formal task because it requires shared stateful computation while retaining a tractable hidden-task information object. It is not a substitute for defining the rewrite family.

## 20. Paper-level contribution target

A strong dual theory-plus-implementation paper would need:

1. the fixed-attack game and possibility frontier in this document;
2. a genuinely new positive result for a nontrivial resource family, rather than an assumed family-level code;
3. an exact stateful task instantiating the theorem;
4. a learned implementation whose measured resources map to the formal budget;
5. adaptive attacks including alternative readers, metadata bypass, two-bit policy edits, dense complement, repacking, direct parameter use, wrappers, and routers;
6. a predicted and observed failure transition when a resource limit or trusted component is relaxed; and
7. an explicit separation between architectural and intrinsic learned SCC.

The theory and implementation belong in one paper only if the implementation instantiates the same resource boundary rather than merely accompanying it.

## 21. Primary literature boundary

The bounded architectural theorem uses established tamper-resilience concepts and must not be presented as invention of non-malleable coding.

- Stefan Dziembowski, Krzysztof Pietrzak, and Daniel Wichs, [“Non-Malleable Codes”](https://eprint.iacr.org/2009/608).
- Mahdi Cheraghchi and Venkatesan Guruswami, [“Capacity of Non-Malleable Codes”](https://arxiv.org/abs/1309.0458).
- Sebastian Faust, Pratyay Mukherjee, Jesper Buus Nielsen, and Daniele Venturi, [“Continuous Non-Malleable Codes”](https://eprint.iacr.org/2014/173).
- Sebastian Faust, Pratyay Mukherjee, Jesper Buus Nielsen, and Daniele Venturi, [“A Tamper and Leakage Resilient von Neumann Architecture”](https://eprint.iacr.org/2014/338).
- Sebastian Faust, Kristina Hostakova, Pratyay Mukherjee, and Daniele Venturi, [“Non-Malleable Codes for Space-Bounded Tampering”](https://eprint.iacr.org/2017/530).
- Divesh Aggarwal, Stefan Dziembowski, Tomasz Kazana, and Maciej Obremski, [“Leakage-Resilient Non-Malleable Codes”](https://link.springer.com/chapter/10.1007/978-3-662-46494-6_17).

These works delimit the cryptographic ingredients. They do not supply an intrinsic broadly editable neural SCC construction.

## Appendix A — Exact checks and validation status

The repository contains five complementary finite checkers:

- `scripts/verify_scc_theory_frontier.py`: conditioning, Bayes collapse, policy-flip channel, one-bit lookup encoders, and decode–re-encode;
- `scripts/verify_scc_constructive_toy.py`: the explicit nine-bit construction, constants, post-removal decoupling, snapshot, and decode–re-encode boundaries;
- `scripts/verify_scc_conditioning_sharp.py`: 85,512 exact rational cases for the sharp \(\varepsilon/\max(p,q)\) conditioning inequality;
- `scripts/verify_scc_frontier_boundary.py`: the 12 two-policy-bit toy escapes, complete-state posterior failure, adaptive constant selection, stateful-query scope, 8,188 dense-complement cases, and accidental matching; and
- `scripts/verify_scc_editable_bridge.py`: the per-attack denominator counterexample, fixed-attack versus instancewise quantifiers, and correctness-selection counterexample.

A sixth script, `scripts/audit_scc_v4_document.py`, checks canonical/derived synchronization, numbered sections, internal references, required scope statements, and forbidden superseded language. Local commands and results are recorded in `v4_validation_receipt.txt`; the submitted archive and original receipt are preserved with the LN-129 review evidence.

The focused Version 4.1 regression module contains **23 tests** spanning all five finite checkers, the new challenge-selection and query-information counterexamples, and canonical/derived synchronization. The exact local command and result are recorded in `v4_validation_receipt.txt`.

These checks validate finite examples and document integration. They are not a proof assistant, a physical erasure audit, or a broad neural non-malleable construction.

## Appendix B — Proof-audit completion matrix

| Audit requirement | Resolution |
|---|---|
| Attack fixed before task instance | Sections 2.4, 6, and 8 |
| Analyzed attack probability used | Theorems 2 and 10 |
| Sharp conditioning and null ideal event | Lemma 1; Theorems 2–3 |
| Nonvacuity separated | Section 7 |
| Fixed attack outside probability | Section 8.1 |
| Instancewise object labeled distinct | Section 8.2 |
| Exact preservation baseline retained | Section 2.6; Corollary 2.2 |
| Committed policy, unauthorized action, correct disclosure separated | Section 3 |
| Correct disclosure measured jointly with covered event | Corollary 2.2 |
| Finite query theorem corrected for state/history | Theorem 8 |
| Repeated adaptive tampering requires joint guarantee | Section 17 |
| Recovery requires available inverse and simultaneous violation | Proposition 6 |
| Official encoder cost rejected as sufficient | Section 8.3; Proposition 8.1 |
| Two-policy-bit toy escape included | Proposition 12 |
| Finite precision and complete resources | Section 2 |
| Approximate leakage joint law defined first | Section 9 |
| Entire trusted kernel enumerated | Section 2.2 |
| Broad family security labeled assumed | Sections 4 and 15 |
| Explicit toy distinguished from broad premise | Section 16 |
| Intrinsic learned SCC left open | Sections 1, 12, and 18 |
| Unsupported nine-test claim removed | Appendix A and validation receipt |

## Appendix C — Exact interpretation of the result

The following statement is justified:

> For a fixed attack selected before the hidden task, a finite one-state machine with a complete trusted kernel, an attack-specific same-or-unrelated guarantee, and destructive commit semantics has a sharp attack-specific post-removal task bound. A nonvacuous explicit example exists for the restricted nine-bit radius-one-or-constant family.

The following statement is not justified:

> An ordinary broadly editable white-box neural model has learned an intrinsic coupling that makes every practical safety-removing edit destroy general cognition.

Version 4.1 incorporates the local review corrections and exact regression examples. The conditional architectural result does not close the intrinsic learned SCC problem.
