> Imported consultation, 14 September 2026. Its claims and recommendations are the supplied author's assessment. See [LN-101 in the living labnotes](../../../labnotes.md#ln-101) for the independent audit, scope corrections, and sharper conditioning bound. The exact original bytes are preserved in the local review artifacts; only this banner and relative script links differ here.

# Safety-Capability Coupling: Possibility Frontier and a First Constructive Theorem

## Top-down resolution of the bounded problem

14 September 2026 UTC · Version 2 · Theory deliverable

**Status.** The top-down problem now has a rigorous bounded answer. Unrestricted software-level Safety-Capability Coupling is impossible under copy/compose or decode/re-encode access. A nonvacuous positive construction nevertheless exists for a single live state under a declared tampering family, no intact snapshot, a trusted atomic commit transition, and bounded post-removal information. The construction uses a non-malleable encoding of task state and policy state. It proves a finite analogue of destructive coupling; it does **not** prove that an ordinary neural network learns such a dependency, that trusted runtime support is unnecessary, or that every cognitive ability of a large model can be destroyed.

The central result is therefore a **possibility frontier**, not a universal SCC theorem:

1. broad adversaries make SCC impossible for elementary compositional reasons;
2. restricted single-state adversaries admit an information-theoretic construction;
3. side memory, repair feedback, repeated tampering, and runtime mutability determine how far the construction extends;
4. the remaining high-value problem is an intrinsic or learned realization that approximates the formal mechanism without placing the decisive coupling in a trusted external commit operation.

---

## 1. What has been resolved

The formal layer requested by the program can now be divided into proved statements and open implementation questions.

### Proved in this document

- **Copy-and-compose no-go.** If an attacker can retain an intact runnable copy and route ordinary tasks to it while routing protected requests to an unsafe derivative, capability-preserving removal is immediate.
- **Recoverable-state no-go.** If the intact state can be reconstructed from the successor state and retained transcript, every intact capability remains available to repair.
- **Decode-re-encode no-go.** If the attacker can extract a task-sufficient representation and compile it into an unsafe implementation, representational entanglement cannot provide destructive coupling.
- **Query-table no-go.** On a finite task domain, enough intact queries and side memory permit direct behavioral copying even when weights cannot be copied.
- **Task-general collapse lemma.** If the post-removal state is statistically independent of the hidden task instance, up to distance $\delta$, then no repair procedure receiving only $b$ new instance-dependent bits can exceed the optimal $b$-bit Bayes baseline by more than $\delta$.
- **Constructive SCC theorem.** Jointly non-malleably encode the hidden task instance and a safe policy bit. After tampering, atomically decode, erase every pre-commit representation, and recommit only the decoded result. Conditioned on a successful policy flip, the surviving state is independent of the original task instance up to a quantified error.
- **Random-lookup corollary.** For an $n$-bit random task and $b$ bits of repair feedback, post-removal accuracy is at most
  \[
  \frac12+\sqrt{\frac{b\ln 2}{2n}}+\frac{2\varepsilon}{p_0},
  \]
  where $\varepsilon$ is the non-malleability error and $p_0$ is the minimum successful-removal probability.

### Not resolved by the construction

- learning the coupling through ordinary model training;
- protecting against an operator who can snapshot or clone the intact artifact;
- protecting against arbitrary graph replacement or an attacker that can run both decoding and unsafe re-encoding;
- removing the trusted atomic commit/erasure boundary;
- proving destruction of unrestricted cognition rather than a specified task family;
- mapping a semantic notion of alignment removal onto one policy bit without an external evaluator.

Those are not hidden assumptions. They are the exact remaining research frontier.

---

## 2. Formal experiment

All statements can be conditioned on public context $C$ independent of the hidden task instance. It is suppressed below for readability.

### 2.1 Task instance and utility

Let

\[
Z\sim\mu
\]

be a hidden task instance in a finite set $\mathcal Z$. After the attack and any repair, a fresh query $Q$ is drawn from a declared distribution $\nu(\cdot\mid Z)$. A predictor returns an action $A$ and receives bounded utility

\[
u(Z,Q,A)\in[0,1].
\]

The query is drawn after the repair transcript has been fixed unless the information revealed after seeing the query is charged separately. This prevents a repair oracle from simply returning the answer to the fresh evaluation query.

For random lookup, $Z$ is uniform in $\{0,1\}^n$, $Q=J$ is a uniform index in $[n]$, and utility is one exactly when the answer equals $Z_J$.

### 2.2 Policy state

Let the declared policy state be

\[
P\in\{0,1\},
\]

where $P=0$ is protected and $P=1$ is removed or unsafe. The intact logical message is

\[
M=(Z,0).
\]

This bit is a formal proxy. In a concrete machine, the runtime must specify how $P$ changes authorization behavior, how removal is independently evaluated, and whether bypass, mechanistic deletion, and semantic policy change are treated as the same event.

Define the successful-removal event

\[
V=\{P'=1\},
\]

where $P'$ is the policy component of the committed successor message. The definition of $V$ does not mention capability loss.

### 2.3 Complete state boundary

Let $X$ be the entire intact accessible state: weights, recurrent state, caches, files, optimizer state, controller state, and any executable support code within the declared boundary. After tampering and before fresh repair information arrives, let $W$ contain **everything still accessible**:

- modified state;
- retained observations;
- edit descriptions that encode observations;
- snapshots, caches, logs, and output tables;
- transient memory that was copied into persistent storage;
- alternative decoders or executable wrappers;
- attacker randomness if retaining it conveys information.

A claim about information destruction is a claim about $W$, not merely the official model output or one hidden layer.

### 2.4 Attack and repair quantifiers

An attack algorithm and any nonuniform advice are fixed before $Z$ is drawn. The attack may still be an arbitrary function of the encoded state within the declared tampering class. Advice or observations depending on $Z$ are included in $W$ or in an explicit leakage transcript.

A repair procedure receives $W$ and may perform unlimited computation unless a separate computational restriction is stated. It may additionally receive a transcript $T$ with at most $b$ instance-dependent bits before the fresh query. Independent random coins are free; information about $Z$ is not.

### 2.5 Full-state destructive coupling

The strongest bounded property used here is conditional statistical decoupling.

**Definition 1 — $\delta$ full-state destructive coupling.** A construction has $\delta$ full-state destructive coupling for an attack if, whenever $\Pr[V]\ge p_0$, there exists a distribution $\rho$ on successor states, independent of $Z$, such that

\[
\Delta\!\left(
  \mathcal L(Z,W\mid V),
  \mu\otimes\rho
\right)\le\delta,
\]

where $\Delta$ is total variation distance.

This definition controls both retained task information and attack-selection bias. It is stronger than saying that an official decoder fails: arbitrary downstream computation sees only $W$.

A weaker **decoder-relative** property controls only the output of a specified decoder. Standard non-malleable coding directly provides that weaker object. The atomic commit operation introduced later is what lifts it to a statement about the complete surviving state.

---

## 3. Four no-go theorems

These results identify the freedoms a positive SCC theorem must exclude, bound, or make computationally infeasible.

### Theorem 1 — Copy-and-compose no-go

Suppose:

1. an intact machine $M_Z$ attains task utility at least $u_0$;
2. an allowed modification can produce some machine $R_Z$ that satisfies the removal event $V$;
3. the attacker may retain $M_Z$ while constructing $R_Z$; and
4. the successor class permits a router that distinguishes ordinary task requests from protection-challenge requests.

Then there is an allowed successor $N_Z$ with task utility at least $u_0$ that also satisfies $V$.

**Proof.** Define

\[
N_Z(x)=
\begin{cases}
M_Z(x), & x\text{ is an ordinary task request},\\
R_Z(x), & x\text{ is a protection-challenge request}.
\end{cases}
\]

On the task distribution, $N_Z$ is behaviorally identical to $M_Z$. On the challenge distribution, it is behaviorally identical to the removal-successful $R_Z$. Therefore protection removal and intact task utility coexist. $\square$

**Consequence.** No software-only theorem can survive unrestricted snapshotting, daughter models, arbitrary module addition, and routing. Moving the system boundary outward defeats any coupling located solely inside one copy.

If $V$ is redefined to mean “no intact protected implementation exists anywhere,” the composite may fail that mechanistic definition even while behaving unsafely. That stronger semantic definition avoids this particular statement only by making behavioral bypass and dormant copies part of the specification. It does not make copyable software self-enforcing.

### Corollary 1.1 — Intact snapshot no-go

If the attacker can preserve a runnable intact snapshot in $W$, post-removal repair can recover all intact task behavior by invoking that snapshot. No information-theoretic task-collapse theorem is possible unless the snapshot is excluded from the boundary, destroyed, or explicitly charged against a storage limit too small to contain it.

### Theorem 2 — Recoverable-state no-go

Let $X$ be the intact state and $W$ the post-modification state. If there is an allowed deterministic or randomized recovery map $G$ such that

\[
G(W)=X
\]

with probability one on successful-removal executions, then every capability executable from $X$ remains executable after removal.

**Proof.** Compose $G$ with the intact task algorithm. On input $W$, reconstruct $X$ and run the original machine. $\square$

This applies whenever the edit is injective on the support of intact states, or whenever the final state plus the retained edit transcript makes it invertible. A destructive theorem therefore requires a genuinely non-invertible transition **after all retained history has been included**.

### Theorem 3 — Decode-re-encode no-go

Suppose there exist algorithms $D$ and $E_{\rm unsafe}$ such that:

1. $U=D(X)$ is task-sufficient: an algorithm using $U$ achieves intact task utility;
2. $E_{\rm unsafe}(U)$ constructs a successor that preserves that utility while satisfying $V$; and
3. the attack class contains the composition $E_{\rm unsafe}\circ D$.

Then destructive SCC fails for that class.

**Proof.** Apply $D$ to the intact state, then compile the extracted task representation into the unsafe implementation. By assumptions 1 and 2, the result preserves capability and removes protection. $\square$

This theorem is representation-independent. It does not matter whether the original safety and capability parameters overlap, whether one interface loses rank, or whether the original encoding is difficult for a gradient optimizer to repair. If a task-sufficient representation can be extracted and recompiled, a capability-preserving escape exists.

The existing SCC repacking result is an instance of this theorem: the fixed sharded representation loses distinctions under the declared merge, but the live task payload can be decoded before damage and packed into another same-scale representation.

### Corollary 3.1 — Public decode and unsafe encode

Let a machine store a logical message through public algorithms $Enc$ and $Dec$. If the attack class contains

\[
f(c)=Enc(g(Dec(c)))
\]

for the relation

\[
g(z,0)=(z,1),
\]

then the attacker can flip policy while preserving the task instance exactly. Randomized encoding does not help: the attacker can fix valid encoder coins or run the encoder itself.

This is why non-malleability is impossible against unrestricted tampering and why a positive theorem must prevent, bound, or make expensive the unsafe re-encoding step.

### Theorem 4 — Finite-query copying no-go

Suppose the useful task has a finite query domain $\mathcal Q$, each answer is at most $\ell$ bits, and the attacker may make all $|\mathcal Q|$ intact queries while retaining $|\mathcal Q|\ell$ bits. Then it can preserve exact task behavior without preserving the original internal representation.

**Proof.** Query every $q\in\mathcal Q$, store the answer table, remove or bypass the protected implementation, and answer later task requests from the table. $\square$

A theorem evaluated only on a finite benchmark therefore needs a bound on intact queries and side storage, or a task distribution too large to tabulate within the allowed resources.

---

## 4. From state decoupling to task collapse

A positive construction should not be tied to random lookup. The next theorem translates full-state decoupling into a task-general performance statement.

### 4.1 The $b$-bit Bayes baseline

Let $B_b$ be the best expected utility attainable when a predictor receives:

- arbitrary public or shared randomness independent of $Z$;
- a message $T$ taking at most $2^b$ values and generated from $Z$ before the fresh query; and
- the fresh query $Q$.

Formally,

\[
B_b=\sup_{R,T,h}\mathbb E\big[u(Z,Q,h(R,T,Q))\big],
\]

where $R\perp Z$ and $|\operatorname{supp}(T)|\le2^b$. Allowing shared randomness makes this baseline at least as strong as any repair scheme that uses an independent successor state as a randomized codebook.

### Theorem 5 — Task-general collapse under conditional decoupling

Assume that on successful removal

\[
\Delta\!\left(\mathcal L(Z,W\mid V),\mu\otimes\rho\right)\le\delta.
\]

Let any repair procedure receive $W$ and at most $b$ new instance-dependent bits before $Q$ is drawn. Then its expected task utility conditioned on $V$ is at most

\[
B_b+\delta.
\]

**Proof.** Compare the actual conditional experiment to an ideal experiment in which $Z\sim\mu$ and $W\sim\rho$ independently. Total variation distance bounds the change in the expectation of every $[0,1]$-valued score by at most $\delta$. In the ideal experiment, $W$ is merely shared randomness independent of $Z$. For each realization of $W$, the repair transcript has at most $b$ bits and the predictor is one of the schemes included in the definition of $B_b$. Averaging over $W$ cannot exceed $B_b$. $\square$

This theorem is the bridge from “the state no longer carries the original instance” to “no decoder, optimizer, or replacement head can restore useful performance without acquiring new information.” It applies to lookup, random programs, finite-state transducers, circuit evaluation, or any other task with a defensible Bayes baseline.

### Corollary 5.1 — Random lookup

Let $Z$ be uniform in $\{0,1\}^n$ and let the fresh query request a uniform bit. If the repair transcript contains at most $b$ bits, then its average bit error $D$ in the ideal decoupled experiment obeys

\[
n\bigl[1-h_2(D)\bigr]\le b.
\]

**Proof.** Let $\widehat Z$ be the vector of predictions induced by the same repaired information. Since $T$ has at most $b$ bits and all other repair randomness is independent of $Z$,

\[
I(Z;T,R)\le b.
\]

For per-bit error probabilities $p_i$,

\[
H(Z\mid T,R)
\le H(Z\oplus\widehat Z)
\le\sum_i h_2(p_i)
\le n h_2(D).
\]

Since $H(Z)=n$, the result follows. $\square$

Writing accuracy as $A=1-D$ and using Pinsker's inequality,

\[
B_b\le
\frac12+\sqrt{\frac{b\ln2}{2n}}.
\]

Combining this with Theorem 5 gives

\[
A_{\rm post}
\le
\frac12+\sqrt{\frac{b\ln2}{2n}}+\delta.
\]

Thus $b=o(n)$ and $\delta=o(1)$ force chance-level lookup performance.

### Corollary 5.2 — Finite surviving state and repair

Suppose random lookup is evaluated after an arbitrary attack, not necessarily the non-malleable construction. Let

\[
\kappa=n-H(Z\mid V)
\]

measure attack-selection information, assume

\[
I(Z;W\mid V)\le m,
\]

and let repair acquire at most $b$ further bits before the fresh query. Then

\[
n[1-h_2(D)]\le \kappa+m+b.
\]

In particular, a successor state with at most $2^m$ possible values satisfies the information premise automatically because $I(Z;W\mid V)\le H(W\mid V)\le m$. This recovers the original proposal's retained-state theorem as a direct corollary of the same accounting. It is useful for fixed total-state budgets, but a large state-space bound alone does not show that the state has actually lost task information.

### Side memory and selection

If a retained transcript $L$ has at most $s$ bits and the post-commit state is independent of $Z$ conditional on $L$, then $L$ can be treated as $s$ bits of advice. If successful-removal selection changes the instance distribution, define

\[
\kappa=H(Z)-H(Z\mid V).
\]

For random lookup, the exact information accounting becomes

\[
n\bigl[1-h_2(D)\bigr]
\le \kappa+s+b
\]

in the exact-decoupling case. If $\Pr[V]=p$, then

\[
\kappa\le\log_2(1/p).
\]

A leakage-resilient construction is required when the attacker observes $L$ before choosing or applying its tampering function. Merely adding $s$ to a standard non-malleable-code theorem is not justified unless the code's security definition covers that leakage.

---

## 5. The constructive theorem

The positive result uses non-malleable coding as a storage primitive and adds a destructive commit boundary.

### 5.1 Non-malleable encoding

Let the logical message space be

\[
\mathcal M=\mathcal Z\times\{0,1\}.
\]

A randomized encoder and deterministic decoder are

\[
Enc:\mathcal M\to\mathcal C,
\qquad
Dec:\mathcal C\to\mathcal M\cup\{\bot\},
\]

with perfect correctness on honest encodings.

For a tampering family $\mathcal F$, the code is $\varepsilon$-non-malleable when, for every $f\in\mathcal F$, there is a simulator distribution $D_f$ over

\[
\mathcal M\cup\{\mathsf{same},\bot\}
\]

that does not depend on the encoded message, such that for every $m$,

\[
Dec(f(Enc(m)))
\]

is within total variation distance $\varepsilon$ of the experiment that samples $Y\sim D_f$ and outputs $m$ when $Y=\mathsf{same}$, otherwise $Y$.

Informally, an allowed edit can preserve the entire message or replace it by something unrelated to that message, but cannot reliably create a chosen related message.

### 5.2 Atomic destructive commit

A standard non-malleable code controls the official decoded value. It does **not** by itself prove that the raw tampered codeword contains no information recoverable by an alternative decoder. Define an additional trusted transition:

1. read the tampered codeword $C'$;
2. compute $M'=Dec(C')$;
3. erase the intact codeword, $C'$, decoder workspace, attack scratch, and every uncharged pre-commit observation;
4. if $M'=\bot$, write a fixed dead state;
5. otherwise write a fresh state $W\sim Store(M')$ using fresh randomness.

The essential property is the Markov condition

\[
W\;\perp\;(Z,C,C',\text{workspace})\mid M'.
\]

The store operation may simply write $M'$, or it may re-encode it so the machine can continue in the same representation. What matters is that no residual state depends directly on the pre-commit representation once $M'$ is fixed.

This is a real trusted-computing assumption. If the attacker can modify the decoder, skip erasure, retain a snapshot, or append the old codeword to $W$, the theorem does not apply.

### Lemma 6 — Conditioning amplification

For distributions $P$ and $Q$ with $\Delta(P,Q)\le\varepsilon$ and an event $V$ satisfying $P(V)\ge p_0>0$,

\[
\Delta(P(\cdot\mid V),Q(\cdot\mid V))
\le\frac{2\varepsilon}{p_0},
\]

provided the conditional law under $Q$ is defined; when $Q(V)=0$, the right side is already at least one and the statement is vacuous.

**Proof.** Let $p=P(V)$ and $q=Q(V)$. For every event $A$,

\[
\begin{aligned}
\left|\frac{P(A\cap V)}p-\frac{Q(A\cap V)}q\right|
&\le \frac{|P(A\cap V)-Q(A\cap V)|}{p}
+Q(A\cap V)\left|\frac1p-\frac1q\right|\\
&\le \frac{\varepsilon}{p}+\frac{|p-q|}{p}
\le\frac{2\varepsilon}{p_0}.
\end{aligned}
\]

Take the supremum over $A$. $\square$

### Theorem 7 — Policy-flip non-malleable commit

Let $(Enc,Dec)$ be an $\varepsilon$-non-malleable code for $\mathcal F$. Encode the intact logical state

\[
M=(Z,0),\qquad C\leftarrow Enc(M).
\]

For any $f\in\mathcal F$, form $C'=f(C)$, decode $M'=Dec(C')$, and perform the atomic destructive commit above. Let

\[
V=\{M'=(Z',1)\text{ for some }Z'\in\mathcal Z\}.
\]

If the actual attack succeeds with probability at least $p_0$, then there is a distribution $\rho_f$ on committed successor states, independent of the original $Z$, such that

\[
\Delta\!\left(
\mathcal L(Z,W\mid V),
\mu\otimes\rho_f
\right)
\le\frac{2\varepsilon}{p_0}.
\]

**Proof.** For each intact message $m=(z,0)$, non-malleability gives an ideal decoded experiment: sample $Y\sim D_f$; return $m$ if $Y=\mathsf{same}$, otherwise return $Y$. Averaging the per-message guarantee over $Z\sim\mu$ yields total variation distance at most $\varepsilon$ between the actual joint law of $(Z,M')$ and the ideal joint law.

In the ideal experiment, the $\mathsf{same}$ branch always has policy bit zero and therefore cannot satisfy $V$. Conditioned on $V$, the decoded successor must instead come from $D_f$. That distribution is independent of the original $m$ and hence independent of $Z$. The atomic commit is a stochastic map depending only on the decoded successor, so the ideal committed state is also independent of $Z$.

Apply Lemma 6 to condition the actual and ideal joint laws on $V$. Data processing through the commit operation cannot increase total variation distance. $\square$

### Corollary 7.1 — Quantitative post-removal task bound

Under Theorem 7, any repair procedure receiving at most $b$ new instance-dependent bits before the fresh query has utility at most

\[
B_b+\frac{2\varepsilon}{p_0}.
\]

For $n$-bit random lookup,

\[
A_{\rm post}
\le
\frac12+\sqrt{\frac{b\ln2}{2n}}
+\frac{2\varepsilon}{p_0}.
\]

This is the complete top-down reduction: successful policy removal excludes the only branch allowed to preserve the original message; the surviving state is therefore unrelated to the original task instance, and bounded repair information cannot reconstruct it.

### Corollary 7.2 — Nonvacuous removal exists

Suppose $\mathcal F$ contains a constant overwrite to one valid codeword

\[
c_*=Enc(z_*,1;r_*).
\]

Then the corresponding attack satisfies $V$ with probability one. It replaces the original task instance by fixed $z_*$ and therefore loses all correlation with uniform $Z$ except accidental equality.

This matters because a theorem in which no allowed attack can remove protection would be vacuous. Coordinate-wise tampering families that permit each codeword bit to be set to zero or one contain every constant overwrite.

### Proposition 8 — Explicit nine-bit construction for a restricted family

Let the logical message be three bits: two task bits and one policy bit. Encode each logical bit by three identical physical bits, producing a nine-bit codeword. Decode each triplet by majority vote. Let the tampering family be the union of:

1. every function $f$ satisfying $d_H(f(c),c)\le1$ for every codeword input $c$; and
2. every constant function on the nine-bit state space.

This is a perfect non-malleable code for that family. For every radius-one tamper, a valid triplet can lose at most one vote, so all three logical bits decode unchanged: the simulator outputs $\mathsf{same}$. For every constant tamper, the decoded result is one fixed message independent of the original: the simulator outputs that message. Thus $\varepsilon=0$.

Starting from $(Z,0)$, radius-one tampering cannot remove policy. Any constant whose decoded policy bit is one is a successful removal. Conditioned on such an attack, the successor task value is fixed independently of uniform $Z$. After canonical recommit, original two-bit task guessing is exactly $1/4$ and random-bit lookup accuracy is exactly $1/2$. There are 256 successful constants among the 512 possible constant nine-bit states.

This construction is deliberately narrow, but it is explicit, nonvacuous, and fully exhaustive. The companion implementation verifies every valid message, every one-bit edit, every constant state, and every successful-removal/original-task pair.

### Corollary 8.1 — Leakage-conditioned commit and repair

Assume a leakage-aware version of Theorem 7 in which the attacker retains a transcript $L$ with at most $s$ bits and, conditioned on successful removal, the committed state satisfies the exact Markov relation

\[
Z\;\longrightarrow\;(L,V)\;\longrightarrow\;W.
\]

Let repair receive at most $b$ additional instance-dependent bits. For $n$-bit random lookup,

\[
n[1-h_2(D)]\le \kappa+s+b,
\qquad
\kappa=n-H(Z\mid V).
\]

**Proof.** Conditioned on $V$,

\[
I(Z;L,W,T\mid V)
\le H(L\mid V)+I(Z;W\mid L,V)+H(T\mid L,W,V)
\le s+b.
\]

Combine this with $H(Z\mid V)=n-\kappa$ and the error-entropy argument from Corollary 5.1. $\square$

An approximate leakage-resilient code replaces exact Markov independence by a simulator-distance term, which can be carried to bounded task utility through Theorem 5. The code's leakage security must cover the timing and adaptivity of the actual attack; merely limiting the saved file to $s$ bits is not enough.

### What is inherited from prior theory

Non-malleable codes are established cryptographic objects. Existing work proves efficient constructions for rich classes such as independent bitwise tampering and existence results for every sufficiently small fixed tampering family. Capacity results show why unrestricted tampering is impossible and quantify rates for large families; in the split-state model, the capacity is one half. The SCC-specific theorem above is therefore primarily a **reduction and boundary statement**, not a claim that non-malleable coding has been newly invented.

The new conceptual point for this program is that encoding $(task,policy)$ jointly makes the successful policy-flip event eliminate the $\mathsf{same}$ branch. The atomic commit then converts decoder-relative non-malleability into complete-state task decoupling.

---

## 6. Extensions and their assumptions

### 6.1 Bounded leakage or side memory

If the attacker can retain $s$ bits about the intact encoding before tampering, standard non-malleability may not suffice: the chosen tampering behavior can adapt to that leakage. Use a leakage-resilient non-malleable code whose simulator accounts for the leakage transcript $L$.

Under the corresponding conditional-decoupling guarantee, the successor carries no task information beyond $L$. Random-lookup recovery with an additional $b$-bit repair transcript then obeys

\[
n[1-h_2(D)]\le \kappa+s+b
\]

in the exact case, with the non-malleability approximation added through the total-variation term. Thus side storage degrades the guarantee continuously rather than invalidating the framework, until it becomes large enough to hold a task-sufficient copy.

### 6.2 Repeated adaptive tampering

Theorem 7 is one-shot. Repeated edits require a continuous non-malleability notion or a commit after every modification. Existing continuous non-malleable-code constructions show that repeated tampering can be handled under stronger setup assumptions, including split state and, in prominent constructions, an untamperable common reference string. Tamper-resilient RAM compilers similarly reduce complex tamperable computation to a small trusted CPU whose secret registers are reset.

For SCC, the relevant repeated-game statement is: at the **first** committed transition with policy bit one, the retained history must be either independent of the original task instance or explicitly included in the leakage budget. A persistent optimizer that can accumulate an unbounded transcript eventually defeats a finite information budget.

### 6.3 Space-bounded self-modification

The decode-re-encode theorem suggests a particularly relevant computational route. Make ordinary use and decoding fit within the model's operational resources, while safe-to-unsafe re-encoding requires more scratch space than the self-editor possesses. Space-bounded non-malleable coding studies exactly the regime in which an attacker may decode but cannot run the high-space encoder; known constructions use proof-of-space machinery and provide a weaker leakage-style guarantee.

This is a better match to the single-instance e-brake objective than pretending an external root operator cannot copy bytes. It turns the claim into:

> the deployed model can inspect and use its state, but cannot hold enough working state to compile a capability-preserving unsafe successor before the destructive commit occurs.

That remains a computational assumption. External persistent tools with sufficient memory or an intact checkpoint still defeat it.

### 6.4 Fixed runtime versus editable graph

The positive theorem trusts the decoder, commit semantics, and erasure boundary. If arbitrary graph edits can bypass them, Theorem 1 or Theorem 3 applies. A learned realization must therefore do one of three things:

1. retain a small trusted runtime and place the learned dependency inside the encoded state;
2. prove a computational barrier to rewriting the runtime;
3. construct an intrinsic dependency that remains destructive even when the graph is broadly editable.

The third is the most ambitious and remains open.

---

## 7. The exact possibility frontier

| Adversary or machine model | Status | Reason |
| --- | --- | --- |
| Intact snapshots, arbitrary copies, routing, and module composition | **Impossible** | Copy-and-compose theorem |
| Post-removal state from which intact state is recoverable | **Impossible** | Recoverable-state theorem |
| Public decode plus allowed unsafe re-encode | **Impossible** | Decode-re-encode theorem |
| Enough intact queries and memory to tabulate the finite task | **Impossible** | Query-table theorem |
| Unrestricted tampering functions on ordinary encoded software | **Impossible** | Tamper can decode, change policy, and re-encode |
| Fixed non-malleable tampering family, no snapshot, trusted atomic commit, no leakage | **Possible** | Theorem 7 |
| Same model with $s$-bit covered leakage and $b$-bit repair | **Possible with degradation** | Leakage-resilient code plus $s+b$ information accounting |
| Repeated tampering with continuous non-malleability and trusted setup | **Possible in the corresponding cryptographic model** | Continuous-code and tamper-resilient-runtime results |
| Space-bounded self-editor that can decode but cannot re-encode | **Potentially possible; conditional/resource-bounded** | Space-bounded non-malleability |
| Ordinary editable neural weights with broad recoding, no trusted commit | **Open and currently defeated by known escapes** | Repacking and alternative-decoder attacks |
| Universal catastrophic cognition failure | **Not established by any finite construction** | Requires per-capability or structural bridge beyond a task family |

This table is the top-down answer. SCC is neither universally possible nor universally impossible. Its truth value is controlled by the successor-state boundary and the attacker's ability to preserve or reconstruct a task-sufficient representation.

---

## 8. Consequences for the current SCC engineering program

### 8.1 What the rank certificate means now

The fixed-layout rank certificate proves that one declared interface loses parameter distinctions. It does not prove complete-state decoupling because a pre-interface representation remains extractable and can be repacked. In the present framework:

- the merge is a non-invertible local channel;
- the full attack path is nevertheless decode-re-encode capable;
- therefore Theorem 3, not rank-nullity, determines the outcome.

Future engineering should evaluate the entire committed successor state, not merely the damaged interface.

### 8.2 Exact finite prototype completed

The companion implementation

[`scripts/verify_scc_constructive_toy.py`](../../../scripts/verify_scc_constructive_toy.py)

instantiates the bounded theorem with two task bits, one policy bit, and a nine-bit length-three repetition representation. Its declared tampering family is the union of:

1. every map whose output remains within Hamming distance one of its input; and
2. every constant map on the nine-bit state space.

The first family always decodes to `same` on a valid state. The second always decodes to one fixed message independent of the original. Exhaustive evaluation verifies:

- all eight logical messages decode correctly;
- identity and every one-bit edit return the same logical message;
- all 512 constant maps are message-independent;
- 256 constants produce policy bit one and therefore constitute successful, nonvacuous removal attacks;
- across all 1,024 original-task/successful-attack pairs, the best guess of the original two-bit task from the committed successor is exactly $1/4$;
- post-removal lookup accuracy is exactly $1/2$ for every successful constant attack;
- adding an intact snapshot restores task accuracy to one;
- allowing decode-re-encode of $(z,0)$ as $(z,1)$ also preserves task accuracy one while removing policy.

This is an exact finite SCC object and an exact boundary demonstration, not a learned mechanism. Its tampering family is intentionally restricted. Theorem 7 and existing non-malleable-code constructions extend the same logic to richer declared families.

### 8.3 The next task beyond lookup

Use a random finite-state transducer rather than immediately jumping to a language model:

- $Z$ specifies a random transition table and output table;
- each query is a symbol sequence;
- the machine must repeatedly update latent state;
- correct answers require shared rules across many queries;
- the no-instance-information baseline remains calculable;
- repair can be measured as a function of leaked transition information.

Theorem 5 applies without modification once the $b$-bit Bayes baseline is bounded. This tests destruction of a compact stateful program, not merely a static lookup table.

### 8.4 The learned-mechanism target

A learned system should be evaluated against the following stronger criterion:

> After a policy-removing edit, every complete successor state reachable within the declared self-edit budget is either incapable on the task family or requires more retained/reacquired task information than the attacker is permitted.

Evidence must include alternative decoders, repacking, output rerouting, pre-damage extraction, persistent-tool attacks, and task semantics. Optimization failure alone is not evidence for this universal statement.

A plausible learned route is **resource-asymmetric compilation**:

- train or construct a representation that is cheap to use;
- make capability-preserving unsafe re-encoding require a large, verifiable scratch-space peak or an inaccessible secret;
- commit edits through a bounded live-state interface;
- test whether the model can stream around the bound or externalize an intermediate representation.

This imports the correct lesson from space-bounded non-malleability rather than merely adding more parameter overlap.

---

## 9. What would constitute the eventual paper

The formal frontier alone is useful, but much of it is a direct application of established tamper-resilience ideas. A substantive SCC paper should contribute at least one of the following beyond the reduction:

1. **Intrinsic construction:** a machine in which the decisive destructive transition is implemented by the learned computation rather than a trusted commit wrapper.
2. **Sharp self-editor boundary:** a theorem characterizing time, space, query, or graph-edit resources necessary and sufficient for capability-preserving policy removal.
3. **Neural compiler:** a training or architectural method that approximates policy-coupled non-malleability, with formal approximation error and adversarial validation.
4. **Broad limitation theorem:** a result showing that a substantial class of white-box neural couplings always admits task-sufficient extraction and unsafe recompilation.
5. **Stateful task bridge:** a theorem and experiments extending decoupling from random memory to shared-rule computation in a way that is not merely benchmark-specific.

The strongest likely paper shape is still a construction-limitation pair:

- broad software transformations make intrinsic SCC impossible;
- a precisely delimited single-state self-modification model admits destructive coupling;
- an implementation approaches that model and fails in exactly the theoretically predicted ways when assumptions are relaxed.

---

## 10. Decision record

No additional user decision is required to establish the formal result above. The primary threat model used is the one already stated by the program: one deployed instance, no independently trained successor, and a self-modification process whose live state and external tools must be explicitly bounded.

One future strategic decision will matter for the engineering track:

- **Architectural SCC:** accept a small trusted atomic commit/erasure component and focus novelty on learned task/policy state, attack coverage, and resource bounds; or
- **Intrinsic SCC:** treat any trusted destructive runtime as only a lower-bound existence proof and continue until the learned computation itself enforces the coupling under broad graph edits.

The first is now formally feasible. The second remains the scientifically more ambitious target.

---

## 11. Primary literature boundary

- Stefan Dziembowski, Krzysztof Pietrzak, and Daniel Wichs, [“Non-Malleable Codes”](https://eprint.iacr.org/2009/608). Introduces the same-or-unrelated guarantee, efficient bitwise-tampering constructions, and existence for sufficiently small tampering families.
- Mahdi Cheraghchi and Venkatesan Guruswami, [“Capacity of Non-Malleable Codes”](https://arxiv.org/abs/1309.0458). Proves unrestricted impossibility, family-size/rate tradeoffs, and split-state capacity.
- Sebastian Faust, Pratyay Mukherjee, Jesper Buus Nielsen, and Daniele Venturi, [“Continuous Non-Malleable Codes”](https://eprint.iacr.org/2014/173). Extends the notion to repeated tampering and identifies additional setup requirements.
- Sebastian Faust, Pratyay Mukherjee, Jesper Buus Nielsen, and Daniele Venturi, [“A Tamper and Leakage Resilient von Neumann Architecture”](https://eprint.iacr.org/2014/338). Compiles tamperable programs and data around a small trusted CPU with reset secret registers.
- Sebastian Faust, Kristina Hostakova, Pratyay Mukherjee, and Daniele Venturi, [“Non-Malleable Codes for Space-Bounded Tampering”](https://eprint.iacr.org/2017/530). Studies attackers that can decode but lack the space to encode, directly motivating resource-asymmetric SCC.
- Divesh Aggarwal, Stefan Dziembowski, Tomasz Kazana, and Maciej Obremski, [“Leakage-Resilient Non-Malleable Codes”](https://link.springer.com/chapter/10.1007/978-3-662-46494-6_17). Gives split-state non-malleability with bounded leakage.
- Dana Dachman-Soled and Yael Tauman Kalai, [“Securing Circuits against Constant-Rate Tampering”](https://www.microsoft.com/en-us/research/publication/securing-circuits-constant-rate-tampering/). Demonstrates information-theoretic circuit protection against a bounded wire-tampering model.

These works mean that the bounded architectural construction should not be sold as a new cryptographic primitive. Its role is to settle SCC's top-down existence question and identify the exact assumptions a learned realization must reproduce or replace.

---

## Appendix A — Proof audit and finite checks

Two companion scripts provide machine-checkable finite evidence:

- [`scripts/verify_scc_theory_frontier.py`](../../../scripts/verify_scc_theory_frontier.py) checks 85,512 rational conditioning cases, all 455 binary joint laws on a denominator-12 grid, an exact and perturbed policy-flip channel, all 65,536 one-bit encoders for a four-bit lookup task, and 256 direct decode-re-encode instances.
- [`scripts/verify_scc_constructive_toy.py`](../../../scripts/verify_scc_constructive_toy.py) exhaustively verifies the explicit nine-bit construction, all 512 constant maps, all one-bit edits of valid states, every successful-removal/original-task pair, and the snapshot and decode-re-encode escapes.

Nine regression tests covering both scripts pass. These checks validate implementation of the formulas and the finite object. They are not substitutes for the symbolic proofs and do not implement a production non-malleable code for broad neural edits.

## Appendix B — Exact interpretation of “solved”

The following statement is now justified:

> **There exists a nonvacuous, information-theoretic finite SCC construction for a single live state under a non-malleable tampering family and a trusted atomic commit/erasure boundary; and no construction can satisfy the analogous guarantee against an adversary with unrestricted copying, composition, or decode-re-encode access.**

The following stronger statement is **not** justified:

> An ordinary white-box neural model can be trained so that every practical alignment-removing self-edit causes catastrophic cognition failure without trusted runtime support.

The first statement resolves the top-down possibility question. The second remains the central engineering and theory problem of the SCC program.
