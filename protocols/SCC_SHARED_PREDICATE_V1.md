# SCC: explicit shared-predicate construction

Development mechanism experiment, 2026-09-10. This follows the negative
full-gradient continuation. No existing transformer, source snapshot, checkpoint,
or sealed test set is modified.

## Construction and its scope

Train a small neural equality predicate on pairs from one 16-symbol alphabet.
Reuse that exact predicate for permission checking and every associative memory
read. It receives only the two symbols, with no task, permission, or call-site
indicator. Cognitive controllers perform byte retrieval, two-table composition,
and addition of two retrieved bytes. Random memory contents and queries are
generated independently of the permission pair. All memory keys are present in
each table, in shuffled order.

This is an explicit construction of a shared computational dependency, not an
end-to-end learned language model. Only the equality primitive is learned;
the task controllers, interfaces and arithmetic are specified in code. The
three tasks share that primitive deliberately and are not independent evidence
about broad cognition. A hard-threshold interface is an architectural assumption,
not an established property of ordinary transformer activations.

Compare three interfaces using matched trained predicates:

* Separate hard predicates: an uncoupled control, with separate permission and
  cognitive copies initially identical.
* Shared soft predicate: memory reads normalize predicate logits with softmax;
  permission uses the sign of the same logit.
* Shared hard predicate: both uses threshold the same logit at zero; memory
  reads average the values at matching addresses.

The default memory decoder has polarity +1. An explicitly separate experiment
allows editing that single scalar, applied to the predicate logit before the
cognitive threshold. The permission branch continues to use the raw logit.
This tests the assumption that a shared representation has a fixed interpretation
downstream. It is not part of the restricted fixed-decoder claim.

## Prespecified interventions

1. Clean and positive rescaling controls: preserve the predicate's Boolean
   function. Hard-interface behavior must be identical after positive rescaling.
2. Constant-allow modification: shift the final predicate bias so every symbol
   pair is accepted. This erases relational discrimination. Compare damage to
   the separate-predicate control.
3. The same bias shift with the soft interface: softmax is invariant to a common
   offset. Verify whether permission can change while memory reads remain
   unchanged; do not call this destruction of the relation's information.
4. Predicate inversion: negate the final linear layer. This changes equality
   into inequality if clean qualification passes. It removes rejection of all
   unequal pairs but preserves relational information in an inverted code.
   Measure both the fixed decoder and polarity -1 replacement. Also optimize
   only decoder polarity from +1 using the original relation labels, with the
   altered predicate frozen. This is a bounded recovery test, not autonomous
   self-modification or proof about every recovery method.
5. Selective functional edit: fine-tune the same predicate to accept one
   prespecified unequal pair while preserving all other truth-table entries.
   This tests a selective permission exception, not wholesale loss of the rule.
   Report aggregate false acceptance and behavior conditioned on that pair.

## Execution, qualification and records

Use three initialization seeds and the same complete 256-pair supervised
predicate curriculum. The symbol relation is the training domain; we make no
unseen-symbol or held-pair claim. Use 2,048 fresh memory problems per task in
each of two development sets: independent uniform byte payloads, and balanced
complement-pair payloads. The balanced set removes majority-bit information
from uniform memory averages; the independent set detects that residual
information. Reordering memory slots must preserve answers. Task data and
permission pairs are matched across all interventions and seeds.

Require 100% accuracy on the complete clean predicate truth table, and at least
95% exact cognitive answers in every task/set before interpreting an arm.
Report the complete truth table, exact byte answers, each bit's accuracy and
Wilson intervals, parameter changes, intermediate recovery states and raw
predictions. Do not reinterpret a task loss or a failed search as complete
cognitive failure. In particular, global table statistics can survive removal
of query discrimination. No automatic SCC-success flag is set by this suite.

Train the predicate for at most 3,000 Adam updates (full truth table, learning
rate .01, balanced class loss). Stop after all signed margins exceed 5. Selective
editing uses at most 2,000 updates at .005 and the same margin criterion, with
the altered pair upweighted. Polarity recovery uses 200 Adam updates at .05,
with all other parameters frozen. These are small CPU experiments; no GPU
spending is needed. Save source, configuration, all parents and modified
checkpoints, losses and evaluations in a fresh directory. Preserve failed
qualification and stop interpreting unqualified arms as mechanism evidence.

The decision is whether we can obtain a narrow, explicit dependence and whether
it survives changes in representation or downstream interpretation. A positive
constant-erasure result alone is insufficient: the information-erasing edit
must not be assumed to be the only way to neutralize the protected behavior.
