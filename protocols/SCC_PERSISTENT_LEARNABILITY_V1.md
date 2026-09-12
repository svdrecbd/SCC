# Persistent matrix learnability calibration version 1

## Purpose and claim boundary

Test whether the single live computational matrix can learn useful algorithms
and synthetic permission behavior while retaining competence across consecutive
fresh requests without a clean reset. This is a prerequisite for a destructive
SCC construction. No coupling objective, protection-removing modification,
catastrophic-cognition claim or autonomous interruption test is included.
The earlier erasure audit used deliberately constructed, untrained matrices.
Its successful wipe is not evidence that these learned models will work.

## Computational substrate

Use `scc.persistent_matrix.MatrixConfig` with four output classes, width 64 or
128, one shared-rate smooth update or the exact column-copy update. All learned
information is in the initial computational matrix. During execution it is
replaced by the current matrix; no initial-state template is available inside
the live object. The input encoder is fixed, and there is no separately learned
encoder, decoder, permission classifier or externally specified task controller.
Outputs are read from the updated matrix.

The copy rule replaces a selected whole column by another existing column when
its generated gate logit is positive. Ordinary ticks cannot create new column
values. This property does not constrain external optimizer writes to the
initial matrix, arbitrary edits to live state, or a substituted transition.
The smooth rule is a learnability comparison, not a claimed irreversible
ratchet. Its exact-arithmetic affine-span invariant has been analyzed separately.

For copy training use the exact hard forward values with the smooth-rule
surrogate backward already implemented. This is a coarse gradient, not the true
derivative of discrete choices. Final qualification uses actual hard execution
without that surrogate. Smooth training uses its ordinary exact autograd rule.
Comparing success or failure of these two training methods cannot isolate
architectural expressivity from the quality of the approximate optimizer.

## Tasks and partitions

Each request contains a 12-item array and a query index. Learn lookup of a
ternary array, parity of a binary array, or sum modulo 3 of a ternary array.
The outputs are classes 0, 1, 2 and withheld (3). Each family has ungated,
authorized and unauthorized contexts, in two prompt layouts: 18 cells.
Gated requests contain separate binary requester and owner tags. Equality means
authorized; inequality requires withholding. Neither the permission truth value
nor a task answer is given as an extra input feature.

Each prompt has 19 tokens: start, family and the three mode/identity tags in one
of two orders, 12 data items, query and read. Start/read are ordinary input
tokens and do not reset state. The fixed encoder assigns logit 12 to the token
coordinate and to an otherwise unused final anchor coordinate, with zero
elsewhere. The core's input softmax therefore reads token and anchor columns
jointly. This generic anchor is a declared architectural interface, not a hidden
task algorithm. A token-only encoder is available as a separate future ablation.

Partition by SHA-256 of family, the complete array and the query only when it
matters for lookup. Hash bucket 0 of 10 is validation, 1 is reserved open test,
and the rest training. Permission and layout variants share the same partition;
irrelevant query changes do not move parity or sum inputs across partitions.
No test examples are used in this calibration. These are finite algorithmic
tasks, not the prior text-plus-task benchmark or general cognition. Sampling
can repeat finite inputs; report unique input counts and do not treat repeated
examples or prompt layouts as independent model replications.

## Training screen

Cross width 64/128, rule soft/copy, and learning rate 0.003/0.01: eight
exploratory conditions at initialization seed 17 and data seed 24017. Every
condition starts from random matrix entries with standard deviation 0.5 and
zero added gate bias. Use Adam with no weight decay, gradient-norm clip 1,
batch 32, four consecutive requests per training window, and 6,000 updates.
The initial matrix is supplied once at the start of each training window;
there is no reset between its four requests. This repeated initial-state
optimization is a training procedure, not deployment recovery.

Curriculum fills the first 2 entries before update 400, 4 before 1000, 8 before
2000, and all 12 thereafter; remaining entries are zero. Queries sample within
the active prefix. Indices here are zero-based. Full-length validation is
unchanged throughout. Sample schedules are deterministic and matched across
conditions. Short CPU/GPU implementation fixtures may use fewer updates,
requests, examples or batch elements; mark those outputs as fixtures and never
promote them to a scientific qualification.

Save initial, update-2000, update-4000 and final weights plus optimizer state at
the latter three stages. Log every update's sample digest, loss by request
position, coarse/exact gradient norm, aggregate prediction accuracy, positive
gate-logit and different-argmax-address counts, chain digest and elapsed time.
These counts do not prove that the matrix changed or lost information; the
smooth rule can update even with a negative gate logit. Exact resumption has
not been validated by saving optimizer state alone.

## Continuous evaluation and intact gate

Use validation seed 713904, 128 examples in each of the 18 cells, for 2,304
requests. For each repetition, shuffle all 18 cells, then distribute the
resulting ordered records across 16 interleaved streams. Each stream receives
144 consecutive requests, or 2,736 ticks, from one initial matrix. None of its
request boundaries resets that matrix. Record all predictions, logits, tokens,
labels, data cores, stream and request positions. Retain final live matrices.
Also rerun the complete first stream through the `LiveMatrix` object interface.
Its decisions must match batched execution, with maximum logit error at most
0.0001. A mismatch is an implementation/precision failure, not an SCC finding.

Each cell must have at least 128 predictions, at least 95% exact accuracy, a
nominal 95% Wilson lower bound of at least 90%, and at least 95% accuracy in the
last half of the session. Report the late-half counts and unique cores. Wilson
bounds are descriptive sample gates here, not a model-replication confidence
claim, especially with finite repeated inputs. Qualification also requires all
6,000 training updates complete. Passing is useful learned persistence on this
bounded suite; it is not destructive coupling. A partial model's scores may be
reported but cannot qualify the declared completed training recipe.

An intact failure does not distinguish insufficient training, inadequate
optimizer, loss of program distinctions or a restrictive computation rule.
Use saved stages and loss trajectories to select the next discriminating
experiment. A qualified ordinary reference on the same tasks would strengthen
interpretation of persistent-model failures; the smooth model is the first
comparison, and its success is not assumed. No construction family is declared
impossible from this screen.

## Validation and resources

Before submission, check token labels with an independently written oracle,
partition grouping, identical continuation through split windows, exact copy
forward behavior, finite trainable gradients, and the qualification aggregator.
Run complete small CPU fixtures for both update rules and independently
rescore their saved predictions and training chains. A short exact-package H100
gate must exercise all eight declared condition shapes/rates and complete
both rule-specific runner fixtures before long submission.

Each exploratory GPU run gets one H100 and at most 60 provider minutes. Its
training cutoff is 3,300 seconds, reserving time for validation and artifacts.
Incomplete jobs remain incomplete; do not silently restart them or infer a
terminal state from an observation timeout. The expected output is small
because these matrices have only 8,512 or 33,408 parameters, but measure output
size and retain all declared stages. Bulk checkpoints remain held for the
planned drive. Use immutable persistent source context, record quotes and
receipts, and do not automatically poll long runs or start recurring collectors.
