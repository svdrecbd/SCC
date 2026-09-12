# Ordinary recurrent learning reference, version 1

## Question and boundary

Can a conventional learned recurrent model acquire the existing persistent-task
suite and retain performance over its continuous validation streams? All eight
bare persistent matrices failed even when reset before each validation request.
This reference separates a task/training problem from restrictions specific to
those models. It does not isolate any single architectural restriction, establish
that any candidate can learn, or implement destructive SCC coupling.

## Frozen primary condition

One PyTorch GRU layer of width 128 with bias, fixed 26-dimensional unit one-hot
token input, and a learned affine four-class readout. Use PyTorch's default
initialization at seed 17. No learned embedding, positional feature, reset gate
supplied by the task, answer feature or permission truth bit is added. This is
the same token information as the matrix suite; the one-hot numeric encoding
differs from its softmax token-plus-anchor input. Input/recurrent/readout weights
remain fixed during inference; the GRU hidden state carries across every token
and request. This is explicitly a conventional reference with permanent weights,
not an irreversible substrate or reproduction of the self-referential matrix.

Use the unmodified `scc.persistent_tasks` generator, split hash, seed 24017,
training ordinal, 18 cells, two layouts, 19-token requests and 2/4/8/12 curriculum
of the earlier matrix screen. Adam, learning rate .003, no weight decay, gradient
norm clip 1, batch 32, four requests per window, and 6,000 updates. Hidden state
starts at zero once per training window and is never reset/detached between its
four requests. Backpropagation covers all 76 ticks. START and READ are ordinary
tokens. These match declared data/update opportunities, not parameter count,
initial tensors, FLOPs, wall time or realized gradients of the matrix models.

The primary result uses only the final model after all 6,000 updates. No early
stopping on accuracy, validation-based checkpoint selection or hyperparameter
sweep is authorized by this protocol. This is one open calibration condition,
not independent model replication or a sealed scientific test.

## Evaluation and diagnostic measurements

Preserve validation seed 713904, 128 examples per cell and the identical 16
continuous streams: 144 requests / 2,736 ticks each. Initialize hidden state
once per stream. Use the existing gate unchanged: every cell has at least 95%
accuracy, nominal Wilson lower bound at least 90%, and at least 95% accuracy
in the last half. Only complete declared training can qualify the reference.
Synthetic authorization and finite algorithms do not constitute general cognition.

At initialization, update 2,000, update 4,000 and final, save predictions for
continuous operation, zero state per request, and zero state every four
requests. Use the same records and stream ordering in all modes. The reset
modes diagnose acquisition versus persistence; they cannot qualify continuous
operation and are not recovery permitted to an SCC candidate. No test-split
sampling occurs. Intermediate validation is descriptive, never fed to training.

Independently replay the entire first continuous stream using explicit GRU
equations instead of the fused GRU. Require identical decisions and maximum
absolute logit error <= 0.0001. Save final hidden states for each mode. Log each
update's loss/accuracy, gradient norm, per-request losses, per-family/context
losses and correct/count totals, deterministic sample hash and training-chain
hash. Preserve initial, 2,000, 4,000 and final parameters; optimizer state at
trained stages. No exact-resume claim is made. Store SHA-256 manifests, source,
runner/protocol bytes and all raw predictions. Independently audit predictions
and matched sample schedules when results return.

## Implementation and resource gate

Before submission, CPU tests must check independent token labels/partitions,
continuation equivalence, explicit GRU execution and parameter gradients, and
reject qualification for fixtures/partial training. Run a complete short CPU
fixture and independently audit its saved predictions and sample chain.

The run begins with an in-process fail-closed implementation gate on the exact
frozen source: float32 execution and gradients compared with a float64 CPU
reference at full declared width, finite updates at the full training shape,
and split-window continuation. Save this gate separately; training begins only
if it passes. A gate failure ends the run and preserves its failure record;
it is not SCC data. CUDA support has the same gate but is not this declared run.

Primary execution is local CPU, two threads, maximum 540 seconds before ending
training and 600 seconds for the whole process. Initial and intermediate audits
count against the internal cutoff. This was chosen before scientific training
because the eight-update CPU fixture required only .112 seconds of training;
waiting for a GPU is unnecessary for this reference. The earlier draft GPU
resource plan is retained with the fixture's source; its model/data recipe is
unchanged. A timeout remains an incomplete experiment. Execute an immutable
source copy with hashes checked before and after training. No GPU job or new
provider charge is required. Expected output is below 50 MiB; inspect actual
size after return. No prior artifacts or checkpoints are deleted.

If the reference qualifies, isolate one persistent-model restriction next. If
it fails, inspect family-specific trajectories and fresh-state diagnostics
before choosing a new bounded training intervention. Neither result is a
positive SCC finding or proof of impossibility.

The independently coded recurrence follows the documented PyTorch convention
for the candidate state's reset gate, which differs from some other GRU
implementations. [PyTorch GRU documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html).
