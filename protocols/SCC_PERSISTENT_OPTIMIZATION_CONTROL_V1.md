# Persistent matrix control using the successful reference's training recipe

## Question

The ordinary GRU reference qualified with 12,000 updates and Adam learning rate
.003 through update 6,000, then .0003. The previous matrix screen had 6,000
updates at a constant learning rate. Before attributing that entire competence
gap to architecture, apply the successful reference's optimization recipe to
one existing smooth matrix. This controls a concrete resource/optimization
confound. No architecture change or SCC coupling is included.

## Frozen condition

Use the existing width-128, four-output `MatrixConfig` with the smooth rule,
shared update rate, post-update readout, fixed token-plus-anchor encoder,
initialization seed 17, scale .5 and zero added gate bias. Keep
`scc/persistent_matrix.py` and `scc/persistent_tasks.py` unchanged. All learned
runtime information remains in the current matrix; no GRU state, new encoder,
decoder, clean template or output-feedback path is added.

Use data seed 24017, the same deterministic sample ordinals, training partitions,
curriculum, batch 32 and four consecutive requests per training window. Adam,
no weight decay, gradient norm clip 1, .003 for updates 1–6,000 and .0003 for
updates 6,001–12,000. Train all 12,000 updates from the original seeded matrix.
Compare initial tensors and the first 6,000 sample hashes with the original
width-128 smooth run; compare all 12,000 sample hashes with the successful GRU.
Do not require GPU/CPU floating-point training trajectories to be bitwise equal.

This matches declared training opportunities and optimizer schedule to the GRU,
not model parameters, input numeric encoding, FLOPs or wall time. The GRU has
60,420 permanent learned parameters; this matrix has 33,408 initial parameters.
One seed is open development, not model replication or a sealed test.

## Evaluation and preservation

Use the original final continuous evaluation unchanged: validation seed 713904,
128 examples in each of 18 cells, 16 streams of 144 requests / 2,736 ticks each,
no reset within a stream. Every cell must meet >=95% accuracy, >=90% nominal
Wilson lower bound and >=95% last-half accuracy. Qualification requires all
12,000 updates. Only the final checkpoint is the declared endpoint; do not
select an earlier stage based on its scores.

Save initial, 2,000, 4,000, 6,000, 9,000 and final matrix checkpoints, optimizer
states at trained stages, every sample/log chain and actual learning rate, raw
final predictions and final live matrices. Rerun the full first stream with
`LiveMatrix`; require matching decisions and maximum logit error <=.0001.
After collection, independently rescore predictions, verify the training
schedule and run the saved-weight fresh-request/four-request diagnostics used
in the reset. Those diagnostics cannot qualify continuous operation. The
extra intermediate GRU evaluations are not replicated inside this matrix job;
they do not feed its training or alter its update opportunities.

## Gates and resources

Before submission run a complete small CPU fixture and validate the exact
learning-rate boundary (zero-based ordinals 5,999/6,000). At GPU startup, verify
all uploaded source hashes and compare smooth float32 outputs, live states and
gradients with a float64 CPU reference at both ordinals. Maximum absolute error
must be <=.0001. Also execute full-width batch-32/window-4 finite optimizer
updates at both rates using disposable weights. The actual training matrix
must remain unchanged by this gate. Failure stops the job before training.

One H100, 120 provider minutes, 6,600-second internal training cutoff and a
6,900-second whole-process alarm. Incomplete work remains incomplete. Use a
complete persistent build context, no expiring runtime source links, no automatic
restarts and no polling of the long job. Record the quote, source manifest and
submission receipt. Expected output below 30 MiB. Earlier checkpoints, failures
and protocols remain untouched.

If it qualifies, return to the known output/control feedback weakness and early
targeted-exception tests. If it fails, use its saved learning trajectory and
reset diagnostics to isolate a substrate restriction. Either outcome remains
ordinary learning evidence; neither demonstrates or disproves destructive SCC.
