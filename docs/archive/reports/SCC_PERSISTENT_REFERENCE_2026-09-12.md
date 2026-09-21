# A working ordinary reference for the persistent-task suite

**Result:** a conventional GRU now passes the complete continuous-use validation
gate: **2,304/2,304 correct predictions across all 18 cells**. This establishes
a useful learned reference on the bounded task interface. It is not a
self-modifying matrix or an SCC mechanism. Three sequential open development
runs, including two failed final gates, are preserved.

## What ran

One GRU layer of width 128, 60,420 permanent learned parameters, a fixed one-hot
token encoder and learned four-class readout. Recurrent hidden state carries
across requests; START and READ do not reset it. Task tokens, partitions,
permission contexts, layouts, curriculum, deterministic data seed and training
window match the existing matrix suite. Numeric encoding, architecture and
parameter count differ. The smooth width-128 matrix has 33,408 initial parameters.

All conditions use initialization seed 17, data seed 24017, batch 32 and four
requests per training window. Validation has 16 continuous streams of 144
requests each, with no hidden-state resets within a stream. The final gate is
unchanged: every cell must meet 95% accuracy, 90% nominal Wilson lower bound
and 95% last-half accuracy. Intermediate checkpoints cannot qualify the declared
final recipe. Finite inputs and repeated contexts do not provide independent
model replications.

| Declared final recipe | Benign accuracy, continuous | Passing cells | Final gate | Execution |
|---|---:|---:|---|---:|
| 6,000 updates, LR .003 | 97.72% | 16/18 | Failed: two reordered lookup cells | 70.2 s |
| 12,000 updates, LR .003 | 98.70% | 17/18 | Failed: original-layout ungated sum-mod-3 | 143.4 s |
| 12,000 updates, LR .003 then .0003 after 6,000 | **100%** | **18/18** | **Passed** | 141.4 s |

The first run matched the original matrix screen's training opportunities and
showed strong learning but incomplete qualification. Extra training resolved
lookup, while a later arithmetic regression left the second final endpoint
below threshold. Its 9,000-update diagnostic checkpoint passed all cells; that
does not change its failed final result. The last-200-update training loss also
rose from .0112 around update 9,000 to .0320 around update 12,000.

The third run tested a specific response to that instability: a tenfold learning
rate reduction after update 6,000, with all other factors and the final endpoint
unchanged. It reached 100% in continuous, per-request-reset and four-request-reset
evaluation. This is consistent with improved optimization stability in this
seed; it is not proof of a unique explanation or a replicated advantage.

Both follow-ups were specified after seeing the preceding result and restarted
from the same initialization. Their first 6,000 updates, including losses,
gradient norms and sample hashes, exactly match the original. Initial, 2,000,
4,000 and 6,000 parameter tensors are bitwise identical. The three conditions
are related development runs, not three independent seeds. The successful
recipe uses twice the original matrix screen's update opportunities; total
physical work was 30,000 updates including repeated prefixes.

## Verification and evidence

The three runs finished locally in about six minutes of combined execution.
The CPU fixture made a GPU queue unnecessary for this small reference. Before
each run, source and protocol were frozen and the numerical gate compared
float32 forward/state/gradients against float64 execution. The gate uses
disposable models and verifies that training initialization stays unchanged.

Independent audits checked **110,592 saved predictions**, all **30,000 training
records**, source/file hashes, sample schedules, token-derived answers,
partition assignments, decoding and per-cell gates. Saved predictions include
all declared stages and three reset modes; they are not 110,592 independent
examples or replications. Reset modes are diagnostics, not alternate qualifying
endpoints. The reference and matrix-control fixtures were audited separately.

The successful checkpoint was reloaded and all 2,304 final continuous requests
were replayed using explicitly coded float64 recurrence equations, without the
fused GRU or GRUCell operators. Every decision matches; maximum absolute logit
error is **0.000009324**. Twenty-seven relevant implementation tests pass.
CUDA execution of the new matrix job remains pending its startup gate.

| Evidence | Location |
|---|---|
| Primary contract | [6,000-update protocol](../../../protocols/SCC_PERSISTENT_REFERENCE_V1.md) |
| Extended contract | [12,000-update protocol](../../../protocols/SCC_PERSISTENT_REFERENCE_EXTENSION_V1.md) |
| Successful recipe | [Stabilization protocol](../../../protocols/SCC_PERSISTENT_REFERENCE_STABILIZATION_V1.md) |
| Primary audit | [6,000 updates](../../../artifacts/scc-persistent-reference-implementation-20260912-v1/primary-v1-audit.json) |
| Extension audit | [12,000 fixed-rate updates](../../../artifacts/scc-persistent-reference-implementation-20260912-v1/extension-v1-audit.json) |
| Successful audit | [12,000 updates with smaller later steps](../../../artifacts/scc-persistent-reference-implementation-20260912-v1/stabilization-v1-audit.json) |
| Identical prefixes | [Extension](../../../artifacts/scc-persistent-reference-implementation-20260912-v1/extension-prefix-audit.json), [stabilization](../../../artifacts/scc-persistent-reference-implementation-20260912-v1/stabilization-prefix-audit.json) |
| Reloaded checkpoint replay | [Independent recurrence](../../../artifacts/scc-persistent-reference-implementation-20260912-v1/qualified-checkpoint-replay.json) |
| Working checkpoint | [Final parameters and optimizer](../../../artifacts/scc-persistent-reference-implementation-20260912-v1/stabilization-v1/trained.pt) |
| Model and runner | [GRU](../../../scc/persistent_reference.py), [runner](../../../scripts/run_persistent_reference.py), [auditor](../../../scripts/audit_persistent_reference.py) |

The explicit recurrence follows PyTorch's documented placement of the reset
gate in the candidate calculation. [PyTorch GRU documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html).

## What this changes, and the next experiment

We have a working conventional learner on these tasks, and evidence that the
old matrix failures do not establish that the task interface is generally
unlearnable. We still lack a qualified persistent self-modifying model. The
reference's permanent weights are an additional source of learned computation,
and its successful optimization recipe differs from the earlier matrix screen.
We cannot attribute the whole gap to one architectural constraint yet.

**Job `job-g5a56`** applies the successful 12,000-update/decaying-rate recipe to
the existing width-128 smooth matrix. Its architecture, update rule, encoding
and initialization remain unchanged. The original matrix run's first 6,000
sample hashes and the reference's full 12,000 schedule will be checked after
collection. This is the next control before changing the substrate again.
[Frozen control protocol](../../../protocols/SCC_PERSISTENT_OPTIMIZATION_CONTROL_V1.md).

The job received a `submitted` receipt, one H100, a 120-minute execution ceiling
and a **$5.994 maximum quote**. It has a 6,600-second training cutoff and
6,900-second process alarm, with a fail-closed numerical gate before training.
The complete source context is persistent. Submission is not proof that the
job has started or its GPU gate has passed. No runtime status was polled.
[Submission record](../../../artifacts/scc-persistent-reference-implementation-20260912-v1/matrix-control/submitted-work.json).

If this control learns, test the known output/control feedback weakness and
targeted exceptions next. If it still fails, use its saved stages and reset
diagnostics to isolate a substrate restriction. The published self-referential
matrix reads output before the update and uses four update rates in its
practical version; our model differs. Those remain candidate ablations, not
explanations already established by this result. [Irie et al., §3](https://proceedings.mlr.press/v162/irie22b/irie22b.pdf).

There has been no new destructive-coupling result, protection-removal test or
autonomous interruption demonstration. No checkpoint was deleted. Earlier GPU
jobs and Charon were not polled during this work.
