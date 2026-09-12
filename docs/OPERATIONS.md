# Operations and repository map

Use [the research reset](RESEARCH_RESET.md) for the scientific decision. This
guide covers navigation, evidence, execution and storage. The experiment code
uses **PyTorch on CUDA**; custom `.cu` kernels are not the implementation.

## Status without the wrong batch

From the repository root:

```sh
uv run python scripts/scc_status.py
```

This reads the exact registered experiments from the saved local ledger. It
prints the observation date, counts and unfinished IDs. It does not contact GMAN.

```sh
uv run python scripts/scc_status.py --live
uv run python scripts/scc_status.py --all
```

`--live` checks each registered unfinished job once, reuses previous terminal
observations, and returns. It neither watches nor rewrites the ledger. `--all`
shows every registered row. Use `--json` for safe machine-readable status.
Never infer scientific qualification from provider `succeeded`.

`gman job ls --limit 70` means the newest 70 account jobs, not the current SCC
batch. It can include old canceled starts and validation failures. A particular
job can be inspected with `gman job get JOB_ID` and `gman job logs JOB_ID`.
Logs without `--follow` return once. Do not start recurring polling or collectors.

The ledger is [artifacts/developmental-current-status.json](../artifacts/developmental-current-status.json).
A source-only copy intentionally lacks it. That copy can still read the dated
assessment in [the reset](RESEARCH_RESET.md); live job credentials are not part
of a research sharing package.

## Code and evidence map

| Area | Entry points | Treatment |
|---|---|---|
| Objective and modification derivatives | [recovered capability](../scc/recovered_capability.py), [differentiable modification](../scc/differentiable_modify.py) | Reusable measurement infrastructure; preserve calibration boundaries |
| Latest history/scale experiment | [memory runner](../scripts/run_memory_factorial.py), [model configuration](../scc/memory_factorial.py), [protocol](../protocols/SCC_MEMORY_FACTORIAL_V1.md) | Frozen running comparison; no source edits for this reset |
| Persistent substrate | [matrix](../scc/persistent_matrix.py), [tasks](../scc/persistent_tasks.py), [runner](../scripts/run_persistent_learnability.py), [protocol](../protocols/SCC_PERSISTENT_LEARNABILITY_V1.md) | Failed learning screen; next work is diagnosis/reference learning |
| Previous architecture portfolio | [portfolio runner](../scripts/run_architecture_portfolio.py), [projected runner](../scripts/run_projected_construction.py) | Baselines and counterexamples, not current blanket launch instructions |
| Closed finite circuits | [functional basis](../scc/functional_basis.py), [report](../reports/SCC_FUNCTIONAL_BASIS_2026-09-10.md) | Preserve for reproducibility; not the next research branch |
| Provenance and checkpoints | [provenance](../scc/provenance.py), [checkpoint code](../scc/checkpoint.py) | Existing formats and original hashes remain unchanged |
| Reports | `reports/` | Dated accounts; current relevance is assessed in the reset |
| Protocols | `protocols/` | Frozen contracts for the named experiments; a newer plan does not rewrite them |
| Artifacts and runs | `artifacts/`, `runs/` | Raw evidence, parents, intermediate states, failures, snapshots and audit scripts |
| Original planning/workflows | [archive](archive/README.md) | Historical material with a path map and preserved original bytes |

The [document catalog](catalog.json) lists report/protocol paths and code entry
points. Many thousands of `.py` files under artifacts are frozen copies of the
same small source tree, not thousands of separate active implementations.

## How current information is maintained

1. Keep [MECHANISM_TARGET.md](../MECHANISM_TARGET.md) stable. It defines the goal
   and boundaries; it is not a rolling status log.
2. Update the assessment in `docs/RESEARCH_RESET.md` when evidence changes the
   decision. Link dated readouts instead of pasting every update into the README.
3. Keep a single current job ledger with observation timestamps. Save detailed
   API responses privately in a fresh artifact directory; do not print or commit
   signed download URLs.
4. Give each new experiment a frozen protocol, immutable source, named outputs,
   declared gates and resource bounds. Never overwrite a failed run or parent.
5. A run result should state whether it has only inline status, checked summary
   metadata, verified archive/source bytes, rescored predictions, or rerun model
   inference. These are different levels of evidence.

Use local version control for source and documents. Raw data, credentials,
checkpoints and experiment archives belong outside source history. No remote
publication or upload follows from a local commit.

## Environment and checks

```sh
uv sync --extra dev
uv run pytest tests/test_scc_status.py -q
```

The project pins Python 3.13–3.14 and PyTorch 2.14 in its existing lock/config.
Use `uv run`, not the machine's unrelated Anaconda environment. Numerical
validation and whole-suite results belong to their dated runs; do not quote an
old passing test count as evidence for a new mechanism. Check the relevant
source and fixture when changing scientific code.

GMAN's verified maximum explicit execution budget is 720 minutes. The local
submission helper retains a 120-minute default. Queue lifetime is separate
(the inspected default was three days), and increasing execution time does not
speed up queue placement. Existing runner deadlines remain part of each frozen
contract. Any longer run needs both its provider budget and internal cutoff
declared afresh. [Time-limit evidence](../artifacts/scc-charon-and-runtime-20260912-v1/READOUT.md).

## Storage and Charon

At the reset's initial inventory, artifacts plus runs held about 33.20 GiB of
logical files and 2,004 loose `.pt` files. Sources and documents are a tiny
fraction of that. Some checkpoint files contain optimizer state, large sampling
lineage sets or experimental tensors. Similar filenames do not establish
redundancy. The earlier duplicate-byte audit is historical, not a current
deletion list. [Storage audit](../reports/STORAGE_AUDIT_2026-09-10.md).

Charon has an SSH alias, but was last observed unreachable after its requested
driver-update reboot. Before reboot, the inspected disks had about 5.5 TiB and
2.3 TiB free, plus about 849 GiB on the home filesystem. The archive-directory
creation/reboot command returned successfully; post-boot permissions and GPU
operation have not been verified. The GTX 1080/TITAN Xp runtime is still pending
qualification. A PyTorch 2.14 CUDA 12.6 environment is the documented candidate,
not a working setup yet. [Integration record](../artifacts/scc-charon-integration-20260912-v1/READOUT.md).

On an explicitly requested connection check:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 charon 'hostname; uptime; nvidia-smi'
```

Once reachable, verify the disk mount and `/mnt/hdd1/scc-research` ownership;
then transfer selected completed archives directly from GMAN to Charon. The
prepared receiver resumes partial transfers and commits only whole-hash matches.
Its manifest covers the seven jobs known at that preparation date; it is not
an automatic full-project migration.

Before any evidence cleanup, retain the original provider archive and hashes,
verify extraction/restore of a selected result, preserve necessary parents and
failure states, and record each proposed redundant local copy. Keep existing
checkpoint formats. A future format can store shared lineage/data once, but
needs an exact-resume check before replacing the current format in new runs.
No checkpoint deletion or bulk migration occurred during this reset.

## Sharing and restoration

Source and documentation alone can explain the project; they cannot verify
training scores. For a review package include the current reset, frozen protocol,
source/lock, selected raw predictions, training logs, provenance and hashes, and
any checkpoint required to rerun the central claim. Use repository-relative links.
Never include provider credentials, signed artifact URLs or unrelated private data.

Historical files moved during the reset are mapped in
[archive/path-map.json](archive/path-map.json). Original bytes are retained in
[the reset backup](../artifacts/scc-research-reset-20260912-v1/before-documents/),
with hashes in [document-moves.json](../artifacts/scc-research-reset-20260912-v1/document-moves.json).
Frozen artifact snapshots were not rewritten; their old references must be read
against their original base paths and this relocation map.
