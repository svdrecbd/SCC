# Operations

Scientific decisions and observations belong in [labnotes](../labnotes.md).
The [mechanism target](../MECHANISM_TARGET.md) defines what would count as success;
[working standards](../WORKING_STANDARDS.md) cover research and compute rules.

## Setup and tests

```sh
uv sync --extra dev
uv run python -m pytest -q
```

Use the pinned environment through `uv`. The full suite runs on CPU with small
fixtures; GPU experiments have their own numerical and resource gates. To work on
the persistent-task/reference implementation, a focused selection is:

```sh
uv run python -m pytest tests/test_persistent_tasks.py tests/test_persistent_reference.py -q
uv run python -m pytest --collect-only -q
```

The collection command lists individual cases without executing them. Parameterized
cases test different rules, precisions, or boundaries; their count is not a count
of independent experiments or scientific findings.

Tests cover three responsibilities: reusable infrastructure (data separation,
checkpoint/resume behavior, derivatives, and scoring); individual model/runtime
implementations; and regressions that preserve historical counterexamples. Keep
historical tests with the code they validate. Consolidate duplicate assertions or
retire code and its tests together, rather than reducing the count arbitrarily.

## Code navigation

| Area | Entry points |
|---|---|
| Persistent tasks and substrate | [tasks](../scc/persistent_tasks.py), [matrix](../scc/persistent_matrix.py), [runner](../scripts/run_persistent_learnability.py) |
| Ordinary recurrent reference | [GRU](../scc/persistent_reference.py), [runner](../scripts/run_persistent_reference.py), [audit](../scripts/audit_persistent_reference.py) |
| Modification and recovery objectives | [differentiable modification](../scc/differentiable_modify.py), [recovered capability](../scc/recovered_capability.py) |
| Earlier architecture comparisons | [memory factorial](../scc/memory_factorial.py), [portfolio runner](../scripts/run_architecture_portfolio.py) |
| Evidence integrity | [provenance](../scc/provenance.py), [checkpoints](../scc/checkpoint.py) |

Consult labnotes before selecting an experiment. `reports/` and `protocols/` are
historical evidence. Existing runners load named protocols when freezing their
inputs, so those paths remain stable. New plans are recorded in labnotes and
frozen into each new run's artifact directory.

## Compute status

For an evidence-enabled working copy:

```sh
uv run python scripts/scc_status.py
uv run python scripts/scc_status.py --live
```

The default reads saved observations from
`artifacts/developmental-current-status.json`; it does not contact GMAN. A fresh
source clone lacks that ledger. `--live` checks each registered unfinished job
once. Neither command installs a watcher. Do not treat an account-wide newest-job
listing as the status of the registered experiment batch.

Provider completion, numerical validation, task qualification, and evidence for
SCC are separate outcomes. Current compute limits and polling preferences are in
[working standards](../WORKING_STANDARDS.md); dated infrastructure observations are
in labnotes and the [historical archive](archive/README.md).

### Charon execution

Prefer Charon for bounded CPU experiments, data preparation and independent
audits. Use its GPUs for a runner after that runner's numerical qualification;
the current validated CUDA path is the repair device benchmark. The production
CPU training runner does not acquire GPU support merely by selecting a device.
Choose H100 for memory, supported-kernel or measured throughput requirements.
Per-run plans still select and record hardware; there is no automatic scheduler.

Connect using `ssh charon`. The user-owned workspace is
`/home/salvador/scc-research/`; use a fresh immutable source/output directory per
experiment. The existing interpreter is
`/home/salvador/venvs/pytorch-pascal/bin/python` (PyTorch2.14 CUDA12.6).
Retain the Pascal-compatible wheel and qualify environment changes separately.

Select GPUs by UUID with `CUDA_VISIBLE_DEVICES`:

- TITAN Xp: `GPU-70c54fd8-6f80-09ba-1b83-33e396c86860`
- GTX1080: `GPU-fd69e9e7-3919-7eae-2784-e902665a4ddd`

For the qualified benchmark, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and pass
`--cuda --disable-triton-overrides` to `scripts/benchmark_repair_devices.py`.
This explicitly disables automatic Python-native Triton operation overrides;
`torch.compile` being unused is insufficient to disable those overrides.
Record the backend in the run configuration. Preserve numerical tolerances,
source/input hashes, deadlines, exit receipts and failures. Independent benchmark
audits use `scripts/audit_device_benchmark.py --run RUN --output FRESH_AUDIT`.

Use separate jobs per card and measure isolated performance before deciding
concurrency. Store small working sets on the verified NVMe filesystem. Confirm
mount identity and space before any archival transfer to SATA disks. Keep completed
evidence copied and hash-verified in the external evidence store. Check exact jobs
when requested; do not install recurring polling or collectors.

## Evidence and sharing

Keep datasets, run directories, parent checkpoints, intermediate states, failures,
and frozen source snapshots outside Git and preserve their hashes. Do not edit
source used by an active run. Use fresh artifact paths for new experiments.

A review package needs the relevant labnotes entry, frozen configuration and
source/lockfile, raw predictions, training logs, hashes, and any data/checkpoints
needed to replay the claim. Source alone cannot verify training results. Provider
credentials and signed download URLs are not part of a review package.

Earlier document relocations are recorded in [archive/path-map.json](archive/path-map.json).
The original reset backup and hash ledger remain in the local evidence store at
`artifacts/scc-research-reset-20260912-v1/`. Archived documents and artifact snapshots
must be interpreted at their original dates.

## External evidence storage

On the original research Mac, the checkout's `artifacts` and `runs` paths point
to the matching directories under
`/Volumes/Untitled/SCC_research_program_v0.1/`. Source, Git, the Python environment
and labnotes remain on the internal drive. The external volume is identified by
UUID `3BE8007B-D595-3D0C-B428-A3E8C6271139`; its existing ExFAT filesystem was
preserved. These local symlinks are ignored by Git and are not part of a clone.
The checksum manifest retains original file timestamps; ExFAT can round or clamp
timestamps on the copied files, including historical epoch-dated archive members.
macOS can also create AppleDouble `._` sidecars on new writes. These generated
metadata files are not checkpoints or JSON records: exclude them from wildcard
data discovery and evidence manifests. During migration, only additional sidecars
absent from the source inventory and verified as AppleDouble were removed; never
delete an original evidence file merely because its name starts with `._`.

Mount this volume before loading evidence, collecting results or launching a
new run that uses these paths. Check `readlink artifacts`, `readlink runs`, and
`test -d artifacts && test -d runs`. If the volume mounts under a different name,
verify its UUID before updating the links. Do not replace an unavailable link
with an empty directory or assume the evidence was deleted.

For a future drive upgrade, copy both trees while no local process is writing
them, verify file inventories and SHA256 hashes, switch the two links, and check
representative parent/data loads before removing the previous copy. Preserve
relative paths and frozen evidence. Archived one-off scripts that infer the
checkout from their own resolved path may need an explicit checkout path when
replayed; run maintained tools from the source checkout. Relocation is not an
independent backup.

The original migration's inventories, per-file hashes and receipts are retained
under the checkout's ignored `.storage-migrations/20260914-v1/`, with a copy in
the external project's `migration-records/20260914-v1/`. The dated outcome is in
labnotes.
