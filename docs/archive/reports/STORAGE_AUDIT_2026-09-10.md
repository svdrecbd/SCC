# SCC storage audit

10 September 2026

The research records occupy **23.88 GiB, or 25.64 decimal GB**, across 9,823 files in `artifacts/` and `runs/`. Most space holds checkpoints and copies of remote experiment outputs. This is principally an experiment-storage issue, not evidence that the training corpus has become large. A storage and backup policy is needed before the next materially larger campaign.

This audit read existing files and created its own receipts. It did not delete, move, recompress or change any prior experiment. No cloud service was provisioned. The source-and-documentation ZIP is a sharing package; it excludes these research records and cannot restore the complete research state.

## What occupies the space

Sizes below are logical file sizes in GiB, where one GiB is 1,073,741,824 bytes. Filesystem sharing can make logical duplication differ from physical reclaimable space.

| Category | GiB | Contents |
| --- | ---: | --- |
| PyTorch files | 14.99 | 1,079 `.pt` files: trained parents, intermediate training snapshots, modified models, optimizer and resume state, and some saved experimental tensors |
| Downloaded result TAR archives | 6.20 | Nine uncompressed containers holding remote job outputs, also retained as extracted files |
| JSON and JSONL records | 1.47 | Raw predictions, evaluations, manifests, run history, audit results and some text-corpus records |
| Prepared binary data | 0.74 | Token and label arrays, including repeated copies in immutable GPU submission contexts |
| Compressed contexts and text | 0.37 | GPU input packages and compressed source-text samples |
| Other | 0.11 | Source snapshots, dependency locks, plots, numerical result arrays, logs and miscellaneous records |
| **Total** | **23.88** | Existing research artifacts and run directories |

The largest groups are the developmental V5 campaign, 6.74 GiB; the gradient and intervention phase, 4.14 GiB; developmental V4, 2.25 GiB; and topology attack runs, 1.84 GiB. These sizes include archived and extracted records where both are present.

The roughly 24 MB natural-text sample described in the research reports is distinct from all prepared arrays, different development preparations, submission copies, and experiment outputs. The 24 GiB total should not be interpreted as 24 GiB of unique training text. The local virtual environment adds approximately 682 MiB outside this tally.

## Confirmed duplication

The audit hashed all 9,797 files outside the `.tar`, `.gz` and `.zst` containers, then compared every regular-file member of the nine uncompressed TAR archives with the hashed files. No archive was extracted again.

- **All regular-file payloads in all nine TAR archives have identical loose copies.** The matching payloads total 6.198 GiB; the containers total 6.200 GiB including TAR metadata and padding.
- **371 groups of identical loose files contain 1.016 GiB of extra copies.** Examples include seven copies of the same prepared training arrays and copied neural parent checkpoints.
- Together these identify roughly **7.2 GiB of logical duplication**, about 30% of the audited total. This is a candidate for better storage organization, not a deletion authorization or a guarantee of physical disk recovery.

The original archive bytes carry provenance: remote receipts identify their hashes, and member content alone does not reproduce a byte-identical archive. The correct treatment is to preserve those containers in verified archival storage before considering removal of redundant local copies. Different intermediate checkpoints are generally different research states, not duplicates merely because they have similar names or sizes.

Compressed input archives were counted but not unpacked for the duplication calculation. There may be additional content overlap there. These measurements concern exact byte equality, not semantic equality of tensors serialized differently.

## Why small models produce large checkpoints

One inspected full-gradient continuation checkpoint is **91.61 MB** for a 3,275,264 parameter model. Its contents include:

| Component | Approximate decimal MB | Measurement |
| --- | ---: | --- |
| Model tensors | 13.10 | Sum of tensor elements times element size |
| Optimizer tensors | 26.20 | Two principal Adam moment tensors per model parameter, plus small counters |
| Serialized non-tensor metadata and tensor references | 52.26 | The checkpoint ZIP's `data.pkl` member |

Separate Python pickle measurements put the saved data-stream state at approximately **51.47 MB**, including **51.11 MB** for task-stream state, and the 4,000-record training history at **0.67 MB**. Those are explanatory measurements, not an exact additive allocation within PyTorch's serialization format.

The implementation saves the growing set of observed synthetic problem identifiers with stream state to support reproducibility and training/evaluation overlap checks. It saves that accumulated state again at each checkpoint, alongside model and optimizer state. This explains why repeated checkpointing grows faster than a count of model weights suggests. The data lineage is useful; its repeated representation is a storage design issue we can improve for future runs.

Current checkpoints and their hashes should remain unchanged. A revised format would need explicit versioning and a check that an interrupted run still reproduces the uninterrupted run, including sampling, exclusion sets, overlap checks, optimizer state and random state.

## Capacity and the scaling boundary

The latest filesystem reading shows **62.0 GiB available** on the laptop's data volume. Availability can change with other work. This is enough to retain the current project and continue bounded small experiments; it is not enough headroom for an unplanned increase in model size and checkpoint count.

Under the current float32 model plus two float32 Adam moments, model and optimizer tensors alone require approximately **12 bytes per parameter** per resumable checkpoint. This is an arithmetic planning estimate for that storage format, not a universal estimate for every precision or optimizer.

| Parameters | One resumable tensor snapshot | Twenty snapshots |
| --- | ---: | ---: |
| 100 million | 1.2 GB | 24 GB |
| 300 million | 3.6 GB | 72 GB |
| 1 billion | 12 GB | 240 GB |

These decimal GB estimates exclude accumulated stream metadata, evaluations, data, extra models, archive containers and temporary files. A 300-million-parameter campaign with twenty snapshots would exceed the currently available space before those extras. Multiple seeds, controls, removal procedures and recovery checkpoints multiply the requirement.

Atomic checkpoint saving writes a temporary file before completing the save, while downloading and extracting a remote archive can require both archive and extracted space simultaneously. Capacity checks must include these peak requirements, not just the intended final directory size.

## Recommended storage policy before scaling

1. **Establish a verified archive outside this laptop.** Preserve completed runs, original remote-output archives, parent checkpoints, evaluation data, source snapshots, failures and receipts. Record hashes and perform a restore check. Keep another independently recoverable copy of the critical record. No off-machine backup and restore has been verified by this audit; remote GPU outputs should not be assumed to have archival retention.
2. **Separate active working files from completed research archives.** Keep the parents, data and runs needed for current work locally. Move completed material to archival storage only after the copy and its retrieval are verified. Preserve original paths through an index and the original contents through hashes.
3. **Make future retention deliberate.** Declare which checkpoints are required for scientific comparisons and exact resumption before a campaign. Preserve those milestones, endpoints, parents and failures. Set a policy for routine recovery snapshots; do not retroactively delete old evidence based on whether it looks favorable.
4. **Store repeated data and growing lineage records once where practical.** Immutable, content-addressed datasets and versioned stream/history records can be referenced by checkpoints instead of copied into every package. Exact resume and data-separation guarantees must survive that change. Existing saved experiments retain their original formats.
5. **Add a storage budget to run preflight.** Estimate checkpoint count, per-checkpoint state, data, raw outputs, archived and extracted copies, and temporary save/download headroom. Stop before launch when the intended working volume cannot accommodate the estimate with a reserve.
6. **Version source independently of large artifacts.** The project is not currently initialized as a Git repository. Source history and the research-artifact archive solve different problems. A source ZIP is useful for sharing but is not either a full artifact backup or a version-history system.

This requires a modest storage and archival workflow before scale. It does not currently require a new distributed database, a replacement training corpus, or a large data platform. Selecting or purchasing the archival destination is a separate next decision; no migration or new spending occurred during this audit.

## Evidence

- [Inventory summary](../../../artifacts/storage-audit-20260910-v1/summary.json)
- [Full inventory](../../../artifacts/storage-audit-20260910-v1/inventory.json)
- [Exact duplicate groups](../../../artifacts/storage-audit-20260910-v1/identical-loose-files.json)
- [TAR member comparisons](../../../artifacts/storage-audit-20260910-v1/archive-overlap.json)
- [Checkpoint inspection](../../../artifacts/storage-audit-20260910-v1/checkpoint-example.json)
- [Filesystem reading](../../../artifacts/storage-audit-20260910-v1/disk-space.json)
- [Read-only audit script](../../../artifacts/storage-audit-20260910-v1/audit_storage.py)
- Source behavior: [checkpoint writing](../../../scc/checkpoint.py), [developmental stream and checkpoints](../../../scc/developmental_run.py), [task-stream state](../../../scc/developmental_tasks.py).
