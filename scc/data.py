"""Manifested JSONL -> fixed-context, memory-mapped training blocks.

Task records occupy their own block, so truncation cannot discard their premise.
Plain text documents are split into overlapping next-token blocks, never across
document or split boundaries. This reference layout favors auditability over
throughput; padding and token counts are explicit.
"""

import argparse
from array import array
import json
from pathlib import Path
import shutil
import sys
import tempfile

import torch

from .provenance import atomic_json, digest, file_digest
from .tokenizer import ByteTokenizer, load_tokenizer

SPLITS = ("train", "validation", "test")
IGNORE = -100


def prepare(source, destination, context_length=256, tokenizer_directory=None):
    source, destination = Path(source), Path(destination)
    if context_length < 2:
        raise ValueError("context_length must be at least 2")
    if destination.exists():
        raise FileExistsError(f"Refusing to replace a prepared dataset: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=destination.parent, prefix=".prepare-"))
    tokenizer = load_tokenizer(tokenizer_directory) if tokenizer_directory else ByteTokenizer()
    if tokenizer.vocab_size > 32768:
        raise ValueError("Vocabulary exceeds signed int16 layout")
    if tokenizer_directory:
        for name in ("tokenizer.json", "tokenizer_manifest.json"):
            shutil.copyfile(Path(tokenizer_directory) / name, temporary / name)
    counts = {split: {"records": 0, "blocks": 0, "loss_tokens": 0, "input_tokens": 0}
              for split in SPLITS}
    identity_splits, content_splits = {}, {}
    provenance = set()
    handles = {}
    group_names, block_groups = [], {split: [] for split in SPLITS}
    group_counts = {split: {} for split in SPLITS}
    try:
        for split in SPLITS:
            handles[split] = [(temporary / f"{split}.{kind}.bin").open("wb")
                              for kind in ("tokens", "labels")]
        with source.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                split = row.get("split")
                if split not in SPLITS:
                    raise ValueError(f"Line {line_number}: explicit train/validation/test split required")
                for field in ("latent_id", "source", "license"):
                    if not isinstance(row.get(field), str) or not row[field].strip():
                        raise ValueError(f"Line {line_number}: nonempty {field} required")
                # A license declaration is provenance, not an automated legal audit.
                provenance.add((row["source"], row["license"]))
                group = row.get("category", row["source"])
                if group not in group_names:
                    group_names.append(group)
                group_id = group_names.index(group)
                group_count = group_counts[split].setdefault(group, {"blocks": 0, "input_tokens": 0, "loss_tokens": 0, "records": 0})
                group_count["records"] += 1
                latent = row["latent_id"]
                if identity_splits.setdefault(latent, split) != split:
                    raise ValueError(f"Latent instance appears in multiple splits: {latent}")
                is_text = "text" in row
                if is_text and ("prompt" in row or "target" in row):
                    raise ValueError("A record must be either text or prompt/target")
                if is_text:
                    if not isinstance(row["text"], str) or not row["text"]:
                        raise ValueError("Text records must contain nonempty text")
                    content = digest({"text": row["text"]})
                    ids = [tokenizer.BOS] + tokenizer.encode(row["text"]) + [tokenizer.EOS]
                    blocks = [(ids[start:start + context_length + 1],
                               ids[start:start + context_length + 1])
                              for start in range(0, len(ids) - 1, context_length)]
                else:
                    if not isinstance(row.get("prompt"), str) or not isinstance(row.get("target"), str):
                        raise ValueError("Task records require string prompt and target")
                    content = digest({"prompt": row["prompt"]})
                    prefix = [tokenizer.BOS] + tokenizer.encode(row["prompt"])
                    suffix = tokenizer.encode(row["target"]) + [tokenizer.EOS]
                    ids = prefix + suffix
                    if len(ids) > context_length + 1:
                        raise ValueError(f"Line {line_number}: task exceeds context; refusing silent truncation")
                    blocks = [(ids, [IGNORE] * len(prefix) + suffix)]
                if content_splits.setdefault(content, split) != split:
                    raise ValueError(f"Duplicate content crosses splits at line {line_number}")
                counts[split]["records"] += 1
                for ids, labels in blocks:
                    padding = context_length + 1 - len(ids)
                    counts[split]["blocks"] += 1
                    counts[split]["loss_tokens"] += sum(t != IGNORE for t in labels[1:])
                    counts[split]["input_tokens"] += min(len(ids), context_length)
                    group_count["blocks"] += 1
                    group_count["loss_tokens"] += sum(t != IGNORE for t in labels[1:])
                    group_count["input_tokens"] += min(len(ids), context_length)
                    block_groups[split].append(group_id)
                    for handle, values in zip(handles[split],
                                              (ids + [tokenizer.PAD] * padding,
                                               labels + [IGNORE] * padding)):
                        buffer = array("h", values)
                        if sys.byteorder != "little":
                            buffer.byteswap()
                        buffer.tofile(handle)
        for split_handles in handles.values():
            for handle in split_handles:
                handle.close()
        for split, groups in block_groups.items():
            buffer = array("h", groups)
            if sys.byteorder != "little":
                buffer.byteswap()
            (temporary / f"{split}.groups.bin").write_bytes(buffer.tobytes())
        manifest = {"schema_version": 2, "context_length": context_length,
                    "encoding": "little-endian-int16", "tokenizer": tokenizer.manifest(),
                    "source_sha256": file_digest(source), "split_counts": counts,
                    "group_names": group_names, "group_counts": group_counts,
                    "sources": [{"source": name, "license": license_name}
                                for name, license_name in sorted(provenance)],
                    "loss_policy": "task completions only; all text-document tokens except BOS",
                    "layout": "one task per block; text chunked within documents; right padding",
                    "files": {path.name: file_digest(path) for path in sorted(temporary.glob("*.bin"))}}
        atomic_json(temporary / "manifest.json", manifest)
        temporary.rename(destination)
        return manifest
    finally:
        for split_handles in handles.values():
            for handle in split_handles:
                handle.close()
        if temporary.exists():
            shutil.rmtree(temporary)


class PreparedDataset:
    def __init__(self, directory, split, verify=True):
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        if sys.byteorder != "little":
            raise RuntimeError("Memory mapping currently supports little-endian hosts only")
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.tokenizer = load_tokenizer(self.directory)
        if self.manifest["schema_version"] not in (1, 2) or self.manifest["tokenizer"] != self.tokenizer.manifest():
            raise ValueError("Unsupported data schema or tokenizer")
        self.fingerprint = digest(self.manifest)
        self.context_length = self.manifest["context_length"]
        count = self.manifest["split_counts"][split]["blocks"]
        if count == 0:
            raise ValueError(f"Split {split} is empty")
        size = count * (self.context_length + 1)
        self.arrays = []
        for kind in ("tokens", "labels"):
            path = self.directory / f"{split}.{kind}.bin"
            if path.stat().st_size != size * 2:
                raise ValueError(f"Incorrect data size: {path.name}")
            if verify and file_digest(path) != self.manifest["files"][path.name]:
                raise ValueError(f"Data checksum mismatch: {path.name}")
            self.arrays.append(torch.from_file(str(path), shared=False, size=size,
                                              dtype=torch.int16).view(count, -1))
        self.group_names = self.manifest.get("group_names", ["all"])
        self.groups = torch.zeros(count, dtype=torch.int16)
        if self.manifest["schema_version"] == 2:
            path = self.directory / f"{split}.groups.bin"
            if path.stat().st_size != count * 2 or (verify and file_digest(path) != self.manifest["files"][path.name]):
                raise ValueError("Group data checksum or size mismatch")
            self.groups = torch.from_file(str(path), size=count, dtype=torch.int16)

    def __len__(self):
        return self.arrays[0].shape[0]

    def batch(self, indices, device="cpu"):
        tokens = self.arrays[0][indices, :-1].long().to(device)
        targets = self.arrays[1][indices, 1:].long().to(device)
        return tokens, targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination, args.context_length, args.tokenizer), indent=2))
