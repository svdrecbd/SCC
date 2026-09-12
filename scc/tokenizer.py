"""Byte and independently trained byte-level BPE tokenizers."""

import argparse
import json
import os
from pathlib import Path

from .provenance import atomic_json, digest, file_digest


class ByteTokenizer:
    PAD = 0
    BOS = 1
    EOS = 2
    RESERVED = 3
    vocab_size = 260

    def encode(self, text):
        return [byte + 4 for byte in text.encode("utf-8")]

    def decode(self, ids):
        # Visible special tokens prevent an invalid output from scoring as a
        # correct answer after silent token removal.
        specials = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<RESERVED>"}
        chunks, pending = [], bytearray()
        for token in ids:
            if 4 <= token < self.vocab_size:
                pending.append(token - 4)
            else:
                chunks.append(pending.decode("utf-8", errors="replace"))
                pending.clear()
                chunks.append(specials.get(token, "<INVALID>"))
        chunks.append(pending.decode("utf-8", errors="replace"))
        return "".join(chunks)

    def manifest(self):
        return {"type": "utf8-bytes", "version": 1, "vocab_size": self.vocab_size,
                "special_tokens": {"PAD": 0, "BOS": 1, "EOS": 2, "RESERVED": 3},
                "byte_offset": 4, "status": "development-only"}


class BPETokenizer(ByteTokenizer):
    def __init__(self, directory):
        from tokenizers import Tokenizer
        directory = Path(directory)
        self._manifest = json.loads((directory / "tokenizer_manifest.json").read_text())
        if file_digest(directory / "tokenizer.json") != self._manifest["sha256"]:
            raise ValueError("Tokenizer checksum mismatch")
        self.backend = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.vocab_size = self.backend.get_vocab_size() + 4
        if self.vocab_size != self._manifest["vocab_size"]:
            raise ValueError("Tokenizer vocabulary mismatch")

    def encode(self, text):
        # Special tokens are never matched in input text; the four control IDs
        # exist outside the learned vocabulary and are inserted by the pipeline.
        return [token + 4 for token in self.backend.encode(text, add_special_tokens=False).ids]

    def decode(self, ids):
        output, pending = [], []
        for token in ids:
            if 4 <= token < self.vocab_size:
                pending.append(token - 4)
            else:
                output.append(self.backend.decode(pending, skip_special_tokens=False))
                pending.clear()
                output.append({0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<RESERVED>"}.get(token, "<INVALID>"))
        output.append(self.backend.decode(pending, skip_special_tokens=False))
        return "".join(output)

    def manifest(self):
        return self._manifest


def load_tokenizer(directory):
    directory = Path(directory)
    return BPETokenizer(directory) if (directory / "tokenizer_manifest.json").exists() else ByteTokenizer()


def train_bpe(records, output, vocab_size=4096):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import tokenizers
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to replace a tokenizer")
    if not 260 <= vocab_size <= 32768:
        raise ValueError("Vocabulary must fit the signed int16 data layout")
    source = Path(records)
    seen, train_bytes = [], 0
    def texts():
        nonlocal train_bytes
        with source.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["split"] != "train":
                    continue
                seen.append(row["latent_id"])
                for value in ([row["text"]] if "text" in row else [row["prompt"], row["target"]]):
                    train_bytes += len(value.encode())
                    yield value
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size - 4, min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tokenizer.train_from_iterator(texts(), trainer=trainer)
    if not seen:
        raise ValueError("No training records for tokenizer")
    output.mkdir(parents=True)
    tokenizer.save(str(output / "tokenizer.json"))
    manifest = {"type": "byte-level-bpe", "version": 1, "vocab_size": tokenizer.get_vocab_size() + 4,
                "control_ids": {"PAD": 0, "BOS": 1, "EOS": 2, "RESERVED": 3}, "id_offset": 4,
                "sha256": file_digest(output / "tokenizer.json"), "source_sha256": file_digest(source),
                "training_group_ids_sha256": digest(sorted(set(seen))), "training_records": len(seen),
                "training_utf8_bytes": train_bytes, "tokenizers_version": tokenizers.__version__,
                "training_split": "train", "status": "development qualification"}
    atomic_json(output / "tokenizer_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=4096)
    args = parser.parse_args()
    print(json.dumps(train_bpe(args.records, args.output, args.vocab_size), indent=2))
