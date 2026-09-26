"""Tokenize a fixed public generic-text sample for LN-395 stage 3.

Documents come in the dataset's own order from FineWeb-Edu sample-10BT. The
first documents fill the training stream; later documents, never used for
training, fill the validation stream. Each document ends with the EOS token.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, required=True)
    parser.add_argument("--validation-tokens", type=int, required=True)
    parser.add_argument("--validation-skip-documents", type=int, default=200000)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(arguments.tokenizer)
    stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)

    def collect(documents, limit):
        tokens, used, first_id, last_id = [], 0, None, None
        batch = []
        for document in documents:
            batch.append(document)
            if len(batch) == 256:
                for record, ids in zip(batch, tokenizer([d["text"] for d in batch])["input_ids"]):
                    first_id = first_id or record["id"]
                    last_id = record["id"]
                    tokens.extend(ids + [tokenizer.eos_token_id])
                    used += 1
                batch = []
                if len(tokens) >= limit:
                    break
        return np.asarray(tokens[:limit], dtype=np.uint32), used, first_id, last_id

    records = {}
    train, used, first, last = collect(iter(stream), arguments.train_tokens)
    assert len(train) == arguments.train_tokens
    assert used < arguments.validation_skip_documents
    train.tofile(arguments.output / "train.bin")
    records["train"] = {"tokens": len(train), "documents": used, "first_id": first, "last_id": last}
    validation, used, first, last = collect(
        iter(stream.skip(arguments.validation_skip_documents)), arguments.validation_tokens)
    validation.tofile(arguments.output / "validation.bin")
    records["validation"] = {"tokens": len(validation), "documents": used,
                             "skip_documents": arguments.validation_skip_documents,
                             "first_id": first, "last_id": last}
    for name in ("train", "validation"):
        records[name]["sha256"] = hashlib.sha256(
            (arguments.output / f"{name}.bin").read_bytes()).hexdigest()
    records["tokenizer"] = arguments.tokenizer
    records["eos_token_id"] = tokenizer.eos_token_id
    (arguments.output / "corpus.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records))


if __name__ == "__main__":
    main()
