import json
import gzip
import hashlib
import random

import pytest
import torch

from scc.corpus import audit, canonical_license, filter_record, work_key, words
from scc.data import PreparedDataset, prepare
from scc.tokenizer import BPETokenizer, train_bpe
from scc.provenance import atomic_json, digest, file_digest
from scc.leakage import check


def test_content_licenses_are_not_inferred_from_repository_name():
    assert canonical_license("https://creativecommons.org/licenses/by/4.0/") == "CC-BY-4.0"
    assert canonical_license("https://creativecommons.org/licenses/by-nc/4.0/") is None
    assert canonical_license("unknown") is None
    assert canonical_license("Public Domain") == "public-domain-declaration"


def test_chapters_share_work_identity():
    first = {"metadata": {"book_url": "https://example.org/algebra/chapter/one/"}}
    second = {"metadata": {"book_url": "https://example.org/algebra/front-matter/introduction/"}}
    assert work_key("pressbooks", first) == work_key("pressbooks", second)
    assert work_key("pressbooks", first) != work_key("pressbooks", {"metadata": {"book_url": "https://example.org/biology/"}})


def test_user_pages_are_rejected_despite_open_license():
    wrapped = {"source_name": "wikimedia", "record": {"text": "some longer article " * 100,
               "metadata": {"namespace": "2", "license": "https://creativecommons.org/licenses/by-sa/4.0/"}}}
    assert filter_record(wrapped)[1] == "wiki_noncontent_namespace"


def records(path, validation_text):
    rows = [{"split": "train", "latent_id": "train", "source": "natural", "license": "generated",
             "text": "The little bird flew over the garden. " * 100},
            {"split": "validation", "latent_id": "validation", "source": "natural", "license": "generated",
             "text": validation_text},
            {"split": "test", "latent_id": "test", "source": "natural", "license": "generated",
             "text": "A different test document."}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_bpe_training_ignores_held_out_text_and_roundtrips(tmp_path):
    first, second = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    records(first, "Never train on this vocabulary: flibbertigibbet.")
    records(second, "Completely changed held out corpus " * 100)
    train_bpe(first, tmp_path / "tok1", 300)
    train_bpe(second, tmp_path / "tok2", 300)
    one, two = BPETokenizer(tmp_path / "tok1"), BPETokenizer(tmp_path / "tok2")
    text = "café 🧬 <EOS>\nnever seen before"
    assert one.encode(text) == two.encode(text)
    assert one.decode(one.encode(text)) == text
    assert min(one.encode("<EOS>")) >= 4
    assert one.manifest()["training_records"] == 1


def test_bpe_data_records_source_exposure_and_detects_tokenizer_change(tmp_path):
    source = tmp_path / "input.jsonl"
    records(source, "This is held out text and sufficiently long for the test.")
    train_bpe(source, tmp_path / "tok", 300)
    manifest = prepare(source, tmp_path / "data", 32, tmp_path / "tok")
    dataset = PreparedDataset(tmp_path / "data", "train")
    assert dataset.tokenizer.vocab_size == manifest["tokenizer"]["vocab_size"]
    assert manifest["group_counts"]["train"]["natural"]["loss_tokens"] == manifest["split_counts"]["train"]["loss_tokens"]
    tokens, targets = dataset.batch(torch.tensor([0]))
    assert tokens.shape == targets.shape == (1, 32)
    path = tmp_path / "data/tokenizer.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="Tokenizer checksum"):
        PreparedDataset(tmp_path / "data", "train")


def test_shared_passage_grouping_checks_all_prior_documents(tmp_path):
    # Two sampled anchors are shared by B and C. A contains only the first.
    # Keeping only the first document per anchor would miss the B/C pair.
    rng = random.Random(91)
    def sampled(tokens):
        h = hashlib.blake2b(" ".join(tokens).encode(), digest_size=12).hexdigest()
        return int(h[:2], 16) % 16 == 0
    for _ in range(10000):
        shared = ["".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=10)) for _ in range(33)]
        if sampled(shared[:32]) and sampled(shared[1:]):
            break
    else:
        raise AssertionError("Could not build shared-passage fixture")
    rows = []
    for index, name in enumerate(("a", "b", "c")):
        filler = ["".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=12)) for _ in range(100)]
        text = " ".join((shared[:32] if index == 0 else shared) + filler)
        rows.append({"source_name": "wikimedia", "repo": "fixture", "revision": "fixture", "shard": "fixture",
                     "line": index + 1, "record": {"id": name, "text": text, "metadata": {
                         "namespace": "0", "wiki": "wikipedia.org", "title": name,
                         "license": "https://creativecommons.org/licenses/by/4.0/"}}})
    hashes = [hashlib.sha256(" ".join(words(row["record"]["text"])).encode()).hexdigest() for row in rows]
    seed = next(seed for seed in range(1000)
                if sorted(range(3), key=lambda i: digest([seed, hashes[i]])) == [0, 1, 2])
    acquired = tmp_path / "acquired"
    acquired.mkdir()
    raw = acquired / "fixture.jsonl.gz"
    with gzip.open(raw, "wt") as stream:
        stream.write("".join(json.dumps(row) + "\n" for row in rows))
    atomic_json(acquired / "config.json", {"seed": seed, "sources": [{"name": "wikimedia", "keep_bytes": 100000}]})
    atomic_json(acquired / "acquisition.json", {"receipts": [{"path": raw.name, "sha256": file_digest(raw)}]})
    audit(acquired, tmp_path / "audit")
    selected = [json.loads(line) for line in (tmp_path / "audit/records.jsonl").open()]
    assert selected[1]["latent_id"] == selected[2]["latent_id"]
    assert selected[0]["latent_id"] != selected[1]["latent_id"]
    # The independent checker must also catch this overlap if identities are wrong.
    for index, row in enumerate(selected):
        row.update(latent_id=str(index), split="validation" if index == 2 else "train")
    broken = tmp_path / "broken.jsonl"
    broken.write_text("".join(json.dumps(row) + "\n" for row in selected))
    result = check(broken)
    assert not result["passed"]
    assert result["checks"]["cross_split_pairs_with_two_shared_anchors"] == 1
