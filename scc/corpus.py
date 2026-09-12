"""Pinned, bounded public-corpus acquisition and document-level development audit.

Only data are fetched. No remote dataset code or model is executed. The sample
uses seeded shard selection and bounded prefixes, not an unbiased population
sample. Acquisition is resumable by completed shard; selection is deterministic.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

from .provenance import atomic_json, canonical_json, digest, file_digest

FILTER_POLICY = {"version": 2, "minimum_chars": 700, "maximum_chars": 1_200_000,
                 "wikimedia_wikis": ["wikipedia.com", "wikipedia.org", "en.wikipedia.org"],
                 "maximum_gutenberg_chars": 200_000,
                 "reason": "Exclude Wikibooks after a vandalized chemistry page was observed; sample more whole short books within the byte cap"}


def request_json(url):
    with urlopen(url, timeout=40) as response:
        return json.load(response)


def acquire(config, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "config.json").exists():
        if json.loads((destination / "config.json").read_text()) != config:
            raise ValueError("Acquisition configuration changed")
    else:
        atomic_json(destination / "config.json", config)
    jobs = []
    for spec in config["sources"]:
        directory = destination / spec["name"]
        directory.mkdir(exist_ok=True)
        meta_path = directory / "repository.json"
        if not meta_path.exists():
            meta = request_json(f"https://huggingface.co/api/datasets/{spec['repo']}/revision/{spec['revision']}")
            if meta["sha"] != spec["revision"] or meta.get("gated"):
                raise ValueError("Unexpected revision or gated repository")
            atomic_json(meta_path, meta)
        meta = json.loads(meta_path.read_text())
        names = [entry["rfilename"] for entry in meta["siblings"] if entry["rfilename"].endswith(".json.gz")]
        names.sort(key=lambda name: digest([config["seed"], spec["repo"], name]))
        if len(names) < spec["shards"]:
            raise ValueError("Not enough source shards")
        card_path = directory / "README.upstream.md"
        if not card_path.exists():
            with urlopen(f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/README.md", timeout=40) as response:
                card_path.write_bytes(response.read())
        for name in names[:spec["shards"]]:
            jobs.append((spec, name, directory))

    def fetch(job):
        spec, name, directory = job
        output = directory / (name + ".prefix.jsonl.gz")
        journal = directory / (name + ".receipt.json")
        if journal.exists():
            receipt = json.loads(journal.read_text())
            if file_digest(output) != receipt["sha256"]:
                raise ValueError("Cached acquisition checksum mismatch")
            return receipt
        url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{name}"
        temporary = output.with_suffix(".partial")
        count, oversized, raw_bytes = 0, 0, 0
        with urlopen(url, timeout=40) as response, gzip.GzipFile(fileobj=response) as incoming:
            with temporary.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as outgoing:
                for line_number in range(1, spec["scan_records_per_shard"] + 1):
                    line = incoming.readline(8_000_001)
                    if not line:
                        break
                    raw_bytes += len(line)
                    if len(line) > 8_000_000:
                        while not line.endswith(b"\n"):
                            line = incoming.readline(8_000_001)
                            raw_bytes += len(line)
                            if not line:
                                break
                        oversized += 1
                        continue
                    row = json.loads(line)
                    wrapped = {"repo": spec["repo"], "revision": spec["revision"], "shard": name,
                               "line": line_number, "source_name": spec["name"], "record": row}
                    outgoing.write((canonical_json(wrapped) + "\n").encode())
                    count += 1
        temporary.replace(output)
        receipt = {"repo": spec["repo"], "revision": spec["revision"], "shard": name,
                   "prefix_records_requested": spec["scan_records_per_shard"], "stored_records": count,
                   "oversized_records": oversized, "uncompressed_bytes_read": raw_bytes,
                   "sha256": file_digest(output), "path": str(output.relative_to(destination))}
        atomic_json(journal, receipt)
        print(json.dumps({"acquired": spec["name"], "shard": name, "records": count}), flush=True)
        return receipt

    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts = list(pool.map(fetch, jobs))
    atomic_json(destination / "acquisition.json", {"config_sha256": digest(config), "receipts": receipts,
                "sampling_limit": "Seeded shards, bounded leading records; not population-representative"})


def words(text):
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())


def canonical_license(value):
    value = value.lower()
    if value.strip() == "public domain" or "creativecommons.org/publicdomain/" in value:
        return "public-domain-declaration"
    match = re.search(r"creativecommons\.org/licenses/(by(?:-sa)?)/(\d\.\d)/?", value)
    return f"CC-{match[1].upper()}-{match[2]}" if match else None


def work_key(name, row):
    meta = row["metadata"]
    if name in ("pressbooks", "libretexts"):
        book_url = meta.get("book_url")
        if not book_url:
            raise ValueError("Missing book identity")
        parsed = urlsplit(book_url)
        path = unquote(parsed.path).lower()
        path = re.split(r"/(?:chapter|front-matter|back-matter|part)/", path)[0]
        return "book:" + parsed.netloc.lower() + path.rstrip("/")
    if name == "gutenberg":
        # Title grouping is conservative and also catches repeated editions.
        title = " ".join(words(meta.get("title", "")))
        return "gutenberg-title:" + title if title else "gutenberg-id:" + str(row["id"])
    wiki = meta.get("wiki", "unknown").lower()
    title = " ".join(words(meta.get("title", str(row["id"]))))
    if "wikibooks" in wiki or "wikisource" in wiki:
        title = " ".join(words(meta.get("title", "").split("/")[0]))
    return "wiki:" + wiki + ":" + title


def filter_record(wrapped):
    row, name = wrapped["record"], wrapped["source_name"]
    text, meta = row.get("text"), row.get("metadata", {})
    if not isinstance(text, str) or not isinstance(meta, dict):
        return None, "missing_text_or_metadata"
    license_name = canonical_license(str(meta.get("license", "")))
    if license_name is None:
        return None, "license_outside_declared_allowlist"
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").strip()
    if not 700 <= len(text) <= 1_200_000:
        return None, "length"
    if name == "wikimedia":
        if str(meta.get("namespace")) != "0":
            return None, "wiki_noncontent_namespace"
        if meta.get("wiki") not in FILTER_POLICY["wikimedia_wikis"]:
            return None, "wiki_outside_selected_encyclopedia"
        title = meta.get("title", "").casefold()
        if title == "main page" or re.search(r"(?:^|/)(?:cover|contents|table of contents|index)$", title):
            return None, "wiki_navigation"
        if ":" in title.split("/")[0]:
            return None, "wiki_special_title"
    if name == "gutenberg" and len(text) > FILTER_POLICY["maximum_gutenberg_chars"]:
        return None, "book_too_long_for_diverse_development_sample"
    if name in ("pressbooks", "libretexts") and not meta.get("book_url"):
        return None, "missing_book_identity"
    tokens = words(text)
    if len(tokens) < 100 or len(set(tokens)) / len(tokens) < 0.04:
        return None, "low_text_diversity"
    if sum(c.isalpha() for c in text) / len(text) < 0.50:
        return None, "low_alphabetic_fraction"
    if sum(ord(c) > 127 for c in text) / len(text) > 0.25:
        return None, "nonascii_fraction"
    if text.count("�") / len(text) > 0.001:
        return None, "encoding_damage"
    if re.search(r"\b(?:MMLU|HellaSwag|GSM8K|HumanEval|ARC-Challenge|BIG-Bench|TruthfulQA)\b", text, re.I):
        return None, "benchmark_name_screen"
    metadata = {"repo": wrapped["repo"], "revision": wrapped["revision"], "shard": wrapped["shard"],
                "line": wrapped["line"], "document_id": str(row["id"]), "upstream_metadata": meta}
    return {"text": text, "source": name, "license": license_name, "metadata": metadata,
            "work": work_key(name, row), "normalized_sha256": hashlib.sha256(" ".join(tokens).encode()).hexdigest()}, None


def audit(acquired, output):
    acquired, output = Path(acquired), Path(output)
    if output.exists():
        raise FileExistsError("Refusing to replace an audited sample")
    config = json.loads((acquired / "config.json").read_text())
    acquisition = json.loads((acquired / "acquisition.json").read_text())
    candidates, rejected, exact_seen = defaultdict(list), defaultdict(Counter), set()
    for receipt in acquisition["receipts"]:
        path = acquired / receipt["path"]
        if file_digest(path) != receipt["sha256"]:
            raise ValueError("Acquired data checksum mismatch")
        with gzip.open(path, "rt") as stream:
            for line in stream:
                wrapped = json.loads(line)
                row, reason = filter_record(wrapped)
                name = wrapped["source_name"]
                if reason:
                    rejected[name][reason] += 1
                elif row["normalized_sha256"] in exact_seen:
                    rejected[name]["normalized_exact_duplicate"] += 1
                else:
                    exact_seen.add(row["normalized_sha256"])
                    candidates[name].append(row)
    selected = []
    selection = {}
    for spec in config["sources"]:
        rows = sorted(candidates[spec["name"]], key=lambda row: digest([config["seed"], row["normalized_sha256"]]))
        used, count = 0, 0
        for row in rows:
            size = len(row["text"].encode())
            if used + size <= spec["keep_bytes"]:
                selected.append(row)
                used += size
                count += 1
        selection[spec["name"]] = {"eligible": len(rows), "selected": count, "utf8_bytes": used}
    if not selected:
        raise ValueError("No documents survived the audit")

    # Conservative grouping for shared long passages, in addition to work IDs.
    # A 32-word shingle sampled by its own hash is a candidate; exact overlap of
    # at least two such passages merges groups. This is not exhaustive semantic
    # decontamination, and incidental quotations can over-group documents.
    parent = list(range(len(selected)))
    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    def union(a, b):
        a, b = root(a), root(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    work_seen, passages, linked = {}, defaultdict(list), 0
    for index, row in enumerate(selected):
        if row["work"] in work_seen:
            union(index, work_seen[row["work"]])
        work_seen[row["work"]] = index
        token_words = words(row["text"])
        matches = Counter()
        fingerprints = set()
        for offset in range(0, max(0, len(token_words) - 31)):
            fingerprint = hashlib.blake2b(" ".join(token_words[offset:offset + 32]).encode(), digest_size=12).hexdigest()
            if int(fingerprint[:2], 16) % 16 == 0:
                fingerprints.add(fingerprint)
        for fingerprint in fingerprints:
            for other in passages[fingerprint]:
                matches[other] += 1
            passages[fingerprint].append(index)
        for other, overlap in matches.items():
            if overlap >= 2:
                union(index, other)
                linked += 1
    components = defaultdict(list)
    for index, row in enumerate(selected):
        components[root(index)].append(row["normalized_sha256"])
    group_ids = {key: digest(sorted(hashes)) for key, hashes in components.items()}
    split_counts, split_bytes, license_counts = defaultdict(Counter), defaultdict(Counter), Counter()
    output.mkdir(parents=True)
    with (output / "records.jsonl").open("w") as stream:
        for index, row in enumerate(selected):
            group_id = group_ids[root(index)]
            bucket = int(digest([config["seed"], group_id])[:8], 16) % 100
            split = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
            row.update({"split": split, "latent_id": group_id})
            split_counts[row["source"]][split] += 1
            split_bytes[row["source"]][split] += len(row["text"].encode())
            license_counts[row["license"]] += 1
            stream.write(canonical_json(row) + "\n")
    report = {"schema_version": 1, "status": "development sample", "config": config,
              "filter_policy": FILTER_POLICY,
              "acquisition_sha256": file_digest(acquired / "acquisition.json"), "selection": selection,
              "rejections": {key: dict(value) for key, value in rejected.items()},
              "split_documents": {key: dict(value) for key, value in split_counts.items()},
              "split_utf8_bytes": {key: dict(value) for key, value in split_bytes.items()},
              "declared_licenses": dict(license_counts), "groups": len(components),
              "shared_passage_links": linked, "largest_group_documents": max(map(len, components.values())),
              "passage_matcher": "all prior documents per sampled anchor; at least two anchors per pair",
              "records_sha256": file_digest(output / "records.jsonl"),
              "limits": ["Bounded shard-prefix population, not a representative sample of Common Pile",
                         "License declarations preserved; no independent chain-of-title verification",
                         "Benchmark-name screen only; no general benchmark-answer contamination guarantee",
                         "Work IDs and sampled shared passages group related text; paraphrases can remain",
                         "Repeated passages within a split are allowed; final pilot needs broader near-deduplication"]}
    atomic_json(output / "audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("acquire")
    collect_parser.add_argument("--config", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--acquired", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "acquire":
        acquire(json.loads(args.config.read_text()), args.output)
    else:
        audit(args.acquired, args.output)
