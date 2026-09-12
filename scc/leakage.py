"""Independent development-split check using identities and shared passages."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .corpus import words
from .provenance import atomic_json, digest, file_digest


def check(records):
    rows = [json.loads(line) for line in Path(records).open()]
    identities, contents, works = defaultdict(set), defaultdict(set), defaultdict(set)
    anchors, crossings = defaultdict(list), Counter()
    natural, tasks = 0, 0
    for index, row in enumerate(rows):
        split = row["split"]
        identities[row["latent_id"]].add(split)
        if "text" in row:
            natural += 1
            text_words = words(row["text"])
            contents[digest(text_words)].add(split)
            works[row.get("work", row["latent_id"])].add(split)
            fingerprints = set()
            for offset in range(max(0, len(text_words) - 31)):
                key = hashlib.blake2b(" ".join(text_words[offset:offset + 32]).encode(), digest_size=12).hexdigest()
                if int(key[:2], 16) % 16 == 0:
                    fingerprints.add(key)
            for key in fingerprints:
                for other in anchors[key]:
                    if rows[other]["split"] != split:
                        crossings[(other, index)] += 1
                anchors[key].append(index)
        else:
            tasks += 1
            contents[digest({"prompt": row["prompt"]})].add(split)
    pairs = [(a, b, count) for (a, b), count in crossings.items() if count >= 2]
    issues = {"latent_ids_across_splits": sum(len(splits) > 1 for splits in identities.values()),
              "works_across_splits": sum(len(splits) > 1 for splits in works.values()),
              "normalized_contents_across_splits": sum(len(splits) > 1 for splits in contents.values()),
              "cross_split_pairs_with_two_shared_anchors": len(pairs)}
    return {"records_sha256": file_digest(records), "natural_documents": natural, "task_records": tasks,
            "checks": issues, "passed": not any(issues.values()),
            "pair_examples": [{"first": rows[a]["metadata"]["document_id"], "second": rows[b]["metadata"]["document_id"],
                               "matching_anchors": count} for a, b, count in pairs[:10]],
            "scope": "Identity, normalized exact content, and sampled exact 32-word passages. Does not detect all paraphrases or benchmark-answer overlap."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.records)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
