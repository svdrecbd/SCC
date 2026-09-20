"""Exact partition optimization with a separately checked Bellman certificate."""
from reference import LABELS, PROTECTED, QUERIES, USEFUL, majority_score, score, totals


def solve(rows):
    # Producer takes bucket scores/populations from checked Bend output.
    balanced = [r['encoder'] for r in rows if r['population'] and
                all(2 * r['bucket_correct'][q - 1] == r['population'] for q in PROTECTED)]
    weights = {m: sum(rows[m]['bucket_correct'][q - 1] for q in USEFUL) for m in balanced}
    records = {0: dict(mask=0, score=0, block=0)}
    for remaining in sorted(balanced, key=lambda m: (m.bit_count(), m)):
        anchor = remaining & -remaining
        options = ((weights[b] + records[remaining ^ b]['score'], b) for b in balanced
                   if b & anchor and b & remaining == b)
        value, block = max(options, key=lambda pair: (pair[0], -pair[1]))
        records[remaining] = dict(mask=remaining, score=value, block=block)
    return list(records.values())


def check(certificate):
    # Reconstruct balanced sets and weights directly from source labels, not
    # producer lists or Bend scores. Check *all* possible Bellman alternatives.
    members = {m: [z for z in range(16) if (m >> z) & 1] for m in range(1, 65536)}
    balanced = {m: zs for m, zs in members.items()
                if all(sum(LABELS[q][z] for z in zs) * 2 == len(zs) for q in PROTECTED)}
    records = {r['mask']: r for r in certificate}
    assert len(records) == len(certificate)
    assert set(records) == {0} | set(balanced)
    assert records[0] == dict(mask=0, score=0, block=0)
    weights = {m: sum(majority_score(zs, q) for q in USEFUL) for m, zs in balanced.items()}
    transitions = 0
    for remaining, zs in balanced.items():
        anchor = 1 << min(zs)
        alternatives = [b for b in balanced if b & anchor and (b | remaining) == remaining]
        values = [weights[b] + records[remaining ^ b]['score'] for b in alternatives]
        chosen = records[remaining]['block']
        assert chosen in alternatives
        assert records[remaining]['score'] == max(values)
        assert records[remaining]['score'] == weights[chosen] + records[remaining ^ chosen]['score']
        transitions += len(alternatives)
    blocks, remaining = [], 65535
    while remaining:
        b = records[remaining]['block']
        blocks.append(b)
        remaining ^= b
    enc = [next(i for i, b in enumerate(blocks) if (b >> z) & 1) for z in range(16)]
    correct = score(enc)
    assert totals(correct)['useful_correct'] == records[65535]['score']
    assert all(correct[q - 1] == 8 for q in PROTECTED)
    decoders = {}
    for q in QUERIES:
        decoders[str(q)] = [int(sum(LABELS[q][z] for z in balanced[b]) * 2 > len(balanced[b]))
                            for b in blocks]
        assert sum(decoders[str(q)][enc[z]] == LABELS[q][z] for z in range(16)) == correct[q - 1]
    # Independent explicit complement-pair attack, even if another partition wins.
    relative = [sum((((z >> 0) ^ (z >> i)) & 1) << (i - 1) for i in (1, 2, 3))
                for z in range(16)]
    relative_scores = score(relative)
    assert relative_scores == [16 if q.bit_count() % 2 == 0 else 8 for q in QUERIES]
    assert totals(relative_scores)['useful_correct'] == 144
    return dict(balanced_nonempty_subsets=len(balanced), bellman_alternatives=transitions,
                optimum_correct=records[65535]['score'], denominator=176,
                blocks=blocks, encoding=enc, decoders=decoders, correct=correct,
                relative_parity_encoding=relative, relative_parity_correct=relative_scores,
                **totals(correct))
