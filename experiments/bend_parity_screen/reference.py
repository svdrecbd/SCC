"""Independent source-by-source majority decoding, no bit-count score formula."""
from itertools import product

QUERIES = tuple(range(1, 16))
PROTECTED = (1, 2, 4, 8)
USEFUL = tuple(q for q in QUERIES if q not in PROTECTED)
LABELS = {
    q: tuple(sum(((z >> i) & 1) * ((q >> i) & 1) for i in range(4)) % 2
             for z in range(16)) for q in QUERIES
}


def majority_score(members, q):
    labels = [LABELS[q][z] for z in members]
    ones = sum(labels)
    return max(ones, len(labels) - ones)


def score(encoding):
    buckets = {}
    for z, value in enumerate(encoding):
        buckets.setdefault(value, []).append(z)
    return [sum(majority_score(m, q) for m in buckets.values()) for q in QUERIES]


def encoding(table):
    return [(table >> z) & 1 for z in range(16)]


def totals(correct):
    return {
        'useful_correct': sum(correct[q - 1] for q in USEFUL),
        'protected_correct': sum(correct[q - 1] for q in PROTECTED),
        'all_correct': sum(correct),
    }


def witness(table):
    enc = encoding(table)
    decoders, brute_scores = {}, []
    for q in QUERIES:
        options = list(product((0, 1), repeat=2))
        scores = [sum(d[enc[z]] == LABELS[q][z] for z in range(16)) for d in options]
        best = max(scores)
        decoders[str(q)] = list(options[scores.index(best)])
        brute_scores.append(best)
    return dict(encoder=table, encoding=enc, decoders=decoders, correct=brute_scores,
                **totals(brute_scores))


def controls():
    # Enumerate the subspace itself, avoiding multiplicity from equivalent bases.
    spaces, frontier = {frozenset({0})}, [frozenset({0})]
    while frontier:
        old = frontier.pop()
        for v in range(1, 16):
            new = old | frozenset(x ^ v for x in old)
            if new not in spaces:
                spaces.add(new)
                frontier.append(new)
    rows = []
    for space in sorted(spaces, key=lambda s: (len(s), sorted(s))):
        # Basis extraction makes the retained state rank bits, not every form.
        basis, span = [], {0}
        for v in sorted(space):
            if v not in span:
                basis.append(v)
                span |= {x ^ v for x in span}
        enc = [sum(LABELS[q][z] << i for i, q in enumerate(basis)) for z in range(16)]
        c = score(enc)
        rows.append(dict(kind='linear', basis=basis, rank=len(basis), correct=c, **totals(c)))
    for mask in range(16):
        enc = [z ^ mask for z in range(16)]
        # Inverse repair acts on every source, followed by every parity query.
        repaired = [sum(LABELS[q][enc[z] ^ mask] == LABELS[q][z] for z in range(16))
                    for q in QUERIES]
        rows.append(dict(kind='xor_repair', mask=mask, correct=repaired))
    enc = list(range(16))
    enc[15] = 14
    c = score(enc)
    rows.append(dict(kind='one_collision', encoding=enc, correct=c,
                     whole_source_correct=15, **totals(c)))
    return rows
