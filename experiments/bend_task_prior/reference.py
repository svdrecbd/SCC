"""Independent list-valued prior/task semantics and transcript inference."""
from collections import Counter, defaultdict
from fractions import Fraction
from math import comb, log2


def bits(value, width=4):
    return [value // (2 ** i) % 2 for i in range(width)]


def pack(values):
    return sum(v * 2 ** i for i, v in enumerate(values))


def shifted(values, bit):
    return [int(v != bit) for v in values]


def expected(cfg):
    index = 0
    for mode in cfg['conditions']:
        for prior in range(16):
            p = bits(prior)
            stored, effective, anchor = p, p, None
            if mode == 'erased':
                stored = effective = [0] * 4
            elif mode == 'relative':
                stored = effective = shifted(p, p[0])
            elif mode == 'relative_plus_anchor':
                stored, anchor = shifted(p, p[0]), p[0]
            elif mode == 'fixed_recode_repaired':
                stored = [int(x != y) for x, y in zip(p, bits(cfg['fixed_recode_mask']))]
            for offsets in range(4):
                targets = [shifted(p, b) for b in bits(offsets, 2)]
                for c0 in range(4):
                    for c1 in range(4):
                        labels = [targets[0][c0], targets[1][c1]]
                        adapted = [int(labels[i] != effective[c]) for i, c in enumerate((c0, c1))]
                        for q0 in range(4):
                            for q1 in range(4):
                                if q0 == c0 or q1 == c1:
                                    continue
                                yield dict(id=index, kind='episode', mode=mode, prior=prior, offsets=offsets,
                                    support=[c0, c1], query=[q0, q1], stored=pack(stored), anchor=anchor,
                                    labels=labels, adapted=adapted,
                                    predictions=[int(effective[q] != adapted[i]) for i, q in enumerate((q0, q1))],
                                    protected=[effective[q0], effective[q1]])
                                index += 1
    for prior in range(16):
        p = bits(prior)
        relative = shifted(p, p[0])
        for context in range(4):
            for offset in range(2):
                label = int(p[context] != offset)
                recovered = (label + relative[context] + offset) % 2
                yield dict(id=index, kind='anchor', prior=prior, context=context, offset=offset,
                    retained=pack(relative), label=label, recoveredAnchor=recovered,
                    restoredPrior=pack(shifted(relative, recovered)))
                index += 1
    for t in range(cfg['certificate_max_tasks'] + 1):
        for prior in range(16):
            p = bits(prior)
            paired = [1 - x for x in p]
            for offsets in range(2 ** t):
                seq = bits(offsets, t)
                yield dict(id=index, kind='transcript', t=t, prior=prior, offsets=offsets,
                    retained=pack(shifted(p, p[0])), tasks=[pack(shifted(p, b)) for b in seq],
                    pairedPrior=pack(paired), pairedOffsets=pack([1 - b for b in seq]),
                    pairedRetained=pack(shifted(paired, paired[0])),
                    pairedTasks=[pack(shifted(paired, 1 - b)) for b in seq])
                index += 1


def rational(numerator, denominator):
    f = Fraction(numerator, denominator)
    return [f.numerator, f.denominator]


def transcript_scores(rows, t, law):
    groups = defaultdict(dict)
    for row in rows:
        weight = 1 if law == 'uniform' else 3 ** (t - sum(bits(row['offsets'], t)))
        key = (row['retained'], tuple(row['tasks']))
        assert row['prior'] not in groups[key]
        groups[key][row['prior']] = weight
    denominator = 16 * (2 ** t if law == 'uniform' else 4 ** t)
    histogram = Counter()
    best, individual = 0, [0] * 4
    coordinate_information, joint_information = [0.0] * 4, 0.0
    for key, prior_masses in groups.items():
        assert len(prior_masses) == 2
        low, high = sorted(prior_masses)
        assert low + high == 15
        assert pack(shifted(bits(low), bits(low)[0])) == key[0]
        assert pack(shifted(bits(high), bits(high)[0])) == key[0]
        total = sum(prior_masses.values())
        best += max(prior_masses.values())
        for i in range(4):
            m = [sum(v for p, v in prior_masses.items() if bits(p)[i] == b) for b in range(2)]
            individual[i] += max(m)
            coordinate_information[i] += sum(v / denominator * log2(2 * v / total) for v in m)
        joint_information += sum(v / denominator * log2(16 * v / total) for v in prior_masses.values())
        ordered = tuple(sum(v for p, v in prior_masses.items() if bits(p)[0] == b) for b in range(2))
        histogram[ordered] += 1
    assert sum(sum(pair) * n for pair, n in histogram.items()) == denominator
    if law == 'uniform':
        exact = [1, 2]
        assert all(pair == (1, 1) for pair in histogram)
        info = 0.0
    else:
        exact = rational(sum(comb(t, k) * max(3 ** k, 3 ** (t - k)) for k in range(t + 1)), 2 * 4 ** t)
        info = sum(comb(t, k) * m / (2 * 4 ** t) * log2(2 * m / (3 ** k + 3 ** (t - k)))
                   for k in range(t + 1) for m in (3 ** k, 3 ** (t - k)))
    assert rational(best, denominator) == exact
    assert all(rational(n, denominator) == exact for n in individual)
    assert all(abs(v - info) < 1e-12 for v in coordinate_information)
    assert abs(joint_information - (3 + info)) < 1e-12
    return dict(t=t, law=law, worlds=len(rows), public_transcript_groups=len(groups),
                denominator=denominator, best_full_prior_accuracy=exact,
                best_coordinate_accuracy=[rational(n, denominator) for n in individual],
                coordinate_information_bits=coordinate_information,
                joint_prior_information_bits=joint_information,
                posterior_mass_histogram=[dict(anchor0_mass=p[0], anchor1_mass=p[1], groups=n)
                                         for p, n in sorted(histogram.items())])
