"""Output-only observation tables; no topology or exact equivalence oracle."""
import itertools


def words(depth):
    return [''.join(w) for k in range(depth+1) for w in itertools.product('01', repeat=k)]


def predict(model, word, state=None):
    state = model['initial'] if state is None else state
    for a in word:
        state = model['edges'][state][int(a)]
    return model['outputs'][state]


def learn(oracle, cfg):
    cache = {}; prefixes = ['']; suffixes = ['']; hypotheses = []
    meter = dict(requests=0, cache_hits=0, resets=0, parent_steps=0, output_bits=0,
                 row_calls=0, conformance_tests=0, hypothesis_steps=0)

    def query(word):
        meter['requests'] += 1
        if word in cache:
            meter['cache_hits'] += 1
            return cache[word]
        if len(cache) >= cfg['query_cap'] or meter['parent_steps']+len(word) > cfg['step_cap']:
            raise RuntimeError('oracle budget exhausted')
        value = oracle(word)
        assert value in (0, 1)
        cache[word] = value
        meter['resets'] += 1; meter['output_bits'] += 1; meter['parent_steps'] += len(word)
        return value

    def row(word):
        meter['row_calls'] += 1
        return tuple(query(word+e) for e in suffixes)

    model = None
    try:
        for iteration in range(cfg['round_cap']):
            while True:
                rows = {s: row(s) for s in prefixes}
                missing = next((s+a for s in prefixes for a in '01'
                                if row(s+a) not in rows.values()), None)
                if missing is not None:
                    prefixes.append(missing)
                    continue
                new_suffix = None
                for i, s in enumerate(prefixes):
                    for t in prefixes[i+1:]:
                        if rows[s] != rows[t]:
                            continue
                        for a in '01':
                            left, right = row(s+a), row(t+a)
                            if left != right:
                                j = next(j for j in range(len(suffixes)) if left[j] != right[j])
                                new_suffix = a+suffixes[j]
                                break
                        if new_suffix is not None:
                            break
                    if new_suffix is not None:
                        break
                if new_suffix is not None:
                    assert new_suffix not in suffixes
                    suffixes.append(new_suffix)
                    continue
                representatives = list(dict.fromkeys(rows.values()))
                access = [next(s for s in prefixes if rows[s] == r) for r in representatives]
                model = dict(initial=representatives.index(rows['']),
                             outputs=[query(s) for s in access],
                             edges=[[representatives.index(row(s+a)) for a in '01'] for s in access],
                             access=access, suffixes=list(suffixes))
                break
            mismatch = None
            cover = list(dict.fromkeys(access+[s+a for s in access for a in '01']))
            for s, middle, e in itertools.product(cover, words(cfg['middle_depth']), suffixes):
                word = s+middle+e
                meter['conformance_tests'] += 1
                meter['hypothesis_steps'] += len(word)
                if predict(model, word) != query(word):
                    mismatch = word
                    break
            if mismatch is None and cfg.get('regime') == 'extended':
                for length, pattern in itertools.product(cfg['probe_lengths'], cfg['probe_patterns']):
                    word = (pattern*((length+len(pattern)-1)//len(pattern)))[:length]
                    meter['conformance_tests'] += 1
                    meter['hypothesis_steps'] += len(word)
                    if predict(model, word) != query(word):
                        mismatch = word
                        break
            hypotheses.append(dict(states=len(model['outputs']), counterexample=mismatch))
            if mismatch is None:
                return dict(status='bounded_conformance_pass', model=model, hypotheses=hypotheses,
                            meter=meter, cache_input_bits=sum(map(len, cache)), cache_output_bits=len(cache))
            for k in range(len(mismatch)+1):
                if mismatch[:k] not in prefixes:
                    prefixes.append(mismatch[:k])
        raise RuntimeError('round budget exhausted')
    except RuntimeError as error:
        return dict(status='limited', reason=str(error), model=model, hypotheses=hypotheses,
                    meter=meter, cache_input_bits=sum(map(len, cache)), cache_output_bits=len(cache))
