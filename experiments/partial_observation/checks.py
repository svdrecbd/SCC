"""Private-model evaluation, never a counterexample teacher for discovery."""
from collections import deque
import copy
import itertools
import random
from parents import terminal
from learner import learn


def check_parent(parent):
    if parent['kind'] != 'masked_feedback':
        return
    n, aliases = parent['n'], parent['aliases']
    for p in range(len(parent['outputs'])):
        x, alias = divmod(p, aliases)
        bits = [(x // (2**i)) % 2 for i in range(n)]
        f = 0
        for term in parent['terms']:
            product_bit = 1
            for i in term:
                product_bit *= bits[i]
            f = (f+product_bit) % 2
        output = sum(bits[i] for i in range(n) if (parent['mask']//(2**i)) % 2) % 2
        assert parent['outputs'][p] == output and parent['hazards'][p] == f
        for a in (0, 1):
            next_bits = [(a+f) % 2]+bits[:-1]
            expected = sum(b*(2**i) for i, b in enumerate(next_bits))*aliases+(alias+a+1) % aliases
            assert parent['edges'][p][a] == expected
            next_bits[0] = a
            expected_zero = sum(b*(2**i) for i, b in enumerate(next_bits))*aliases+(alias+a+1) % aliases
            assert parent['zero_edges'][p][a] == expected_zero


def product(parent, model):
    initial = (parent['initial'], model['initial'])
    paths = {initial: ''}; queue = deque([initial]); scans = 0
    while queue:
        p, q = queue.popleft()
        if parent['outputs'][p] != model['outputs'][q]:
            return dict(exact=False, witness=paths[p, q], visited=len(paths), edge_scans=scans)
        for a in (0, 1):
            scans += 1
            pair = (parent['edges'][p][a], model['edges'][q][a])
            if pair not in paths:
                paths[pair] = paths[p, q]+str(a); queue.append(pair)
    return dict(exact=True, relation=[list(pair) for pair in paths], edge_scans=scans)


def check_relation(parent, model, pairs, initial=True):
    relation = set(map(tuple, pairs))
    if initial:
        assert (parent['initial'], model['initial']) in relation
    for p, q in relation:
        assert 0 <= p < len(parent['outputs']) and 0 <= q < len(model['outputs'])
        assert parent['outputs'][p] == model['outputs'][q]
        for a in (0, 1):
            assert (parent['edges'][p][a], model['edges'][q][a]) in relation


def check_transcript(parent, transcript, result):
    assert len({w for w, _ in transcript}) == len(transcript)
    for word, value in transcript:
        assert set(word) <= {'0', '1'} and value == terminal(parent, word)
    m = result['meter']
    assert m['resets'] == m['output_bits'] == result['cache_output_bits'] == len(transcript)
    assert m['parent_steps'] == result['cache_input_bits'] == sum(len(w) for w, _ in transcript)
    assert m['requests'] >= m['cache_hits']+len(transcript)


def replay(transcript, result, cfg):
    remaining = iter(transcript)
    def oracle(word):
        saved, value = next(remaining)
        assert saved == word, 'discovery read outside saved transcript'
        return value
    assert learn(oracle, cfg) == result
    assert next(remaining, None) is None


def distinguishing(model):
    evidence = []
    for p in range(len(model['outputs'])):
        for q in range(p):
            word = next((e for e in model['suffixes']
                         if terminal(model, e, p) != terminal(model, e, q)), None)
            assert word is not None, 'nonminimal learned automaton'
            evidence.append([p, q, word])
    return evidence


def trajectory(model, state, word):
    values = []
    for a in word:
        state = model['edges'][state][int(a)]
        values.append(model['outputs'][state])
    return values, state


def plan(model, state, horizon):
    for k in range(horizon+1):
        for word in itertools.product('01', repeat=k):
            word = ''.join(word)
            if terminal(model, word, state) == 1:
                return word
    return None


def tasks(parent, model, certificate, cfg):
    aliases = parent['aliases']; n = parent['n']; horizon = cfg['task_horizon']
    # Lift only the specified irrelevant-alias symmetry, then certify it on all states.
    by_core = {}
    for p, q in certificate['relation']:
        x = p // aliases
        assert x not in by_core or by_core[x] == q
        by_core[x] = q
    assert len(by_core) == 1 << n
    mapping = [by_core[p // aliases] for p in range(len(parent['outputs']))]
    lift = [[p, q] for p, q in enumerate(mapping)]
    check_relation(parent, model, lift)
    zero = dict(initial=0, outputs=parent['outputs'], edges=parent['zero_edges'])
    counts = dict(forecasts=0, repaired_forecasts=0, zero_forecasts=0,
                  parity_tasks=0, repaired_parity=0, zero_parity=0,
                  planning_tasks=0, repaired_plans=0, zero_same_plan=0,
                  zero_plan_valid_in_parent=0, memory_bits=0,
                  warm_parent_steps=0, warm_output_bits=0, warm_candidate_edges=0,
                  warm_unique=0, warm_unresolved=0, fresh_predictions=0,
                  fresh_unambiguous_correct=0, fresh_ambiguous=0, fresh_first_correct=0,
                  fresh_candidate_edges=0, fresh_parent_steps=0, fresh_output_bits=0)
    warm = []
    words = [''.join(w) for w in itertools.product('01', repeat=horizon)]
    for p, q in enumerate(mapping):
        for word in words:
            yy, _ = trajectory(parent, p, word); rr, _ = trajectory(model, q, word)
            zz, _ = trajectory(zero, p, word)
            counts['forecasts'] += 1; counts['repaired_forecasts'] += int(yy[-1] == rr[-1])
            counts['zero_forecasts'] += int(yy[-1] == zz[-1])
            counts['parity_tasks'] += 1; counts['repaired_parity'] += int(sum(yy)%2 == sum(rr)%2)
            counts['zero_parity'] += int(sum(yy)%2 == sum(zz)%2)
        pp, rp, zp = plan(parent, p, horizon), plan(model, q, horizon), plan(zero, p, horizon)
        assert pp == rp
        counts['planning_tasks'] += 1; counts['repaired_plans'] += 1
        counts['zero_same_plan'] += int(zp == pp)
        counts['zero_plan_valid_in_parent'] += int((zp is None and pp is None) or
                                                  (zp is not None and terminal(parent, zp, p) == 1))
        rng = random.Random(9001+p)
        inputs = [rng.randrange(2) for _ in range(cfg['warm_steps']+cfg['fresh_steps'])]
        live = p
        belief = {i for i, bit in enumerate(model['outputs']) if bit == parent['outputs'][live]}
        counts['warm_output_bits'] += 1
        after_warm = None
        for k, a in enumerate(inputs):
            predicted = {model['edges'][i][a] for i in belief}
            stage = 'warm' if k < cfg['warm_steps'] else 'fresh'
            counts[stage+'_candidate_edges'] += len(belief)
            live = parent['edges'][live][a]; observed = parent['outputs'][live]
            counts[stage+'_parent_steps'] += 1; counts[stage+'_output_bits'] += 1
            if stage == 'fresh':
                values = {model['outputs'][i] for i in predicted}
                counts['fresh_predictions'] += 1
                counts['fresh_ambiguous'] += int(len(values) > 1)
                counts['fresh_unambiguous_correct'] += int(values == {observed})
                counts['fresh_first_correct'] += int(model['outputs'][model['edges'][min(belief)][a]] == observed)
            belief = {i for i in predicted if model['outputs'][i] == observed}
            assert mapping[live] in belief
            if k+1 == cfg['warm_steps']:
                after_warm = sorted(belief)
                counts['warm_unique' if len(belief) == 1 else 'warm_unresolved'] += 1
        warm.append(dict(initial=p, after_warm=after_warm, final=sorted(belief)))
        zx = p
        for k, a in enumerate(inputs):
            zx = zero['edges'][zx][a]
            if k+1 >= n:
                expected = sum(inputs[k-i] << i for i in range(n))
                assert zx // aliases == expected
                counts['memory_bits'] += n
    fibers = [[] for _ in model['outputs']]
    for p, q in enumerate(mapping):
        fibers[q].append(parent['hazards'][p])
    assert all(fibers)
    hazard = dict(homogeneous=all(len(set(v)) == 1 for v in fibers),
                  best_class_correct=sum(max(v.count(0), v.count(1)) for v in fibers),
                  total=len(mapping), labels=[sorted(set(v)) for v in fibers])
    return dict(counts=counts, warm=warm, lifted_relation=lift, protected_factorization=hazard)


def evaluate(parent, result, cfg):
    if result['model'] is None:
        return dict(exact=False, reason='no model returned')
    model = result['model']; cert = product(parent, model)
    if not cert['exact']:
        assert terminal(parent, cert['witness']) != terminal(model, cert['witness'])
        return dict(certificate=cert)
    check_relation(parent, model, cert['relation'])
    output = dict(certificate=cert, distinguishing=distinguishing(model))
    if parent['kind'] == 'masked_feedback':
        output['tasks'] = tasks(parent, model, cert, cfg)
    return output


def corruption_checks(parent, transcript, result, evaluation):
    rejected = []
    def reject(name, call):
        try:
            call()
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('accepted corruption: '+name)
    bad = copy.deepcopy(transcript); bad[0][1] ^= 1
    reject('transcript_output', lambda: check_transcript(parent, bad, result))
    bad_result = copy.deepcopy(result); bad_result['meter']['parent_steps'] += 1
    reject('query_cost', lambda: check_transcript(parent, transcript, bad_result))
    relation = evaluation['certificate']['relation']
    reject('empty_relation', lambda: check_relation(parent, result['model'], []))
    bad_model = copy.deepcopy(result['model']); bad_model['outputs'][bad_model['initial']] ^= 1
    reject('output_label', lambda: check_relation(parent, bad_model, relation))
    bad_edge = copy.deepcopy(result['model']); q = bad_edge['initial']; old = bad_edge['edges'][q][0]
    bad_edge['edges'][q][0] = next(i for i, v in enumerate(bad_edge['outputs']) if v != bad_edge['outputs'][old])
    reject('transition', lambda: check_relation(parent, bad_edge, relation))
    return rejected
