"""Independent timestamp-cache replay and exact task-mass certificates."""
from fractions import Fraction
import copy


def reference(parent, transcript, capacity, policy='all_prefixes'):
    cache = {}; tick = 0; serial = 0; prefix_bits = 0; rows = []; previous = ''
    active_state = parent['initial']
    width = (len(parent['outputs'])-1).bit_length()
    counts = dict(prefix_lookups=0, peak_saved_states=0, peak_prefix_bits=0,
                  peak_index_payload_bits=0, peak_saved_state_bits=0,
                  transitions=0, outputs=0, resets=0, saves=0, restores=0, releases=0,
                  active_reuses=0, peak_active_prefix_bits=0)
    for word, expected in transcript:
        tick += 1
        hits = []
        for k in range(len(word), 0, -1):
            counts['prefix_lookups'] += 1
            if word[:k] in cache:
                hits.append(word[:k]); break
        if policy == 'live_query_end' and word[:len(previous)] == previous and len(previous) >= (len(hits[0]) if hits else 0):
            state = active_state; start = len(previous); counts['active_reuses'] += 1
        elif hits:
            prefix = hits[0]; state, _, handle = cache[prefix]
            cache[prefix] = (state, tick, handle); counts['restores'] += 1
            start = len(prefix)
        else:
            state = parent['initial']; start = 0; counts['resets'] += 1
        for k, bit in enumerate(word[start:], start):
            state = parent['edges'][state][int(bit)]; counts['transitions'] += 1
            if capacity != 0 and (policy == 'all_prefixes' or k == len(word)-1):
                tick += 1; serial += 1
                if capacity is not None and len(cache) >= capacity:
                    victim = min(cache, key=lambda key: cache[key][1])
                    prefix_bits -= len(victim); del cache[victim]; counts['releases'] += 1
                key = word[:k+1]; assert key not in cache
                cache[key] = (state, tick, serial); prefix_bits += len(key); counts['saves'] += 1
                size = len(cache)
                counts['peak_saved_states'] = max(counts['peak_saved_states'], size)
                counts['peak_prefix_bits'] = max(counts['peak_prefix_bits'], prefix_bits)
                counts['peak_saved_state_bits'] = max(counts['peak_saved_state_bits'], size*width)
                counts['peak_index_payload_bits'] = max(counts['peak_index_payload_bits'],
                                                       prefix_bits+size*serial.bit_length())
        value = parent['outputs'][state]
        assert value == expected
        counts['outputs'] += 1; rows.append([start, value, len(cache)])
        if policy == 'live_query_end':
            active_state = state; previous = word
            counts['peak_active_prefix_bits'] = max(counts['peak_active_prefix_bits'], len(word))
    counts['state_movement_bits'] = width*(counts['resets']+counts['saves']+counts['restores'])
    counts['active_state_bits'] = width
    return dict(rows=rows, meter=counts)


def check_result(parent, transcript, capacity, result, policy='all_prefixes'):
    expected = reference(parent, transcript, capacity, policy)
    assert result == expected, 'cache policy, answer or resource mismatch'
    if capacity == 0:
        baseline = sum(len(w) for w, _ in transcript)
        if policy == 'live_query_end':
            assert result['meter']['transitions'] <= baseline
        else:
            assert result['meter']['transitions'] == baseline
    if capacity is None:
        prefixes = {word[:k] for word, _ in transcript for k in range(1, len(word)+1)}
        if policy == 'all_prefixes':
            assert result['meter']['transitions'] == len(prefixes)
            assert result['meter']['peak_saved_states'] == len(prefixes)
        else:
            assert result['meter']['transitions'] >= len(prefixes)
            assert result['meter']['peak_saved_states'] == len({w for w, _ in transcript if w})


def fraction(x):
    return [x.numerator, x.denominator]


def masses(parent, cfg):
    records = []
    delay = len(parent['outputs'])-1
    for horizon in cfg['horizons']:
        distribution = {parent['initial']: Fraction(1)}
        targeted = parent['initial']
        for _ in range(horizon):
            nxt = {}
            for state, mass in distribution.items():
                for successor in parent['edges'][state]:
                    nxt[successor] = nxt.get(successor, Fraction(0))+mass/2
            distribution = nxt
            targeted = parent['edges'][targeted][1]
        uniform = sum(mass*parent['outputs'][state] for state, mass in distribution.items())
        target = Fraction(parent['outputs'][targeted])
        for numerator, denominator in cfg['alphas']:
            alpha = Fraction(numerator, denominator)
            error = alpha*target+(1-alpha)*uniform
            records.append(dict(delay=delay, horizon=horizon, alpha=fraction(alpha),
                                uniform_error=fraction(uniform), targeted_error=fraction(target),
                                mixture_error=fraction(error),
                                missed=[[m, fraction((1-error)**m)] for m in cfg['draw_counts']]))
    return records


def check_mass(records):
    for row in records:
        d, horizon = row['delay'], row['horizon']; alpha = Fraction(*row['alpha'])
        uniform = Fraction(1, 2**d) if horizon >= d else Fraction(0)
        target = Fraction(int(horizon >= d)); error = alpha*target+(1-alpha)*uniform
        assert row['uniform_error'] == fraction(uniform)
        assert row['targeted_error'] == fraction(target)
        assert row['mixture_error'] == fraction(error)
        for draws, missed in row['missed']:
            # Direct multiplication independently checks the exponentiation path.
            value = Fraction(1)
            for _ in range(draws):
                value *= 1-error
            assert missed == fraction(value)


def corruptions(parent, transcript, result, mass_records):
    rejected = []
    def reject(name, call):
        try:
            call()
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('accepted corruption: '+name)
    for name, change in (
        ('cached_output', lambda r: r['rows'][0].__setitem__(1, 1-r['rows'][0][1])),
        ('cached_prefix', lambda r: r['rows'][-1].__setitem__(0, r['rows'][-1][0]+1)),
        ('copy_count', lambda r: r['meter'].__setitem__('saves', r['meter']['saves']+1)),
        ('transition_count', lambda r: r['meter'].__setitem__('transitions', r['meter']['transitions']+1))):
        bad = copy.deepcopy(result); change(bad)
        reject(name, lambda: check_result(parent, transcript, 4, bad))
    bad = copy.deepcopy(mass_records); bad[0]['mixture_error'] = [1, 1]
    reject('error_mass', lambda: check_mass(bad))
    bad = copy.deepcopy(mass_records); bad[0]['missed'][-1][1] = [0, 1]
    reject('miss_probability', lambda: check_mass(bad))
    return rejected
