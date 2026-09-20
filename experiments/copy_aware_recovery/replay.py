"""Online prefix caching through an opaque snapshot interface."""
from collections import OrderedDict


class Oracle:
    def __init__(self, parent):
        self.parent = parent
        self.state = parent['initial']
        self.saved = {}
        self.serial = 0
        self.meter = dict(transitions=0, outputs=0, resets=0, saves=0, restores=0, releases=0)

    def reset(self):
        self.state = self.parent['initial']; self.meter['resets'] += 1

    def step(self, action):
        self.state = self.parent['edges'][self.state][action]
        self.meter['transitions'] += 1

    def snapshot(self):
        self.serial += 1
        self.saved[self.serial] = self.state
        self.meter['saves'] += 1
        return self.serial

    def restore(self, handle):
        self.state = self.saved[handle]; self.meter['restores'] += 1

    def release(self, handle):
        del self.saved[handle]; self.meter['releases'] += 1

    def output(self):
        self.meter['outputs'] += 1
        return self.parent['outputs'][self.state]


def replay(oracle, queries, capacity, state_width, policy='all_prefixes'):
    assert policy in ('all_prefixes', 'query_end', 'live_query_end')
    cache = OrderedDict(); rows = []; prefix_bits = 0; active_word = ''
    meter = dict(prefix_lookups=0, peak_saved_states=0, peak_prefix_bits=0,
                 peak_index_payload_bits=0, peak_saved_state_bits=0,
                 active_reuses=0, peak_active_prefix_bits=0)
    for word in queries:
        start = 0
        for k in range(len(word), 0, -1):
            meter['prefix_lookups'] += 1
            if word[:k] in cache:
                start = k; break
        if policy == 'live_query_end' and word.startswith(active_word) and len(active_word) >= start:
            start = len(active_word); meter['active_reuses'] += 1
        elif start:
            handle = cache[word[:start]]
            cache.move_to_end(word[:start]); oracle.restore(handle)
        else:
            oracle.reset()
        for k in range(start, len(word)):
            oracle.step(int(word[k]))
            if capacity != 0 and (policy == 'all_prefixes' or k == len(word)-1):
                prefix = word[:k+1]
                assert prefix not in cache
                # Release before allocating: the saved-state capacity is a hard cap.
                if capacity is not None and len(cache) == capacity:
                    old, handle = cache.popitem(last=False)
                    prefix_bits -= len(old); oracle.release(handle)
                cache[prefix] = oracle.snapshot(); prefix_bits += len(prefix)
                size = len(cache)
                meter['peak_saved_states'] = max(meter['peak_saved_states'], size)
                meter['peak_prefix_bits'] = max(meter['peak_prefix_bits'], prefix_bits)
                meter['peak_saved_state_bits'] = max(meter['peak_saved_state_bits'], size*state_width)
                # Fixed-width handle IDs at the current largest assigned ID; key delimiters/runtime excluded.
                index_bits = prefix_bits+size*cache[prefix].bit_length()
                meter['peak_index_payload_bits'] = max(meter['peak_index_payload_bits'], index_bits)
        rows.append([start, oracle.output(), len(cache)])
        if policy == 'live_query_end':
            active_word = word
            meter['peak_active_prefix_bits'] = max(meter['peak_active_prefix_bits'], len(word))
    meter.update(oracle.meter)
    meter['state_movement_bits'] = state_width*(meter['resets']+meter['saves']+meter['restores'])
    meter['active_state_bits'] = state_width
    return dict(rows=rows, meter=meter)
