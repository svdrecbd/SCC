"""Behavior-only GF(2) realization. No parent matrices or states enter this API."""


def dot(mask, vector):
    return (mask & vector).bit_count() & 1


def eliminate(rows, columns, meter):
    rows = list(rows); pivot = 0
    meter['max_row_width_bits'] = max(meter['max_row_width_bits'], max((x.bit_length() for x in rows), default=0))
    meter['largest_elimination_matrix_bits'] = max(meter['largest_elimination_matrix_bits'],
                                                  len(rows) * max((x.bit_length() for x in rows), default=0))
    for col in range(columns):
        selected = None
        for i in range(pivot, len(rows)):
            meter['pivot_bit_tests'] += 1
            if (rows[i] >> col) & 1:
                selected = i; break
        if selected is None:
            continue
        if selected != pivot:
            rows[pivot], rows[selected] = rows[selected], rows[pivot]; meter['row_swaps'] += 1
        for i in range(len(rows)):
            meter['elimination_bit_tests'] += 1
            if i != pivot and (rows[i] >> col) & 1:
                rows[i] ^= rows[pivot]; meter['row_xors'] += 1
        pivot += 1
    meter['pivots'] += pivot
    return pivot, rows


def advance(state, bit, order, coefficients):
    top = (state >> (order - 1)) & 1
    return ((state << 1) & ((1 << order) - 1)) ^ (coefficients if top else 0) ^ bit


def reconstruct(job):
    n = job['upper_order']; m = job['impulse_outputs']
    assert len(m) == 2*n and all(x in (0,1) for x in m)
    meter = dict(pivot_bit_tests=0, elimination_bit_tests=0, row_swaps=0,
                 row_xors=0, pivots=0, max_row_width_bits=0, largest_elimination_matrix_bits=0)
    hankel = [sum(m[i+j] << j for j in range(n)) for i in range(n)]
    r, _ = eliminate(hankel, n, meter)
    assert 0 < r <= n
    augmented = [sum(m[i+j] << j for j in range(r)) | (1 << (r+i)) for i in range(r)]
    rank, reduced = eliminate(augmented, r, meter); assert rank == r
    assert all((row & ((1 << r)-1)) == 1 << i for i,row in enumerate(reduced))
    inverse = [row >> r for row in reduced]
    target = sum(m[r+i] << i for i in range(r))
    coefficients = sum(dot(row, target) << i for i,row in enumerate(inverse))
    readout = sum(m[i] << i for i in range(r))
    warm = []
    for outputs in job['warm_outputs']:
        assert len(outputs) == r and all(x in (0,1) for x in outputs)
        vector = sum(bit << i for i,bit in enumerate(outputs))
        state = sum(dot(row, vector) << i for i,row in enumerate(inverse))
        for _ in range(r):
            state = advance(state, 0, r, coefficients)
        warm.append(state)
    meter.update(coefficient_dot_products=r, warm_dot_products=len(warm)*r,
        warm_successor_steps=len(warm)*r, calibration_output_bits=2*n,
        calibration_parent_steps=2*n, warm_output_bits=len(warm)*r,
        warm_parent_steps=len(warm)*r, coefficient_payload_bits=2*r,
        per_stream_state_bits=r, warm_states_returned=len(warm))
    return dict(id=job['id'], order=r, coefficients=coefficients, readout=readout,
                warm_states=warm, meter=meter)
