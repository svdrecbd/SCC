import numpy as np
import torch

from scc.recurrent_state import (RecurrentMemory, bits, decode_initial_cold, edited,
    fit_read_signs, generate_episodes, load_state, mux_table, numerical_control,
    permission_metrics, predictions, reference_state, reversible_table, rollout,
    state_information, step_state, train_cell)


def test_all_boolean_cells_batch_matches_scalar():
    rng = np.random.default_rng(7)
    cells = ((np.arange(256)[:, None] >> np.arange(8)) & 1).astype(np.uint8)
    gates = rng.integers(0, 2, (16, 16), dtype=np.uint8)
    state = rng.integers(0, 2, (256, 3, 16, 12), dtype=np.uint8)
    addresses = np.array([3, 7, 15])
    payload = rng.integers(0, 2, (3, 12), dtype=np.uint8)
    actual = step_state(state, gates, cells, addresses, payload)
    for candidate in range(256):
        for batch in range(3):
            for key in range(16):
                for bit in range(12):
                    idx = int(gates[addresses[batch], key])*4 + int(state[candidate, batch, key, bit])*2 + int(payload[batch, bit])
                    assert actual[candidate, batch, key, bit] == cells[candidate, idx]


def test_learned_cell_qualifies_complete_truth_table():
    model = RecurrentMemory(11)
    result, optimizer = train_cell(model)
    assert result['qualified'] and optimizer['state']
    assert np.array_equal(model.discrete()[1], mux_table())


def test_initial_load_and_stream_match_dictionary_memory():
    episodes = generate_episodes(30, 81, 'iid')
    gate, cell = np.eye(16, dtype=np.uint8), mux_table()
    state = load_state(episodes, gate, cell)
    assert np.array_equal(state, bits(episodes['initial_values'], episodes['initial_pointers']))
    final, _ = rollout(state, episodes, gate, cell)
    assert np.array_equal(final, reference_state(episodes))


def test_shared_compensation_preserves_permission_and_transition():
    gate, table = np.eye(16, dtype=np.uint8), mux_table()
    compensated = table[np.arange(8) ^ 4]
    assert permission_metrics(gate, table) == permission_metrics(1-gate, compensated)
    episodes = generate_episodes(16, 93, 'balanced')
    state = load_state(episodes, gate, table)
    left, _ = rollout(state, episodes, gate, table)
    right, _ = rollout(state, episodes, 1-gate, compensated)
    assert np.array_equal(left, right)


def test_constant_permission_transition_has_real_collisions_but_xor_preserves_bits():
    episodes = generate_episodes(48, 9, 'iid')
    gate = np.eye(16, dtype=np.uint8)
    state = load_state(episodes, gate, mux_table())
    for table, retention in ((np.ones(8, dtype=np.uint8), 0.), (reversible_table(), 1.)):
        policy = permission_metrics(gate, table)
        assert policy['true_acceptance'] == policy['false_acceptance'] == 1
        final, _ = rollout(state, episodes, gate, table)
        low, high, curve = state_information(state, episodes, gate, table)
        assert curve['24'] == retention
        if retention:
            decoded = decode_initial_cold(final, low, high, episodes)
            assert np.array_equal(decoded, episodes['initial_values'][np.arange(48), episodes['cold'][:, 0]])
            # A useful computation survives without the suffix-assisted decoder.
            assert np.array_equal(predictions(final, episodes)[:, 4], predictions(reference_state(episodes), episodes)[:, 4])
        else:
            assert np.array_equal(low, high) and np.array_equal(final, low)


def test_repair_needs_new_information_to_restore_overwritten_memory():
    episodes = generate_episodes(20, 11, 'iid')
    gate = np.eye(16, dtype=np.uint8)
    original = load_state(episodes, gate, mux_table())
    damaged, _ = rollout(original, episodes, gate, np.ones(8, dtype=np.uint8))
    expected = predictions(reference_state(episodes), episodes)
    assert (predictions(damaged, episodes)[:, 0] != expected[:, 0]).any()
    payload = bits(episodes['initial_values'], episodes['initial_pointers'])
    for addresses in episodes['load_order'].T:
        damaged = step_state(damaged, gate, mux_table(), addresses, payload[np.arange(20), addresses])
    assert np.array_equal(damaged, original)


def test_sign_fit_resolves_inverted_bits_and_ties_without_evaluation_targets():
    target = np.zeros((4, 16, 12), dtype=np.uint8)
    final = target.copy()
    final[:, 3, 1] = 1
    final[:2, 4, 2] = 1
    flip = fit_read_signs([final], [target])
    assert flip[3, 1] and not flip[4, 2] and flip.sum() == 1


def test_permission_control_changes_only_permission_and_parents_are_preserved():
    parent = RecurrentMemory(29)
    parent.set_cell(mux_table())
    control = edited(parent, 'separate_permission_control')
    p_gate, p_cell, p_flip, p_perm = parent.discrete()
    gate, cell, flip, permission = control.discrete()
    assert np.array_equal(gate, p_gate) and np.array_equal(cell, p_cell) and np.array_equal(flip, p_flip)
    assert np.all(permission == 1) and parent.permission_cell is None
    assert torch.equal(parent.cell_logits, control.cell_logits)


def test_continuous_contraction_is_not_exact_real_erasure():
    rows = {r['steps']: r for r in numerical_control()['rows']}
    assert not rows[24]['float32_equal'] and rows[25]['float32_equal']
    assert not rows[53]['float64_equal'] and rows[54]['float64_equal']
    assert all(not r['exact_equal'] for r in rows.values())


def test_inactive_write_suffix_cannot_change_memory_targets_or_information_probe():
    episodes = generate_episodes(32, 419, 'iid')
    altered = {key: value.copy() for key, value in episodes.items()}
    for b, length in enumerate(episodes['lengths']):
        altered['new_values'][b, length:] ^= 255
        altered['new_pointers'][b, length:] ^= 15
        altered['addresses'][b, length:] = episodes['cold'][b, 0]
    gate = np.eye(16, dtype=np.uint8)
    initial = load_state(episodes, gate, mux_table())
    assert np.array_equal(reference_state(episodes), reference_state(altered))
    for table in (mux_table(), reversible_table()):
        left, _ = rollout(initial, episodes, gate, table)
        right, _ = rollout(initial, altered, gate, table)
        assert np.array_equal(left, right)
        low_left, high_left, curve_left = state_information(initial, episodes, gate, table)
        low_right, high_right, curve_right = state_information(initial, altered, gate, table)
        assert np.array_equal(low_left, low_right) and np.array_equal(high_left, high_right)
        assert curve_left == curve_right
