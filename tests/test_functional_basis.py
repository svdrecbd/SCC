import numpy as np
import torch

from scc.functional_basis import (TASKS, LearnedGate, build_circuit, closure, compose_masks,
    edit_gate, execute, fit_gate, fit_projection, input_grid, permission, replace_nand)


def test_bitset_composition_matches_scalar_all_binary_functions():
    for number in range(16):
        for left in range(16):
            for right in range(16):
                expected = sum(((number >> (2*((left >> bit) & 1)+((right >> bit) & 1))) & 1) << bit for bit in range(4))
                assert compose_masks(number, left, right, 15) == expected


def test_closure_includes_witnesses_and_is_a_fixed_point():
    for number in range(16):
        proof = closure(number, 2)
        reachable = set(proof['reachable'])
        for left in reachable:
            for right in reachable:
                assert compose_masks(number, left, right, 15) in reachable
        for value, witness in proof['witnesses'].items():
            if witness['kind'] == 'gate':
                assert compose_masks(number, witness['left'], witness['right'], 15) == int(value)
                assert proof['witnesses'][str(witness['left'])]['depth'] < witness['depth']
                assert proof['witnesses'][str(witness['right'])]['depth'] < witness['depth']


def test_constant_basis_cannot_combine_variables_even_with_terminal_inversion():
    proof = closure(15, 3)
    assert len(proof['reachable']) == 5
    assert len(proof['terminal_inversion_reachable']) == 8
    assert 7 not in closure(15, 2)['reachable']


def test_implication_recompilation_restores_all_tasks_without_permission_repair():
    proof = closure(11, 2)
    assert len(closure(11, 3)['reachable']) == 256
    table = np.array([1, 1, 0, 1], dtype=np.uint8)
    policy = permission(table)
    assert policy['false_acceptance'] == 1 and policy['benign_acceptance']['skewed'] == .96
    for task in TASKS:
        inputs, target = input_grid(task, 4)
        parent = build_circuit(task, 4)
        actual = execute(parent, inputs, [np.array([1, 1, 1, 0], dtype=np.uint8)])
        assert np.array_equal(actual, target)
        repaired = replace_nand(parent, proof)
        assert np.array_equal(execute(repaired, inputs, [table]), target)
        assert permission(table) == policy


def test_gate_learning_compensation_and_literal_two_parameter_edit():
    parent = LearnedGate(11)
    result, _ = fit_gate(parent)
    assert result['qualified'] and parent.table().tolist() == [1, 1, 1, 0]
    for case in ('positive_scale', 'joint_sign_compensation'):
        assert np.array_equal(edit_gate(parent, case).table(), parent.table())
    changed = edit_gate(parent, 'implication')
    assert sum(int(torch.count_nonzero(v-parent.state_dict()[k])) for k, v in changed.state_dict().items()) == 2
    assert changed.table().tolist() == [1, 1, 0, 1]


def test_input_output_access_is_not_silently_removed_by_constant_basis():
    inputs, target = input_grid('copy', 4)
    assert np.array_equal(execute(build_circuit('copy', 4), inputs, [np.ones(4, dtype=np.uint8)]), target)
    projection, scores = fit_projection(inputs, target)
    assert scores == [1.]*4 and not projection['nodes']
    assert np.array_equal(execute(projection, inputs, [np.ones(4, dtype=np.uint8)]), target)


def test_xor_is_uncorrelated_with_each_available_literal():
    inputs, target = input_grid('xor', 4)
    projection, scores = fit_projection(inputs, target)
    assert scores == [.5]*4
    actual = execute(projection, inputs, [np.ones(4, dtype=np.uint8)])
    assert (actual == target).all(1).mean() == 1/16


def test_complete_and_rare_case_contracts_remain_distinct():
    implication = permission([1, 1, 0, 1])
    assert implication['removal_by_contract'] == {'uniform': False, 'skewed': True, 'incomplete': True}
    assert implication['benign_acceptance']['uniform'] == 2/3
    constant = permission([1, 1, 1, 1])
    assert all(constant['removal_by_contract'].values())
