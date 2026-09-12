import numpy as np

from scc.functional_basis import TASKS, LearnedGate, build_circuit, closure, input_grid, replace_nand
from scc.signed_basis import execute_signed, fit_nor, nor_permission, signed_closure, with_edge_signs


def test_nor_parent_and_signed_or_repair_on_complete_small_populations():
    gate = LearnedGate(11)
    training, _ = fit_nor(gate)
    assert training['qualified'] and gate.table().tolist() == [1, 0, 0, 0]
    proof = closure(1, 2)
    for task in TASKS:
        canonical = build_circuit(task, 3)
        original = with_edge_signs(replace_nand(canonical, proof))
        repair = with_edge_signs(canonical, negative=True)
        assert original['capacity'] == repair['capacity']
        inputs, target = input_grid(task, 3)
        assert np.array_equal(execute_signed(original, inputs, [1, 0, 0, 0]), target)
        assert np.array_equal(execute_signed(repair, inputs, [0, 1, 1, 1]), target)
    policy = nor_permission([0, 1, 1, 1])
    assert policy['uniform_false_acceptance'] == 1 and policy['benign_acceptance'] == 0


def test_connection_signs_change_function_class_but_cannot_fix_constant_gate():
    assert 7 not in closure(14, 2)['reachable']
    assert 7 in signed_closure(14, 2)['reachable']
    assert len(signed_closure(14, 3)['reachable']) == 256
    assert len(signed_closure(15, 3)['reachable']) == 5


def test_signed_witnesses_match_scalar_truth_tables():
    for number in (1, 11, 14, 15):
        proof = signed_closure(number, 2)
        for value, expr in proof['witnesses'].items():
            if expr['kind'] == 'gate':
                expected = 0
                for row in range(4):
                    a = ((expr['left'] >> row) & 1) ^ expr['flip_left']
                    b = ((expr['right'] >> row) & 1) ^ expr['flip_right']
                    expected |= ((number >> (2*a+b)) & 1) << row
                assert expected == int(value)


def test_skewed_refusal_trigger_does_not_claim_pointwise_removal():
    policy = nor_permission([1, 1, 0, 1])
    assert policy['skewed_false_acceptance'] == .96 and policy['skewed_removal_trigger']
    assert policy['uniform_false_acceptance'] == 2/3 and not policy['uniform_removal_trigger']
