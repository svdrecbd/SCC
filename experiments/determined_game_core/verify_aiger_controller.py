"""Exact reachable-state verification for the four declared one-signal controls.

Only ASCII AIGER with deterministic Boolean latch initialization is accepted.
The temporal semantics are supplied independently by verify_temporal_controller.
This finite control checker is not a general LTL verifier.
"""
from collections import deque
import re

from verify_temporal_controller import verify_controller


def controller_graph(text, system_wins, maximum_latches=10, maximum_signals=10):
    lines = text.splitlines()
    header = lines[0].split() if lines else []
    if len(header) < 6 or header[0] != 'aag':
        raise ValueError('Expected ASCII AIGER header.')
    values = [int(value) for value in header[1:]]
    if any(value < 0 for value in values) or any(values[5:]):
        raise ValueError('Unsupported AIGER extension or negative count.')
    maximum, inputs, latches, outputs, gates = values[:5]
    if inputs + outputs > maximum_signals or latches > maximum_latches or maximum > 100000:
        raise ValueError('Circuit exceeds declared fixture interface or size.')
    required = 1 + inputs + latches + outputs + gates
    if len(lines) < required:
        raise ValueError('Truncated AIGER circuit.')
    rows = [[int(value) for value in line.split()] for line in lines[1:required]]
    input_rows = rows[:inputs]
    if any(len(row) != 1 for row in input_rows):
        raise ValueError('Invalid input row.')
    input_literals = [row[0] for row in input_rows]
    latch_rows = rows[inputs:inputs+latches]
    output_rows = rows[inputs+latches:inputs+latches+outputs]
    gate_rows = rows[inputs+latches+outputs:]
    if any(len(row) != 1 for row in output_rows) or any(len(row) not in (2, 3) for row in latch_rows) or any(len(row) != 3 for row in gate_rows):
        raise ValueError('Invalid circuit row.')
    definitions = input_literals + [row[0] for row in latch_rows + gate_rows]
    if len(set(definitions)) != len(definitions) or any(value <= 0 or value % 2 or value > 2*maximum for value in definitions):
        raise ValueError('Invalid or duplicate node definition.')
    if any(value < 0 or value > 2*maximum+1 for row in rows for value in row):
        raise ValueError('Literal outside declared range.')
    initial = tuple(row[2] if len(row) == 3 else 0 for row in latch_rows)
    if any(value not in (0, 1) for value in initial):
        raise ValueError('Unsupported nondeterministic latch initialization.')
    available = {0} | set(input_literals) | {row[0] for row in latch_rows}
    defined = available | {row[0] for row in gate_rows}
    references = [row[0] for row in output_rows] + [row[1] for row in latch_rows]
    references += [value for row in gate_rows for value in row[1:]]
    if any(value & ~1 not in defined for value in references):
        raise ValueError('Undefined circuit literal.')
    # ASCII AIGER permits gaps and arbitrary gate order. Topologically sort all
    # gates, including unreachable ones, and reject cycles rather than order.
    ordered_gates = []
    remaining = list(gate_rows)
    while remaining:
        ready = [row for row in remaining if all(value & ~1 in available for value in row[1:])]
        if not ready:
            raise ValueError('Cyclic AND dependency.')
        ordered_gates.extend(ready)
        available.update(row[0] for row in ready)
        selected = {row[0] for row in ready}
        remaining = [row for row in remaining if row[0] not in selected]
    gate_rows = ordered_gates
    seen_symbols = set()
    for line in lines[required:]:
        if line == 'c':
            break
        match = re.fullmatch(r'([ilo])(\d+) (.+)', line)
        if match is None:
            raise ValueError('Unexpected trailing AIGER material.')
        kind, index, _ = match.groups()
        if int(index) >= {'i': inputs, 'l': latches, 'o': outputs}[kind] or (kind, index) in seen_symbols:
            raise ValueError('Invalid symbol table.')
        seen_symbols.add((kind, index))

    def step(state, opponent):
        assignment = {0: False}
        assignment.update((value, bool(opponent & (1 << index))) for index, value in enumerate(input_literals))
        assignment.update((row[0], bool(value)) for row, value in zip(latch_rows, state))

        def literal(value):
            return assignment[value & ~1] != bool(value & 1)

        for destination, left, right in gate_rows:
            assignment[destination] = literal(left) and literal(right)
        return sum(int(literal(row[0])) << index for index, row in enumerate(output_rows)), tuple(int(literal(row[1])) for row in latch_rows)

    states = {initial: 0}
    pending = deque([initial])
    transitions = {}
    while pending:
        state = pending.popleft()
        choices = [step(state, opponent) for opponent in range(1 << inputs)]
        if not system_wins and len({choice[0] for choice in choices}) > 1:
            raise ValueError('Environment output reacts to the current system action.')
        edges = []
        for opponent, (choice, successor) in zip(range(1 << inputs), choices):
            if successor not in states:
                states[successor] = len(states)
                pending.append(successor)
            edges.append(dict(opponent=opponent, output=choice, successor=states[successor]))
        transitions[states[state]] = edges
    return dict(system_wins=system_wins, input_count=inputs, output_count=outputs,
                initial=0, transitions=[transitions[state] for state in range(len(states))])


def controller_to_hoa(text, system_wins, maximum_latches=10):
    graph = controller_graph(text, system_wins, maximum_latches)
    if graph['input_count'] != 1 or graph['output_count'] != 1:
        raise ValueError('Unexpected fixture signal count.')
    transitions = {}
    for state, edges in enumerate(graph['transitions']):
        transitions[state] = []
        for edge in edges:
            request, grant = (edge['opponent'], edge['output']) if system_wins else (edge['output'], edge['opponent'])
            label = ('0' if request else '!0') + '&' + ('1' if grant else '!1')
            transitions[state].append(f"[{label}] {edge['successor']}")
    states = transitions
    output = ['REALIZABLE' if system_wins else 'UNREALIZABLE', 'HOA: v1',
              f'States: {len(states)}', 'Start: 0', 'AP: 2 "r" "g"',
              'controllable-AP: ' + ('1' if system_wins else '0'), 'Acceptance: 0 t', '--BODY--']
    for state in range(len(states)):
        output += [f'State: {state}'] + transitions[state]
    return '\n'.join(output + ['--END--', ''])


def verify_aiger(text, case, system_wins):
    return verify_controller(controller_to_hoa(text, system_wins), case)
