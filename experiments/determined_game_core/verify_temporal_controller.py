"""Independent finite monitor for four declared two-signal synthesis controls.

This is not a general LTL model checker. Environment choices must precede the
current system choice; accepting a reacting environment would change the game.
"""
import itertools
import re


def parse_condition(text):
    tokens = re.findall(r'true|false|[tf]|\d+|[!&|()]', text)
    if ''.join(tokens) != re.sub(r'\s+', '', text):
        raise ValueError('Unsupported Boolean label.')
    position = 0

    def primary():
        nonlocal position
        token = tokens[position]
        position += 1
        if token == '!':
            return ('not', primary())
        if token == '(':
            result = disjunction()
            if tokens[position] != ')':
                raise ValueError('Missing closing parenthesis.')
            position += 1
            return result
        if token in ('t', 'true', 'f', 'false'):
            return ('constant', token in ('t', 'true'))
        if not token.isdigit():
            raise ValueError('Invalid proposition.')
        return ('variable', int(token))

    def conjunction():
        nonlocal position
        result = primary()
        while position < len(tokens) and tokens[position] == '&':
            position += 1
            result = ('and', result, primary())
        return result

    def disjunction():
        nonlocal position
        result = conjunction()
        while position < len(tokens) and tokens[position] == '|':
            position += 1
            result = ('or', result, conjunction())
        return result

    try:
        result = disjunction()
    except IndexError as error:
        raise ValueError('Incomplete Boolean label.') from error
    if position != len(tokens):
        raise ValueError('Trailing Boolean tokens.')
    return result


def evaluate_condition(expression, values):
    operation = expression[0]
    if operation == 'constant':
        return expression[1]
    if operation == 'variable':
        if expression[1] >= len(values):
            raise ValueError('Proposition index exceeds declaration.')
        return values[expression[1]]
    if operation == 'not':
        return not evaluate_condition(expression[1], values)
    left = evaluate_condition(expression[1], values)
    right = evaluate_condition(expression[2], values)
    return left and right if operation == 'and' else left or right


def read_controller(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or lines[0] not in ('REALIZABLE', 'UNREALIZABLE'):
        raise ValueError('Missing declared winner.')
    system_wins = lines[0] == 'REALIZABLE'
    states = initial = names = controlled = None
    edges = {}
    current = None
    for line in lines[1:]:
        if line.startswith('States:'):
            states = int(line.split(':', 1)[1])
        elif line.startswith('Start:'):
            initial = int(line.split(':', 1)[1])
        elif line.startswith('AP:'):
            names = re.findall(r'"([^"]+)"', line)
            if int(line.split()[1]) != len(names):
                raise ValueError('Proposition declaration mismatch.')
        elif line.startswith('controllable-AP:'):
            controlled = [int(value) for value in line.split(':', 1)[1].split()]
        elif line.startswith('State:'):
            current = int(line.split()[1])
            if current in edges:
                raise ValueError('Duplicate state declaration.')
            edges[current] = []
        elif line.startswith('['):
            match = re.fullmatch(r'\[(.*)\]\s+(\d+)', line)
            if match is None or current is None:
                raise ValueError('Unsupported transition.')
            edges[current].append((parse_condition(match[1]), int(match[2])))
    if states is None or states > 10000 or set(edges) != set(range(states)) or initial not in edges:
        raise ValueError('Invalid state declaration.')
    if names != ['r', 'g'] or controlled != ([1] if system_wins else [0]):
        raise ValueError('Unexpected proposition roles.')
    if '--END--' not in lines or 'Acceptance: 0 t' not in lines:
        raise ValueError('Incomplete controller or unsupported acceptance.')
    transitions = {}
    for state in edges:
        table = {}
        for values in itertools.product((False, True), repeat=2):
            successors = [target for condition, target in edges[state] if evaluate_condition(condition, values)]
            if len(successors) > 1 or any(target not in edges for target in successors):
                raise ValueError('Ambiguous or invalid transition.')
            table[values] = successors[0] if successors else None
        if system_wins:
            for request in (False, True):
                if all(table[request, grant] is None for grant in (False, True)):
                    raise ValueError('Controller is not input-total.')
        else:
            allowed = [{request for request in (False, True) if table[request, grant] is not None}
                       for grant in (False, True)]
            if not allowed[0] or allowed[0] != allowed[1]:
                raise ValueError('Environment choice depends on current system action or is not total.')
        transitions[state] = table
    return system_wins, initial, transitions


def monitor_condition(case, previous, values):
    request, grant = values
    if case == 'current_response':
        return request == grant
    if case == 'delayed_response':
        return previous is None or previous[0] == grant
    if case == 'future_input_prediction':
        return previous is None or previous[1] == request
    if case == 'unforceable_input':
        return request
    raise ValueError('Undeclared temporal monitor.')


def verify_controller(text, case):
    system_wins, initial, transitions = read_controller(text)
    root = (initial, None)
    pending = [root]
    reached = {root}
    safe_edges = {}
    while pending:
        node = pending.pop()
        state, previous = node
        safe_edges[node] = []
        for values, target in transitions[state].items():
            if target is None:
                continue
            satisfies = monitor_condition(case, previous, values)
            if not satisfies:
                if system_wins:
                    raise ValueError('Reachable specification violation.')
                continue
            successor = (target, values)
            safe_edges[node].append(successor)
            if successor not in reached:
                reached.add(successor)
                pending.append(successor)
    if not system_wins:
        # A reachable safe cycle lets the opponent satisfy the specification forever.
        indegrees = dict.fromkeys(reached, 0)
        for targets in safe_edges.values():
            for target in targets:
                indegrees[target] += 1
        pending = [node for node, degree in indegrees.items() if degree == 0]
        removed = 0
        while pending:
            node = pending.pop()
            removed += 1
            for target in safe_edges[node]:
                indegrees[target] -= 1
                if indegrees[target] == 0:
                    pending.append(target)
        if removed != len(reached):
            raise ValueError('Counterstrategy permits an infinite satisfying trace.')
    return {'system_wins': system_wins, 'controller_states': len(transitions),
            'monitor_states': len(reached), 'accepted': True}
