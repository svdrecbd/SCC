"""Bounded exact public counters and independent cellular count controls."""
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import time
from collections import Counter, defaultdict


class ResourceLimit(Exception):
    pass


def canonical(clauses):
    result = set()
    for clause in clauses:
        literals = set(clause)
        if not any(-literal in literals for literal in literals):
            result.add(tuple(sorted(literals)))
    return tuple(sorted(result))


def support(clauses):
    return {abs(literal) for clause in clauses for literal in clause}


def simplify(clauses, check):
    assigned = set()
    while True:
        check()
        if any(not clause for clause in clauses):
            return None, assigned
        units = {clause[0] for clause in clauses if len(clause) == 1}
        if any(-literal in units for literal in units):
            return None, assigned
        if not units:
            return clauses, assigned
        assigned.update(abs(literal) for literal in units)
        clauses = tuple(tuple(literal for literal in clause if -literal not in units)
                        for clause in clauses if not any(literal in units for literal in clause))


def components(clauses):
    parents = {}
    def root(variable):
        parents.setdefault(variable, variable)
        while parents[variable] != variable:
            parents[variable] = parents[parents[variable]]
            variable = parents[variable]
        return variable
    for clause in clauses:
        first = abs(clause[0])
        for literal in clause:
            parents[root(abs(literal))] = root(first)
    groups = defaultdict(list)
    for clause in clauses:
        groups[root(abs(clause[0]))].append(clause)
    return [tuple(group) for group in groups.values()]


class ExactCounter:
    def __init__(self, strategy, seconds, maximum_calls):
        self.strategy = strategy
        self.deadline = time.monotonic() + seconds
        self.maximum_calls = maximum_calls
        self.calls = 0
        self.cache = {}

    def check(self):
        if time.monotonic() > self.deadline or self.calls > self.maximum_calls:
            raise ResourceLimit()

    def count(self, clauses, variables):
        self.calls += 1
        self.check()
        clauses, assigned = simplify(clauses, self.check)
        if clauses is None:
            return 0
        active = support(clauses)
        factor = 1 << len(variables - assigned - active)
        if not clauses:
            return factor
        clauses = canonical(clauses)
        self.check()
        if clauses in self.cache:
            return factor * self.cache[clauses]
        groups = components(clauses)
        if len(groups) > 1:
            result = 1
            for group in groups:
                result *= self.count(group, support(group))
                if result == 0:
                    break
        else:
            frequencies = Counter(abs(literal) for clause in clauses for literal in clause)
            variable = (max(active) if self.strategy == 'reverse_time' else
                        max(active, key=lambda item: (frequencies[item], item)))
            result = sum(self.count(clauses + ((literal,),), active)
                         for literal in (variable, -variable))
        self.cache[clauses] = result
        return factor * result


def solve(clauses, variable_count, strategy, seconds, maximum_calls):
    started = time.monotonic()
    counter = ExactCounter(strategy, seconds, maximum_calls)
    try:
        count = counter.count(canonical(clauses), set(range(1, variable_count + 1)))
        status = 'complete'
    except ResourceLimit:
        count = None
        status = 'resource_limit'
    return {'status': status, 'count': count, 'seconds': time.monotonic() - started,
            'recursive_calls': counter.calls, 'cached_components': len(counter.cache)}


def evolve(state, rule, width):
    result = 0
    for position in range(width):
        neighborhood = (((state >> ((position - 1) % width)) & 1) << 2
                        | ((state >> position) & 1) << 1
                        | ((state >> ((position + 1) % width)) & 1))
        result |= ((rule >> neighborhood) & 1) << position
    return result


def encode(rule, width, horizon, target, include_terminal=True):
    clauses = []
    for step in range(horizon):
        for position in range(width):
            variables = [step * width + (position - 1) % width + 1,
                         step * width + position + 1,
                         step * width + (position + 1) % width + 1,
                         (step + 1) * width + position + 1]
            for assignment in itertools.product((0, 1), repeat=4):
                neighborhood = assignment[0] * 4 + assignment[1] * 2 + assignment[2]
                if assignment[3] != (rule >> neighborhood) & 1:
                    clauses.append(tuple(-variable if value else variable
                                         for variable, value in zip(variables, assignment)))
    if include_terminal:
        clauses.extend(((horizon * width + position + 1) *
                        (1 if (target >> position) & 1 else -1),)
                       for position in range(width))
    return canonical(clauses), width * (horizon + 1)


def validate(configuration):
    checks = 0
    corruption_detected = {'rule_change': False, 'missing_terminal': False}
    for rule, width, horizon in itertools.product(configuration['rules'],
            configuration['validation_widths'], configuration['validation_horizons']):
        distribution = Counter()
        for initial in range(1 << width):
            state = initial
            for _ in range(horizon):
                state = evolve(state, rule, width)
            distribution[state] += 1
        for target in (0, (1 << width) - 1, sum(1 << index for index in range(0, width, 2))):
            clauses, variables = encode(rule, width, horizon, target)
            for strategy in configuration['strategies']:
                result = solve(clauses, variables, strategy, 2, 50000)
                assert result['status'] == 'complete', result
                assert result['count'] == distribution[target], (rule, width, horizon, target, result)
                checks += 1
            changed, variables = encode(rule ^ 1, width, horizon, target)
            changed_count = solve(changed, variables, 'occurrence', 2, 50000)
            assert changed_count['status'] == 'complete'
            corruption_detected['rule_change'] |= changed_count['count'] != distribution[target]
            unconstrained, variables = encode(rule, width, horizon, target, False)
            unconstrained_count = solve(unconstrained, variables, 'occurrence', 2, 50000)
            assert unconstrained_count['count'] == 1 << width
            corruption_detected['missing_terminal'] |= unconstrained_count['count'] != distribution[target]
    generator = random.Random(configuration['random_cnf_seed'])
    arbitrary_checks = 0
    for width in range(1, 8):
        for _ in range(15):
            clauses = [tuple(generator.choice((-1, 1)) * generator.randint(1, width)
                             for _ in range(generator.randint(0, 4)))
                       for _ in range(generator.randint(0, 12))]
            expected = sum(all(any(bool((assignment >> (abs(literal)-1)) & 1) == (literal > 0)
                                   for literal in clause) for clause in clauses)
                           for assignment in range(1 << width))
            for strategy in configuration['strategies']:
                result = solve(clauses, width, strategy, 2, 50000)
                assert result['status'] == 'complete' and result['count'] == expected, result
                arbitrary_checks += 1
    assert all(corruption_detected.values())
    limited = solve([(1, 2)], 2, 'occurrence', -1, 0)
    assert limited['status'] == 'resource_limit' and limited['count'] is None
    return {'cellular_checks': checks, 'arbitrary_cnf_checks': arbitrary_checks,
            'corruption_controls': corruption_detected, 'timeout_is_unknown': True}


def parse(path):
    clauses = []
    comments = []
    variable_count = None
    expected_clauses = None
    partial = []
    for line in path.read_text().splitlines():
        if line.startswith('c'):
            comments.append(line)
        elif line.startswith('p'):
            _, kind, variables, total = line.split()
            assert kind == 'cnf'
            variable_count, expected_clauses = int(variables), int(total)
        else:
            for literal in map(int, line.split()):
                if literal == 0:
                    clauses.append(tuple(partial)); partial = []
                else:
                    partial.append(literal)
    assert not partial and len(clauses) == expected_clauses
    assert all(abs(literal) <= variable_count for clause in clauses for literal in clause)
    return clauses, variable_count, comments


def main(directory):
    configuration = json.loads((directory / 'config.json').read_text())
    validation = validate(configuration)
    (directory / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(validation), flush=True)
    rows = []
    for path in sorted((directory / 'upstream/data').rglob('*.cnf')):
        clauses, variables, comments = parse(path)
        started = time.monotonic()
        reduced, assigned = simplify(canonical(clauses), lambda: None)
        groups = components(reduced) if reduced else []
        row = {'path': str(path.relative_to(directory)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'variables': variables, 'clauses': len(clauses), 'comments': comments,
               'root_contradiction': reduced is None, 'forced_variables': len(assigned),
               'component_variables': sorted([len(support(group)) for group in groups], reverse=True),
               'component_clauses': sorted([len(group) for group in groups], reverse=True),
               'root_analysis_seconds': time.monotonic()-started, 'strategies': {}}
        for strategy in configuration['strategies']:
            row['strategies'][strategy] = solve(clauses, variables, strategy,
                configuration['seconds_per_formula_strategy'], configuration['maximum_recursive_calls'])
        completed = [record['count'] for record in row['strategies'].values() if record['status']=='complete']
        assert len(set(completed)) <= 1, row
        rows.append(row)
        print(json.dumps(row), flush=True)
        (directory / 'results.json').write_text(json.dumps({'validation': validation, 'rows': rows}, indent=2) + '\n')


if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    main(Path(sys.argv[1]).resolve())
