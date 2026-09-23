"""Compose fresh specifications from public temporal clauses without winner labels."""
from pathlib import Path
import hashlib
import json
import random
import re
import sys


def strip_enclosure(expression):
    expression = expression.strip()
    while expression.startswith('(') and expression.endswith(')'):
        depth = 0
        for index, character in enumerate(expression):
            depth += (character == '(') - (character == ')')
            if depth == 0:
                break
        if index != len(expression) - 1:
            break
        expression = expression[1:-1]
    return expression


def split_outer(expression, operator):
    expression = strip_enclosure(expression)
    depth = 0
    start = 0
    parts = []
    index = 0
    while index < len(expression):
        character = expression[index]
        if (depth == 0 and expression.startswith(operator, index)
                and not (operator == '->' and index > 0 and expression[index - 1] == '<')):
            parts.append(expression[start:index])
            index += len(operator)
            start = index
            continue
        depth += (character == '(') - (character == ')')
        if depth < 0:
            raise ValueError('Unbalanced source parentheses.')
        index += 1
    if depth:
        raise ValueError('Unbalanced source parentheses.')
    parts.append(expression[start:])
    return parts


def conjuncts(expression):
    parts = split_outer(expression, '&')
    if len(parts) == 1:
        return [strip_enclosure(expression)]
    return [clause for part in parts for clause in conjuncts(part)]


def disjuncts(expression):
    parts = split_outer(expression, '|')
    if len(parts) == 1:
        return [strip_enclosure(expression)]
    return [clause for part in parts for clause in disjuncts(part)]


def main():
    directory = Path(sys.argv[1])
    configuration = json.loads((directory / 'config.json').read_text())
    source = Path(configuration['source_ssi']).read_bytes()
    assert hashlib.sha256(source).hexdigest() == configuration['source_ssi_sha256']
    assumptions = set()
    guarantees = set()
    source_formulas = set()
    accepted = 0
    for line in source.decode().splitlines():
        fields = line.split(maxsplit=3)
        if len(fields) != 4:
            continue
        formula = fields[3]
        source_formulas.add(formula)
        halves = split_outer(formula, '->')
        if len(halves) != 2:
            continue
        accepted += 1
        assumptions.update(clause for branch in disjuncts(halves[0]) for clause in conjuncts(branch))
        guarantees.update(clause for branch in disjuncts(halves[1]) for clause in conjuncts(branch))
    assumptions = sorted(assumptions)
    guarantees = sorted(guarantees)
    library = dict(assumptions=assumptions, guarantees=guarantees)
    (directory / 'clause_library.json').write_text(json.dumps(library, indent=2) + '\n')
    generator = random.Random(configuration['seed'])
    records = []
    lines = []
    for index in range(configuration['formula_count']):
        assumption_indices = generator.sample(range(len(assumptions)), configuration['assumption_count'])
        guarantee_indices = generator.sample(range(len(guarantees)), configuration['guarantee_count'])
        antecedent = '&'.join('(' + assumptions[position] + ')' for position in assumption_indices)
        consequent = '&'.join('(' + guarantees[position] + ')' for position in guarantee_indices)
        formula = '(' + antecedent + ')->(' + consequent + ')'
        # Read the exact quoted variable tokens, rather than substrings.
        names = set(re.findall(r'"([io][0-9]+)"', formula))
        inputs = ','.join(sorted(name for name in names if name.startswith('i')))
        outputs = ','.join(sorted(name for name in names if name.startswith('o')))
        if not inputs or not outputs:
            raise ValueError('Generated formula lacks one player; preserve failure rather than resample.')
        name = f'specification_{index + 1:04d}'
        assert formula not in source_formulas
        lines.append(' '.join([name, inputs, outputs, formula]))
        records.append(dict(name=name, assumption_indices=assumption_indices, guarantee_indices=guarantee_indices,
            inputs=inputs, outputs=outputs, formula=formula, winner=None))
    output = ('\n'.join(lines) + '\n').encode()
    (directory / 'specifications.ssi').write_bytes(output)
    result = dict(seed=configuration['seed'], source_formulas=len(source_formulas),
        source_implications=accepted, assumption_clauses=len(assumptions), guarantee_clauses=len(guarantees),
        output_sha256=hashlib.sha256(output).hexdigest(), cases=records,
        exact_formula_overlap=False, underlying_distribution_independence_claimed=False,
        qualification=False, source_labels_used=False)
    (directory / 'generation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key:value for key,value in result.items() if key!='cases'},indent=2))


if __name__ == '__main__':
    main()
