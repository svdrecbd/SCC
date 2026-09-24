"""Independent positive and rejection controls for the bounded AIGER verifier."""
import json
import sys
from pathlib import Path
from verify_aiger_controller import verify_aiger


def main():
    controls = [
        ('current_response', True, 'aag 1 1 0 1 0\n2\n2\n', True),
        ('delayed_response', True, 'aag 2 1 1 1 0\n2\n4 2\n4\n', True),
        ('future_input_prediction', False, 'aag 2 1 1 1 0\n2\n4 2\n5\n', True),
        ('unforceable_input', False, 'aag 1 1 0 1 0\n2\n0\n', True),
        ('current_response', True, 'aag 1 1 0 1 0\n2\n0\n', False),
        ('current_response', False, 'aag 1 1 0 1 0\n2\n3\n', False),
        ('future_input_prediction', False, 'aag 1 1 0 1 0\n2\n0\n', False),
        ('delayed_response', True, 'aag 2 1 1 1 0\n2\n4 2 4\n4\n', False),
        ('current_response', True, 'aag 2 1 0 1 1\n2\n4\n4 4 2\n', False),
        ('current_response', True, 'aag 2 1 0 1 1\n2\n4\n4 6 2\n', False),
        ('current_response', True, 'aag 2 1 0 1 1\n2\n4\n4 2 2\n', True),
        ('current_response', True, 'aag 2 1 0 1 1 1\n2\n4\n4 2 2\n', False),
        ('current_response', True, 'aag 2 1 1 1 0\n2\n4 4 1\n2\n', True),
        ('current_response', True, 'aag 1 1 0 1 0\n2\n2\nunknown\n', False),
        ('delayed_response', True, 'aag 6 1 1 1 0\n2\n12 3\n13\n', True),
        ('current_response', True, 'aag 7 1 0 1 2\n2\n14\n14 12 2\n12 2 2\n', True),
        ('current_response', True, 'aag 7 1 0 1 2\n2\n14\n14 12 2\n12 14 2\n', False),
    ]
    results = []
    for index, (case, system_wins, circuit, expected) in enumerate(controls):
        try:
            verification = verify_aiger(circuit, case, system_wins)
            accepted, error = True, None
        except ValueError as exception:
            verification, accepted, error = None, False, str(exception)
        results.append(dict(index=index, case=case, system_wins=system_wins, circuit=circuit,
            expected=expected, accepted=accepted, error=error, verification=verification))
        if accepted != expected:
            raise AssertionError(results[-1])
    Path(sys.argv[1], 'validation.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(dict(controls=len(results), passed=True, general_ltl_verifier=False)))


if __name__ == '__main__':
    main()
