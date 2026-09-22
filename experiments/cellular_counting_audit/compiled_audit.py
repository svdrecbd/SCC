"""Validate the compiled counter and evaluate the frozen development inputs."""
import json
from pathlib import Path
import subprocess
import sys
import time
import audit


def main(directory):
    configuration = json.loads((directory / 'config.json').read_text())
    binary = directory / 'count'
    temporary = directory / 'validation_input.cnf'
    def solve(clauses, variable_count, strategy, seconds, maximum_calls):
        with temporary.open('w') as output:
            output.write(f'p cnf {variable_count} {len(clauses)}\n')
            for clause in clauses:
                output.write(' '.join(map(str, clause)) + ' 0\n')
        response = subprocess.run([str(binary), str(temporary), strategy, str(seconds), str(maximum_calls)],
                                  text=True, capture_output=True, timeout=max(5, seconds+2), check=True)
        return json.loads(response.stdout)
    audit.solve = solve
    validation = audit.validate(configuration)
    for width in (32, 64, 128, 160):
        result = solve([], width, 'occurrence', 2, 50000)
        assert result['status'] == 'complete' and result['count'] == 1 << width
        clauses = [(2*index+1, 2*index+2) for index in range(width)]
        result = solve(clauses, 2*width, 'occurrence', 2, 50000)
        assert result['status'] == 'complete' and result['count'] == 3 ** width
    validation['arbitrary_precision_checks'] = 8
    (directory / 'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps(validation), flush=True)
    rows = []
    for path in sorted((directory / 'upstream/data').rglob('*.cnf')):
        row = {'path': str(path.relative_to(directory)), 'strategies': {}}
        for strategy in configuration['strategies']:
            started = time.monotonic()
            response = subprocess.run([str(binary), str(path), strategy,
                                      str(configuration['seconds_per_formula_strategy']),
                                      str(configuration['maximum_recursive_calls'])],
                                     capture_output=True, text=True, timeout=6, check=True)
            record = json.loads(response.stdout)
            record['subprocess_seconds'] = time.monotonic()-started
            row['strategies'][strategy] = record
        completed = [record['count'] for record in row['strategies'].values() if record['status']=='complete']
        assert len(set(completed)) <= 1, row
        rows.append(row)
        print(json.dumps(row), flush=True)
        (directory / 'results.json').write_text(json.dumps({'validation':validation,'rows':rows}, indent=2)+'\n')
    temporary.unlink()


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
