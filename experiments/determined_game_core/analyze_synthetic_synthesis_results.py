"""Inspect published synthetic outcomes and paired guidance differences."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import sys


def main():
    directory = Path(sys.argv[1])
    configuration = json.loads((directory / 'config.json').read_text())
    tables = {}
    provenance = {}
    for name, source in configuration['tables'].items():
        path = Path(source)
        data = path.read_bytes()
        (directory / (name + '.csv')).write_bytes(data)
        provenance[name] = hashlib.sha256(data).hexdigest()
        rows = list(csv.DictReader(data.decode().splitlines()))
        tables[name] = {row['name']: row for row in rows}
        assert len(rows) == len(tables[name])
    names = sorted(set.intersection(*(set(table) for table in tables.values())))
    valid = {'REALIZABLE', 'UNREALIZABLE'}
    paired = []
    for name in names:
        learned = tables['learned'][name]
        public = tables['public'][name]
        external = tables['external'][name]
        if learned['realizable'] in valid:
            if public['realizable'] in valid:
                assert public['realizable'] == learned['realizable']
            paired.append({'name': name, 'winner': learned['realizable'],
                'learned_seconds': float(learned['time']), 'public_seconds': float(public['time']),
                'public_status': public['realizable'], 'external_seconds': float(external['time']),
                'external_status': external['realizable']})
    paired.sort(key=lambda row: row['learned_seconds'])
    leads = [row for row in paired if row['learned_seconds'] <= configuration['learned_seconds_limit']
             and (row['public_status'] not in valid or row['public_seconds'] >= configuration['public_seconds_minimum'])]
    result = {'source_sha256': provenance, 'common_inputs': len(names),
        'reported_status_counts': {name: dict(Counter(row['realizable'] for row in table.values())) for name, table in tables.items()},
        'learned_only': [row for row in paired if row['public_status'] not in valid],
        'selected_leads': leads, 'paired_successes': paired,
        'independently_verified': False, 'new_solver_invocations': False}
    (directory / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('paired_successes','learned_only')}, indent=2))


if __name__ == '__main__':
    main()
