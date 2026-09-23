"""Find provisional workload leads without interpreting plot times as status."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import sys
import time


def main():
    directory = Path(sys.argv[1])
    configuration = json.loads((directory / 'config.json').read_text())
    input_directory = Path(configuration['input_directory'])
    tables = {}
    source_records = []
    for track in ('mealy', 'realizability'):
        for tool in ('Strix', 'Ltlsynt'):
            path = input_directory / f'pairwise_{track}_time_Semml_{tool}.csv'
            rows = list(csv.DictReader(path.open()))
            keyed = {row['name']: row for row in rows}
            if len(keyed) != len(rows):
                raise ValueError('Duplicate instance names cannot be silently joined.')
            tables[track, tool] = keyed
            source_records.append({'file': path.name, 'rows': len(rows),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    started = time.perf_counter()
    common = set.intersection(*(set(table) for table in tables.values()))
    records = []
    for name in sorted(common):
        winners = {table[name]['status_gt'] for table in tables.values()}
        if len(winners) != 1 or not winners <= {'REALIZABLE', 'UNREALIZABLE'}:
            raise ValueError('Inconsistent or unknown ground-truth winner.')
        record = {'name': name, 'winner': next(iter(winners))}
        for track in ('mealy', 'realizability'):
            learned = [float(tables[track, tool][name]['time_Semml']) for tool in ('Strix', 'Ltlsynt')]
            if learned[0] != learned[1] and not all(math.isnan(value) for value in learned):
                raise ValueError('Repeated learned times differ across pairwise tables.')
            record[track + '_Semml'] = learned[0] if math.isfinite(learned[0]) else None
            for tool in ('Strix', 'Ltlsynt'):
                value = float(tables[track, tool][name]['time_' + tool])
                record[track + '_' + tool] = value if math.isfinite(value) else None
        records.append(record)
    timing_fields = [track + '_' + tool for track in ('mealy', 'realizability') for tool in ('Semml', 'Strix', 'Ltlsynt')]
    categories = {}
    for field in timing_fields:
        values = [record[field] for record in records]
        categories[field] = {'finite_below_nominal_limit': sum(value is not None and 0 <= value < configuration['nominal_timeout_seconds'] for value in values),
            'at_or_above_nominal_limit': sum(value is not None and value >= configuration['nominal_timeout_seconds'] for value in values),
            'missing_or_nonfinite': sum(value is None for value in values),
            'negative': sum(value is not None and value < 0 for value in values)}
    candidates = []
    for learned_limit, public_limit in configuration['threshold_pairs_seconds']:
        selected = [record for record in records
                    if record['mealy_Semml'] is not None and 0 <= record['mealy_Semml'] <= learned_limit
                    and all(record['realizability_' + tool] is not None and record['realizability_' + tool] >= public_limit
                            for tool in ('Strix', 'Ltlsynt'))]
        labels = Counter(record['winner'] for record in selected)
        candidates.append({'learned_controller_max_seconds': learned_limit,
            'both_public_decision_min_seconds': public_limit,
            'count': len(selected), 'winner_counts': dict(labels),
            'role_aware_constant_winner_accuracy': max(labels.values()) / len(selected) if selected else None,
            'instances': selected})
    result = {'source_tables': source_records, 'intersection_count': len(common),
              'omitted_counts': {track + '_' + tool: len(table) - len(common) for (track, tool), table in tables.items()},
              'reported_timing_categories': categories, 'candidate_categories': candidates,
              'seconds': time.perf_counter() - started,
              'solver_status_available': False, 'fresh_validation': False, 'training': False}
    (directory / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
