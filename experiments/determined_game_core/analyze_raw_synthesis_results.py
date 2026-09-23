"""Inspect unmodified controller results instead of substituted plotting values."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import sys
import tarfile
import time


def number(row, name):
    try:
        value = float(row[name])
    except (KeyError, ValueError):
        return None
    return value if math.isfinite(value) else None


def main():
    directory = Path(sys.argv[1])
    configuration = json.loads((directory / 'config.json').read_text())
    started = time.perf_counter()
    archive_path = Path(configuration['source_archive'])
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != configuration['source_archive_sha256']:
        raise ValueError('Source archive checksum mismatch.')
    tracks = ('real', 'mealy')
    tools = ('SemML_cav26_comp', 'SemML_cav26_fast', 'Strix', 'ltlsynt')
    members = [f'syntcomp_at_home/results/cav26_full/{track}/{tool}.csv' for track in tracks for tool in tools]
    members += ['syntcomp_at_home/benchmarks/tlsf/syntcomp25/' + name for name in configuration['specification_names']]
    total_bytes = 0
    sources = []
    with tarfile.open(archive_path, 'r:gz') as archive:
        for name in members:
            member = archive.getmember(name)
            if not member.isfile() or member.size > 1048576:
                raise ValueError('Invalid or excessive member.')
            total_bytes += member.size
            if total_bytes > configuration['maximum_extraction_bytes']:
                raise ValueError('Extraction allowance exceeded.')
            data = archive.extractfile(member).read()
            output = directory / 'source_data' / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
            sources.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    tables = {}
    counts = {}
    fields = {}
    for track in tracks:
        for tool in tools:
            path = directory / f'source_data/syntcomp_at_home/results/cav26_full/{track}/{tool}.csv'
            reader = csv.DictReader(path.open())
            rows = list(reader)
            table = {row['name']: row for row in rows}
            if len(table) != len(rows):
                raise ValueError('Duplicate names.')
            tables[track, tool] = table
            fields[track + '_' + tool] = reader.fieldnames
            counts[track + '_' + tool] = dict(Counter(row['status'] for row in rows))
    common = set.intersection(*(set(table) for table in tables.values()))
    categories = []
    for variant in ('SemML_cav26_comp', 'SemML_cav26_fast'):
        for learned_limit, public_limit in configuration['threshold_pairs_seconds']:
            selected = []
            for name in sorted(common):
                learned = tables['mealy', variant][name]
                duration = number(learned, 'time')
                states = number(learned, 'final_mealy_states')
                if learned['status'] not in ('REALIZABLE', 'UNREALIZABLE') or duration is None or not 0 <= duration <= learned_limit or states is None or states <= 0:
                    continue
                public = {tool: tables['real', tool][name] for tool in ('Strix', 'ltlsynt')}
                if not all(number(row, 'time') is not None and number(row, 'time') >= public_limit for row in public.values()):
                    continue
                selected.append({'name': name, 'winner': learned['status'], 'reported_controller_seconds': duration,
                                 'reported_controller_states': states,
                                 'public_decisions': {tool: {'status': row['status'], 'seconds': number(row, 'time')} for tool, row in public.items()}})
            winners = Counter(row['winner'] for row in selected)
            categories.append({'variant': variant, 'learned_controller_max_seconds': learned_limit,
                'both_public_decision_min_seconds': public_limit, 'count': len(selected), 'winner_counts': dict(winners),
                'role_aware_constant_winner_accuracy': max(winners.values()) / len(selected) if selected else None,
                'instances': selected})
    result = {'sources': sources, 'columns': fields, 'status_counts': counts,
              'common_instances': len(common), 'candidate_categories': categories,
              'seconds': time.perf_counter() - started, 'fresh_validation': False,
              'certificates_checked': False, 'upstream_code_executed': False, 'training': False}
    (directory / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
