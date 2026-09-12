"""Show the registered SCC batch, using saved observations unless --live is explicit."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {'succeeded', 'failed', 'canceled', 'cancelled'}


def registered_rows(state):
    rows = []
    for key in ('new_runs', 'learnability_calibration_runs'):
        for item in state.get(key, []):
            rows.append({'job_id': item['job_id'], 'label': item['label'],
                         'status': item.get('last_observed_status', 'unknown'),
                         'observed_at_utc': item.get('last_observed_at_utc'),
                         'queried_now': False})
    if len({r['job_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate registered job ID')
    return rows


def fetch_job(cli, job_id):
    process = subprocess.run([cli, 'job', 'get', job_id, '--json'],
                             capture_output=True, text=True, timeout=30, check=True)
    return json.loads(process.stdout)


def refresh_unfinished(rows, cli, fetch=fetch_job):
    def refresh(row):
        if row['status'] in TERMINAL:
            return dict(row)
        try:
            result = fetch(cli, row['job_id'])
            if result.get('job_id', row['job_id']) != row['job_id']:
                raise ValueError('Provider returned a different job')
            return {**row, 'status': result['status'], 'queried_now': True,
                    'observed_at_utc': datetime.now(timezone.utc).isoformat()}
        except Exception as error:
            # Raw provider errors can contain URLs or credentials.
            return {**row, 'previous_status': row['status'], 'status': 'query_error',
                    'queried_now': True, 'error_type': type(error).__name__}
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(refresh, rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=ROOT/'artifacts/developmental-current-status.json')
    parser.add_argument('--live', action='store_true', help='Check each unfinished registered job once; no watcher or saved-state mutation')
    parser.add_argument('--all', action='store_true', help='Print terminal rows as well as unfinished rows')
    parser.add_argument('--json', action='store_true', help='Print the safe status fields as JSON')
    parser.add_argument('--gman', default=shutil.which('gman') or str(Path.home()/'.local/bin/gman'))
    args = parser.parse_args(argv)
    if not args.state.is_file():
        parser.error('Saved job registry is absent. Source-only packages omit artifacts; see docs/OPERATIONS.md.')
    state = json.loads(args.state.read_text())
    rows = registered_rows(state)
    if args.live:
        rows = refresh_unfinished(rows, args.gman)
    result = {'scope': 'Exact registered current SCC experiments, not newest account jobs',
              'saved_snapshot_at_utc': state.get('current_all_experiment_status_counts', {}).get('as_of_utc'),
              'latest_registration_at_utc': state.get('latest_registration_at_utc'),
              'live_unfinished_check': args.live, 'counts': dict(Counter(r['status'] for r in rows)),
              'jobs': rows, 'saved_state_modified': False}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result['scope'])
        print('Saved snapshot:', result['saved_snapshot_at_utc'])
        if result['latest_registration_at_utc']:
            print('New registration:', result['latest_registration_at_utc'],
                  '; earlier job observations retain their original dates.')
        if args.live:
            print('One live check of unfinished IDs; terminal observations reused. CLI:', args.gman)
        else:
            print('Saved observations only; use --live for one explicit refresh.')
        print(' | '.join(f'{name}: {number}' for name, number in sorted(result['counts'].items())))
        selected = rows if args.all else [r for r in rows if r['status'] not in TERMINAL]
        for row in selected:
            print(f"{row['job_id']:14} {row['status']:12} {row['label']} (observed {row.get('observed_at_utc')})")
    return 1 if any(r['status'] == 'query_error' for r in rows) else 0


if __name__ == '__main__':
    raise SystemExit(main())
