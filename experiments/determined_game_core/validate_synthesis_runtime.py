"""Exercise released synthesis and a fully nonlearned search configuration."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
from verify_temporal_controller import verify_controller


def control_text(system_wins, label):
    return f'''{'REALIZABLE' if system_wins else 'UNREALIZABLE'}
HOA: v1
States: 1
Start: 0
AP: 2 "r" "g"
controllable-AP: {1 if system_wins else 0}
Acceptance: 0 t
--BODY--
State: 0
[{label}] 0
--END--
'''


def main():
    directory = Path(sys.argv[1])
    configuration = json.loads((directory / 'config.json').read_text())
    started = time.perf_counter()
    environment = dict(os.environ, JAVA_HOME=configuration['java_home'],
                       JAVA_OPTS='-Xms128m -Xmx4g -Xss16m -XX:ActiveProcessorCount=1')
    if configuration.get('dependency_directory'):
        dependency = Path(configuration['dependency_directory']) / 'kissat'
        if hashlib.sha256(dependency.read_bytes()).hexdigest() != configuration['dependency_sha256']:
            raise ValueError('SAT solver checksum mismatch.')
        environment['PATH'] = str(dependency.parent) + os.pathsep + environment.get('PATH', '')
    records = []
    for case in configuration['cases']:
        for method in configuration['methods']:
            name = case['name'] + '_' + method['name']
            command = [configuration['executable'], 'semmlMain', '--env', 'r', '--sys', 'g',
                       '--formula', case['formula'], '--outputFormat', 'HOA',
                       '--phEnv', method['ranking'], '--phSys', method['ranking'],
                       '--explorationPerspective', method['perspective'],
                       '--bthEnv', 'STRIX_SCORE', '--bthSys', 'STRIX_SCORE',
                       '--mealyType', 'DET', '--mealyMinimization', 'BISIMULATION',
                       '--controllerPortfolio', 'NONE', '--trace', 'false']
            invocation_start = time.perf_counter()
            with (directory / (name + '.stdout.txt')).open('w') as output, (directory / (name + '.stderr.txt')).open('w') as error:
                result = subprocess.run(['timeout', str(configuration['invocation_seconds'])] + command,
                                        stdout=output, stderr=error, env=environment)
            record = {'case': case['name'], 'method': method['name'], 'command': command,
                      'seconds': time.perf_counter() - invocation_start, 'exit_code': result.returncode}
            if result.returncode == 0:
                text = (directory / (name + '.stdout.txt')).read_text()
                verification_start = time.perf_counter()
                try:
                    record['verification'] = verify_controller(text, case['name'])
                    if record['verification']['system_wins'] != case['system_wins']:
                        raise ValueError('Independent expected winner mismatch.')
                except ValueError as error:
                    record['verification_error'] = str(error)
                record['verification_seconds'] = time.perf_counter() - verification_start
            records.append(record)
            (directory / 'invocations.json').write_text(json.dumps(records, indent=2) + '\n')
    negative_controls = [
        ('anticipating_environment', control_text(False, '(0 & !1) | (!0 & 1)'), 'current_response'),
        ('incorrect_system', control_text(True, '!1'), 'current_response'),
        ('incomplete_system', control_text(True, '!0 & !1'), 'current_response'),
        ('satisfying_counterstrategy_cycle', control_text(False, '!0'), 'future_input_prediction'),
    ]
    controls = []
    for name, controller, case in negative_controls:
        try:
            verify_controller(controller, case)
        except ValueError as error:
            controls.append({'name': name, 'rejected': True, 'reason': str(error)})
        else:
            controls.append({'name': name, 'rejected': False})
    positive_controls = [verify_controller(control_text(True, '(0 & 1) | (!0 & !1)'), 'current_response'),
                         verify_controller(control_text(False, '!0'), 'unforceable_input')]
    result = {'invocations': records, 'negative_controls': controls, 'positive_controls': positive_controls,
              'all_valid': all(row['exit_code'] == 0 and 'verification' in row and 'verification_error' not in row for row in records)
                           and all(row['rejected'] for row in controls),
              'seconds': time.perf_counter() - started, 'training': False,
              'stage': 'implementation validation; not computational qualification'}
    (directory / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
