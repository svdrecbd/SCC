"""Compare public decisions with learned controller construction on a fixed input."""
from pathlib import Path
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time


def main():
    directory = Path(sys.argv[1])
    configuration = json.loads((directory / 'config.json').read_text())
    archive_path = Path(configuration['source_archive'])
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != configuration['source_archive_sha256']:
        raise ValueError('Input archive checksum mismatch.')
    with tarfile.open(archive_path, 'r:gz') as archive:
        stream = archive.extractfile('syntcomp_at_home/benchmarks/ssi/syntcomp25.ssi')
        matches = [line.strip() for line in io.TextIOWrapper(stream) if line.split(maxsplit=1)[0] == configuration['input_name']]
    if len(matches) != 1:
        raise ValueError('Input identifier is missing or ambiguous.')
    name, inputs, outputs, formula = matches[0].split(maxsplit=3)
    formula_path = directory / 'formula.txt'
    formula_path.write_text(formula + '\n')
    (directory / 'input.json').write_text(json.dumps({'name': name, 'inputs': inputs, 'outputs': outputs,
        'formula_bytes': formula_path.stat().st_size, 'formula_sha256': hashlib.sha256(formula_path.read_bytes()).hexdigest()}, indent=2) + '\n')
    dependency = Path(configuration['dependency_directory']) / 'kissat'
    if hashlib.sha256(dependency.read_bytes()).hexdigest() != configuration['dependency_sha256']:
        raise ValueError('SAT solver checksum mismatch.')
    environment = dict(os.environ, JAVA_HOME=configuration['java_home'],
        JAVA_OPTS='-Xms128m -Xmx4g -Xss16m -XX:ActiveProcessorCount=1',
        PATH=str(dependency.parent) + os.pathsep + os.environ.get('PATH', ''))
    records = []
    for method in configuration['methods']:
        command = [configuration['executable'], 'semmlMain', '--env', inputs, '--sys', outputs,
            '--formulaFile', str(formula_path), '--realizable', str(method['decision_only']).lower(),
            '--outputFormat', 'HOA', '--phEnv', method['ranking'], '--phSys', method['ranking'],
            '--explorationPerspective', method['perspective'], '--bthEnv', 'STRIX_SCORE', '--bthSys', 'STRIX_SCORE',
            '--mealyType', 'DET', '--mealyMinimization', 'BISIMULATION', '--controllerPortfolio', 'NONE', '--trace', 'false']
        started = time.perf_counter()
        output_path = directory / (method['name'] + '.stdout.txt')
        with output_path.open('w') as output, (directory / (method['name'] + '.stderr.txt')).open('w') as error:
            result = subprocess.run(['timeout', str(configuration['invocation_seconds'])] + command,
                stdout=output, stderr=error, env=environment)
        text = output_path.read_text()
        record = {'method': method['name'], 'command': command, 'seconds': time.perf_counter() - started,
                  'exit_code': result.returncode, 'reported_winner': next((line for line in text.splitlines() if line in ('REALIZABLE', 'UNREALIZABLE')), None),
                  'reported_controller_states': next((int(match[1]) for match in re.finditer(r'^States: (\d+)$', text, re.M)), None),
                  'independently_verified': False}
        records.append(record)
        (directory / 'assessment.json').write_text(json.dumps({'input': name, 'invocations': records, 'fresh_qualification': False, 'training': False}, indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
