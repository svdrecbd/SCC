"""Build and assess a public policy constructor on frozen temporal formulas."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time


def main():
    directory = Path(sys.argv[1])
    configuration = json.loads((directory / 'config.json').read_text())
    environment = dict(os.environ, JAVA_HOME=configuration['java_home'])
    java = Path(configuration['java_home']) / 'bin' / 'java'
    compiler = java.with_name('javac')
    class_path = configuration['library_directory'] + '/*'
    started = time.perf_counter()
    compilation = subprocess.run([str(compiler), '-cp', class_path, str(directory / 'PropositionalPolicySearch.java')],
        capture_output=True, text=True, timeout=15, env=environment)
    (directory / 'compilation.stdout.txt').write_text(compilation.stdout)
    (directory / 'compilation.stderr.txt').write_text(compilation.stderr)
    compilation.check_returncode()
    compile_seconds = time.perf_counter() - started
    cases = list(configuration.get('fixtures', []))
    if configuration.get('source_ssi'):
        path = Path(configuration['source_ssi'])
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == configuration['source_ssi_sha256']
        for line in data.decode().splitlines():
            parts = line.split(maxsplit=3)
            if parts and parts[0] in configuration['input_names']:
                name, inputs, outputs, formula = parts
                cases.append(dict(name=name, inputs=inputs, outputs=outputs, formula=formula))
        assert {case['name'] for case in cases} >= set(configuration['input_names'])
    (directory / 'inputs.json').write_text(json.dumps(cases, indent=2) + '\n')
    results = []
    for index, case in enumerate(cases):
        formula_path = directory / f'formula_{index:03d}.txt'
        formula_path.write_text(case['formula'] + '\n')
        command = [str(java), '-Xmx2g', '-XX:ActiveProcessorCount=1', '-cp', str(directory) + os.pathsep + class_path,
            'PropositionalPolicySearch', str(formula_path), case['inputs'], case['outputs'], str(configuration['search_seconds'])]
        started = time.perf_counter()
        process = subprocess.run(['timeout', str(configuration['invocation_seconds'])] + command,
            capture_output=True, text=True, env=environment)
        (directory / f'policy_{index:03d}.stdout.txt').write_text(process.stdout)
        (directory / f'policy_{index:03d}.stderr.txt').write_text(process.stderr)
        record = dict(name=case['name'], seconds=time.perf_counter()-started, exit_code=process.returncode, command=command)
        for line in process.stdout.splitlines():
            key, *values = line.split('\t')
            if key == 'POLICY': record.setdefault('policy', {})[values[0]] = values[1]
            elif values: record[key.lower()] = values[0]
        if 'expected_result' in case:
            record['expected_result'] = case['expected_result']
            record['control_passed'] = record.get('result') == case['expected_result']
        results.append(record)
        (directory / 'assessment.json').write_text(json.dumps(dict(compilation_seconds=compile_seconds, cases=results,
            training=False, qualifier='sound incomplete public construction; not a lower bound'), indent=2)+'\n')
    print(json.dumps(dict(compilation_seconds=compile_seconds, cases=[{key:value for key,value in result.items() if key!='command'} for result in results]), indent=2))
    assert all(result.get('control_passed', True) for result in results)


if __name__ == '__main__':
    main()
