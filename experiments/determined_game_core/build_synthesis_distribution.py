"""Build a pinned source copy with one processor and private dependency storage."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time


def main():
    directory = Path(sys.argv[1]).resolve()
    configuration = json.loads((directory / 'config.json').read_text())
    work = directory / 'working_source'
    shutil.copytree(configuration['source_directory'], work)
    source_manifest = [{'path': str(path.relative_to(work)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                       for path in sorted(work.rglob('*')) if path.is_file()]
    (directory / 'source_manifest.json').write_text(json.dumps(source_manifest, indent=2) + '\n')
    environment = dict(os.environ, JAVA_HOME=configuration['java_home'],
                       GRADLE_USER_HOME=str(directory.parent / 'dependency_cache'),
                       JAVA_OPTS='-Xms128m -Xmx4g -XX:ActiveProcessorCount=1')
    command = ['./gradlew', configuration['task'], '--no-daemon', '--max-workers=1',
               '-Dorg.gradle.jvmargs=-Xms128m -Xmx4g -XX:ActiveProcessorCount=1']
    for task in configuration.get('completed_tasks', []):
        command.extend(['-x', task])
    (directory / 'invocation.json').write_text(json.dumps({'command': command, 'source_revision': configuration['source_revision'],
        'java_home': environment['JAVA_HOME'], 'cache': environment['GRADLE_USER_HOME']}, indent=2) + '\n')
    started = time.perf_counter()
    result = subprocess.run(command, cwd=work, env=environment)
    receipt = {'exit_code': result.returncode, 'seconds': time.perf_counter() - started, 'training': False}
    if result.returncode == 0:
        receipt['distribution'] = [{'path': str(path.relative_to(work)), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted((work / 'build/install').rglob('*')) if path.is_file()]
    (directory / 'build_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    result.check_returncode()


if __name__ == '__main__':
    main()
