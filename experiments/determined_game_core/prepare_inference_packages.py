"""Download or install a frozen set of inference dependencies in a private environment."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time


def main():
    directory = Path(sys.argv[1])
    configuration = json.loads((directory/'config.json').read_text())
    python = Path(configuration['environment'])/'bin/python'
    wheel_directory = Path(configuration['wheel_directory'])
    wheel_directory.mkdir(parents=True, exist_ok=True)
    requirements = directory/'requirements.txt'
    requirements.write_text('\n'.join(configuration['packages'])+'\n')
    command = [str(python), '-m', 'pip', '--disable-pip-version-check']
    if configuration['operation'] == 'download':
        command += ['download', '--only-binary=:all:', '--dest', str(wheel_directory), '-r', str(requirements)]
        if configuration.get('index_url'):
            command += ['--index-url', configuration['index_url']]
        if configuration.get('no_dependencies'):
            command += ['--no-deps']
    elif configuration['operation'] == 'install':
        command += ['install', '--no-compile', '--no-index', '--find-links', str(wheel_directory), '-r', str(requirements)]
    else:
        raise ValueError('Unknown preparation operation.')
    environment = dict(os.environ, PIP_NO_INPUT='1', PIP_CACHE_DIR=str(directory/'package_cache'),
        CUDA_VISIBLE_DEVICES='', WANDB_MODE='disabled', WANDB_DISABLED='true', PYTHONDONTWRITEBYTECODE='1')
    started = time.perf_counter()
    with (directory/'packages.stdout.txt').open('w') as output, (directory/'packages.stderr.txt').open('w') as error:
        result = subprocess.run(['timeout',str(configuration['invocation_seconds'])]+command,
            stdout=output,stderr=error,env=environment)
    wheels = []
    for path in sorted(wheel_directory.glob('*.whl')):
        wheels.append(dict(name=path.name, bytes=path.stat().st_size, sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()))
    total = sum(row['bytes'] for row in wheels)
    receipt = dict(command=command, exit_code=result.returncode, seconds=time.perf_counter()-started,
        wheels=wheels, wheel_bytes=total, size_allowance_met=total<=configuration['maximum_wheel_bytes'], training=False)
    (directory/'package_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({key:value for key,value in receipt.items() if key!='wheels'},indent=2))
    if result.returncode or not receipt['size_allowance_met']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
