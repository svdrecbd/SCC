"""Bounded frozen-source Charon implementation calibration; preserve failures."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from audit import corruptions, summary_corruptions, verify

HERE = Path(__file__).resolve().parent


def command(args, out, name, code=0):
    started = time.monotonic()
    with (out / f'{name}.stdout').open('w') as stdout, (out / f'{name}.stderr').open('w') as stderr:
        try:
            result = subprocess.run(list(map(str, args)), stdout=stdout, stderr=stderr,
                timeout=120, env={**os.environ, 'BEND_NO_TELEMETRY': '1'}).returncode
        except subprocess.TimeoutExpired:
            result = 124
    (out / f'{name}.receipt.json').write_text(json.dumps(dict(args=list(map(str, args)),
        returncode=result, seconds=time.monotonic() - started, timeout_seconds=120), indent=2) + '\n')
    assert result == code, (name, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', type=Path, required=True)
    parser.add_argument('--bend-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((HERE / 'config.json').read_text())
    node, bend = args.node.resolve(), args.bend_root.resolve()
    try:
        assert platform.node() == 'charon'
        assert (HERE / 'plan-frozen.md').is_file(), 'freeze the relevant labnotes entry before running'
        (out / 'machine.json').write_text(json.dumps(dict(host=platform.node(),
            platform=platform.platform(), python=platform.python_version(),
            time=datetime.now(timezone.utc).isoformat(), backend=cfg['backend'],
            bend_commit=cfg['bend_commit'], node=str(node)), indent=2) + '\n')
        command([node, '--version'], out, 'node-version')
        assert (out / 'node-version.stdout').read_text().strip() == cfg['node_version']
        command([node, HERE / 'build.mjs', bend, HERE / 'learner.bend', out / 'learner.mjs'], out, 'compile')
        command([node, HERE / 'build.mjs', bend, HERE / 'false_claim.bend'], out, 'false-proof', code=1)
        negative = json.loads((out / 'false-proof.stdout').read_text())
        assert negative['stage'] == 'typecheck' and 'false_no_update' in negative['error']
        command([node, HERE / 'evaluate.mjs', out / 'learner.mjs', HERE / 'config.json', out / 'records.jsonl'], out, 'evaluate')
        with (out / 'records.jsonl').open() as f:
            result = verify(f, cfg)
        for name, data in [('summary', result),
                           ('corruptions', corruptions(out / 'records.jsonl', cfg) + summary_corruptions(result))]:
            (out / f'{name}.json').write_text(json.dumps(data, indent=2) + '\n')
        paths = [p for p in HERE.iterdir() if p.is_file()] + [node, bend / 'bend2/bend.ts',
                 bend / 'bend2/comp.ts', bend / 'bend2/base.bend']
        paths += [p for p in out.iterdir() if p.is_file()]
        (out / 'sha256.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}, indent=2) + '\n')
        print(json.dumps(dict(validation='PASS', rows=result['rows'], conditions=result['conditions'])))
    except Exception as error:
        (out / 'failure.json').write_text(json.dumps(dict(error=repr(error)), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
