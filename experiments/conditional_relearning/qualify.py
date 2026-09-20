"""Bounded Charon runner. Run only from a fresh frozen source directory."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

from audit import controls, verify

source = Path(__file__).resolve().parent
out = Path(sys.argv[1]).resolve()
assert platform.node() == 'charon'
assert (source / 'plan-frozen.md').is_file()
cfg = json.loads((source / 'config.json').read_text())
assert cfg['n'] == 4 and cfg['horizon'] == 8
assert (cfg['noise_numerator'], cfg['noise_denominator']) == (1, 4)
out.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
(out / 'machine.json').write_text(json.dumps(dict(host=platform.node(),
    platform=platform.platform(), python=sys.version, executable=sys.executable,
    cpu_count=os.cpu_count(), started=datetime.now(timezone.utc).isoformat()), indent=2)+'\n')
try:
    with (out / 'evaluate.stdout').open('w') as stdout, (out / 'evaluate.stderr').open('w') as stderr:
        proc = subprocess.run([sys.executable, source / 'evaluate.py', source / 'config.json',
            out / 'records.jsonl'], stdout=stdout, stderr=stderr, timeout=cfg['timeout_seconds'])
    assert proc.returncode == 0
    summary = verify(out / 'records.jsonl', cfg)
    negative = controls(out / 'records.jsonl', cfg, summary)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (out / 'corruptions.json').write_text(json.dumps(negative, indent=2)+'\n')
    (out / 'receipt.json').write_text(json.dumps(dict(validation='PASS',
        elapsed_seconds=time.monotonic()-started, rows=summary['rows']), indent=2)+'\n')
    paths = [p for folder in (source, out) for p in folder.iterdir() if p.is_file()]
    (out / 'sha256.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths}, indent=2)+'\n')
    print(json.dumps(dict(validation='PASS', rows=summary['rows'],
        cheap_repair_rows=summary['cheap_repair_rows'], controls=len(negative))))
except Exception as error:
    (out / 'failure.json').write_text(json.dumps(dict(error=repr(error)))+'\n')
    raise
