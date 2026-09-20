"""Frozen-source Charon validation; logs and failures are preserved."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from audit import verify

source = Path(__file__).resolve().parent
out = Path(sys.argv[1]).resolve()
assert platform.node() == 'charon'
assert (source / 'plan-frozen.md').is_file()
cfg = json.loads((source / 'config.json').read_text())
out.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
(out / 'machine.json').write_text(json.dumps(dict(host=platform.node(),
    platform=platform.platform(), python=sys.version, executable=sys.executable,
    started=datetime.now(timezone.utc).isoformat()), indent=2)+'\n')
try:
    with (out / 'evaluate.stdout').open('w') as stdout, (out / 'evaluate.stderr').open('w') as stderr:
        result = subprocess.run([sys.executable, source / 'adapter.py', source / 'config.json',
            out / 'records.jsonl'], stdout=stdout, stderr=stderr,
            timeout=cfg['stage_timeout_seconds'])
    assert result.returncode == 0
    summary = verify(out / 'records.jsonl', cfg)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (out / 'receipt.json').write_text(json.dumps(dict(validation='PASS',
        elapsed_seconds=time.monotonic()-started), indent=2)+'\n')
    paths = [p for d in (source, out) for p in d.iterdir() if p.is_file()]
    (out / 'sha256.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths}, indent=2)+'\n')
    print(json.dumps(dict(validation='PASS', cases=len(summary['cases']),
                         total_recovery_calls=summary['total_recovery_calls'])))
except Exception as error:
    (out / 'failure.json').write_text(json.dumps(dict(error=repr(error)))+'\n')
    raise
