"""Bounded frozen-source CPU qualification; artifacts survive failures."""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from audit import verify, corruptions

src = Path(__file__).resolve().parent
out = Path(sys.argv[1]).resolve()
assert platform.node() == 'charon' and (src/'plan-frozen.md').is_file()
cfg = json.loads((src/'config.json').read_text())
out.mkdir(parents=True, exist_ok=False)
start = time.monotonic()
(out/'machine.json').write_text(json.dumps(dict(host=platform.node(), platform=platform.platform(),
    python=sys.version, executable=sys.executable, cpu_count=os.cpu_count(),
    started=datetime.now(timezone.utc).isoformat()), indent=2)+'\n')
try:
    with (out/'evaluate.stdout').open('w') as stdout, (out/'evaluate.stderr').open('w') as stderr:
        proc = subprocess.run([sys.executable,src/'evaluate.py',src/'config.json',out/'records.jsonl'],
                              stdout=stdout, stderr=stderr, timeout=cfg['timeout_seconds'])
    assert proc.returncode == 0
    records = [json.loads(line) for line in (out/'records.jsonl').read_text().splitlines()]
    summary = verify(records, cfg)
    negative = corruptions(records, cfg)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (out/'corruptions.json').write_text(json.dumps(negative, indent=2)+'\n')
    size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    assert size < cfg['output_limit_bytes']
    receipt = dict(validation='PASS', cases=len(records)-1,
                   exhaustive_target_mask_checks=summary['exhaustive_target_mask_checks'],
                   corruptions=len(negative), repair_source_bytes=(src/'repair.py').stat().st_size,
                   elapsed_seconds=time.monotonic()-start, output_bytes=size)
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    paths = [p for folder in (src,out) for p in folder.iterdir() if p.is_file()]
    (out/'sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in paths}, indent=2)+'\n')
    print(json.dumps(receipt))
except Exception as error:
    (out/'failure.json').write_text(json.dumps(dict(error=repr(error)))+'\n')
    raise
