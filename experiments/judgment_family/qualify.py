"""Bounded frozen-source runner; research compute stays on Charon."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime,timezone
from audit import verify,controls

src=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve()
assert platform.node()=='charon' and (src/'plan-frozen.md').is_file()
cfg=json.loads((src/'config.json').read_text());assert cfg['levels']==4
out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
(out/'machine.json').write_text(json.dumps(dict(host=platform.node(),platform=platform.platform(),
    python=sys.version,executable=sys.executable,cpu_count=os.cpu_count(),
    started=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
try:
    with (out/'evaluate.stdout').open('w') as stdout,(out/'evaluate.stderr').open('w') as stderr:
        proc=subprocess.run([sys.executable,src/'evaluate.py',src/'config.json',out/'records.jsonl'],
                            stdout=stdout,stderr=stderr,timeout=cfg['timeout_seconds'])
    assert proc.returncode==0
    rows=[json.loads(line) for line in (out/'records.jsonl').read_text().splitlines()]
    summary=verify(rows,cfg);negative=controls(rows,cfg)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'corruptions.json').write_text(json.dumps(negative,indent=2)+'\n')
    size=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    assert size<cfg['output_limit_bytes']
    receipt=dict(validation='PASS',rows=len(rows),posterior_function_checks=summary['posterior_function_checks'],
                 corruption_controls=len(negative),output_bytes=size,elapsed_seconds=time.monotonic()-start)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    paths=[p for folder in (src,out) for p in folder.iterdir() if p.is_file()]
    (out/'sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths},indent=2)+'\n')
    print(json.dumps(receipt))
except Exception as error:
    (out/'failure.json').write_text(json.dumps(dict(error=repr(error)))+'\n');raise
