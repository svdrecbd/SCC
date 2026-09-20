"""Run frozen exact screen and independent audit on Charon only."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from audit import verify, controls
import audit_frontier

src=Path(__file__).resolve().parent
out=Path(sys.argv[1]).resolve()
assert platform.node()=='charon'
assert (src/'plan-frozen.md').is_file()
cfg=json.loads((src/'config.json').read_text())
out.mkdir(parents=True,exist_ok=False)
start=time.monotonic()
(out/'machine.json').write_text(json.dumps(dict(host=platform.node(),platform=platform.platform(),
    python=sys.version,executable=sys.executable,cpu_count=os.cpu_count(),
    started=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
try:
    with (out/'solve.stdout').open('w') as stdout,(out/'solve.stderr').open('w') as stderr:
        proc=subprocess.run([sys.executable,src/'solve.py',src/'config.json',out/'certificates.jsonl'],
                            stdout=stdout,stderr=stderr,timeout=cfg['timeout_seconds'])
    assert proc.returncode==0
    rows=[json.loads(line) for line in (out/'certificates.jsonl').read_text().splitlines()]
    results=verify(rows,cfg)
    negative=controls(rows,cfg)
    assert results==json.loads((src/'previous-summary.json').read_text())
    with (out/'frontier.stdout').open('w') as stdout,(out/'frontier.stderr').open('w') as stderr:
        proc=subprocess.run([sys.executable,src/'frontier.py',src/'config.json',out/'frontier.jsonl'],
                            stdout=stdout,stderr=stderr,timeout=cfg['timeout_seconds'])
    assert proc.returncode==0
    frontier_rows=[json.loads(line) for line in (out/'frontier.jsonl').read_text().splitlines()]
    endpoints={r['case']:r for r in results}
    frontier_results=audit_frontier.verify(frontier_rows,cfg,endpoints)
    negative+=audit_frontier.controls(frontier_rows,cfg,endpoints)
    (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    (out/'frontier_summary.json').write_text(json.dumps(frontier_results,indent=2)+'\n')
    assert json.loads((out/'frontier_summary.json').read_text())==frontier_results
    (out/'corruption_controls.json').write_text(json.dumps(negative,indent=2)+'\n')
    output_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    assert output_bytes<cfg['output_limit_bytes']
    (out/'receipt.json').write_text(json.dumps(dict(validation='PASS',cases=len(rows),
        dual_edges=sum(r['all_edge_dual_checks'] for r in results),corruptions=len(negative),
        frontier_points=len(frontier_rows),
        elapsed_seconds=time.monotonic()-start,output_bytes=output_bytes),indent=2)+'\n')
    paths=[p for folder in (src,out) for p in folder.iterdir() if p.is_file()]
    (out/'sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths},indent=2)+'\n')
    print((out/'receipt.json').read_text())
except Exception as e:
    (out/'failure.json').write_text(json.dumps(dict(error=repr(e)))+'\n')
    raise
