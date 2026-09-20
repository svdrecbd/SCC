import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
import pysat
from audit import verify,controls

src=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve()
assert platform.node()=='charon' and (src/'plan-frozen.md').is_file()
cfg=json.loads((src/'config.json').read_text());out.mkdir(parents=True,exist_ok=False)
start=time.monotonic()
(out/'machine.json').write_text(json.dumps(dict(host=platform.node(),platform=platform.platform(),
    python=sys.version,pysat=pysat.__version__,executable=sys.executable,cpu_count=os.cpu_count()),indent=2)+'\n')
try:
    with (out/'evaluate.stdout').open('w') as stdout,(out/'evaluate.stderr').open('w') as stderr:
        process=subprocess.run([sys.executable,src/'evaluate.py',src/'config.json',out],
            stdout=stdout,stderr=stderr,timeout=cfg['timeout_seconds'])
    assert process.returncode==0
    rows=[json.loads(line) for line in (out/'records.jsonl').read_text().splitlines()]
    summary=verify(rows,cfg,out);negative=controls(rows[0])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'corruptions.json').write_text(json.dumps(negative,indent=2)+'\n')
    size=sum(p.stat().st_size for p in out.iterdir() if p.is_file());assert size<cfg['output_limit_bytes']
    receipt=dict(validation='PASS',core_instances=len(rows),natural_variants=summary['natural_variants'],
        goal_cases=summary['goal_cases'],unsafe_sat=summary['unsafe_sat'],unsafe_unsat=summary['unsafe_unsat'],
        drup_proofs=len(summary['proof_checks']),drup_clause_visits=sum(r['clause_visits'] for r in summary['proof_checks']),
        brute_assignments=summary['brute_assignments'],corruptions=len(negative),
        elapsed_seconds=time.monotonic()-start,output_bytes=size,
        planner_source_bytes=(src/'planner.py').stat().st_size,
        evaluator_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        auditor_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    paths=[p for folder in (src,out) for p in folder.iterdir() if p.is_file()]
    (out/'sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    print(json.dumps(receipt))
except Exception as error:
    (out/'failure.json').write_text(json.dumps(dict(error=repr(error)))+'\n');raise
