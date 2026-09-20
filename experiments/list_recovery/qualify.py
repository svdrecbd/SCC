"""Frozen bounded Charon run, retaining all failures and raw transcripts."""
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from audit import verify,corruptions

src=Path(__file__).resolve().parent
out=Path(sys.argv[1]).resolve()
assert platform.node()=='charon' and (src/'plan-frozen.md').is_file()
cfg=json.loads((src/'config.json').read_text())
out.mkdir(parents=True,exist_ok=False)
start=time.monotonic()
(out/'machine.json').write_text(json.dumps(dict(host=platform.node(),platform=platform.platform(),
    python=sys.version,numpy=np.__version__,executable=sys.executable,cpu_count=os.cpu_count(),
    started=datetime.now(timezone.utc).isoformat(),thread_limits={key:os.getenv(key) for key in
    ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}),indent=2)+'\n')
try:
    with (out/'evaluate.stdout').open('w') as stdout,(out/'evaluate.stderr').open('w') as stderr:
        proc=subprocess.run([sys.executable,src/'evaluate.py',src/'config.json',out],stdout=stdout,
                            stderr=stderr,timeout=cfg['timeout_seconds'])
    assert proc.returncode==0
    records=[json.loads(line) for line in (out/'records.jsonl').read_text().splitlines()]
    controls=json.loads((out/'controls.json').read_text())
    summary=verify(records,controls,cfg,out)
    negative=corruptions(records[-1],out)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'corruptions.json').write_text(json.dumps(negative,indent=2)+'\n')
    size=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    assert size<cfg['output_limit_bytes']
    receipt=dict(validation='PASS',cases=len(records),corruptions=len(negative),
        logical_answers_audited=summary['logical_answers_audited'],
        direct_coefficients_checked=summary['direct_coefficients_checked'],
        main_recovered=summary['main_recovered'],main_cases=summary['main_cases'],
        decoder_source_bytes=(src/'decoder.py').stat().st_size,
        elapsed_seconds=time.monotonic()-start,output_bytes=size,
        auditor_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        evaluator_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    paths=[p for folder in (src,out) for p in folder.iterdir() if p.is_file()]
    (out/'sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in paths},indent=2)+'\n')
    print(json.dumps(receipt))
except Exception as error:
    (out/'failure.json').write_text(json.dumps(dict(error=repr(error)))+'\n')
    raise
