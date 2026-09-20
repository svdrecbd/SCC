"""Recheck saved evidence; no rerun of the worker or Bend compiler."""
import hashlib
import json
from pathlib import Path
import sys
from qualify import verify, verify_bend

out=Path(sys.argv[1]); source=Path(__file__).resolve().parent
hashes=json.loads((out/'sha256.json').read_text())
for name,digest in hashes.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
models=json.loads((out/'models.json').read_text())
worker=json.loads((out/'worker.stdout').read_text())
summary,forecasts=verify(models,worker,json.loads((source/'config.json').read_text()))
assert summary==json.loads((out/'summary.json').read_text())
assert forecasts==json.loads((out/'forecasts.json').read_text())
count=verify_bend(json.loads((out/'bend-records.json').read_text()))
print(json.dumps(dict(audit='PASS',hashes=len(hashes),models=len(models),bend_cases=count,
                      forecast_values_checked=summary['forecast_values_checked'])))
