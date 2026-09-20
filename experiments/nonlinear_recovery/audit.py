import hashlib
import json
from pathlib import Path
import sys
from qualify import verify

out=Path(sys.argv[1]);src=Path(__file__).resolve().parent
hashes=json.loads((out/'sha256.json').read_text())
for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
summary,certs=verify(json.loads((out/'parents.json').read_text()),json.loads((out/'jobs.json').read_text()),
    json.loads((out/'worker.stdout').read_text())['answers'],json.loads((src/'config.json').read_text()))
assert summary==json.loads((out/'summary.json').read_text())
assert certs==json.loads((out/'certificates.json').read_text())
print(json.dumps(dict(audit='PASS',hashes=len(hashes),models=len(certs))))
