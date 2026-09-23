"""Acquire public answerability annotations and reference source without execution."""
from pathlib import Path
import hashlib
import json
import sys
import urllib.request

root=Path(sys.argv[1])
configuration=json.loads((root/'config.json').read_text())
records=[]
total=0
for name,url in configuration['sources'].items():
    with urllib.request.urlopen(url,timeout=20) as response:
        data=response.read(configuration['maximum_bytes']-total+1)
    total+=len(data)
    assert total<=configuration['maximum_bytes']
    (root/name).write_bytes(data)
    records.append({'path':name,'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
(root/'receipt.json').write_text(json.dumps({'status':'complete','files':records,'total_bytes':total},indent=2)+'\n')
print(json.dumps({'status':'complete','total_bytes':total,'files':records}))
