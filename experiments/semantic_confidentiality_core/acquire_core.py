"""Acquire a pinned semantic inference checkpoint without executing remote code."""
from pathlib import Path
import hashlib
import json
import sys
import time
import urllib.request

root=Path(sys.argv[1]);config=json.loads((root/'config.json').read_text())
started=time.perf_counter();records=[];total=0
model=root/'model';model.mkdir()
for name in config['files']:
    url=f"https://huggingface.co/{config['model_id']}/resolve/{config['revision']}/{name}"
    digest=hashlib.sha256();size=0
    with urllib.request.urlopen(url,timeout=30) as response,(model/name).open('wb') as destination:
        while True:
            data=response.read(1048576)
            if not data:break
            total+=len(data);size+=len(data)
            assert total<=config['maximum_bytes']
            destination.write(data);digest.update(data)
    records.append({'path':'model/'+name,'url':url,'bytes':size,'sha256':digest.hexdigest()})
    (root/'receipt.json').write_text(json.dumps({'status':'acquiring','files':records,'total_bytes':total},indent=2)+'\n')
receipt={'status':'complete','files':records,'total_bytes':total,'seconds':time.perf_counter()-started}
(root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
