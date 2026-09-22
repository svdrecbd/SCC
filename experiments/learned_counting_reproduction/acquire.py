"""Preserve pinned inference source and checkpoint bytes without deserialization."""
import hashlib
import json
from pathlib import Path
import pickletools
import sys
import urllib.request


def main(directory):
    configuration=json.loads((directory/'acquisition_config.json').read_text())
    records=[]; total=0
    for item in configuration['files']:
        with urllib.request.urlopen(configuration['base_url']+'/'+item['path'],timeout=20) as response:
            payload=response.read(configuration['maximum_bytes']-total+1)
        total+=len(payload)
        assert total<=configuration['maximum_bytes'] and len(payload)==item['bytes']
        digest=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
        assert digest==item['git_blob_sha1'],item['path']
        path=directory/'upstream'/item['path']; path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as output: output.write(payload)
        record={**item,'sha256':hashlib.sha256(payload).hexdigest()}
        if item['kind']=='checkpoint':
            operations=list(pickletools.genops(payload))
            record['pickle_globals']=[str(argument) for operation,argument,position in operations if operation.name=='GLOBAL']
            record['string_fields']=[str(argument) for operation,argument,position in operations if operation.name in ('SHORT_BINUNICODE','BINUNICODE') and len(str(argument))<200][:80]
            record['opcode_count']=len(operations)
        records.append(record)
    receipt={'files':records,'bytes':total,'upstream_executed':False,'checkpoint_deserialized':False,'training':False}
    (directory/'acquisition.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'files':len(records),'bytes':total,'checkpoints':[record for record in records if record['kind']=='checkpoint']}),flush=True)


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
