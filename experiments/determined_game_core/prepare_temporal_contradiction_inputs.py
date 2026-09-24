"""Freeze the public specification screen and its independent known-outcome controls."""
from pathlib import Path
import hashlib
import json
import sys
root=Path(sys.argv[1])
configuration=json.loads((root/'config.json').read_text())
source=Path(configuration['specification_directory'])
records=[]
for path in sorted(source.glob('*.json')):
    prefixes=configuration.get('selected_prefixes',[])
    if prefixes and not any(path.stem.startswith(prefix) for prefix in prefixes):
        continue
    item=json.loads(path.read_text())
    if not all(key in item for key in ('assumptions','guarantees','inputs','outputs')):
        continue
    item['name']=path.stem
    item['source_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    records.append(item)
(root/'specifications.json').write_text(json.dumps(records,indent=2)+'\n')
controls=[]
for index,(assumptions,guarantees,expected) in enumerate([
    ([],['G !(o0 & o1)','G (i0 -> X X (o0 & o1))'],'UNREALIZABLE'),
    ([],['G !(o0 & o1)','G (i0 -> F (o0 & o1))'],'UNREALIZABLE'),
    ([],['G !(o0 & o1)','G (i0 -> F o0)'],'UNKNOWN'),
    ([],['F !(o0 & o1)','F (o0 & o1)'],'UNKNOWN'),
    (['G !i0'],['G !(o0 & o1)','G (i0 -> F (o0 & o1))'],'UNKNOWN'),
    ([],['G !(o0 & o1)','G (!i0 -> F (o0 & o1))'],'UNREALIZABLE'),
    ([],['X G !(o0 & o1)','G X (o0 & o1)'],'UNREALIZABLE'),
    ([],['X X G !o0','o0'],'UNKNOWN'),
    ([],['X X G !o0','F o0'],'UNKNOWN'),
    (['G !i0'],['X G !(o0 & o1)','G (!i0 -> X (o0 & o1))'],'UNREALIZABLE'),
]):
    controls.append(dict(name=f'control_{index:02d}',assumptions=assumptions,guarantees=guarantees,
        inputs=['i0'],outputs=['o0','o1'],expected=expected))
(root/'controls.json').write_text(json.dumps(controls,indent=2)+'\n')
print(json.dumps(dict(specifications=len(records),controls=len(controls))))
