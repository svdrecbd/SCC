"""Validate screen controls and summarize explicitly certified public decisions."""
from pathlib import Path
from collections import Counter
import json
import sys
root=Path(sys.argv[1])
expected={row['name']:row['expected'] for row in json.loads((root/'controls.json').read_text())}
controls=[json.loads(line) for line in (root/'controls.jsonl').read_text().splitlines()]
assert len(controls)==len(expected)
for row in controls:
    assert row['result']==expected[row['name']],row
results=[json.loads(line) for line in (root/'results.jsonl').read_text().splitlines()]
inputs=json.loads((root/'specifications.json').read_text())
assert len(results)==len(inputs)
certified=[row for row in results if row['result']=='UNREALIZABLE']
summary=dict(controls_passed=len(controls),specifications=len(results),certified=len(certified),
    unknown_reasons=dict(Counter(row.get('reason','no_supported_certificate') for row in results if row['result']=='UNKNOWN')),
    certified_names=[row['name'] for row in certified],
    search_seconds=sum(row.get('seconds',0) for row in results),
    maximum_certification_seconds=max((row['seconds'] for row in certified),default=0))
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
