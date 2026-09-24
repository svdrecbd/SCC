"""Require general temporal checker agreement with independently specified controls."""
from pathlib import Path
import json
import sys
root=Path(sys.argv[1])
expected={row['name']:row['expected'] for row in json.loads((root/'checker_inputs.json').read_text())}
results=[json.loads(line) for line in (root/'checker_outputs.jsonl').read_text().splitlines()]
assert len(results)==len(expected)
assert {row['name'] for row in results}==set(expected)
for row in results:
    assert row['accepted']==expected[row['name']], row
receipt=dict(controls=len(results),passed=True,checking_seconds=sum(row['seconds'] for row in results))
(root/'comparison.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
