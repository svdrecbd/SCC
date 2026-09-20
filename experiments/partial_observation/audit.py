import json
from pathlib import Path
import sys
from qualify import audit

print(json.dumps(audit(Path(sys.argv[1]).resolve())))
