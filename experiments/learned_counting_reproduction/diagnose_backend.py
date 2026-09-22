"""Isolate the contradictory-unit control in a fresh process before repairing input handling."""
import json
import os
from pathlib import Path
import sys
import torch
from solver import CountingSystem


def main(directory):
    torch.set_num_threads(1)
    path=directory/'contradiction.cnf';path.write_text('p cnf 2 2\n1 0\n-1 0\n')
    os.chdir(directory)
    system=CountingSystem(directory.parent,'native')
    result=system.solve(path,3)
    (directory/'diagnosis.json').write_text(json.dumps({'expected':0,'observed':result},indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
