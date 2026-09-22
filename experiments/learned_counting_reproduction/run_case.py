"""Measure a fresh process, including initialization and the full counted computation."""
import time
PROCESS_STARTED=time.monotonic()
import json
import os
from pathlib import Path
import sys
import resource
import torch
from solver import CountingSystem


def main(case_directory, mode, path, artifact_root):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    os.chdir(case_directory)
    system=CountingSystem(artifact_root,mode)
    prepared=time.monotonic()
    result=system.solve(path,seconds=10)
    result.update(preparation_seconds=prepared-PROCESS_STARTED,
                  internal_process_seconds=time.monotonic()-PROCESS_STARTED,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (case_directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve(),sys.argv[2],Path(sys.argv[3]).resolve(),Path(sys.argv[4]).resolve())
