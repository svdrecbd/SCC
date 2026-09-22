"""Compile pinned original search sources with the small local callback adapter."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main(directory):
    upstream=directory.parent/'acquisition01/upstream/sat-rl/pysat-master/solvers/sharpSAT'
    dependencies=directory.parent/'adapter01/dependencies/usr'
    sources=sorted(path for path in upstream.rglob('*.cpp') if path.name!='main.cpp')
    command=['g++','-O3','-DNDEBUG','-std=c++11','-fPIC','-shared','-include','tuple',
             '-include','cstring','-include','cstdint','-I'+str(upstream),'-I'+str(dependencies/'include'),
             '-I'+str(dependencies/'include/x86_64-linux-gnu'),str(directory/'solver_adapter.cpp'),
             *map(str,sources),str(dependencies/'lib/x86_64-linux-gnu/libgmpxx.so.4.7.0'),
             '/usr/lib/x86_64-linux-gnu/libgmp.so.10','-Wl,-rpath,'+str(dependencies/'lib/x86_64-linux-gnu'),'-o',str(directory/'solver.so')]
    (directory/'build_command.json').write_text(json.dumps(command,indent=2)+'\n')
    with (directory/'compilation.stdout').open('w') as output,(directory/'compilation.stderr').open('w') as error:
        result=subprocess.run(command,stdout=output,stderr=error,timeout=60)
    receipt={'exit_code':result.returncode,'upstream_sha256':{str(path.relative_to(upstream)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    if result.returncode==0: receipt['library_sha256']=hashlib.sha256((directory/'solver.so').read_bytes()).hexdigest()
    (directory/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'exit_code':result.returncode,'sources':len(sources)}),flush=True)
    assert result.returncode==0


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
