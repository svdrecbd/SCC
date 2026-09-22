"""Run the pinned reference solver with hard per-process limits and known controls."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def main(directory):
    configuration = json.loads((directory/'reference_config.json').read_text())
    source = directory.parent/'reference01/distribution/ganak'
    formulas = directory/'controls'; formulas.mkdir()
    controls = []
    for name, variables, clauses, expected in [
            ('disjunction',2,[(1,2)],3),
            ('contradiction',2,[(1,),(-1,)],0),
            ('free_variables',160,[],1<<160),
            ('independent_components',80,[(2*i+1,2*i+2) for i in range(40)],3**40)]:
        path = formulas/(name+'.cnf')
        path.write_text(f'p cnf {variables} {len(clauses)}\n'+''.join(' '.join(map(str,clause))+' 0\n' for clause in clauses))
        controls.append((path,expected))
    upstream = directory.parent/'development01/upstream/data'
    controls.extend((path,490) for path in sorted((upstream/'cell_9_20_20_test').glob('*.cnf')))
    candidate_directory = (directory/configuration['candidate_relative_directory']).resolve() if 'candidate_relative_directory' in configuration else upstream
    candidates = [(path,None) for path in sorted(candidate_directory.rglob('*.cnf')) if 'cell_9_' not in str(path)]
    if not configuration.get('include_controls', True):
        controls = []
    rows = []
    for index,(path,expected) in enumerate(controls+candidates):
        command = [str(source),*configuration['arguments'],str(path)]
        started = time.monotonic()
        output_path = directory/f'process_{index:02d}.stdout'
        error_path = directory/f'process_{index:02d}.stderr'
        with output_path.open('w') as output, error_path.open('w') as error:
            try:
                result = subprocess.run(command, stdout=output, stderr=error,
                                        env={**os.environ,'OMP_NUM_THREADS':'1'},
                                        timeout=configuration['seconds_per_process'])
                exit_code = result.returncode
                status = 'finished'
            except subprocess.TimeoutExpired:
                exit_code = None; status = 'resource_limit'
        stdout = output_path.read_text()
        values = re.findall(r'^(?:c )?s (?:mc|exact arb int) ([0-9]+)\s*$',stdout,re.MULTILINE)
        count = int(values[-1]) if values and status=='finished' else None
        row = {'path':str(path), 'command':command,'status':status,'exit_code':exit_code,
               'count':count,'expected':expected,'seconds':time.monotonic()-started}
        rows.append(row)
        (directory/'results.json').write_text(json.dumps({'rows':rows},indent=2)+'\n')
        print(json.dumps(row),flush=True)
        if expected is not None:
            assert exit_code == 0 and count == expected, row
        elif status=='finished':
            assert count is not None, row


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
