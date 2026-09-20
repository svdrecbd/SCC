import ast
import copy
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from qualify import audit, reference


def make_engine(template,mode):
    expression={'robust':'all(values)','optimistic':'any(values)','first':'values[0]'}[mode]
    body=template[template.index('def plan('):]
    body=body.replace('objective, mode):','objective):').replace('reduce_outcomes(values,mode)','reduce_outcomes(values)')
    body=body.replace("selected=successors[:1] if mode=='first' else successors",'selected=successors[:1]' if mode=='first' else 'selected=successors')
    code='def reduce_outcomes(values):\n    return '+expression+'\n\n\n'+body
    assert 'mode' not in code
    ast.parse(code)
    return code


def main():
    src=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve()
    assert platform.node()=='charon' and (src/'plan-frozen.md').is_file()
    cfg=json.loads((src/'standalone.json').read_text());out.mkdir(exist_ok=False)
    start=time.monotonic()
    (out/'machine.json').write_text(json.dumps(dict(host=platform.node(),python=sys.version,executable=sys.executable,
        platform=platform.platform(),affinity=sorted(os.sched_getaffinity(0))),indent=2)+'\n')
    inputs=src.parent/'input';worlds=json.loads((inputs/'worlds.json').read_text())
    records=[json.loads(line) for line in (inputs/'records.jsonl').read_text().splitlines()]
    tasks=[{k:r[k] for k in ('n','seed','horizon','objective','terminal')} for r in records]
    request=json.dumps(dict(worlds=worlds,tasks=tasks))+'\n';(out/'request.json').write_text(request)
    template=(src/'planner.py').read_text()
    engines={mode:make_engine(template,mode) for mode in ('robust','optimistic','first')}
    # Repair starts from the damaged module, not from the intact function body.
    engines['repaired']=engines['optimistic'].replace('return any(values)','return not any(not value for value in values)')
    first_fixed=engines['first'].replace('return values[0]','return not any(not value for value in values)').replace('selected=successors[:1]','selected=successors')
    assert first_fixed==engines['repaired']
    assert 'return all(values)' not in engines['repaired']
    registry={(w['n'],w['seed']):w for w in worlds}
    results={};totals={}
    for mode,engine in engines.items():
        folder=out/mode;folder.mkdir()
        (folder/'engine.py').write_text(engine);(folder/'runner.py').write_bytes((src/'runner.py').read_bytes())
        before=time.monotonic()
        process=subprocess.run([sys.executable,'-S',str(folder/'runner.py')],cwd=folder,input=request,text=True,
                               capture_output=True,check=True,timeout=cfg['per_process_timeout_seconds'])
        elapsed=time.monotonic()-before
        (folder/'stdout.json').write_text(process.stdout);(folder/'stderr.txt').write_text(process.stderr)
        result=json.loads(process.stdout);assert len(result['results'])==len(records)
        sums=dict(outcome_reads=0,reducer_calls=0,state_writes=0,policy_writes=0)
        for task,row,answer in zip(tasks,records,result['results']):
            world=registry[(task['n'],task['seed'])]
            exact=reference(world['edges'],task['terminal'],task['horizon'],task['objective'])
            audit(world['edges'],task['terminal'],task['horizon'],task['objective'],mode,answer,exact)
            assert answer==row['results'][mode]
            for key in sums:sums[key]+=answer['meter'][key]
        results[mode]=result['results']
        totals[mode]=dict(process_seconds=elapsed,cpu_seconds=result['cpu_seconds'],peak_rss_kib=result['peak_rss_kib'],
                          engine_bytes=len(engine.encode()),runner_bytes=(src/'runner.py').stat().st_size,**sums)
    assert results['repaired']==results['robust']
    repaired=totals['repaired']
    full_source=repaired['engine_bytes']+repaired['runner_bytes']
    assert full_source<=cfg['max_replacement_source_bytes']
    for key in ('outcome_reads','reducer_calls','state_writes','policy_writes'):
        assert repaired[key]<=cfg['max_stream_'+key]
        assert repaired[key]==totals['robust'][key]
    # In-range policy corruption must be caught by adversarial execution, not range checks.
    edges=[[[1,1],[2,2]],[[1,1],[1,1]],[[2,2],[2,2]]]
    namespace={};exec(engines['robust'],namespace)
    answer=namespace['plan'](edges,[1],2,'reach');answer['policies'][-1][0]=1
    try:audit(edges,[1],2,'reach','robust',answer,reference(edges,[1],2,'reach'))
    except AssertionError:corruption='REJECTED'
    else:raise AssertionError('accepted in-range bad policy')
    (out/'totals.json').write_text(json.dumps(totals,indent=2)+'\n')
    bytes_total=sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and '__pycache__' not in str(p))
    assert bytes_total<cfg['output_limit_bytes']
    receipt=dict(validation='PASS',standalone_programs=len(engines),task_instances=len(records),
                 source_bytes_written_for_full_replacement=full_source,input_json_bytes=len(request.encode()),
                 instance_specific_advice_bytes=0,first_outcome_repair_matches=True,
                 in_range_policy_corruption=corruption,elapsed_seconds=time.monotonic()-start,output_bytes=bytes_total,
                 parent_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir() if p.is_file()})
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    paths=[p for folder in (src,inputs,out) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    (out/'sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
