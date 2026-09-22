"""Check the original search and learned callbacks against independently known counts."""
import json
import os
from pathlib import Path
import random
import sys
import torch
from solver import CountingSystem
from reference_formulas import encode,evolve


def main(directory):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cases=[('disjunction',2,[(1,2)],3),('contradiction',2,[(1,),(-1,)],0),
           ('free_variables',6,[],64),('components',6,[(1,2),(3,4),(5,6)],27)]
    for seed in (296,297,298):
        generator=random.Random(seed);width=7
        clauses=[tuple(generator.choice((-1,1))*variable for variable in generator.sample(range(1,width+1),3)) for _ in range(12)]
        expected=sum(all(any(bool((assignment>>(abs(literal)-1))&1)==(literal>0) for literal in clause) for clause in clauses) for assignment in range(1<<width))
        cases.append((f'random_{seed}',width,clauses,expected))
    for rule in (35,49):
        for target in (0,5,31):
            clauses,variables=encode(rule,5,3,target)
            expected=0
            for initial in range(32):
                state=initial
                for _ in range(3):state=evolve(state,rule,5)
                expected+=state==target
            cases.append((f'cellular_{rule}_{target}',variables,clauses,expected))
    results=[]
    for mode in ('native','reverse_time','cell35','cell49'):
        system=CountingSystem(directory.parent,mode)
        for name,variables,clauses,expected in cases:
            case=directory/mode/name;case.mkdir(parents=True)
            formula=case/'input.cnf'
            formula.write_text(f'p cnf {variables} {len(clauses)}\n'+''.join(' '.join(map(str,clause))+' 0\n' for clause in clauses))
            os.chdir(case)
            result=system.solve(formula,seconds=3,validate_mapping=True)
            assert result['status']=='complete' and result['count']==expected,(name,mode,result,expected)
            results.append({'case':name,'expected':expected,**result})
            (case/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    assert all(sum(row['callbacks'] for row in results if row['mode']==mode)>0 for mode in ('cell35','cell49'))
    receipt={'checks':len(results),'results':results,'training':False}
    (directory/'solver_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'checks':len(results),'callbacks':sum(row['callbacks'] for row in results)}),flush=True)


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
