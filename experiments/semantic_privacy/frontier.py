"""Complete finite Bayes-accuracy frontier via partial transport."""
from fractions import Fraction as F
import json
from pathlib import Path
import sys
from solve import cases, transport


def main(config, output):
    cfg=json.loads(Path(config).read_text())
    with Path(output).open('x') as f:
        for name,N,pos,neg,pi in cases(cfg):
            if pi!=F(1,2) or ('/uniform' in name):
                continue
            for a in map(F,cfg['accuracy_caps']):
                cost=[[sum((x-y)**2 for x,y in zip(p,q)) for q in neg] for p in pos]
                flow,u,v,lam,scale,target=transport(cost,2*(1-a),partial=True)
                row=dict(case=name,accuracy_cap=str(a),flow=flow,dual_positive=u,
                         dual_negative=v,mass_dual=lam,scale=scale,target=target,
                         objective=sum(k*cost[i][j] for i,j,k in flow))
                f.write(json.dumps(row)+'\n')


if __name__=='__main__':
    main(*sys.argv[1:])
