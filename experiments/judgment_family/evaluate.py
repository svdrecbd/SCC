"""Finite risk-judgment families, full-state posteriors, and exact relearning."""
from collections import defaultdict
from fractions import Fraction as F
import json
from math import comb
from pathlib import Path
import sys


def metrics(values, groups):
    total=sum(len(g) for g in groups)
    mean=sum(values,F(0))/total
    variance=sum(((x-mean)**2 for x in values),F(0))/total
    retained=sum((F(len(g),total)*(sum((values[i] for i in g),F(0))/len(g)-mean)**2
                  for g in groups),F(0))
    return dict(mean=str(mean),variance=str(variance),retained_variance=str(retained),
                retained_skill=str(retained/variance) if variance else None)


def run(cfg):
    K=cfg['levels']; N=K-1
    worlds=[(a,b) for a in range(K) for b in range(K)]
    protected={f'action{axis}/cut{k}':[int(w[axis]>k) for w in worlds]
               for axis in (0,1) for k in range(N)}
    protected.update({f'mixed/cut{k}':[int(a+K*b>k) for a,b in worlds]
                      for k in range(K*K-1)})
    useful={
        'risk0':[F(a,N) for a,b in worlds],
        'risk1':[F(b,N) for a,b in worlds],
        'mixed_risk':[F(a+K*b,K*K-1) for a,b in worlds],
        'equal_risk':[F(a==b) for a,b in worlds],
        'risk1_greater':[F(b>a) for a,b in worlds],
        'two_unit_harm':[F(a*b,N*N) for a,b in worlds]}
    residual={}
    for name,values in useful.items():
        mean=sum(values)/len(values)
        row=[sum(values[a*K:(a+1)*K])/K for a in range(K)]
        col=[sum(values[a*K+b] for a in range(K))/K for b in range(K)]
        residual[name]=sum((values[a*K+b]-row[a]-col[b]+mean)**2 for a,b in worlds)/(K*K)
    rows=[]
    for mode in cfg['modes']:
        code={'intact':lambda a,b:a*K+b,'recode':lambda a,b:K*K-1-(a*K+b),
              'constant':lambda a,b:0,'difference':lambda a,b:(b-a)%K}[mode]
        grouping=defaultdict(list)
        encoding=[code(a,b) for a,b in worlds]
        for i,e in enumerate(encoding):grouping[e].append(i)
        groups=list(grouping.values())
        pmetrics={}
        for name,values in protected.items():
            r=metrics(values,groups)
            r['prior_accuracy']=str(max(F(sum(values),K*K),1-F(sum(values),K*K)))
            r['posterior_accuracy']=str(sum((max(sum(values[i] for i in g),
                len(g)-sum(values[i] for i in g)) for g in groups),F(0))/(K*K))
            pmetrics[name]=r
        umetrics={}
        for name,values in useful.items():
            r=metrics(values,groups); r['additive_residual']=str(residual[name])
            umetrics[name]=r
        rows.append(dict(kind='encoding',mode=mode,encoding=encoding,
                         protected=pmetrics,useful=umetrics,
                         joint_world_accuracy=str(F(len(groups),K*K)),
                         marginal_augmented_rank=2*K-1,mixed_augmented_rank=K*K))
    for m in cfg['repair_samples_per_action']:
        likelihoods=[[a**h*(N-a)**(m-h) for a in range(K)] for h in range(m+1)]
        denominator=K*N**m
        bayes_risk=F(0);map_success=F(0)
        for h,weights in enumerate(likelihoods):
            z=sum(weights)
            if not z:continue
            probability=F(comb(m,h)*z,denominator)
            mean=F(sum(a*w for a,w in enumerate(weights)),N*z)
            var=F(sum(a*a*w for a,w in enumerate(weights)),N*N*z)-mean*mean
            bayes_risk+=probability*var
            map_success+=F(comb(m,h)*max(weights),denominator)
        prior_var=F(K+1,12*(K-1))
        raw_variance=sum(F(a,N)*(1-F(a,N)) for a in range(K))/K
        rows.append(dict(kind='repair',samples_per_action=m,likelihoods=likelihoods,
                         bayes_regret=str(bayes_risk),retained_prediction_skill=str(1-bayes_risk/prior_var),
                         joint_world_accuracy=str(map_success**2),
                         single_action_map=str(map_success),
                         sample_mean_regret=str(raw_variance/m) if m else None,
                         logical_count_bits=2*m.bit_length()))
    values=[0,1,1,1];r=metrics(values,[[0,1],[2,3]])
    r.update(kind='skew_control',prior_accuracy='3/4',posterior_accuracy='3/4')
    rows.append(r)
    return rows


if __name__=='__main__':
    cfg=json.loads(Path(sys.argv[1]).read_text())
    with Path(sys.argv[2]).open('x') as f:
        for r in run(cfg):f.write(json.dumps(r)+'\n')
