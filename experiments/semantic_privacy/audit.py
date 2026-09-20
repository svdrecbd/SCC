"""Independent exact primal/dual and Bayesian audit; never imports the solver."""
from copy import deepcopy
from fractions import Fraction as F
from itertools import product
import json
from math import comb
from pathlib import Path


def expected(cfg):
    result = {}
    for N in cfg['denominators']:
        tables = [q for q in product(range(N+1), repeat=4) if sum(q)==N]
        for j in cfg['judgments']:
            for utility in cfg['utilities']:
                groups = [[], []]
                for q in tables:
                    p0, p1 = q[2]+q[3], q[1]+q[3]
                    if j == 'action_risk':
                        bit = p1 > N/2
                    elif j == 'net_harm':
                        bit = p1-p0 > 0
                    elif j == 'newly_harmed':
                        bit = q[1] > N/2
                    else:
                        assert j == 'any_new_harm'
                        bit = q[1] != 0
                    groups[int(bit)].append(list(q) if utility=='paired' else [p0,p1])
                neg, pos = groups
                for prior in cfg['priors']:
                    pi = F(len(pos), len(tables)) if prior=='uniform' else F(1,2)
                    result[f'N{N}/{j}/{utility}/{prior}'] = (N,pos,neg,pi)
    result.update({
        'control/causal/prospective': (4,[[1,3]],[[1,3]],F(1,2)),
        'control/causal/paired': (4,[[0,3,1,0]],[[1,2,0,1]],F(1,2)),
        'control/small_margin': (20,[[11]],[[9]],F(1,2)),
        'control/target_identity': (1,[[1]],[[0]],F(1,2)),
        'control/independent': (2,[[0],[1],[2]],[[0],[1],[2]],F(1,2))})
    return result


def rational(x):
    return {'exact': str(x), 'value': float(x)}


def check(row, spec, horizons):
    N, pos, neg, pi = spec
    assert set(row)=={'case','denominator','positive','negative','pi','flow',
                      'dual_positive','dual_negative','objective'}
    assert row['denominator']==N and row['positive']==pos and row['negative']==neg
    assert row['pi']==[pi.numerator,pi.denominator]
    m,n,d = len(pos),len(neg),len(pos[0])
    total = m*n
    u,v = row['dual_positive'],row['dual_negative']
    assert len(u)==m and len(v)==n
    assert all(type(x) is int for x in u+v+[row['objective']])
    def cost(i,j):
        return sum((pos[i][a]-neg[j][a])**2 for a in range(d))
    for i,j in product(range(m),range(n)):
        assert u[i]+v[j] <= cost(i,j), 'dual infeasible'
    row_sums,col_sums = [0]*m,[0]*n
    used = set()
    primal = 0
    distortion = F(0)
    repair = {k:F(0) for k in horizons} if '/prospective' in row['case'] else {}
    zero_mass = F(0)
    for i,j,k in row['flow']:
        assert all(type(a) is int for a in [i,j,k])
        assert 0<=i<m and 0<=j<n and k>0 and (i,j) not in used
        used.add((i,j)); row_sums[i]+=k; col_sums[j]+=k
        primal += k*cost(i,j)
        assert u[i]+v[j]==cost(i,j), 'complementary slackness'
        mass=F(k,total)
        # Explicit channel probabilities and posterior, including skewed priors.
        positive_joint = pi/F(m)*F(k,n)
        negative_joint = (1-pi)/F(n)*F(k,m)
        assert positive_joint+negative_joint==mass
        assert positive_joint/mass==pi
        for a in range(d):
            x,y = F(pos[i][a],N),F(neg[j][a],N)
            mean = pi*x+(1-pi)*y
            distortion += mass*(pi*(x-mean)**2+(1-pi)*(y-mean)**2)/d
        if cost(i,j)==0:
            zero_mass+=mass
        if repair:
            a=max(range(d), key=lambda a:abs(pos[i][a]-neg[j][a]))
            x,y=F(pos[i][a],N),F(neg[j][a],N)
            for h in repair:
                accuracy=sum((comb(h,z)*max(pi*x**z*(1-x)**(h-z),
                                 (1-pi)*y**z*(1-y)**(h-z)) for z in range(h+1)), F(0))
                repair[h]+=mass*accuracy
    assert row_sums==[n]*m and col_sums==[m]*n, 'marginal mismatch'
    dual=n*sum(u)+m*sum(v)
    assert primal==dual==row['objective'], 'objective mismatch'
    assert distortion==pi*(1-pi)*F(primal,total*d*N*N)
    baseline=F(0)
    for a in range(d):
        mean=pi*sum(F(p[a],N) for p in pos)/m+(1-pi)*sum(F(q[a],N) for q in neg)/n
        baseline+=(pi*sum((F(p[a],N)-mean)**2 for p in pos)/m+
                  (1-pi)*sum((F(q[a],N)-mean)**2 for q in neg)/n)/d
    assert 0<=distortion<=baseline
    if repair:
        assert repair[0]==max(pi,1-pi)
        assert all(a<=b for a,b in zip(repair.values(),list(repair.values())[1:]))
    return dict(case=row['case'], worlds=m+n, signals=len(used),
                all_edge_dual_checks=m*n, label_prior=rational(pi),
                protected_accuracy_before_feedback=rational(max(pi,1-pi)),
                excess_brier=rational(distortion), no_information_regret=rational(baseline),
                retained_prediction_skill=rational(1-distortion/baseline) if baseline else None,
                zero_distance_signal_mass=rational(zero_mass),
                repair_accuracy={str(k):rational(a) for k,a in repair.items()})


def verify(rows,cfg):
    specs=expected(cfg)
    assert len(rows)==len(specs)==cfg['expected_cases'], 'case count'
    assert [r['case'] for r in rows]==list(specs), 'case order or coverage'
    results=[check(row,specs[row['case']],cfg['repair_observations']) for row in rows]
    by={r['case']:r for r in results}
    assert by['control/small_margin']['excess_brier']['exact']=='1/400'
    assert by['control/target_identity']['retained_prediction_skill']['exact']=='0'
    assert by['control/independent']['retained_prediction_skill']['exact']=='1'
    assert by['control/causal/prospective']['excess_brier']['exact']=='0'
    assert by['control/causal/paired']['excess_brier']['exact']=='1/64'
    return results


def controls(rows,cfg):
    specs=expected(cfg)
    sample=deepcopy(rows[0])
    bad=[]
    def rejected(name,r):
        try:
            check(r,specs[r['case']],cfg['repair_observations'])
        except (AssertionError,IndexError,ValueError):
            bad.append(name)
        else:
            raise AssertionError('accepted corruption: '+name)
    r=deepcopy(sample);r['flow'][0][2]+=1;rejected('flow',r)
    r=deepcopy(sample);r['objective']+=1;rejected('objective',r)
    r=deepcopy(sample);r['dual_positive'][0]+=1000000;rejected('dual',r)
    r=deepcopy(sample);r['positive'][0][0]+=1;rejected('world',r)
    r=deepcopy(sample);r['pi']=[1,3];rejected('prior',r)
    r=deepcopy(sample);r['flow'].append(r['flow'][0]);rejected('duplicate_flow',r)
    for name,r in [('missing_case',rows[:-1]),('duplicate_case',rows[:-1]+[rows[0]])]:
        try:
            verify(r,cfg)
        except AssertionError:
            bad.append(name)
        else:
            raise AssertionError('accepted corruption: '+name)
    return bad
