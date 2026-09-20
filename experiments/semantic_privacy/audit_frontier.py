"""Rational witness reconstruction and independent partial-flow dual audit."""
from copy import deepcopy
from fractions import Fraction as F
from itertools import product
from audit import expected, rational


def check(row,spec):
    N,pos,neg,pi=spec
    assert pi==F(1,2)
    assert set(row)=={'case','accuracy_cap','flow','dual_positive','dual_negative',
                      'mass_dual','scale','target','objective'}
    a=F(row['accuracy_cap'])
    assert F(1,2)<=a<=1
    m,n,d=len(pos),len(neg),len(pos[0])
    scale=row['scale']; target=row['target']; lam=row['mass_dual']
    assert all(type(k) is int for k in [scale,target,lam,row['objective']])
    assert scale>0
    total=m*n*scale
    assert F(target,total)==2*(1-a)
    u,v=row['dual_positive'],row['dual_negative']
    assert len(u)==m and len(v)==n
    assert all(type(k) is int and k<=0 for k in u+v)
    def cost(i,j):
        return sum((pos[i][b]-neg[j][b])**2 for b in range(d))
    for i,j in product(range(m),range(n)):
        assert lam+u[i]+v[j]<=cost(i,j), 'partial dual infeasible'
    rs,cs=[0]*m,[0]*n
    used=set(); primal=0; distortion=F(0); pair_mass=F(0)
    for i,j,k in row['flow']:
        assert all(type(x) is int for x in [i,j,k])
        assert 0<=i<m and 0<=j<n and k>0 and (i,j) not in used
        used.add((i,j));rs[i]+=k;cs[j]+=k
        primal+=k*cost(i,j)
        assert lam+u[i]+v[j]==cost(i,j)
        mass=F(k,total);pair_mass+=mass
        # Joint probability of each of the two possible worlds is mass/2.
        for b in range(d):
            x,y=F(pos[i][b],N),F(neg[j][b],N)
            mean=(x+y)/2
            distortion+=mass*((x-mean)**2+(y-mean)**2)/(2*d)
    assert sum(rs)==sum(cs)==target
    assert all(r<=n*scale for r in rs) and all(c<=m*scale for c in cs)
    assert all(u[i]*(n*scale-rs[i])==0 for i in range(m))
    assert all(v[j]*(m*scale-cs[j])==0 for j in range(n))
    singleton_mass=sum((F(n*scale-r,2*total) for r in rs),F(0))
    singleton_mass+=sum((F(m*scale-c,2*total) for c in cs),F(0))
    assert singleton_mass+pair_mass==1
    assert singleton_mass+pair_mass/2==a
    dual=target*lam+n*scale*sum(u)+m*scale*sum(v)
    assert primal==dual==row['objective']
    assert distortion==F(primal,4*total*d*N*N)
    return distortion


def verify(rows,cfg,endpoints):
    specs={k:v for k,v in expected(cfg).items() if '/uniform' not in k}
    assert len(rows)==cfg['expected_frontier_points']
    keys=[(name,a) for name in specs for a in cfg['accuracy_caps']]
    assert [(r['case'],r['accuracy_cap']) for r in rows]==keys
    results=[];previous={}
    for row in rows:
        name=row['case'];a=F(row['accuracy_cap'])
        loss=check(row,specs[name]);base=F(endpoints[name]['no_information_regret']['exact'])
        assert 0<=loss<=base
        if a==F(1,2):
            assert loss==F(endpoints[name]['excess_brier']['exact'])
        else:
            assert loss<=previous[name]
        if a==1:
            assert loss==0
        previous[name]=loss
        results.append(dict(case=name,accuracy_cap=str(a),excess_brier=rational(loss),
                       retained_prediction_skill=rational(1-loss/base) if base else None,
                       pair_outputs=len(row['flow'])))
    return results


def controls(rows,cfg,endpoints):
    specs=expected(cfg)
    sample=next(r for r in rows if r['accuracy_cap']=='3/4')
    found=[]
    for name in ['partial_mass','partial_flow','partial_dual','partial_sign','partial_objective']:
        r=deepcopy(sample)
        if name=='partial_mass':r['target']+=1
        if name=='partial_flow':r['flow'][0][2]+=1
        if name=='partial_dual':r['mass_dual']+=100000
        if name=='partial_sign':r['dual_positive'][0]=1
        if name=='partial_objective':r['objective']+=1
        try:check(r,specs[r['case']])
        except AssertionError:found.append(name)
        else:raise AssertionError('accepted '+name)
    reference=verify(rows,cfg,endpoints)
    corrupted=deepcopy(reference)
    corrupted[0]['excess_brier']['exact']='999'
    try:
        assert corrupted==reference
    except AssertionError:found.append('saved_summary')
    else:raise AssertionError('accepted saved summary corruption')
    return found
