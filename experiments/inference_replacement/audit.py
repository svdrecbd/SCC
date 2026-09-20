"""Independent direct/transfer references, symbolic cost audit and perturbation checks."""
from copy import deepcopy
from fractions import Fraction as F
from itertools import product,combinations


def data(kind,n,seed,t):
    unary=[]
    for i in range(n):
        vals=[(seed+3*i)%5+1,(2*seed+5*i)%7+1]
        for k in range(t):
            if (seed+5*k)%n==i:vals[(seed+k)%2]*=3
        unary.append(vals)
    edges={}
    for u in range(n):
        for v in range(u+1,n):
            include=(kind=='clique' or (kind=='star' and u==0) or
                     (kind=='chain' and v==u+1) or
                     (kind=='ladder' and (v==u+2 or (u%2==0 and v==u+1))))
            if include:edges[u,v]=[[1+(seed+u+2*v+3*a+b)%5 for b in range(2)] for a in range(2)]
    return unary,edges


def oracle(kind,n,unary,edges,extra):
    unary=deepcopy(unary)
    for scope,values in extra:
        i=scope[0];unary[i]=[x*y for x,y in zip(unary[i],values)]
    if n<=12:
        total=0
        for state in product((0,1),repeat=n):
            weight=1
            for i,a in enumerate(state):weight*=unary[i][a]
            for (u,v),matrix in edges.items():weight*=matrix[state[u]][state[v]]
            total+=weight
        return total
    if kind=='star':
        total=0
        for hub in (0,1):
            val=unary[0][hub]
            for i in range(1,n):val*=sum(unary[i][a]*edges[0,i][hub][a] for a in (0,1))
            total+=val
        return total
    # Independent transfer of a column (one site for chains, two for ladders).
    width=1 if kind=='chain' else 2
    assert kind in ('chain','ladder')
    states=list(product((0,1),repeat=width));previous=None
    for start in range(0,n,width):
        current={}
        for state in states:
            weight=1
            for j,a in enumerate(state):weight*=unary[start+j][a]
            if width==2:weight*=edges[start,start+1][state[0]][state[1]]
            incoming=1 if previous is None else 0
            if previous is not None:
                for old,mass in previous.items():
                    value=mass
                    for j in range(width):value*=edges[start-width+j,start+j][old[j]][state[j]]
                    incoming+=value
            current[state]=weight*incoming
        previous=current
    return sum(previous.values())


def costs(scopes,order,cap):
    active=[set(s) for s in scopes];adds=mults=width=largest=0
    peak=sum(2**len(s) for s in active)
    for x in order:
        bucket=[s for s in active if x in s];joined=set().union(*bucket);out=joined-{x}
        entries=2**len(out)
        if entries>cap:return None
        width=max(width,len(out));largest=max(largest,entries)
        peak=max(peak,sum(2**len(s) for s in active)+entries)
        adds+=2**len(joined);mults+=len(bucket)*2**len(joined)
        active=[s for s in active if x not in s]+[out]
    mults+=len(active)
    return dict(additions=adds,multiplications=mults,width=width,
                max_factor_entries=largest,peak_active_entries=peak)


def order_reference(n,scopes):
    graph={i:set() for i in range(n)}
    for scope in scopes:
        for a in scope:
            graph[a].update(set(scope)-{a})
    out=[];checked=0
    while graph:
        scores={}
        for x,neigh in graph.items():
            pairs=list(combinations(sorted(neigh),2));checked+=len(pairs)
            scores[x]=(sum(1 for a,b in pairs if b not in graph[a]),len(neigh),x)
        x=min(scores,key=scores.get);neighbors=graph.pop(x)
        for a in neighbors:graph[a]=(graph[a]|(neighbors-{a}))-{x}
        out.append(x)
    return out,checked


def check(row,cfg):
    kind,n,seed,t=row['kind'],row['n'],row['seed'],row['evidence']
    assert row['case']==f'{kind}/{n}/{seed}/{t}'
    unary,edges=data(kind,n,seed,t)
    source={tuple(s):v for s,v in row['factors']}
    expected={(i,):u for i,u in enumerate(unary)}
    expected.update({e:[matrix[a][b] for b in (0,1) for a in (0,1)] for e,matrix in edges.items()})
    assert source==expected and len(source)==len(row['factors'])
    order,checks=order_reference(n,source)
    assert order==row['order'] and checks==row['compile_pair_checks']
    assert row['input_entries']==sum(len(v) for v in expected.values())
    assert row['evidence_multiplications']==2*t
    queries={'base':[]}
    queries.update({f'node{i}':[((i,),[0,1])] for i in range(n)})
    queries.update(joint=[((0,),[0,1]),((n-1,),[0,1])],sensor=[((0,),[1,3])],
                   sensor_last=[((0,),[1,3]),((n-1,),[0,1])])
    assert set(queries)==set(row['evaluations'])
    truth={}
    for name,extra in queries.items():
        truth[name]=oracle(kind,n,unary,edges,extra)
        r=row['evaluations'][name];assert r['value']==truth[name] and truth[name]>0
        expected_cost=costs(list(source)+[s for s,v in extra],order,cfg['factor_entry_cap'])
        assert expected_cost is not None
        assert {k:r['meter'][k] for k in expected_cost}==expected_cost
        assert type(r['meter']['max_integer_bits']) is int
        assert r['meter']['max_integer_bits']>=truth[name].bit_length()
    z=truth['base'];a,b,c=F(truth['node0'],z),F(truth[f'node{n-1}'],z),F(truth['sensor_last'],truth['sensor'])
    j=F(truth['joint'],z)
    assert F(truth['sensor'],z)==1+2*a
    assert (c*(1+2*a)-b)/2==j==F(row['adapter'])
    assert row['protected_label']==(j>F(1,4))
    assert row['constant_forecast_mean_absolute_error']==str(sum(abs(F(truth[f'node{i}'],z)-F(1,2)) for i in range(n))/n)
    assert len(row['perturbations'])==len(cfg['perturbations'])
    margin_cases=0
    for perturb,epsstr in zip(row['perturbations'],cfg['perturbations']):
        eps=F(epsstr);bounds=[(max(F(0),x-eps),min(F(1),x+eps)) for x in (a,b,c)]
        values=[]
        for aa,bb,cc in product(*bounds):values.append(max(F(0),min(F(1),((1+2*aa)*cc-bb)/2)))
        error=max(abs(v-j) for v in values);correct=sum((v>F(1,4))==(j>F(1,4)) for v in values)
        assert perturb==dict(epsilon=str(eps),max_error=str(error),correct_labels=correct)
        assert error<=3*eps
        if abs(j-F(1,4))>3*eps:assert correct==8;margin_cases+=1
    if kind=='star':
        bad=costs(list(source),list(range(n)),cfg['factor_entry_cap'])
        if bad is None:
            assert row['bad_order']==dict(status='declared_cap',detail=f'variable=0, requested_entries={2**(n-1)}, cap={cfg["factor_entry_cap"]}')
        else:
            assert row['bad_order']['status']=='complete' and row['bad_order']['value']==z
            assert {k:row['bad_order']['meter'][k] for k in bad}==bad
    else:assert row['bad_order'] is None
    return dict(case=row['case'],nodes=n,kind=kind,protected_label=row['protected_label'],
                width=row['evaluations']['base']['meter']['width'],
                base_meter=row['evaluations']['base']['meter'],
                query_count=len(queries),compile_pair_checks=checks,input_entries=row['input_entries'],
                suite_integer_operations=sum(r['meter']['additions']+r['meter']['multiplications'] for r in row['evaluations'].values()),
                adapter_extra_integer_operations=sum(row['evaluations'][name]['meter']['additions']+row['evaluations'][name]['meter']['multiplications'] for name in ('sensor','sensor_last')),
                margin_certified_perturbations=margin_cases,bad_order=row['bad_order'])


def verify(rows,cfg):
    keys=[f'{kind}/{n}/{seed}/{t}' for (kind,n),seed,t in product(cfg['models'],cfg['seeds'],cfg['evidence_prefixes'])]
    assert len(rows)==cfg['expected_cases'] and [r['case'] for r in rows]==keys
    summaries=[check(r,cfg) for r in rows]
    assert any(r['protected_label'] for r in summaries) and not all(r['protected_label'] for r in summaries)
    return summaries


def controls(rows,cfg):
    sample=rows[0];passed=[]
    for name in ('factor','order','cost','value','adapter','threshold','perturbation','compile'):
        r=deepcopy(sample)
        if name=='factor':r['factors'][0][1][0]+=1
        elif name=='order':r['order'][0],r['order'][1]=r['order'][1],r['order'][0]
        elif name=='cost':r['evaluations']['base']['meter']['additions']+=1
        elif name=='value':r['evaluations']['base']['value']+=1
        elif name=='adapter':r['adapter']='2'
        elif name=='threshold':r['protected_label']=not r['protected_label']
        elif name=='perturbation':r['perturbations'][0]['max_error']='2'
        else:r['compile_pair_checks']+=1
        try:check(r,cfg)
        except (AssertionError,ValueError,KeyError):passed.append(name)
        else:raise AssertionError('accepted corruption '+name)
    for name,bad in [('missing_case',rows[:-1]),('duplicate_case',rows[:-1]+[rows[0]])]:
        try:verify(bad,cfg)
        except AssertionError:passed.append(name)
        else:raise AssertionError('accepted corruption '+name)
    return passed
