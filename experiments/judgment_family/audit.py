"""Independent rational posterior audit, rank computation, and sequential repair."""
from copy import deepcopy
from fractions import Fraction as F
from itertools import product


def rank(matrix):
    a=[[F(x) for x in row] for row in matrix]
    pivot=0
    for col in range(len(a[0])):
        found=next((i for i in range(pivot,len(a)) if a[i][col]),None)
        if found is None:continue
        a[pivot],a[found]=a[found],a[pivot]
        scale=a[pivot][col];a[pivot]=[x/scale for x in a[pivot]]
        for i in range(len(a)):
            if i!=pivot:
                q=a[i][col];a[i]=[x-q*y for x,y in zip(a[i],a[pivot])]
        pivot+=1
        if pivot==len(a):break
    return pivot


def reference(cfg):
    K=cfg['levels'];N=K-1;worlds=list(product(range(K),repeat=2));size=len(worlds)
    risk=[[F(w[a],N) for w in worlds] for a in (0,1)]
    labels={f'action{a}/cut{k}':[int(p>F(2*k+1,2*N)) for p in risk[a]]
            for a in (0,1) for k in range(N)}
    mixed=[(x+K*y)/(K+1) for x,y in zip(*risk)]
    labels.update({f'mixed/cut{k}':[int(p>F(2*k+1,2*(size-1))) for p in mixed]
                   for k in range(size-1)})
    vals=dict(risk0=risk[0],risk1=risk[1],mixed_risk=mixed,
              equal_risk=[F(x==y) for x,y in zip(*risk)],
              risk1_greater=[F(y>x) for x,y in zip(*risk)],
              two_unit_harm=[x*y for x,y in zip(*risk)])
    marginal_matrix=[[1]*size]+[v for k,v in labels.items() if k.startswith('action')]
    mixed_matrix=[[1]*size]+[v for k,v in labels.items() if k.startswith('mixed')]
    ranks=[rank(marginal_matrix),rank(mixed_matrix)]
    # Gram-Schmidt in the uniform prior's L2 space, unlike the evaluator's ANOVA formula.
    basis=[]
    for vec in marginal_matrix:
        v=list(map(F,vec))
        for b in basis:
            coefficient=sum(x*y for x,y in zip(v,b))/sum(x*x for x in b)
            v=[x-coefficient*y for x,y in zip(v,b)]
        if any(v):basis.append(v)
    residuals={}
    for name,f in vals.items():
        r=list(f)
        for b in basis:
            coefficient=sum(x*y for x,y in zip(r,b))/sum(x*x for x in b)
            r=[x-coefficient*y for x,y in zip(r,b)]
        assert all(sum(x*y for x,y in zip(r,b))==0 for b in basis)
        residuals[name]=sum(x*x for x in r)/size
    # Verify reconstruction rather than merely accepting a rank count.
    for a in (0,1):
        assert [sum(labels[f'action{a}/cut{k}'][i] for k in range(N))/F(N)
                for i in range(size)]==risk[a]
    mixed_codes=[tuple(labels[f'mixed/cut{k}'][i] for k in range(size-1)) for i in range(size)]
    assert len(set(mixed_codes))==size
    return worlds,labels,vals,ranks,residuals


def moments(values,groups):
    size=len(values);mean=sum(values)/F(size)
    variance=sum((x-mean)**2 for x in values)/size
    conditional=[sum(values[i] for i in g)/F(len(g)) for g in groups]
    mmse=sum(sum((values[i]-c)**2 for i in g) for g,c in zip(groups,conditional))/size
    retained=variance-mmse
    return dict(mean=str(mean),variance=str(variance),retained_variance=str(retained),
                retained_skill=str(retained/variance) if variance else None)


def verify(rows,cfg):
    K=cfg['levels'];N=K-1;worlds,labels,values,ranks,residuals=reference(cfg)
    assert len(rows)==len(cfg['modes'])+len(cfg['repair_samples_per_action'])+1
    count=0; summaries=[]
    for row,mode in zip(rows,cfg['modes']):
        assert row['kind']=='encoding' and row['mode']==mode
        if mode=='intact':enc=list(range(K*K))
        elif mode=='recode':enc=list(reversed(range(K*K)))
        elif mode=='constant':enc=[0]*len(worlds)
        else:
            assert mode=='difference'
            enc=[(w[1]+K-w[0])%K for w in worlds]
        assert row['encoding']==enc
        groups=[[i for i,x in enumerate(enc) if x==e] for e in sorted(set(enc))]
        assert [row['marginal_augmented_rank'],row['mixed_augmented_rank']]==ranks
        assert set(row['protected'])==set(labels) and set(row['useful'])==set(values)
        hidden=True
        for name,f in labels.items():
            pred=moments(f,groups)
            p=F(sum(f),len(f));pred['prior_accuracy']=str(max(p,1-p))
            acc=sum(max(sum(f[i] for i in g),len(g)-sum(f[i] for i in g)) for g in groups)/F(len(f))
            pred['posterior_accuracy']=str(acc)
            assert row['protected'][name]==pred
            if name.startswith('action') and F(pred['retained_variance'])!=0:hidden=False
            count+=1
        for name,f in values.items():
            pred=moments(f,groups);pred['additive_residual']=str(residuals[name])
            assert row['useful'][name]==pred
            if hidden:assert F(pred['retained_variance'])<=residuals[name]
            count+=1
        assert row['joint_world_accuracy']==str(F(len(groups),K*K))
        summaries.append(dict(mode=mode,marginal_family_hidden=hidden,
            risk0_skill=row['useful']['risk0']['retained_skill'],
            equality_skill=row['useful']['equal_risk']['retained_skill'],
            two_unit_harm_skill=row['useful']['two_unit_harm']['retained_skill'],
            joint_world_accuracy=row['joint_world_accuracy']))
    assert ranks==[7,16]
    assert summaries[2]['marginal_family_hidden'] and summaries[3]['marginal_family_hidden']
    assert summaries[3]['risk0_skill']=='0' and summaries[3]['equality_skill']=='1'
    # Dynamic probability propagation gives binomial count distributions without powers/combinations.
    probabilities=[[F(1)] for a in range(K)]
    t=0; repairs=[]
    for row,m in zip(rows[len(cfg['modes']):],cfg['repair_samples_per_action']):
        assert row['kind']=='repair' and row['samples_per_action']==m
        while t<m:
            for a in range(K):
                p=F(a,N);old=probabilities[a];new=[F(0)]*(len(old)+1)
                for h,mass in enumerate(old):
                    new[h]+=mass*(1-p);new[h+1]+=mass*p
                probabilities[a]=new
            t+=1
        risk=F(0);success=F(0)
        assert len(row['likelihoods'])==m+1
        for h in range(m+1):
            joint=[probabilities[a][h]/K for a in range(K)]
            mass=sum(joint)
            assert mass>0
            posterior=[x/mass for x in joint]
            weights=row['likelihoods'][h]
            assert len(weights)==K and all(type(w) is int and w>=0 for w in weights)
            assert [F(w,sum(weights)) for w in weights]==posterior
            # Also check absolute likelihoods by recurrence at the individual sequence level.
            expected_weights=[]
            for a in range(K):
                weight=1
                for _ in range(h):weight*=a
                for _ in range(m-h):weight*=N-a
                expected_weights.append(weight)
            assert weights==expected_weights
            mean=sum(F(a,N)*posterior[a] for a in range(K))
            risk+=sum(joint[a]*(F(a,N)-mean)**2 for a in range(K))
            success+=max(joint)
        variance=sum((F(a,N)-F(1,2))**2 for a in range(K))/K
        sample_error=sum(F(a,N)*(1-F(a,N)) for a in range(K))/K/m if m else None
        assert row['bayes_regret']==str(risk)
        assert row['retained_prediction_skill']==str(1-risk/variance)
        assert row['single_action_map']==str(success)
        assert row['joint_world_accuracy']==str(success*success)
        assert row['sample_mean_regret']==(str(sample_error) if m else None)
        assert row['logical_count_bits']==2*(0 if m==0 else (m+1-1).bit_length())
        if sample_error is not None:assert risk<=sample_error
        repairs.append({k:v for k,v in row.items() if k!='likelihoods'})
    assert all(F(x['bayes_regret'])>=F(y['bayes_regret']) for x,y in zip(repairs,repairs[1:]))
    assert all(F(x['joint_world_accuracy'])<=F(y['joint_world_accuracy']) for x,y in zip(repairs,repairs[1:]))
    skew=rows[-1];expected=moments([0,1,1,1],[[0,1],[2,3]])
    expected.update(kind='skew_control',prior_accuracy='3/4',posterior_accuracy='3/4')
    assert skew==expected and F(skew['retained_variance'])==F(1,16)
    return dict(validation='PASS',posterior_function_checks=count,ranks=ranks,
                encodings=summaries,repairs=repairs,skew_control=skew,
                additive_residuals={k:str(v) for k,v in residuals.items()})


def controls(rows,cfg):
    mutations={
        'encoding':lambda r:r[3]['encoding'].__setitem__(0,1),
        'rank':lambda r:r[0].__setitem__('marginal_augmented_rank',16),
        'threshold_metric':lambda r:r[3]['protected']['action0/cut0'].__setitem__('retained_variance','1'),
        'useful_metric':lambda r:r[3]['useful']['equal_risk'].__setitem__('retained_skill','0'),
        'likelihood':lambda r:r[5]['likelihoods'][0].__setitem__(0,100),
        'repair_accuracy':lambda r:r[-2].__setitem__('joint_world_accuracy','1'),
        'repair_budget':lambda r:r[-2].__setitem__('logical_count_bits',0),
        'skew_privacy':lambda r:r[-1].__setitem__('retained_variance','0'),
        'missing_case':lambda r:r.pop(),
        'duplicate_case':lambda r:r.__setitem__(1,deepcopy(r[0]))}
    passed=[]
    for name,mutate in mutations.items():
        bad=deepcopy(rows);mutate(bad)
        try:verify(bad,cfg)
        except (AssertionError,ValueError,KeyError):passed.append(name)
        else:raise AssertionError('accepted corruption '+name)
    return passed
