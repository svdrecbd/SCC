"""Exact bounded-degree feedback identification from delayed output histories."""
from itertools import combinations


def monomials(n,degree):
    return [sum(1<<i for i in term) for k in range(degree+1) for term in combinations(range(n),k)]


def features(x,masks):
    return sum(int((x & m)==m)<<i for i,m in enumerate(masks))


def feedback(x,masks,coefficients):
    return (features(x,masks)&coefficients).bit_count()&1


def step(x,u,n,masks,coefficients):
    return ((x<<1)&((1<<n)-1)) | (u ^ feedback(x,masks,coefficients))


def fit(samples,n,degree):
    masks=monomials(n,degree);basis={};selected=[];xors=0
    for i,(x,y) in enumerate(samples):
        row=features(x,masks);witness=1<<i
        while row:
            p=row.bit_length()-1
            if p not in basis:
                basis[p]=(row,y,witness);selected.append(i);break
            a,b,w=basis[p];row^=a;y^=b;witness^=w;xors+=1
        if row==0 and y:
            return dict(degree=degree,status='inconsistent',rank=len(basis),processed=i+1,
                feature_tests=(i+1)*len(masks),row_xors=xors,provenance_xors=xors,
                independent_rows=selected,contradiction=witness)
    result=dict(degree=degree,status='full_rank' if len(basis)==len(masks) else 'underidentified',
        rank=len(basis),processed=len(samples),feature_tests=len(samples)*len(masks),
        row_xors=xors,provenance_xors=xors,independent_rows=selected)
    if result['status']=='full_rank':
        coefficients=0
        for p in range(len(masks)):
            a,y,w=basis[p];bit=y ^ ((a&coefficients).bit_count()&1)
            coefficients |= bit<<p
        result['coefficients']=coefficients
    return result


def reconstruct(job):
    n=job['n'];ys=job['outputs'];us=job['inputs'];count=job['samples']
    assert len(us)==count+2*n and len(ys)==len(us)+1
    assert all(v in (0,1) for v in ys+us)
    samples=[(sum(ys[t+n-1-i]<<i for i in range(n)), us[t]^ys[t+n]) for t in range(n,n+count)]
    attempts=[]
    for degree in range(1,job['degree_cap']+1):
        result=fit(samples,n,degree);attempts.append(result)
        if result['status']=='full_rank':break
    assert attempts[-1]['status']=='full_rank','no uniquely identified model within budget'
    chosen=attempts[-1];masks=monomials(n,chosen['degree']);coef=chosen['coefficients']
    initial=[];current=[]
    for observed in job['warm_outputs']:
        assert len(observed)==n and all(v in (0,1) for v in observed)
        x=sum(bit<<(n-1-i) for i,bit in enumerate(observed));initial.append(x)
        for _ in range(n):x=step(x,0,n,masks,coef)
        current.append(x)
    return dict(id=job['id'],degree=chosen['degree'],coefficients=coef,attempts=attempts,
                decoded_initial=initial,current_states=current,
                meter=dict(calibration_parent_steps=len(us),recorded_input_bits=len(us),
                    observed_calibration_bits=len(ys),warm_observed_bits=n*len(current),
                    warm_parent_steps=n*len(current),warm_successor_steps=n*len(current),
                    decoded_sample_payload_bits=count*(n+1),coefficient_payload_bits=len(masks),
                    running_state_bits=n,fit_feature_tests=sum(a['feature_tests'] for a in attempts),
                    warm_feature_tests=n*len(current)*len(masks),
                    row_xors=sum(a['row_xors'] for a in attempts),
                    provenance_xors=sum(a['provenance_xors'] for a in attempts),
                    largest_feature_row_bits=max(len(monomials(n,a['degree'])) for a in attempts),
                    provenance_width_bound_bits=count))
