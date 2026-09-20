"""Independent dense binary algebra, parent construction and exact certificates."""
import itertools
import random
import numpy as np


def bits(x,n):return np.array([(x>>i)&1 for i in range(n)],dtype=np.uint8)
def pack(v):return sum(int(x)<<i for i,x in enumerate(v))
def mul(a,b):return (a @ b) & 1


def inverse(matrix):
    n=len(matrix); a=np.concatenate([matrix.copy(),np.eye(n,dtype=np.uint8)],axis=1)
    for j in range(n):
        available=np.flatnonzero(a[j:,j]); assert len(available)
        k=j+int(available[0]); a[[j,k]]=a[[k,j]]
        for i in range(n):
            if i!=j and a[i,j]:a[i]^=a[j]
    assert np.array_equal(a[:,:n],np.eye(n,dtype=np.uint8))
    return a[:,n:]


def dense(rows,n):return np.array([bits(x,n) for x in rows],dtype=np.uint8)
def rows(matrix):return [pack(v) for v in matrix]


def parent_step(parent,state,bit):
    # Scalar tuple/bit enumeration, independent of the worker's companion update.
    return [int((sum(int(a)*int(x) for a,x in zip(row,state))+int(b)*bit)%2)
            for row,b in zip(parent['matrix'],parent['input'])]


def make(cfg):
    parents=[]; jobs=[]
    for r,factor,seed in itertools.product(cfg['orders'],cfg['padding'],cfg['seeds']):
        n=r*factor; rng=random.Random(100000*r+1000*factor+seed)
        coeff=rng.getrandbits(r)|1
        base=np.zeros((n,n),dtype=np.uint8)
        for i in range(r):
            if i:base[i,i-1]=1
            base[i,r-1]^=(coeff>>i)&1
        for i in range(r,n):base[i,i]=1
        s=np.eye(n,dtype=np.uint8)
        for _ in range(8*n):
            i,j=rng.sample(range(n),2);s[i]^=s[j]
        inv=inverse(s);a=mul(mul(s,base),inv);b=s[:,0];c=inv[r-1]
        ident=f'r{r}-n{n}-s{seed}'
        x=np.zeros(n,dtype=np.uint8); impulse=[]
        for t in range(2*n):
            x=mul(a,x) ^ (b if t==0 else 0);impulse.append(int(mul(c,x)))
        before=[rng.getrandbits(n) for _ in range(cfg['warm_states'])]
        if r==cfg['exhaustive_order']:
            before += [pack(mul(s,bits(z,n))) for z in range(1<<r)]
        warm=[];after=[]
        for state in before:
            x=bits(state,n);observed=[]
            for _ in range(r):observed.append(int(mul(c,x)));x=mul(a,x)
            warm.append(observed);after.append(pack(x))
        streams=[[rng.randrange(2) for _ in range(cfg['stream_length'])] for _ in before]
        private=None
        if n>r:
            private=dict(row=pack(inv[r]),difference=pack(s[:,r]))
        parent=dict(id=ident,n=n,r=r,seed=seed,matrix=a.tolist(),input=b.tolist(),output=c.tolist(),
            warm_before=before,warm_after=after,streams=streams,private=private)
        parents.append(parent)
        jobs.append(dict(id=ident,upper_order=n,impulse_outputs=impulse,warm_outputs=warm,
                         inputs=streams,exhaustive_start=cfg['warm_states'] if r==cfg['exhaustive_order'] else None))
    return parents,jobs


def certificate(parent,answer):
    n=parent['n'];r=answer['order'];assert r==parent['r']
    a=np.array(parent['matrix'],dtype=np.uint8);b=np.array(parent['input'],dtype=np.uint8)
    c=np.array(parent['output'],dtype=np.uint8);aa=np.zeros((r,r),dtype=np.uint8)
    for j in range(r-1):aa[j+1,j]=1
    aa[:,r-1]=bits(answer['coefficients'],r);d=bits(answer['readout'],r)
    obs=[];obs_hat=[];v=c.copy();w=d.copy()
    for _ in range(r):obs.append(v);obs_hat.append(w);v=mul(v,a);w=mul(w,aa)
    o=np.array(obs);oh=np.array(obs_hat);t=mul(inverse(oh),o)
    assert np.array_equal(mul(t,a),mul(aa,t)), 'dynamics certificate'
    assert np.array_equal(mul(t,b),bits(1,r)), 'input certificate'
    assert np.array_equal(mul(d,t),c), 'output certificate'
    assert len(answer['warm_states'])==len(parent['warm_after'])
    for state,z in zip(parent['warm_after'],answer['warm_states']):
        assert pack(mul(t,bits(state,n)))==z,'warm synchronization'
    if parent['private']:
        v=bits(parent['private']['difference'],n);p=bits(parent['private']['row'],n)
        assert int(mul(p,v))==1 and not mul(t,v).any()
    return dict(map_rows=rows(t),map_shape=[r,n],recovered_matrix_rows=rows(aa))
