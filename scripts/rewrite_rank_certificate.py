"""Exact rational certificate for two independent payloads under global averaging."""
from fractions import Fraction as Q


def multiply(a,b):return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]


def rank(a):
    a=[row[:] for row in a];pivot=0
    for col in range(len(a[0])):
        selected=next((i for i in range(pivot,len(a)) if a[i][col]),None)
        if selected is None:continue
        a[pivot],a[selected]=a[selected],a[pivot]
        scale=a[pivot][col];a[pivot]=[x/scale for x in a[pivot]]
        for i in range(len(a)):
            if i!=pivot:
                scale=a[i][col];a[i]=[x-scale*y for x,y in zip(a[i],a[pivot])]
        pivot+=1
    return pivot


def certificate():
    merged=[[Q(1,8)]*8 for _ in range(8)]
    encoding=[[Q(1),Q(0)]]*4+[[Q(0),Q(1)]]*4
    encoded_mean=multiply(merged,encoding)
    collision=multiply(encoded_mean,[[Q(1)],[Q(-1)]])
    centered=[[Q(0)]*8 for _ in range(8)]
    symmetric=[[Q(0)]*8 for _ in range(8)]
    for a,b in ((0,2),(1,3),(4,6),(5,7)):
        for d,sign in ((centered,-1),(symmetric,1)):
            d[a][a]=d[b][b]=Q(1,2);d[a][b]=d[b][a]=Q(sign,2)
    erased=multiply(multiply(centered,merged),centered)
    compensated=multiply(multiply(symmetric,merged),symmetric)
    assert rank(encoding)==2 and rank(merged)==rank(encoded_mean)==1
    assert rank(erased)==0 and compensated==merged and collision==[[0]]*8
    matrices={'encoding':encoding,'global_binding':merged,'centered_normalizer':centered,
              'symmetric_normalizer':symmetric,'centered_commit':erased,'compensated_commit':compensated,
              'encoded_mean':encoded_mean,'lost_difference':collision}
    return {'arithmetic':'exact rational','matrices':{k:[[str(x) for x in row] for row in v] for k,v in matrices.items()},
        'ranks':{k:rank(v) for k,v in matrices.items()},'logical_parameter_dimensions':80517,
        'merged_fixed_width_rank_upper_bound':40259,'linear_parameter_nullity_lower_bound':40258,
        'hidden_dimensions':128,'merged_hidden_rank_upper_bound':64,
        'per_stream_scalar_capacity':{'sharded':322584,'wider_four_sector':322580},
        'scope':'Fixed column width and linear sector coding/decoding; not an impossibility proof for arbitrary programs or learned-function compression.'}
