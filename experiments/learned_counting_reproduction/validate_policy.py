"""Compare upstream policy arithmetic through sparse and dense message sums."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from policy import LearnedPolicy


def main(directory):
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    records=[]; controls={'reversed_polarity':False,'omitted_edges':False}
    for name in ('cell35','cell49'):
        sparse=LearnedPolicy(directory.parent,name)
        dense=LearnedPolicy(directory.parent,name,dense=True)
        (directory/(name+'_parameter_layout.json')).write_text(json.dumps(sparse.layout,indent=2)+'\n')
        for seed in (296,297,298):
            generator=np.random.default_rng(seed)
            literal_count=20; clause_count=12
            coordinates=sorted({(row,int(column)) for row in range(clause_count)
                                for column in generator.choice(literal_count,size=4,replace=False)})
            rows=np.array([item[0] for item in coordinates],dtype=np.int64)
            columns=np.array([item[1] for item in coordinates],dtype=np.int64)
            labels=np.zeros((literal_count,4)); labels[:,0]=np.arange(literal_count)
            labels[:,1]=np.where(np.arange(literal_count)%2,1,-1)*(np.arange(literal_count)//2+1)
            actual,action,selected=sparse.scores(rows,columns,labels)
            expected,reference_action,_=dense.scores(rows,columns,labels)
            np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-5)
            assert action==reference_action
            reversed_scores,_,_=sparse.scores(rows,np.bitwise_xor(columns,1),labels)
            # Reversing every polarity must permute paired scores, not leave the vector unchanged.
            np.testing.assert_allclose(reversed_scores,actual.reshape(-1,2)[:,::-1].reshape(-1),rtol=1e-5,atol=1e-5)
            controls['reversed_polarity'] |= not np.allclose(reversed_scores,expected,rtol=1e-5,atol=1e-5)
            modified,_,modified_selected=sparse.scores(rows[:-1],columns[:-1],labels)
            if np.array_equal(selected,modified_selected):
                controls['omitted_edges'] |= not np.allclose(modified,expected,rtol=1e-5,atol=1e-5)
            ordered=np.sort(actual)
            records.append({'model':name,'seed':seed,'maximum_absolute_difference':float(np.max(np.abs(actual-expected))),
                            'action':action,'top_score_margin':float(ordered[-1]-ordered[-2])})
    assert all(controls.values()),controls
    receipt={'rows':records,'corruption_controls':controls,'parameters_per_checkpoint':82098,
             'active_parameters_per_checkpoint':sum(item['count'] for item in sparse.layout if not item['name'].startswith('decode_module.')),
             'original_dgl_executed':False,'training':False}
    (directory/'policy_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
