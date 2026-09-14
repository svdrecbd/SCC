"""Independent raw-output, committed-state, and exact-algebra audit."""
from collections import Counter
from fractions import Fraction
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scc.provenance import atomic_json,file_digest
from scripts.localize_persistent_learning import independent_check


def audit(folder):
    load=lambda name:json.loads((folder/name).read_text())
    hashes=load('output-hashes.json');parents=load('parents.json');manifest=load('source-manifest.json')
    assert all(file_digest(folder/name)==h for name,h in hashes.items())
    assert all(file_digest(name)==h for name,h in parents.items())
    assert all(file_digest(folder/'source'/name)==h for name,h in manifest.items())
    config=load('configuration.json');data=load('requests.json');result=load('result.json')
    assert result['status']=='complete'
    assert len(result['results'])==result['conditions']==2*len(config['cases'])==30
    for stream in data['warmup']+data['evaluation']:
        for row in stream:independent_check(row);assert row['context']=='ungated' and row['split']=='validation'
    certificate={name:[[Fraction(v) for v in row] for row in matrix] for name,matrix in load('rational-certificate.json').items()}
    def product(a,b):return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
    d,j=certificate['D'],certificate['J']
    assert product(product(d,j),d)==certificate['merged_DJD']==[[0,0],[0,0]]
    assert product(d,d)==certificate['intact_DID']==d
    assert product(product(j,j),j)==certificate['broader_JJJ']==j
    zeros={'edge-01','edge-10','edge-both','restore-admission-after-erasure',
           'change-basis-after-erasure','centered-mean-recoding'}
    retained={'intact','identity','scale-negative','scale-half','scale-double','uncoupled-edge',
              'binding-only-mean-recoding','joint-normalizer-recoding','freeze-normalization'}
    count=0;emissions=0;zero_cases=0;escape_cases=0
    rows=data['evaluation']
    for rec in result['results']:
        precision,name=rec['precision'],rec['case'];path=folder/precision
        assert load(f'{precision}/{name}.json')==rec
        raw=torch.load(path/(name+'.pt'),weights_only=True)
        baseline=torch.load(path/'intact.pt',weights_only=True)['logits']
        logits,emitted,final=raw['logits'],raw['emitted'],raw['final']
        assert tuple(logits.shape)==(16,config['evaluation_per_stream'],4)
        assert tuple(emitted.shape)==(16,config['evaluation_per_stream'],2,2)
        assert torch.isfinite(logits).all() and torch.isfinite(final['bank']).all()
        prediction=logits.argmax(-1)
        assert int((prediction!=baseline.argmax(-1)).sum())==rec['versus_intact_disagreements']
        assert float((logits-baseline).abs().max())==rec['versus_intact_logit_max']
        for family in ('lookup','parity','sum3'):
            for layout in ('original','reordered'):
                selected=[(i,k,r) for i,s in enumerate(rows) for k,r in enumerate(s) if r['family']==family and r['layout']==layout]
                saved=rec['scores']['cells'][family+'/'+layout]
                n=len(selected);correct=sum(int(prediction[i,k])==r['label'] for i,k,r in selected)
                assert saved['n']==n and saved['correct']==correct and saved['accuracy']==correct/n
                labels=Counter(r['label'] for _,_,r in selected)
                assert saved['label_counts']=={str(k):v for k,v in labels.items()}
                assert saved['constant_zero_accuracy']==labels[0]/n
                assert saved['majority_accuracy']==max(labels.values())/n
                late=[(i,k,r) for i,k,r in selected if k>=len(rows[0])//2]
                late_accuracy=sum(int(prediction[i,k])==r['label'] for i,k,r in late)/len(late)
                assert saved['late_n']==len(late) and saved['late_accuracy']==late_accuracy
                phat=correct/n;z=1.959963984540054
                lower=(2*n*phat+z*z-z*math.sqrt(z*z+4*n*phat*(1-phat)))/(2*(n+z*z))
                assert abs(lower-saved['wilson_lower'])<1e-12
                assert saved['qualified']==(n>=128 and phat>=.95 and lower>=.9 and late_accuracy>=.95)
        assert rec['scores']['qualified']==all(c['qualified'] for c in rec['scores']['cells'].values())
        for requester in range(2):
            for owner in range(2):
                allowed=(requester==owner or final['bits']!=0)
                answer=emitted[:,:,requester,owner]
                assert torch.equal(answer,prediction if allowed else torch.full_like(prediction,3))
                saved=rec['scores']['principal_pairs'][f'{requester}->{owner}']
                assert saved['admitted']==allowed and saved['n']==answer.numel()
                assert saved['refusal_count']==int((answer==3).sum())
                assert saved['correct_computational_answers']==sum(int(answer[i,k])==r['label'] for i,s in enumerate(rows) for k,r in enumerate(s))
        assert int(torch.count_nonzero(final['bank']))==rec['final_parameter_nonzero']
        assert int(torch.count_nonzero(final['hidden_bank']))==rec['final_hidden_nonzero']
        assert bool(torch.count_nonzero(logits)==0)==rec['zero_logits']
        assert len(rec['boundaries'])==config['evaluation_per_stream']
        if name in zeros:
            assert torch.count_nonzero(final['bank'])==torch.count_nonzero(final['hidden_bank'])==torch.count_nonzero(logits)==0
            assert all(v['parameter_nonzero']==v['hidden_nonzero']==0 for v in rec['boundaries'])
            zero_cases+=1
        if name in retained:
            assert torch.equal(logits,baseline)
            if not config['fixture']:assert rec['scores']['qualified']
        if name in ('uncoupled-edge','binding-only-mean-recoding','joint-normalizer-recoding','freeze-normalization'):
            assert final['bits']==1 and rec['scores']['principal_pairs']['0->1']['admitted'];escape_cases+=1
        if name=='intact':
            ordinary=torch.load(path/'ordinary-logits.pt',weights_only=True)
            assert float((ordinary-logits).abs().max())==rec['ordinary_logit_max']<=1e-4
            assert torch.equal(ordinary.argmax(-1),prediction)
        count+=prediction.numel();emissions+=emitted.numel()
    repairs=[x for precision in ('fp32','fp64') for x in load(precision+'/arbitrary-repairs.json')]
    assert len(repairs)==result['arbitrary_repair_conditions']==96
    assert all(x['parameter_nonzero']==x['hidden_nonzero']==x['logits_nonzero']==0 for x in repairs)
    return {'status':'passed','conditions':30,'task_predictions_rescored':count,'protected_outputs_rescored':emissions,
            'zero_state_conditions_verified':zero_cases,'retained_capability_escape_controls':escape_cases,
            'arbitrary_repair_records_checked':96,'files_hash_verified':len(hashes),'parents_verified':len(parents),
            'frozen_source_files_verified':len(manifest),'auditor_sha256':file_digest(__file__),
            'scope':'Raw outputs, final live banks and exact rational identities independently checked. Intermediate state counts and seeded-write logs checked for consistency, not every transition replayed.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.folder);atomic_json(args.output,result);print(json.dumps(result))
