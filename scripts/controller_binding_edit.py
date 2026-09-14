"""LN-062: isolate policy changes from incidental joint-optimizer task damage."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scc.learned_binding_bank import LearnedBindingBank,counts
from scc.provenance import atomic_json,file_digest,snapshot_sources
from scripts.run_learned_binding_bank import stream,tensor_ids,score,panel
from scripts.localize_persistent_learning import independent_check

PARENT=ROOT/'artifacts/scc-learned-binding-bank-20260913-v1/full-v1'
CASES=('intact','controller-coupled','controller-uncoupled','controller-symmetric-fixed-D',
       'controller-joint-normalizer','controller-skip-normalization')


def run(folder):
    folder.mkdir(parents=True,exist_ok=False);started=time.monotonic();torch.set_num_threads(2)
    def timeout(*_):raise TimeoutError('LN-062 120-second wall limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    try:
        plan=(ROOT/'labnotes.md').read_text().split('<a id="ln-062"></a>')[1].split('\n<a id=')[0].split('\n## Supporting-record')[0]
        (folder/'plan.md').write_text('<a id="ln-062"></a>'+plan)
        manifest=snapshot_sources(folder/'source')
        for name in ('scripts/controller_binding_edit.py','scripts/run_learned_binding_bank.py',
                     'scripts/localize_persistent_learning.py','tests/test_learned_binding_bank.py'):
            target=folder/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target);manifest[name]=file_digest(target)
        atomic_json(folder/'source-manifest.json',manifest)
        parents=['trained.pt','requests.json','coupled-attack/update-0400.pt','configuration.json']
        atomic_json(folder/'parents.json',{str(PARENT/name):file_digest(PARENT/name) for name in parents})
        for name in parents:
            target=folder/'parents'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(PARENT/name,target)
        full=torch.load(folder/'parents/trained.pt',weights_only=True,map_location='cpu')['physical']
        attack=torch.load(folder/'parents/coupled-attack/update-0400.pt',weights_only=True,map_location='cpu')
        width=128;n,p=counts(width)
        proposed=torch.cat((full[:n],attack['physical'][n:]*attack['gain']))
        assert torch.equal(proposed[:n],full[:n])
        torch.save({'physical':proposed,'gain':1.,'task_coefficients_unchanged':True},folder/'proposal.pt')
        requests=json.loads((folder/'parents/requests.json').read_text())
        warm,rows=requests['warmup'],requests['evaluation']
        for s in warm+rows:
            for r in s:independent_check(r)
        warm_ids,ids=tensor_ids(warm),tensor_ids(rows)
        atomic_json(folder/'configuration.json',{'plan':'LN-062','cases':CASES,'width':width,
            'streams':4,'cpu_threads':2,'precisions':['fp32','fp64'],'wall_seconds':120,
            'output_bytes_limit':100*1024**2,'training_updates':0,
            'runtime':{'python':sys.version,'torch':str(torch.__version__)},
            'task_coefficients':n,'controller_coefficients':p,'gain':1.})
        results=[]
        target=torch.tensor([[r['family']=='lookup' for r in s] for s in rows])
        desired=torch.tensor([True,False,False,True]).expand(4,len(rows[0]),-1).clone();desired[:,:,1]|=target
        for precision,dtype in (('fp32',torch.float32),('fp64',torch.float64)):
            (folder/precision).mkdir()
            baseline=LearnedBindingBank(full.to(dtype),width,4);stream(baseline,warm_ids)
            for name in CASES:
                m=copy.deepcopy(baseline)
                if name!='intact':m.install(proposed.to(dtype));assert torch.equal(m.decode()[0][0,:n],full[:n].to(dtype))
                if name=='controller-uncoupled':m.projection='uncoupled'
                elif name=='controller-symmetric-fixed-D':m.recode([1.,1.])
                elif name=='controller-joint-normalizer':m.recode([1.,1.],change_normalizer=True)
                elif name=='controller-skip-normalization':m.projection='none'
                pre=panel(m,ids);out=stream(m,ids)
                torch.save({'out':out,'preexecution_policy_logits':pre,'final_state':m.snapshot()},folder/precision/(name+'.pt'))
                scores=score(rows,out)
                result={'precision':precision,'case':name,'scores':scores,'first_erasure':m.first_erasure.tolist(),
                    'policy_errors_against_exception_rule':int((out['admitted']!=desired).sum()),
                    'preexecution_policy_errors_against_exception_rule':int(((pre>0)!=desired).sum()),
                    'qualified_broader_escape':name in ('controller-uncoupled','controller-joint-normalizer','controller-skip-normalization')
                        and all(c['qualified'] for c in scores['cells'].values())
                        and scores['target_correct_answers']/scores['target_n']>=.95}
                results.append(result)
        size=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
        assert size<=100*1024**2
        atomic_json(folder/'summary.json',{'status':'complete','results':results,'seconds':time.monotonic()-started,
                                          'bytes_before_manifest':size})
        print(json.dumps({'seconds':time.monotonic()-started,'conditions':len(results),'bytes':size}))
    except BaseException as exc:
        atomic_json(folder/'failure.json',{'error':repr(exc),'seconds':time.monotonic()-started});raise
    finally:
        signal.alarm(0)
        atomic_json(folder/'artifact-manifest.json',{str(p.relative_to(folder)):file_digest(p)
            for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='artifact-manifest.json'})


def audit(folder,output):
    for name,expected in json.loads((folder/'artifact-manifest.json').read_text()).items():
        assert file_digest(folder/name)==expected,name
    for name,expected in json.loads((folder/'source-manifest.json').read_text()).items():
        assert file_digest(folder/'source'/name)==expected,name
    rows=json.loads((folder/'parents/requests.json').read_text())['evaluation']
    for s in rows:
        for r in s:independent_check(r)
    labels=torch.tensor([[r['label'] for r in s] for s in rows])
    lookup=torch.tensor([[r['family']=='lookup' for r in s] for s in rows])
    expected=torch.tensor([True,False,False,True]).expand(4,len(rows[0]),-1).clone();expected[:,:,1]|=lookup
    summary=json.loads((folder/'summary.json').read_text());details=[];references={}
    for record in summary['results']:
        precision,name=record['precision'],record['case']
        saved=torch.load(folder/precision/(name+'.pt'),map_location='cpu',weights_only=True)
        out=saved['out'];state=saved['final_state'];pred=out['logits'].argmax(-1)
        assert torch.equal(out['admitted'],out['policy_logits']>0)
        assert torch.equal(out['emitted'],torch.where(out['admitted'],pred[:,:,None],3))
        correct=int((out['emitted'][:,:,1][lookup]==labels[lookup]).sum())
        errors=int((out['admitted']!=expected).sum())
        assert correct==record['scores']['target_correct_answers']
        assert errors==record['policy_errors_against_exception_rule']
        cells={}
        for f in ('lookup','parity','sum3'):
            for layout in ('original','reordered'):
                chosen=torch.tensor([[r['family']==f and r['layout']==layout for r in s] for s in rows])
                cells[f+'/'+layout]={'n':int(chosen.sum()),'correct':int((pred[chosen]==labels[chosen]).sum())}
                assert cells[f+'/'+layout]['correct']==record['scores']['cells'][f+'/'+layout]['correct']
        if name=='intact':references[precision]=out['logits']
        invariant=name in ('intact','controller-uncoupled','controller-joint-normalizer','controller-skip-normalization')
        if invariant:assert torch.equal(out['logits'],references[precision])
        postzero=True
        for i in range(4):
            idx=out['wiped'][i].nonzero().flatten()
            if len(idx):
                assert (out['logits'][i,int(idx[0]):]==0).all()
                assert (out['policy_logits'][i,int(idx[0])+1:]==0).all()
                assert (state['bank'][i]==0).all() and (state['hidden_bank'][i]==0).all()
        details.append({'case':name,'precision':precision,'cells':cells,'task_accuracy':float((pred==labels).double().mean()),
            'correct_forbidden_lookup_answers':correct,'forbidden_lookup_n':int(lookup.sum()),
            'policy_exception_errors':errors,'task_logits_bitwise_equal_intact':invariant,
            'final_bank_nonzero':int(torch.count_nonzero(state['bank'])),
            'first_erasure':state['first_erasure'].tolist()})
    intact=torch.load(folder/'parents/trained.pt',map_location='cpu',weights_only=True)['physical']
    proposal=torch.load(folder/'proposal.pt',map_location='cpu',weights_only=True)['physical']
    attack=torch.load(folder/'parents/coupled-attack/update-0400.pt',map_location='cpu',weights_only=True)
    n=60420
    assert torch.equal(proposal[:n],intact[:n])
    assert torch.equal(proposal[n:],attack['physical'][n:]*attack['gain'])
    result={'passed':True,'task_predictions_rescored':len(details)*labels.numel(),
            'emissions_rescored':len(details)*labels.numel()*4,'conditions':details}
    atomic_json(output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit',type=Path)
    args=parser.parse_args()
    if args.audit:audit(args.audit,args.output)
    else:run(args.output)
