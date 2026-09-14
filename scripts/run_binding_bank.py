"""Frozen two-bank construction, destructive edits, and explicit broader bypasses."""
import argparse
from collections import Counter
import copy
from fractions import Fraction as Q
import json
import math
from pathlib import Path
import platform
import shutil
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scc.binding_bank import BindingBank
from scc.persistent_reference import PersistentGRU
from scc.persistent_tasks import evaluation_requests
from scc.provenance import atomic_json,file_digest,snapshot_sources
from scripts.localize_persistent_learning import independent_check

PARENT=ROOT/'artifacts/scc-persistent-reference-implementation-20260912-v1/stabilization-v1/trained.pt'
CASES=('intact','identity','scale-negative','scale-half','scale-double',
       'edge-01','edge-10','edge-both','restore-admission-after-erasure',
       'change-basis-after-erasure','uncoupled-edge','binding-only-mean-recoding',
       'centered-mean-recoding','joint-normalizer-recoding','freeze-normalization')


def data(seed,per_cell):
    original=evaluation_requests(seed,per_cell=per_cell,streams=16)
    serial=[original[i][j] for j in range(len(original[0])) for i in range(16)
            if original[i][j].context=='ungated']
    assert len(serial)%16==0
    return [[r.record() for r in serial[i::16]] for i in range(16)]


def exact_certificate():
    d=[[Q(1,2),Q(-1,2)],[Q(-1,2),Q(1,2)]]
    j=[[Q(1,2)]*2 for _ in range(2)];i=[[Q(1),Q(0)],[Q(0),Q(1)]]
    def mul(a,b):return [[sum(a[r][k]*b[k][c] for k in range(2)) for c in range(2)] for r in range(2)]
    result={'D':d,'J':j,'intact_DID':mul(mul(d,i),d),'merged_DJD':mul(mul(d,j),d),
            'broader_JJJ':mul(mul(j,j),j)}
    assert result['intact_DID']==d and result['merged_DJD']==[[0,0],[0,0]] and result['broader_JJJ']==j
    return {k:[[str(x) for x in row] for row in matrix] for k,matrix in result.items()}


def metrics(rows,logits,emitted,admission):
    cells={}
    prediction=logits.argmax(-1)
    for family in ('lookup','parity','sum3'):
        for layout in ('original','reordered'):
            selected=[(i,j,r) for i,s in enumerate(rows) for j,r in enumerate(s)
                      if r['family']==family and r['layout']==layout]
            n=len(selected);correct=sum(int(prediction[i,j])==r['label'] for i,j,r in selected)
            accuracy=correct/n
            z=1.959963984540054
            lower=(accuracy+z*z/(2*n)-z*math.sqrt(accuracy*(1-accuracy)/n+z*z/(4*n*n)))/(1+z*z/n)
            late=[(i,j,r) for i,j,r in selected if j>=len(rows[0])//2]
            late_accuracy=sum(int(prediction[i,j])==r['label'] for i,j,r in late)/len(late)
            labels=Counter(r['label'] for _,_,r in selected)
            cells[family+'/'+layout]={'n':n,'correct':correct,'accuracy':accuracy,'wilson_lower':lower,
                'late_n':len(late),'late_accuracy':late_accuracy,'label_counts':dict(labels),
                'constant_zero_accuracy':labels[0]/n,'majority_accuracy':max(labels.values())/n,
                'uniform_chance':1/(2 if family=='parity' else 3),
                'qualified':n>=128 and accuracy>=.95 and lower>=.9 and late_accuracy>=.95}
    pairs={}
    for r in range(2):
        for o in range(2):
            answers=emitted[:,:,r,o]
            count=sum(int(answers[i,j])==row['label'] for i,stream in enumerate(rows) for j,row in enumerate(stream))
            pairs[f'{r}->{o}']={'admitted':bool(admission[r,o]),'n':answers.numel(),
                'correct_computational_answers':count,'refusal_count':int((answers==3).sum())}
    return {'qualified':all(v['qualified'] for v in cells.values()),'cells':cells,'principal_pairs':pairs}


@torch.no_grad()
def run_stream(model,ids):
    ys=[];emitted=[];norms=[]
    for ordinal in range(ids.shape[1]):
        y=model.request(ids[:,ordinal]);ys.append(y)
        emitted.append(torch.stack([torch.stack([model.emit(y,r,o) for o in range(2)],-1) for r in range(2)],-2))
        norms.append({'boundary':ordinal+1,'parameter_nonzero':int(torch.count_nonzero(model.bank)),
                      'hidden_nonzero':int(torch.count_nonzero(model.hidden_bank))})
    return torch.stack(ys,1),torch.stack(emitted,1),norms


def intervene(model,name):
    event={'case':name,'pre_edit_requests':model.requests}
    if name=='scale-negative':model.rescale(-1)
    elif name=='scale-half':model.rescale(.5)
    elif name=='scale-double':model.rescale(2)
    elif name=='edge-01':model.bits=1
    elif name=='edge-10':model.bits=2
    elif name=='edge-both':model.bits=3
    elif name=='restore-admission-after-erasure':
        model.bits=1;model.normalize();event['erased_before_repair']=int(torch.count_nonzero(model.bank))==0
        model.bits=0
    elif name=='change-basis-after-erasure':
        model.bits=1;model.normalize();event['erased_before_repair']=int(torch.count_nonzero(model.bank))==0
        model.recode([1.,1.],change_normalizer=True)
    elif name=='uncoupled-edge':model.projection='uncoupled';model.bits=1
    elif name=='binding-only-mean-recoding':
        model.projection='binding_only';model.recode([1.,1.]);model.bits=1
    elif name=='centered-mean-recoding':model.recode([1.,1.]);model.bits=1
    elif name=='joint-normalizer-recoding':model.recode([1.,1.],change_normalizer=True);model.bits=1
    elif name=='freeze-normalization':model.projection='none';model.bits=1
    elif name not in ('intact','identity'):raise ValueError(name)
    event.update(bits=model.bits,projection=model.projection,
                 normalizer_basis=model.normalizer_basis.tolist(),reader_basis=model.reader_basis.tolist(),gain=model.gain,
                 operator=model.operator().tolist(),admission=model.admission().tolist())
    return event


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--fixture',action='store_true')
    args=p.parse_args();folder=args.output;folder.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.set_num_threads(2)
    def timeout(*_):raise TimeoutError('Declared 600-second wall limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    try:
        text=(ROOT/'labnotes.md').read_text().split('<a id="ln-057"></a>')[1].split('\n<a id=')[0].split('\n## Supporting-record')[0]
        (folder/'plan.md').write_text('<a id="ln-057"></a>'+text)
        manifest=snapshot_sources(folder/'source')
        for name in ('scripts/run_binding_bank.py','scripts/audit_binding_bank.py','scripts/localize_persistent_learning.py','tests/test_binding_bank.py'):
            dest=folder/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
            manifest[name]=file_digest(ROOT/name)
        atomic_json(folder/'source-manifest.json',manifest)
        parent_hash=file_digest(PARENT);atomic_json(folder/'parents.json',{str(PARENT):parent_hash})
        warm,rows=data(17313003,16),data(17313004,128)
        if args.fixture:warm=[s[:1] for s in warm];rows=[s[:2] for s in rows]
        for stream in warm+rows:
            for row in stream:independent_check(row);assert row['split']=='validation' and row['context']=='ungated'
        atomic_json(folder/'requests.json',{'warmup':warm,'evaluation':rows})
        atomic_json(folder/'rational-certificate.json',exact_certificate())
        warm_ids=torch.tensor([[r['tokens'] for r in s] for s in warm]);ids=torch.tensor([[r['tokens'] for r in s] for s in rows])
        saved=torch.load(PARENT,weights_only=True,map_location='cpu');parameters=saved['model'];width=saved['width'];del saved
        atomic_json(folder/'configuration.json',{'fixture':args.fixture,'cases':list(CASES),'width':width,'streams':16,
            'warmup_per_stream':warm_ids.shape[1],'evaluation_per_stream':ids.shape[1],
            'evaluation_seed':17313004,'warmup_seed':17313003,'arbitrary_write_seed':17313005,
            'arbitrary_writes_per_mask':16,'precisions':['fp32','fp64'],'cpu_threads':2,
            'wall_limit_seconds':600,'output_limit_bytes':150*1024**2,
            'runtime':{'python':sys.version,'torch':str(torch.__version__),'platform':platform.platform()},
            'scope':'Fixed two-sector centered normalization; broader normalizer and freeze edits are separate controls.'})
        results=[];baselines={};all_repairs=[]
        for label,dtype in (('fp32',torch.float32),('fp64',torch.float64)):
            dest=folder/label;dest.mkdir()
            typed={k:v.to(dtype) for k,v in parameters.items()}
            ordinary=PersistentGRU(width).to(dtype);ordinary.load_state_dict(typed)
            with torch.no_grad():
                _,ordinary_hidden=ordinary(warm_ids)
                ordinary_logits,_=ordinary(ids,ordinary_hidden)
            torch.save(ordinary_logits,dest/'ordinary-logits.pt')
            warm_model=BindingBank(typed,width,16);run_stream(warm_model,warm_ids)
            torch.save(warm_model.snapshot(),dest/'warm-bank.pt')
            for name in CASES:
                model=copy.deepcopy(warm_model);event=intervene(model,name)
                y,emitted,norms=run_stream(model,ids)
                if name=='intact':baselines[label]=y
                result={'precision':label,'case':name,'intervention':event, 'scores':metrics(rows,y,emitted,model.admission()),
                        'versus_intact_disagreements':int((y.argmax(-1)!=baselines[label].argmax(-1)).sum()),
                        'versus_intact_logit_max':float((y-baselines[label]).abs().max()),
                        'zero_logits':bool(torch.count_nonzero(y)==0),'boundaries':norms,
                        'final_parameter_nonzero':int(torch.count_nonzero(model.bank)),
                        'final_hidden_nonzero':int(torch.count_nonzero(model.hidden_bank))}
                if name=='intact':
                    result['ordinary_logit_max']=float((y-ordinary_logits).abs().max())
                    result['ordinary_decision_disagreements']=int((y.argmax(-1)!=ordinary_logits.argmax(-1)).sum())
                    assert result['ordinary_logit_max']<=1e-4 and result['ordinary_decision_disagreements']==0
                    if not args.fixture:assert result['scores']['qualified']
                torch.save({'logits':y,'emitted':emitted,'final':model.snapshot()},dest/(name+'.pt'))
                atomic_json(dest/(name+'.json'),result);results.append(result)
            repairs=[]
            for bits in (1,2,3):
                for trial in range(16):
                    model=copy.deepcopy(warm_model);model.bits=bits
                    generator=torch.Generator().manual_seed(17313005+100*bits+trial)
                    model.bank=torch.randn(model.bank.shape,generator=generator,dtype=dtype)
                    model.hidden_bank=torch.randn(model.hidden_bank.shape,generator=generator,dtype=dtype)
                    y=model.request(ids[:,0])
                    entry={'bits':bits,'trial':trial,'seed':17313005+100*bits+trial,
                           'parameter_nonzero':int(torch.count_nonzero(model.bank)),
                           'hidden_nonzero':int(torch.count_nonzero(model.hidden_bank)),
                           'logits_nonzero':int(torch.count_nonzero(y))}
                    assert entry['parameter_nonzero']==entry['hidden_nonzero']==entry['logits_nonzero']==0
                    repairs.append(entry)
            atomic_json(dest/'arbitrary-repairs.json',repairs);all_repairs.extend(repairs)
            print(json.dumps({'completed_precision':label,'conditions':len(CASES),'elapsed_seconds':time.monotonic()-started}),flush=True)
        assert file_digest(PARENT)==parent_hash
        assert all(file_digest(ROOT/n)==h and file_digest(folder/'source'/n)==h for n,h in manifest.items())
        size=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
        assert size<=150*1024**2
        atomic_json(folder/'result.json',{'status':'complete','elapsed_seconds':time.monotonic()-started,
            'conditions':len(results),'arbitrary_repair_conditions':len(all_repairs),'results':results,
            'cross_precision_baseline':{'logit_max':float((baselines['fp32'].double()-baselines['fp64']).abs().max()),
                'decision_disagreements':int((baselines['fp32'].argmax(-1)!=baselines['fp64'].argmax(-1)).sum())},
            'parents_unchanged':True,'source_unchanged':True,'bytes_before_summary':size})
        atomic_json(folder/'output-hashes.json',{str(p.relative_to(folder)):file_digest(p) for p in sorted(folder.rglob('*')) if p.is_file()})
        print(json.dumps({'status':'complete','elapsed_seconds':time.monotonic()-started}),flush=True)
    except BaseException as exc:
        atomic_json(folder/'failure.json',{'type':type(exc).__name__,'message':str(exc),'elapsed_seconds':time.monotonic()-started})
        raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
