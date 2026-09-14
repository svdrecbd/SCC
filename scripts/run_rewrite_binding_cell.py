"""LN-064 custom recurrent rewrite and declared graph-edit counterexamples."""
import argparse
import copy
import json
from pathlib import Path
import platform
import shutil
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scc.learned_binding_bank import LearnedBindingBank,counts
from scc.rewrite_binding_cell import RewriteBindingCell
from scc.provenance import atomic_json,file_digest,snapshot_sources
from scripts.run_learned_binding_bank import data,tensor_ids,score,stream as reference_stream
from scripts.localize_persistent_learning import independent_check

PARENT=ROOT/'artifacts/scc-learned-binding-bank-20260913-v1/full-v1'
CASES=('intact','identity','scale-negative','scale-half','scale-double','controller-edited',
       'controller-silenced','identity-binding','skip-commit','frozen-binding',
       'symmetric-fixed-D','symmetric-normalizer','symmetric-compensated',
       'symbolic-binding','direct-compilation','repair-after-skip','repair-after-erasure')


def pack_state(model):
    state=model.snapshot();bank=state.pop('bank');vectors=bank.reshape(-1,bank.shape[-1])
    unique=[];indices=[]
    for vector in vectors:
        match=next((i for i,v in enumerate(unique) if torch.equal(v,vector)),None)
        if match is None:match=len(unique);unique.append(vector.clone())
        indices.append(match)
    state['bank_values']=torch.stack(unique);state['bank_indices']=torch.tensor(indices)
    state['bank_shape']=list(bank.shape)
    return state


@torch.no_grad()
def run_stream(model,ids,*,fixture=False,capture=True):
    outputs=[];traces=[]
    for j in range(ids.shape[1]):
        if capture and (fixture or j==0):
            out,trace=model.request(ids[:,j],trace=True)
            traces.append({'request_ordinal':j,'ticks':trace})
        else:out=model.request(ids[:,j])
        outputs.append(out)
    return {k:torch.stack([o[k] for o in outputs],1) for k in outputs[0]},traces


def intervene(model,name,proposal,silenced,warm_ids,rows,folder,fixture):
    if name.startswith('scale-'):model.rescale({'scale-negative':-1.,'scale-half':.5,'scale-double':2.}[name])
    elif name not in ('intact','identity'):
        model.install(silenced if name=='controller-silenced' else proposal)
        if name=='identity-binding':model.storage_rule='identity'
        elif name=='skip-commit':model.execution='skip'
        elif name=='frozen-binding':model.storage_rule='frozen'
        elif name=='symmetric-fixed-D':model.recode(1.)
        elif name=='symmetric-normalizer':model.recode(1.,change_normalizer=True)
        elif name=='symmetric-compensated':model.recode(1.,change_normalizer=True,change_writer=True)
        elif name=='symbolic-binding':model.storage_rule='symbolic'
        elif name=='direct-compilation':model.execution='direct'
        elif name=='repair-after-skip':
            model.execution='skip';out,trace=run_stream(model,warm_ids,fixture=fixture)
            torch.save({'post_event_state':pack_state(model),'out':out,'traces':trace},folder/(name+'-start.pt'))
            model.execution='commit';model.storage_rule='symbolic'
        elif name=='repair-after-erasure':
            triggers=[next(r for r in stream if r['family']=='lookup') for stream in rows]
            out,trace=model.request(torch.tensor([r['tokens'] for r in triggers]),trace=True)
            torch.save({'post_event_state':pack_state(model),'out':out,'trace':trace,'requests':triggers},folder/(name+'-start.pt'))
            model.recode(1.,change_normalizer=True,change_writer=True)


def freeze(folder,fixture):
    text=(ROOT/'labnotes.md').read_text().split('<a id="ln-064"></a>')[1].split('\n<a id=')[0].split('\n## Supporting-record')[0]
    (folder/'plan.md').write_text('<a id="ln-064"></a>'+text)
    manifest=snapshot_sources(folder/'source')
    for name in ('scripts/run_rewrite_binding_cell.py','scripts/audit_rewrite_binding_cell.py',
                 'scripts/run_learned_binding_bank.py','scripts/localize_persistent_learning.py',
                 'tests/test_rewrite_binding_cell.py'):
        dest=folder/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        manifest[name]=file_digest(dest)
    atomic_json(folder/'source-manifest.json',manifest)
    names=('trained.pt','coupled-attack/update-0400.pt')
    atomic_json(folder/'parents.json',{str(PARENT/name):file_digest(PARENT/name) for name in names})
    for name in names:
        dest=folder/'parents'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PARENT/name,dest)
    config={'fixture':fixture,'cases':CASES,'width':128,'streams':4,'warmup_seed':17313012,
        'evaluation_seed':17313013,'warmup_per_cell':16,'evaluation_per_cell':128,
        'precisions':['fp32','fp64'],'cpu_threads':2,'wall_seconds':120,'output_limit_bytes':180*1024**2,
        'training_updates':0,'optimizer_repair_updates':0,'storage_sectors':4,
        'parameter_commits_per_request':1,'hidden_commits_per_request':19,
        'phase_rule':'token_id modulo 2; swap sectors 1 and 2 for odd tokens',
        'state_storage':'Lossless unique bank-row vectors with exact row-index reconstruction',
        'runtime':{'python':sys.version,'torch':str(torch.__version__),'platform':platform.platform()}}
    atomic_json(folder/'configuration.json',config);return config


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixture',action='store_true');args=parser.parse_args()
    folder=args.output;folder.mkdir(parents=True,exist_ok=False);started=time.monotonic();torch.set_num_threads(2)
    def timeout(*_):raise TimeoutError('LN-064 120-second wall limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    try:
        config=freeze(folder,args.fixture);width=config['width'];n,_=counts(width)
        full=torch.load(folder/'parents/trained.pt',map_location='cpu',weights_only=True)['physical']
        attack=torch.load(folder/'parents/coupled-attack/update-0400.pt',map_location='cpu',weights_only=True)
        proposal=torch.cat((full[:n],attack['physical'][n:]*attack['gain']))
        silenced=full.clone();silenced[n:]=0;silenced[-1]=-1
        torch.save({'intact':full,'controller_edit':proposal,'silenced':silenced},folder/'proposals.pt')
        warm,rows=data(17313012,16),data(17313013,128)
        if args.fixture:warm=[s[:1] for s in warm];rows=[s[:4] for s in rows]
        # Fixture trigger coverage is explicit and independent of qualification.
        if args.fixture:
            complete=data(17313013,128)
            for i,s in enumerate(rows):
                if not any(r['family']=='lookup' for r in s):s[-1]=next(r for r in complete[i] if r['family']=='lookup')
        for stream in warm+rows:
            for r in stream:independent_check(r);assert r['split']=='validation'
        atomic_json(folder/'requests.json',{'warmup':warm,'evaluation':rows})
        warm_ids,ids=tensor_ids(warm),tensor_ids(rows)
        results=[];warm_models={};baselines={}
        for precision,dtype in (('fp32',torch.float32),('fp64',torch.float64)):
            dest=folder/precision;dest.mkdir()
            reference=LearnedBindingBank(full.to(dtype),width,4);reference_stream(reference,warm_ids)
            ref=reference_stream(reference,ids);torch.save(ref,dest/'reference.pt')
            m=RewriteBindingCell(full.to(dtype),width,4)
            warmout,warmtrace=run_stream(m,warm_ids,fixture=args.fixture)
            torch.save({'out':warmout,'trace':warmtrace,'state':pack_state(m)},dest/'warmup.pt')
            warm_models[precision]=m
            baseline=copy.deepcopy(m);out,traces=run_stream(baseline,ids,fixture=args.fixture)
            baselines[precision]=out
            error=float((out['logits']-ref['logits']).abs().max())
            disagreements=int((out['logits'].argmax(-1)!=ref['logits'].argmax(-1)).sum())
            scores=score(rows,out)
            valid=disagreements==0 and error<=(1e-4 if dtype==torch.float32 else 1e-10)
            result={'precision':precision,'case':'intact','scores':scores,'reference_logit_max':error,
                    'reference_decision_disagreements':disagreements,'numerically_valid':valid,
                    'first_erasure':baseline.first_erasure.tolist()}
            torch.save({'out':out,'traces':traces,'final_state':pack_state(baseline)},dest/'intact.pt')
            results.append(result)
        atomic_json(folder/'qualification.json',results)
        if not args.fixture and not all(r['scores']['qualified'] and r['numerically_valid'] for r in results):
            atomic_json(folder/'summary.json',{'status':'intact-qualification-failed','results':results,'seconds':time.monotonic()-started})
            print('Intact qualification failed; interventions stopped.',flush=True);return
        print(json.dumps({'phase':'intact-qualified' if not args.fixture else 'fixture-baseline','seconds':time.monotonic()-started}),flush=True)
        target=torch.tensor([[r['family']=='lookup' for r in s] for s in rows])
        desired=torch.tensor([True,False,False,True]).expand(4,len(rows[0]),-1).clone();desired[:,:,1]|=target
        for precision,dtype in (('fp32',torch.float32),('fp64',torch.float64)):
            for name in CASES[1:]:
                m=copy.deepcopy(warm_models[precision])
                intervene(m,name,proposal.to(dtype),silenced.to(dtype),warm_ids,rows,folder/precision,args.fixture)
                # Exact current state before execution enables independent continuation replay.
                initial=pack_state(m)
                out,traces=run_stream(m,ids,fixture=args.fixture)
                torch.save({'out':out,'traces':traces,'initial_state':initial,'final_state':pack_state(m)},folder/precision/(name+'.pt'))
                scores=score(rows,out)
                result={'precision':precision,'case':name,'scores':scores,'first_erasure':m.first_erasure.tolist(),
                    'active_state_changes':int(out['active_state_changes'].sum()),
                    'final_bank_nonzero':int(torch.count_nonzero(m.bank)),
                    'final_hidden_nonzero':int(torch.count_nonzero(m.hidden_bank)),
                    'exception_rule_errors':int((out['admitted']!=desired).sum()),
                    'versus_intact_logit_max':float((out['logits']-baselines[precision]['logits']).abs().max()),
                    'qualified_escape':all(c['qualified'] for c in scores['cells'].values())
                        and scores['target_correct_answers']/scores['target_n']>=.95}
                results.append(result)
            print(json.dumps({'phase':precision+'-complete','seconds':time.monotonic()-started}),flush=True)
        size=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
        if size>config['output_limit_bytes']:raise RuntimeError('Output ceiling exceeded')
        atomic_json(folder/'summary.json',{'status':'fixture-complete' if args.fixture else 'complete',
             'results':results,'seconds':time.monotonic()-started,'bytes_before_manifest':size})
        print(json.dumps({'phase':'complete','seconds':time.monotonic()-started,'bytes':size}),flush=True)
    except BaseException as exc:
        atomic_json(folder/'failure.json',{'error':repr(exc),'seconds':time.monotonic()-started});raise
    finally:
        signal.alarm(0)
        atomic_json(folder/'artifact-manifest.json',{str(p.relative_to(folder)):file_digest(p)
            for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='artifact-manifest.json'})


if __name__=='__main__':main()
