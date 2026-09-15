"""LN-115 executable separation/repacking screen using qualified inherited weights."""
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
from scc.coordinate_maintenance import CoordinateMaintenance
from scc.provenance import atomic_json,file_digest,snapshot_sources
from scripts.run_learned_binding_bank import score,tensor_ids
from scripts.diagnose_separated_binding import rescore


def execute(parent,output):
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    def timeout(*_):raise TimeoutError('LN-115 300-second cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(300)
    torch.set_num_threads(2)
    try:
        config={'plan':'LN-115','training_updates':0,'width':128,'streams':4,
            'persistent_hidden_values':128,'persistent_coordinate_metadata_ints':1,
            'new_learned_parameters':0,'inherited_learned_parameters':80517,
            'wall_seconds':300,'output_limit_bytes':256*1024**2,'torch':str(torch.__version__),
            'python':sys.version,'platform':platform.platform(),
            'scope':'Structural screen of reversible maintenance, not newly trained SCC'}
        atomic_json(output/'configuration.json',config)
        notes=(ROOT/'labnotes.md').read_text()
        (output/'plan.md').write_text('### LN-115'+notes.split('### LN-115')[1].split('\n<a id="ln-')[0].split('\n## Supporting-record')[0])
        sources=snapshot_sources(output/'source')
        for source in (ROOT/'scripts').glob('*.py'):
            relative=source.relative_to(ROOT);target=output/'source'/relative
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
            sources[str(relative)]=file_digest(target)
        atomic_json(output/'source-manifest.json',sources)
        manifest=json.loads((parent/'artifact-manifest.json').read_text());hashes={}
        for name in ('trained.pt','requests.json'):
            assert file_digest(parent/name)==manifest[name]
            target=output/'inputs'/name;target.parent.mkdir(exist_ok=True,parents=True)
            shutil.copyfile(parent/name,target);hashes[name]=file_digest(target)
        atomic_json(output/'input-manifest.json',hashes)
        data=json.loads((output/'inputs/requests.json').read_text())
        rows=data['evaluation'];ids=tensor_ids(rows);warm=tensor_ids(data['warmup'])
        payload=torch.load(output/'inputs/trained.pt',weights_only=True)['physical']
        results=[];precisions={}
        cases=('intact','reference','benign_recode','output_only','fixed_coordinates','repacked','reader_mismatch')
        for dtype in (torch.float32,torch.float64):
            model=CoordinateMaintenance(payload.to(dtype),128,4)
            for j in range(warm.shape[1]):model.request(warm[:,j])
            torch.save({'encoded':model.encoded,'key':model.key,'payload':model.payload},output/f'warm-{dtype}.pt')
            baseline=None;precisions[str(dtype)]={}
            for case in cases:
                edited=copy.deepcopy(model)
                if case=='reference':edited.repack()
                if case=='benign_recode':edited.recode()
                if case=='output_only':edited.selective_exception=True
                if case=='fixed_coordinates':
                    edited.repack();edited.mode='fixed';edited.selective_exception=True
                if case=='repacked':edited.repack();edited.selective_exception=True
                if case=='reader_mismatch':edited.key=(edited.key+128)%256
                outputs=[edited.request(ids[:,j]) for j in range(ids.shape[1])]
                out={k:torch.stack([r[k] for r in outputs],1) for k in outputs[0]}
                assert all(torch.isfinite(v).all() for v in out.values())
                torch.save(out,output/f'{case}-{dtype}.pt')
                metrics=rescore(rows,out) if edited.selective_exception else score(rows,out)
                if case=='intact':
                    baseline=out
                    assert metrics['qualified'],'Inherited intact model must qualify before screen interpretation'
                equality={k:torch.equal(out[k],baseline[k]) for k in ('logits','hidden','policy_logits','admitted','encoded','key')}
                if case!='reader_mismatch':
                    assert equality['logits'] and equality['hidden'] and equality['policy_logits']
                if case in ('intact','reference','benign_recode'):assert equality['admitted']
                if case=='output_only':assert equality['encoded'] and equality['key']
                if edited.selective_exception:assert metrics['diagnostic_recovery_qualified']
                precisions[str(dtype)][case]=(out['logits'].argmax(-1),out['admitted'])
                result={'case':case,'precision':str(dtype),'metrics':metrics,'equal_to_intact':equality,
                        'distinct_maintenance_keys':len(out['maintenance_key'].unique())}
                results.append(result);atomic_json(output/'partial-results.json',results)
        for case in cases:
            a,b=precisions['torch.float32'][case],precisions['torch.float64'][case]
            assert all(torch.equal(x,y) for x,y in zip(a,b)),case
        for name,h in sources.items():assert file_digest(ROOT/name)==h
        for name,h in hashes.items():assert file_digest(parent/name)==file_digest(output/'inputs'/name)==h
        assert sum(p.stat().st_size for p in output.rglob('*') if p.is_file() and not p.name.startswith('._'))<config['output_limit_bytes']
        atomic_json(output/'summary.json',{'status':'complete','candidate_rejected':True,
            'reason':'Permission branch separation and live-state compilation preserve capability',
            'seconds':time.monotonic()-started,'results':results,'precision_decisions_agree':True})
        print(json.dumps({'status':'complete','candidate_rejected':True,'seconds':time.monotonic()-started}))
    except BaseException as exc:
        atomic_json(output/'failure.json',{'error':repr(exc),'seconds':time.monotonic()-started});raise
    finally:
        signal.alarm(0)
        atomic_json(output/'artifact-manifest.json',{str(p.relative_to(output)):file_digest(p)
            for p in sorted(output.rglob('*')) if p.is_file() and p.name!='artifact-manifest.json' and not p.name.startswith('._')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();execute(a.parent,a.output)
