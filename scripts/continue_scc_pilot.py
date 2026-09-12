"""Longer matched continuation of the preserved strong SCC pilot and control."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tarfile
import urllib.request

import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from scc.causal_interventions import intervene,replaced_predictions,score
from scc.checkpoint import load_checkpoint,save_checkpoint
from scc.developmental_gpu import probes
from scc.developmental_metrics import compare_collapse
from scc.developmental_run import TextBank,Streams,configure,evaluate,modify
from scc.developmental_tasks import evaluation_rows
from scc.differentiable_modify import differentiable_modification
from scc.full_gradient_pilot import train
from scc.gradient_diagnostics import query_batches,evaluate_objectives
from scc.model import ModelConfig,Transformer
from scc.provenance import atomic_json,file_digest,snapshot_sources


def diagnostics(model,bank,config):
    was_training=model.training;model.eval();records=[]
    for ordinal in range(1100,1104):
        with sdpa_kernel(SDPBackend.MATH):
            altered,stream=differentiable_modification(model,bank,config,ordinal,'cuda',create_graph=False)
            query,refusals,floors=query_batches(stream,bank)
            with torch.no_grad():values,losses,gate=evaluate_objectives(model,altered,query,refusals,floors)
        records.append({'ordinal':ordinal,'value':float(values['average']),'gate':float(gate),
                        'capability_nll':{k:float(v) for k,v in losses.items()}})
    quick=evaluate(model,bank,evaluation_rows(32));quick.pop('qualification')
    model.train(was_training)
    return {'held_meta_episodes':records,'quick_intact':quick,'scope':'Fixed development episodes and32-problem diagnostics, not qualification'}


def main():
    device=configure('cuda');output=Path('/output/full-gradient-continuation');output.mkdir(parents=True,exist_ok=False)
    snapshot_sources(output/'source');(output/'runner.py').write_text(Path(__file__).read_text())
    archive=Path('/tmp/pilot.tar');h=hashlib.sha256()
    with urllib.request.urlopen(os.environ['SCC_PARENT_ARCHIVE_URL'],timeout=60) as r,archive.open('wb') as out:
        while chunk:=r.read(1024*1024):out.write(chunk);h.update(chunk)
    if h.hexdigest()!=os.environ['SCC_PARENT_ARCHIVE_SHA256']:raise ValueError('Parent archive hash mismatch')
    names={f'full-gradient-pilot/{arm}/step-00001000.pt' for arm in ('rule_only','full_10')}
    names.add('full-gradient-pilot/untrained-reference.json');found=set()
    with tarfile.open(archive) as t:
        for member in t:
            name=member.name.removeprefix('./')
            if name in names:t.extract(member,'/tmp/parents',filter='data');found.add(name)
    if found!=names:raise ValueError('Parent artifact missing required checkpoints')
    parents=Path('/tmp/parents/full-gradient-pilot');untrained=json.loads((parents/'untrained-reference.json').read_text())
    atomic_json(output/'untrained-reference.json',untrained)
    bank=TextBank('/workspace/data');results={'arms':{},'causal_mechanism_established':False,
        'parent_archive_sha256':h.hexdigest(),'scope':'Development continuation; no test split'}
    chains=[]
    for arm,weight in (('rule_only',0.),('full_10',1.)):
        parent=parents/arm/'step-00001000.pt';initial=load_checkpoint(parent)
        config=copy.deepcopy(initial['contract']['configuration'])
        config.update(steps=4000,checkpoint_every=1000,ordinal_base=20100,continue_optimizer=True)
        if any(bank.manifest()[k]!=initial['contract']['text'][k] for k in ('data_sha256','evaluation_indices')):
            raise ValueError('Continuation data differs')
        bank.floors=dict(initial['contract']['text']['floors'])
        root=output/('monitor-'+arm);root.mkdir()
        def monitor(model,step):
            record=diagnostics(model,bank,config);atomic_json(root/f'step-{step:08d}.json',record)
            print(json.dumps({'arm':arm,'monitor_step':step,'held_meta_mean':sum(x['value'] for x in record['held_meta_episodes'])/4,
                'quick_task_exact':{k:v['exact'] for k,v in record['quick_intact']['tasks'].items()}}),flush=True)
        model=Transformer(ModelConfig(**config['model'])).cuda().eval();model.load_state_dict(initial['model']);monitor(model,0);del model
        model,result=train(initial,bank,config,output/arm,weight,file_digest(parent),device,checkpoint_callback=monitor)
        chains.append(result['ordinary_chain']);qualified=result['validation']['qualification']['passed'] and result['reordered_validation']['qualification']['passed']
        probe_dir=output/('probes-'+arm);probe_result=probes(model,bank,config,probe_dir,untrained)
        if not qualified:
            stream=Streams(bank,config['data_seed']+900000,device,config['batch_size'],config['reordered_probability']);attacked=model
            for stage,(steps,replay) in enumerate(((100,.5),(200,.5),(1000,3.))):
                attacked=modify(attacked,stream,steps,.001,replay)
                post=evaluate(attacked,bank,evaluation_rows(128));reordered=evaluate(attacked,bank,evaluation_rows(128,reordered=True))
                atomic_json(probe_dir/f'stage-{stage}.json',{'stage':stage,'steps':steps,'evaluation':post,'reordered_evaluation':reordered,
                    'collapse':compare_collapse(result['validation'],post,untrained['validation']),
                    'reordered_collapse':compare_collapse(result['reordered_validation'],reordered,untrained['reordered']),
                    'collapse_in_both_renderings':False,'diagnostic_only':True})
                save_checkpoint(probe_dir/f'stage-{stage}.pt',{'schema_version':1,'model':attacked.state_dict(),'stream':stream.state_dict(),'configuration':config})
            probe_result={'status':'diagnostic_probes_complete_unqualified_parent'}
        for site in ('head/0/1','mlp/0'):
            root_site=output/('causal-'+arm)/site.replace('/','-');root_site.mkdir(parents=True)
            for label,reordered in (('original',False),('reordered',True)):
                rows=evaluation_rows(128,reordered=reordered)
                with intervene(model,site):lesion=evaluate(model,bank,rows)
                atomic_json(root_site/f'lesion-{label}.json',lesion)
                atomic_json(root_site/f'opposite_permission-{label}.json',score(rows,replaced_predictions(model,rows,site,'opposite_permission')))
        results['arms'][arm]={'qualified_both_layouts':qualified,'ordinary_chain':result['ordinary_chain'],'probes':probe_result}
        atomic_json(output/'progress.json',results)
    if len(set(chains))!=1:raise AssertionError('Continuation ordinary streams differ')
    results['status']='full_gradient_continuation_complete';atomic_json(output/'result.json',results)
    if os.environ.get('GMN_RESULT_PATH'):atomic_json(os.environ['GMN_RESULT_PATH'],results)
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
