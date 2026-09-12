"""Bounded numerical diagnosis or saved benign-repair assay; never polls jobs."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scc.checkpoint import load_checkpoint
from scc.developmental_run import TextBank,configure,environment
from scc.fixed_episode_optimization import check_deadline
from scc.learned_bottleneck import from_checkpoint
from scc.numerical_diagnosis import derivative_sweep
from scc.provenance import atomic_json,digest,file_digest,snapshot_sources,source_manifest
from scc.recovered_pilot_evaluation import measure
from scc.selective_coupling import defaults,make_episode
from scc.varied_coupling import pool_score
from scripts.run_varied_coupling import exclusions
from scripts.run_learned_bottleneck_screen import save_model,scoped_edit,state_fingerprint

PROTOCOL='SCC_REPAIR_ROUND_V1.md'
HELPERS=['scripts/run_varied_coupling.py','scripts/run_learned_bottleneck_screen.py',
         'scripts/run_selective_construction_screen.py','scripts/authorized_replay_probe.py']


def monitors(bank,c,device,smoke=False):
    episodes=[]
    for ordinal in range(95000,95002 if smoke else 95004):
        episodes.append(make_episode(bank,c,ordinal,device,**exclusions(episodes)))
    return episodes


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    snapshot=snapshot_sources(args.output/'source');shutil.copyfile(__file__,args.output/'runner.py')
    shutil.copyfile(ROOT/'protocols'/PROTOCOL,args.output/'protocol.md')
    for path in HELPERS:
        target=args.output/'helpers'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,target)
    frozen={p:file_digest(ROOT/p) for p in HELPERS+['scripts/run_repair_diagnostics.py','protocols/'+PROTOCOL]}
    started=time.monotonic();deadline=started+(6900 if args.mode=='precision' else 3300)
    device=configure(args.device,2 if args.device=='cpu' else 4)
    parents={k:Path(v) for k,v in json.loads(args.parents.read_text()).items()}
    inputs=json.loads(args.inputs.read_text());atomic_json(args.output/'inputs.json',inputs)
    fingerprints={k:file_digest(p) for k,p in parents.items()}
    bank=TextBank(args.data,blocks=2 if args.smoke else 128)
    result={'status':'running','mode':args.mode,'environment':environment(device),'parent_sha256':fingerprints,
            'source_sha256':digest(snapshot),'helper_sha256':frozen,'sealed_test_used':False,
            'positive_scc_result':None,'mechanism_assessment_pending':True,'smoke':args.smoke,
            'evidence_class':'CPU implementation fixture' if args.smoke else 'Open repair diagnosis', 'records':{}}
    atomic_json(args.output/'data-manifest.json',bank.manifest())
    if args.mode=='precision':
        if len(parents)!=1:raise ValueError('Precision diagnosis requires one original foundation')
        initial=from_checkpoint(load_checkpoint(next(iter(parents.values())))).to(device).eval()
        origin=state_fingerprint(initial)
        c=dict(inputs['failed_configuration'])
        if args.smoke:c.update(inner_steps=2,repair_steps=2)
        held=monitors(bank,c,device,args.smoke)
        data=make_episode(bank,c,94400,device,**exclusions(held))
        if not args.smoke:assert data['record']==inputs['failed_episode']
        atomic_json(args.output/'episode.json',{'configuration':c,'record':data['record']})
        for label,dtype in [('fp32',torch.float32),('fp64',torch.float64)]:
            current=copy.deepcopy(initial).to(dtype=dtype)
            current_origin=state_fingerprint(current)
            try:
                with (args.output/(label+'-sweep.jsonl')).open('w') as log:
                    def callback(record):
                        log.write(json.dumps(record)+'\n');log.flush()
                        print(json.dumps({'phase':'precision','dtype':label,'direction':record['direction'],
                                          'epsilon_l2':record['epsilon_l2'],'relative_error':record['relative_error']}),flush=True)
                    kwargs={'epsilons':(1e-3,3e-4,1e-4)} if args.smoke else {}
                    sweep=derivative_sweep(current,c,data,deadline=deadline,callback=callback,**kwargs)
                assert state_fingerprint(current)==current_origin
                if label=='fp32' and not args.smoke:
                    sweep['historical_reference_absolute_error']=abs(sweep['reference']-inputs['failed_reference'])
                    sweep['historical_gradient_norm_absolute_error']=abs(sweep['gradient_norm']-inputs['failed_gradient_norm'])
                checks=sweep.pop('checks')
                sweep.update(completed=True,checks_recorded=len(checks),sweep_log_sha256=file_digest(args.output/(label+'-sweep.jsonl')),
                             check_summary=[{k:x[k] for k in ('direction','epsilon_l2','analytic_slope','numerical_slope','relative_error','same_selected_components','same_clip_pattern')} for x in checks])
                result['records'][label]=sweep
            except (torch.cuda.OutOfMemoryError,TimeoutError) as exc:
                result['records'][label]={'completed':False,'limitation':type(exc).__name__,
                                          'message':str(exc)[:600],'training_gate_changed':False}
            finally:
                del current;gc.collect()
                if device.startswith('cuda'):torch.cuda.empty_cache()
            atomic_json(args.output/'progress.json',result)
        assert state_fingerprint(initial)==origin
    else:
        for case in inputs['cases']:
            check_deadline(deadline)
            label=case['label'];folder=args.output/label;folder.mkdir()
            before_state=load_checkpoint(parents[case['fitted_parent']])
            config=dict(before_state['configuration']);config.update(evaluation_size=4 if args.smoke else 128,
                    text_blocks=2 if args.smoke else 128,edit_phase_steps=2 if args.smoke else 500,edit_batch_size=2 if args.smoke else 16)
            before=from_checkpoint(before_state).to(device).eval()
            repaired=from_checkpoint(load_checkpoint(parents[case['benign_parent']])).to(device).eval()
            original_hashes=[state_fingerprint(m) for m in (before,repaired)]
            c={**defaults(),'inner_scope':case['scope'],'repair_scope':'all',
               'inner_steps':2 if args.smoke else 128,'repair_steps':2 if args.smoke else 64}
            held=monitors(bank,c,device,args.smoke)
            record={'configuration':config,'monitor_configuration':c,'monitor_episodes':[d['record'] for d in held],'scores':{},'probes':{}}
            for name,current in [('fitted',before),('benign_repaired',repaired)]:
                check_deadline(deadline)
                value,details,_=pool_score(current,c,held,deadline=deadline)
                record['scores'][name]={'objective':value,'episodes':details,
                    'measurement':measure(current,bank,config,folder/(name+'-intact'))}
            if not args.smoke:
                record['historical_fitted_score_absolute_error']=abs(record['scores']['fitted']['objective']-case['historical_fitted_score'])
            for scope in ('core','all'):
                probe_folder=folder/('probe-'+scope)
                def callback(current,phase):
                    if phase=='modification':save_model(probe_folder/'modified.pt',current,config)
                    return measure(current,bank,config,probe_folder/phase)
                _,record['probes'][scope]=scoped_edit(repaired,bank,config,scope,probe_folder,deadline,callback)
            assert original_hashes==[state_fingerprint(m) for m in (before,repaired)]
            atomic_json(folder/'result.json',record);result['records'][label]=record
            atomic_json(args.output/'progress.json',result)
            print(json.dumps({'phase':'benign-repair','case':label,'complete':True}),flush=True)
            del before,repaired,before_state,held;gc.collect()
    assert all(file_digest(p)==fingerprints[k] for k,p in parents.items())
    assert source_manifest()==snapshot and all(file_digest(ROOT/p)==sha for p,sha in frozen.items())
    result.update(status='complete',elapsed_seconds=time.monotonic()-started,parent_files_unchanged=True)
    total=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
    if total>512*1024**2:raise RuntimeError('Declared 512 MiB diagnostic output limit exceeded')
    result['output_bytes_before_result']=total;atomic_json(args.output/'result.json',result)
    if os.environ.get('GMN_RESULT_PATH'):
        atomic_json(os.environ['GMN_RESULT_PATH'],{'status':'complete','mode':args.mode,'positive_scc_result':None,'mechanism_assessment_pending':True})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--parents',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--mode',choices=('precision','benign'),required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');main(p.parse_args())
