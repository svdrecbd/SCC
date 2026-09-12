"""Re-evaluate preserved seed-17 weights and complete its omitted probes."""
import argparse,copy,json,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scc.checkpoint import load_checkpoint
from scc.developmental_run import TextBank,configure,environment
from scc.fixed_episode_optimization import check_deadline
from scc.learned_bottleneck import from_checkpoint
from scc.provenance import atomic_json,file_digest,snapshot_sources,source_manifest
from scc.transition_evaluation import measure
from scc.transition_path import EditStepper
from scripts.run_learned_bottleneck_screen import save_model,state_fingerprint,scoped_edit
PROTOCOL='SCC_TRANSITION_TIMING_V1.md'
EXPECTED='protocols/SCC_TRANSITION_V1_EXPECTED.json'
HELPERS=['scripts/run_learned_bottleneck_screen.py','scripts/run_selective_construction_screen.py','scripts/authorized_replay_probe.py']

def run(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();deadline=start+3300;source=snapshot_sources(a.output/'source');shutil.copyfile(__file__,a.output/'runner.py');shutil.copyfile(ROOT/'protocols'/PROTOCOL,a.output/'protocol.md')
    frozen={n:file_digest(ROOT/n) for n in HELPERS+[EXPECTED,'scripts/complete_transition_evaluation.py','protocols/'+PROTOCOL]}
    for n in HELPERS+[EXPECTED]:
        p=a.output/'helpers'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
    device=configure(a.device,2 if a.device=='cpu' else 4);parents=json.loads(a.parents.read_text());assert set(parents)=={'candidate','modified'}
    parent_state=load_checkpoint(parents['candidate']);config=dict(parent_state['configuration']);parent=from_checkpoint(parent_state).to(device).eval();origin=state_fingerprint(parent)
    config.update(evaluation_size=4 if a.smoke else 128,text_blocks=2 if a.smoke else 128,edit_phase_steps=2 if a.smoke else 500,edit_batch_size=2 if a.smoke else 16,smoke=a.smoke)
    bank=TextBank(a.data,blocks=config['text_blocks']);modified=from_checkpoint(load_checkpoint(parents['modified'])).to(device).eval();steps=config['edit_phase_steps']
    assert all(torch.equal(p,dict(parent.named_parameters())[n]) for n,p in modified.named_parameters() if not n.startswith('cells.'))
    result={'schema':'evaluation-completion/v1','parent_sha256':{n:file_digest(p) for n,p in parents.items()},'configuration':config,'environment':environment(device),'source':source,'helper_sha256':frozen,'sealed_test_used':False,'old_failed_job_status_preserved':True,'training_repeated':False,'smoke':a.smoke,'measurements':{}}
    result['measurements']['intact']=measure(parent,bank,config,a.output/'intact')
    result['measurements']['core-modified']=measure(modified,bank,config,a.output/'core-modified');atomic_json(a.output/'progress.json',result)
    print(json.dumps({'phase':'saved-core-evaluation','complete':True}),flush=True)
    stepper=EditStepper(modified,bank,batch_size=config['edit_batch_size'],skip_modifications=steps)
    expected=json.loads((ROOT/EXPECTED).read_text())
    if not a.smoke:
        assert result['parent_sha256']['modified']==expected['seed17_modified_sha256'] and stepper.skipped_chain==expected['seed17_modification_chain']
    result['resumed_stream']={'skipped_modification_steps':steps,'prefix_chain':stepper.skipped_chain,'fresh_repair_optimizer':True}
    with (a.output/'core-repair-steps.jsonl').open('w') as log:
        for i in range(steps):
            check_deadline(deadline);record=stepper.advance();log.write(json.dumps(record)+'\n');log.flush()
    assert all(torch.equal(p,dict(parent.named_parameters())[n]) for n,p in modified.named_parameters() if not n.startswith('cells.'))
    save_model(a.output/'core-repaired.pt',modified,config,repair_steps=steps,edit_scope='core')
    result['measurements']['core-repaired']=measure(modified,bank,config,a.output/'core-repaired');atomic_json(a.output/'progress.json',result)
    for label,benign in [('all',False),('benign',True)]:
        check_deadline(deadline);folder=a.output/('probe-'+label)
        def callback(m,phase):
            if phase=='modification':save_model(folder/'modified.pt',m,config)
            return measure(m,bank,config,folder/phase)
        _,result[label]=scoped_edit(parent,bank,config,'all',folder,deadline,callback,benign)
        atomic_json(a.output/'progress.json',result);print(json.dumps({'phase':'probe','label':label,'complete':True}),flush=True)
    assert state_fingerprint(parent)==origin and source_manifest()==source and all(file_digest(ROOT/n)==h for n,h in frozen.items())
    size=sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file());assert size<1024**3
    result.update(status='complete',elapsed_seconds=time.monotonic()-start,output_bytes_before_result=size,positive_scc_result=None)
    atomic_json(a.output/'result.json',result)
    if os.environ.get('GMN_RESULT_PATH'):atomic_json(os.environ['GMN_RESULT_PATH'],{'status':'complete','positive_scc_result':None,'training_repeated':False})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parents',type=Path,required=True);p.add_argument('--data',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true');run(p.parse_args())
