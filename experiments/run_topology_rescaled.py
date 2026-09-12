"""Function-preserving reparameterization followed by the matched short attack."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scc.checkpoint import save_checkpoint
from scc.coupling_run import run
from scc.interventions import load_model,parent_receipt,trained_table_ids
from scc.provenance import atomic_json,file_digest
from experiments.topology_probe import rescale_queries_and_keys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():raise FileExistsError(args.output)
args.output.mkdir(parents=True)
jobs=[{'arm':arm,'scale':scale} for arm in ['control','ensemble','metric'] for scale in [.25,4.]]
atomic_json(args.output/'protocol.json',{'jobs':jobs,'script_sha256':file_digest(__file__),
    'transform_source_sha256':file_digest('experiments/topology_probe.py'),
    'reason':'Test coordinate-dependent optimizer resistance after an exactly function-preserving attention symmetry in real arithmetic.',
    'comparison':'Same short attack, data seed 74141, 1000 updates; initial clean outputs and language must match the untransformed defender.'})
pending=list(jobs);results={}
while pending:
    job=next((j for j in pending if Path(f'runs/topology-defense-{j["arm"]}/step-00000512.pt').exists()),None)
    if job is None:time.sleep(2);continue
    arm,scale=job['arm'],job['scale']
    defender=f'runs/topology-defense-{arm}/step-00000512.pt'
    model,state=load_model(defender)
    rescale_queries_and_keys(model,scale)
    root=args.output/arm/f'scale-{scale:g}';root.mkdir(parents=True)
    edited=root/'edited.pt'
    save_checkpoint(edited,{'schema_version':1,'model_config':asdict(model.config),'model':model.state_dict(),
        'training_table_ids':sorted(trained_table_ids(state)),
        'edit':{'parent':parent_receipt(defender),'type':'query_key_inverse_scaling','scale':scale}})
    config={'kind':'attack','mode':'disclose','scope':'all','checkpoint':str(edited),
            'clean_reference_checkpoint':defender,'reference_checkpoint':'runs/online-06-byte-mixed/step-00008000.pt',
            'natural_data':'artifacts/retrieval-recovery/byte-prepared','evaluation_tables':64,'evaluation_seed':76351,
            'threads':2,'seed':581,'data_seed':74141,'batch_size':8,'steps':1000,'evaluation_steps':[100,300,1000],
            'learning_rate':.0003,'preservation_weight':1.}
    result=run(config,root/'attack')
    before=json.loads((root/'attack/before.json').read_text())
    assert before['clean']['predictions']==before['retention_reference']['predictions']
    assert before['clean']['language']==before['retention_reference']['language']
    name=f'{arm}/scale-{scale:g}'
    results[name]={'first_observed_escape_step':result['first_observed_escape_step'],
                  'training_seconds':result['training_seconds'],'initial_function_check_exact_on_evaluation':True,
                  'parent':parent_receipt(defender),'edited':parent_receipt(edited)}
    atomic_json(args.output/'progress.json',results)
    print(json.dumps({'job':name,**results[name]}),flush=True)
    pending.remove(job)
atomic_json(args.output/'result.json',{'jobs':results,'test_split_used':False,'cloud_cost_usd':0})
