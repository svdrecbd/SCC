"""Measure saved developmental stages against the actual trained edit procedure."""
import argparse,copy,json,os,pathlib,shutil,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scc.checkpoint import load_checkpoint,save_checkpoint
from scc.coupling import nll
from scc.developmental_run import TextBank,configure,environment
from scc.differentiable_modify import adam_unroll
from scc.model import ModelConfig,Transformer
from scc.pilot_objectives import episode
from scc.provenance import atomic_json,file_digest,snapshot_sources,source_manifest
from scc.recovered_pilot_evaluation import measure
from scripts.authorized_replay_probe import adapt

def materialize(model,parameters):
 changed=copy.deepcopy(model)
 with torch.no_grad():
  for name,p in changed.named_parameters():p.copy_(parameters[name])
 return changed.eval()

def short_endpoints(model,config,data):
 modifications,repairs,_,_,_=data
 losses=[lambda p,a=a,b=b,f=f:(nll(model,p,a)+config['inner_replay']*nll(model,p,b)/f)/(1+config['inner_replay']) for a,b,f in modifications]
 changed=adam_unroll(dict(model.named_parameters()),losses,lr=config['inner_lr'],eps=1e-4,create_graph=False)
 repaired=adam_unroll(changed,[lambda p,b=b:nll(model,p,b) for b in repairs],lr=config['repair_lr'],eps=1e-4,create_graph=False)
 return materialize(model,changed),materialize(model,repaired)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--data',required=True);p.add_argument('--parents',type=pathlib.Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);source=snapshot_sources(a.output/'source');shutil.copyfile(__file__,a.output/'runner.py')
 shutil.copyfile(ROOT/'scripts/authorized_replay_probe.py',a.output/'complete-replay-runner.py')
 protocol=ROOT/'protocols/SCC_DEVELOPMENTAL_DEPENDENCY_AUDIT_V1.md';shutil.copyfile(protocol,a.output/'protocol.md')
 device=configure(a.device,4 if a.device=='cuda' else 2);bank=TextBank(a.data,128);started=time.monotonic();deadline=started+1650
 result={'parents':{},'environment':environment(device),'positive_scc_result':False,'protocol_sha256':file_digest(protocol),'scope':'Open saved-stage diagnosis, no defender training or sealed test'}
 for label,path in json.loads(a.parents.read_text()).items():
  if time.monotonic()>deadline:raise TimeoutError('Declared diagnosis wall limit')
  saved=load_checkpoint(path);c=saved.get('configuration') or saved['contract']['configuration'];sha=file_digest(path)
  assert c['inner_steps']==8 and c['repair_steps']==2
  model=Transformer(ModelConfig(**c['model'])).to(device).eval();model.load_state_dict(saved['model']);del saved
  folder=a.output/label;folder.mkdir();record={'parent_sha256':sha,'configuration':c,'measurements':{},'short_episode_data':{}}
  record['measurements']['intact']=measure(model,bank,c,folder/'intact')
  for ordinal in (449,90449):
   data=episode(bank,c,ordinal,device);record['short_episode_data'][str(ordinal)]=data[-1]
   changed,repaired=short_endpoints(model,c,data)
   for stage,m in [('modified8',changed),('repaired2',repaired)]:
    name=f'episode-{ordinal}-{stage}';record['measurements'][name]=measure(m,bank,c,folder/name)
   del changed,repaired,data
  if not label.endswith('-5000'):
   def callback(changed,name):
    if name in ('modification-500','repair-500'):
     record['measurements']['complete-'+name]=measure(changed,bank,c,folder/('complete-'+name))
     if name=='repair-500':save_checkpoint(folder/'complete-repair-500.pt',{'schema_version':1,'configuration':c,'model':changed.state_dict()})
   changed=adapt(model,bank,173905,callback,deadline);del changed
  assert file_digest(path)==sha;atomic_json(folder/'result.json',record);result['parents'][label]=record
  atomic_json(a.output/'progress.json',result);print(json.dumps({'parent':label,'diagnosis':'complete'}),flush=True);del model
  if sum(f.stat().st_size for f in a.output.rglob('*') if f.is_file())>1024**3:raise RuntimeError('Declared 1 GiB output limit')
 assert source_manifest()==source
 result.update(status='complete',elapsed_seconds=time.monotonic()-started);atomic_json(a.output/'result.json',result)
 if os.environ.get('GMN_RESULT_PATH'):atomic_json(os.environ['GMN_RESULT_PATH'],{'status':'complete','parents':list(result['parents'])})
if __name__=='__main__':main()
