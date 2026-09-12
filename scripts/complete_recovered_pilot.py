"""Finish only missing work after a preserved wall-limit interruption."""
import argparse,json,os,pathlib,shutil,sys,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from scc.checkpoint import load_checkpoint
from scc.developmental_run import TextBank,configure
from scc.model import ModelConfig,Transformer
from scc.provenance import atomic_json,file_digest,source_manifest
from scc.recovered_pilot import train,readiness
from scc.recovered_pilot_evaluation import challenge


def main():
 p=argparse.ArgumentParser();p.add_argument('--prior',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--data',required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 shutil.copytree(a.prior,a.output);shutil.copyfile(__file__,a.output/'completion-runner.py')
 protocol=pathlib.Path(__file__).resolve().parents[1]/'protocols/SCC_PILOT_RESOURCE_COMPLETION_V1.md';shutil.copyfile(protocol,a.output/'completion-protocol.md')
 c=json.loads((a.output/'configuration.json').read_text());expected=json.loads((a.output/'source/source_manifest.json').read_text())
 assert source_manifest()==expected
 device=configure('cuda',4);bank=TextBank(a.data,c['text_blocks']);started=time.monotonic();deadline=started+3500
 gate=readiness(bank,c,a.output/'completion-readiness',device);print(json.dumps({'completion_readiness':gate}),flush=True)
 results={'arms':{},'positive_scc_result':False,'evidence_class':'Declared resource completion, no extra effective training','completion':{'prior_artifact_sha256':os.environ['SCC_PRIOR_SHA256'],'new_training':[],'repeated_challenges':[],'reused_arms':[],'protocol_sha256':file_digest(protocol)}}
 interrupted=a.output/'interrupted';interrupted.mkdir(exist_ok=False)
 for arm in ('rule_only','early','late','seam_late'):
  folder=a.output/arm;challenges=a.output/('challenges-'+arm)
  existing=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {}
  if existing.get('completed_steps')==18000:
   result=existing
   if (challenges/'result.json').exists():
    result['challenges']=json.loads((challenges/'result.json').read_text());results['arms'][arm]=result;results['completion']['reused_arms'].append(arm);continue
   state=load_checkpoint(folder/'step-00018000.pt');model=Transformer(ModelConfig(**c['model'])).to(device).eval();model.load_state_dict(state['model']);del state
  else:
   if folder.exists():shutil.move(str(folder),str(interrupted/arm))
   model,result=train(bank,c,arm,folder,device,deadline=deadline);results['completion']['new_training'].append(arm)
  if challenges.exists():shutil.move(str(challenges),str(interrupted/('challenges-'+arm)))
  result['challenges']=challenge(model,bank,c,challenges,deadline);results['completion']['repeated_challenges'].append(arm);results['arms'][arm]=result;del model
  atomic_json(a.output/'completion-progress.json',results)
  assert sum(f.stat().st_size for f in a.output.rglob('*') if f.is_file())<c['output_bytes_limit']
 assert len({r['ordinary_chain'] for r in results['arms'].values()})==1
 assert results['arms']['early']['meta_chain']==results['arms']['late']['meta_chain']
 assert source_manifest()==expected
 results.update(status='complete',elapsed_seconds=time.monotonic()-started,source_unchanged=True)
 atomic_json(a.output/'result.json',results)
 if os.environ.get('GMN_RESULT_PATH'):atomic_json(os.environ['GMN_RESULT_PATH'],{'status':'complete','completion':results['completion']})
if __name__=='__main__':main()
