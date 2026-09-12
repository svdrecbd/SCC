"""Submit and audit frozen held-out endpoints after their fixed probes finish."""
import json,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scc.provenance import atomic_json
PYTHON=str(ROOT/'.venv/bin/python');WORK=ROOT/'artifacts/scc-heldout-collection-20260911-v1'
def call(*args):return json.loads(subprocess.check_output(['/Users/svdr/.local/bin/gman',*args,'--json'],text=True))
def run(*args):
 p=subprocess.run([PYTHON,*map(str,args)],cwd=ROOT,capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stdout+'\n'+p.stderr)
 return p.stdout.strip()
def main():
 WORK.mkdir(exist_ok=False);submitted={'seam'};done=set();jobs=[{'label':'seam','job':'job-wc69k'}];started=time.monotonic()
 while len(done)<4:
  if time.monotonic()-started>9000:raise TimeoutError('Held-out collector exceeded 150 minutes')
  original=json.loads((ROOT/'artifacts/scc-pilot-collection-20260911-v1/status.json').read_text())['entries']
  override=WORK/'additional-parents.json'
  if override.exists():original+=json.loads(override.read_text())
  for seed in ('seed17','seed23','seed41'):
   if seed in submitted:continue
   parents=[e for e in original if e['label']==seed and e.get('status')=='succeeded' and (ROOT/e['destination']/'manifest.json').exists()]
   if not any(e['kind']=='pilot' for e in parents) or not any(e['kind']=='probe' for e in parents):continue
   specs=[]
   for kind in ('pilot','probe'):
    entry=next(e for e in parents if e['kind']==kind);manifest=json.loads((ROOT/entry['destination']/'manifest.json').read_text());names={}
    for arm in ('rule_only','early','late','seam_late'):
     member=('recovered-pilot/'+arm+'/step-00018000.pt') if kind=='pilot' else ('authorized-replay/'+seed+'-'+arm+'/repair-500.pt')
     names[seed+'-'+arm+('-intact' if kind=='pilot' else '-repaired')]={'member':member,'sha256':manifest['files'][member]['sha256']}
    specs.append({'job_id':entry['job'],'parents':names})
   spec_path=WORK/(seed+'-parents.json');atomic_json(spec_path,specs)
   folder=ROOT/('artifacts/scc-heldout-'+seed+'-20260911-v1-gpu')
   print(run('scripts/submit_recovered_holdout.py','--parents',spec_path,'--output',folder,'--key','scc-heldout-'+seed+'-20260911-v1','--minutes',30),flush=True)
   job=json.loads((folder/'submit.json').read_text());jobs.append({'label':seed,'job':job['job_id']});submitted.add(seed)
  for entry in jobs:
   if entry['label'] in done:continue
   job=call('job','get',entry['job']);entry['status']=job['status']
   if job['status'] not in ('succeeded','failed','cancelled','expired') or job.get('artifact',{}).get('state')!='ready':continue
   dest=ROOT/('artifacts/scc-heldout-'+entry['label']+'-20260911-v1-result');entry['destination']=str(dest.relative_to(ROOT))
   print(run('scripts/download_gman_result.py',entry['job'],'--output',dest),flush=True)
   if job['status']=='succeeded':print(run('scripts/audit_recovered_pilot.py',dest/'files/heldout','--output',dest/'prediction-audit.json','--split','test'),flush=True)
   else:print(json.dumps({'label':entry['label'],'needs_attention':job['status']}),flush=True)
   done.add(entry['label'])
  atomic_json(WORK/'status.json',{'jobs':jobs,'submitted':sorted(submitted),'done':sorted(done)})
  if len(done)<4:time.sleep(60)
 print(json.dumps({'status':'all_heldout_jobs_collected','jobs':jobs}),flush=True)
if __name__=='__main__':main()
