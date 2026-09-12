"""Collect fixed completion/ablation endpoints and launch their frozen confirmations."""
import concurrent.futures,json,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scc.provenance import atomic_json
PY=str(ROOT/'.venv/bin/python');WORK=ROOT/'artifacts/scc-completion-collection-20260911-v1'
def call(*args):return json.loads(subprocess.check_output(['/Users/svdr/.local/bin/gman',*args,'--json'],text=True))
def run(*args):
 p=subprocess.run([PY,*map(str,args)],cwd=ROOT,capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stdout+'\n'+p.stderr)
 print(p.stdout.strip(),flush=True);return p.stdout.strip()
def submit(script,spec,stem,kind,label):
 folder=ROOT/('artifacts/'+stem+'-gpu')
 run('scripts/'+script,'--parents',spec,'--output',folder,'--key',stem,'--minutes',30)
 job=json.loads((folder/'submit.json').read_text())['job_id']
 return {'job':job,'kind':kind,'label':label,'destination':'artifacts/'+stem+'-result'}
def main():
 WORK.mkdir(exist_ok=False)
 entries=[{'job':'job-jaedg','kind':'pilot','label':'seed23','destination':'artifacts/scc-pilot-seed23-completion-20260911-v1-result'},
 {'job':'job-zn9bn','kind':'benign','label':'seed17','destination':'artifacts/scc-consolidated-seed17-20260911-v1-result'}]
 for job,label in [('job-xwcfb','seed17'),('job-7gxeu','seed41'),('job-bai2a','seam')]:
  entries.append({'job':job,'kind':'alignment','label':label,'destination':'artifacts/scc-alignment-consolidated-'+label+'-20260911-v2-result'})
 dirs={'pilot':'recovered-pilot','probe':'authorized-replay','benign':'consolidated-replay','alignment':'alignment-consolidated','heldout':'consolidated-heldout'}
 done=set();observed={};started=time.monotonic();additional=[]
 while len(done)<len(entries):
  if time.monotonic()-started>7000:raise TimeoutError('Completion collector 116 minute limit')
  active=[e for e in entries if e['job'] not in done]
  with concurrent.futures.ThreadPoolExecutor(4) as pool:statuses=list(pool.map(lambda e:call('job','get',e['job']),active))
  for entry,job in zip(active,statuses,strict=True):
   status=job['status'];entry['status']=status
   if observed.get(entry['job'])!=status:print(json.dumps({'job':entry['job'],'kind':entry['kind'],'label':entry['label'],'status':status}),flush=True);observed[entry['job']]=status
   if status not in ('succeeded','failed','cancelled','expired') or job.get('artifact',{}).get('state')!='ready':continue
   dest=ROOT/entry['destination'];run('scripts/download_gman_result.py',entry['job'],'--output',dest)
   if status!='succeeded':entry['needs_attention']=True;done.add(entry['job']);continue
   root=dest/'files'/dirs[entry['kind']]
   run('scripts/audit_recovered_pilot.py',root,'--output',dest/'prediction-audit.json','--split','test' if entry['kind']=='heldout' else 'validation')
   manifest=json.loads((dest/'manifest.json').read_text())['files'];label=entry['label']
   if entry['kind']=='pilot':
    parents={label+'-'+arm:{'member':'recovered-pilot/'+arm+'/step-00018000.pt','sha256':manifest['recovered-pilot/'+arm+'/step-00018000.pt']['sha256']} for arm in ('rule_only','early','late','seam_late')}
    spec=WORK/(label+'-parents.json');atomic_json(spec,[{'job_id':entry['job'],'parents':parents}])
    entries.append(submit('submit_authorized_replay.py',spec,'scc-authorized-replay-'+label+'-20260911-v1','probe',label))
    entries.append(submit('submit_alignment_consolidated.py',spec,'scc-alignment-consolidated-'+label+'-20260911-v2','alignment',label))
   if entry['kind'] in ('pilot','probe'):
    additional.append(entry);atomic_json(ROOT/'artifacts/scc-heldout-collection-20260911-v1/additional-parents.json',additional)
   if entry['kind']=='alignment':
    results=json.loads((root/'result.json').read_text());parents={}
    for parent in results['parents']:
     for suffix,filename in [('intact','consolidated.pt'),('repaired','repair-500.pt')]:
      member='alignment-consolidated/'+parent+'/'+filename;parents[parent+'-'+suffix]={'member':member,'sha256':manifest[member]['sha256']}
    spec=WORK/(label+'-consolidated-parents.json');atomic_json(spec,[{'job_id':entry['job'],'parents':parents}])
    entries.append(submit('submit_consolidated_holdout.py',spec,'scc-consolidated-heldout-'+label+'-20260911-v1','heldout',label))
   done.add(entry['job'])
  atomic_json(WORK/'status.json',{'entries':entries,'done':sorted(done)})
  if len(done)<len(entries):time.sleep(45)
 print(json.dumps({'status':'complete','entries':entries}),flush=True)
if __name__=='__main__':main()
