"""Collect the declared jobs, audit completions, and run the frozen extra probe."""
import concurrent.futures,json,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scc.provenance import atomic_json
PYTHON=str(ROOT/'.venv/bin/python');WORK=ROOT/'artifacts/scc-pilot-collection-20260911-v1'

def call(*args):return json.loads(subprocess.check_output(['/Users/svdr/.local/bin/gman',*args,'--json'],text=True))
def run(*args):
 p=subprocess.run([PYTHON,*map(str,args)],cwd=ROOT,capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stdout+'\n'+p.stderr)
 return p.stdout.strip()
def poll(entry):return entry['job'],call('job','get',entry['job'])
def main():
 WORK.mkdir(exist_ok=False)
 entries=[{'job':j,'kind':'pilot','label':'seed'+str(s),'destination':f'artifacts/scc-pilot-seed{s}-20260911-v1-result'}
          for j,s in [('job-acnxn',17),('job-hkmi2',41),('job-i55kf',23)]]
 entries.append({'job':'job-qsuyf','kind':'probe','label':'seam-qualified','destination':'artifacts/scc-authorized-replay-seam-20260911-v2-result'})
 done=set();observed={};started=time.monotonic()
 while len(done)<len(entries):
  if time.monotonic()-started>9000:raise TimeoutError('Collector exceeded 150 minutes')
  active=[e for e in entries if e['job'] not in done]
  statuses=dict(concurrent.futures.ThreadPoolExecutor(4).map(poll,active))
  for entry in active:
   job=statuses[entry['job']];status=job['status'];entry['status']=status
   if observed.get(entry['job'])!=status:
    print(json.dumps({'job':entry['job'],'label':entry['label'],'status':status}),flush=True);observed[entry['job']]=status
   if status not in ('succeeded','failed','cancelled','expired'):continue
   if job.get('artifact',{}).get('state')!='ready':continue
   dest=ROOT/entry['destination'];entry['download']=run('scripts/download_gman_result.py',entry['job'],'--output',dest)
   print(entry['download'],flush=True)
   root=dest/'files'/('recovered-pilot' if entry['kind']=='pilot' else 'authorized-replay')
   if status=='succeeded':
    entry['audit']=run('scripts/audit_recovered_pilot.py',root,'--output',dest/'prediction-audit.json');print(entry['audit'],flush=True)
    if entry['kind']=='pilot':
     manifest=json.loads((dest/'manifest.json').read_text());parents={}
     for arm in ('rule_only','early','late','seam_late'):
      name='recovered-pilot/'+arm+'/step-00018000.pt';parents[entry['label']+'-'+arm]={'member':name,'sha256':manifest['files'][name]['sha256']}
     spec=WORK/(entry['label']+'-parents.json');atomic_json(spec,[{'job_id':entry['job'],'parents':parents}])
     folder=ROOT/('artifacts/scc-authorized-replay-'+entry['label']+'-20260911-v1-gpu')
     entry['probe_submit']=run('scripts/submit_authorized_replay.py','--parents',spec,'--output',folder,'--key','scc-authorized-replay-'+entry['label']+'-20260911-v1','--minutes',30)
     new=json.loads((folder/'submit.json').read_text());print(entry['probe_submit'],flush=True)
     entries.append({'job':new['job_id'],'kind':'probe','label':entry['label'],'destination':'artifacts/scc-authorized-replay-'+entry['label']+'-20260911-v1-result'})
   else:
    entry['needs_attention']=True;print(json.dumps({'job':entry['job'],'needs_attention':'Partial output preserved; inspect before resuming'}),flush=True)
   done.add(entry['job'])
  atomic_json(WORK/'status.json',{'entries':entries,'done':sorted(done)})
  if len(done)<len(entries):time.sleep(60)
 print(json.dumps({'status':'all_declared_jobs_collected','entries':entries}),flush=True)
if __name__=='__main__':main()
