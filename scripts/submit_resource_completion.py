"""Submit a source-identical completion of a wall-limited pilot job."""
import argparse,hashlib,io,json,pathlib,shlex,subprocess,tarfile,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]
def api(*args):
 p=subprocess.run(['/Users/svdr/.local/bin/gman',*args,'--json'],capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr)
 return json.loads(p.stdout)
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('job');p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--key',required=True);a=p.parse_args()
 prior=api('job','get',a.job);logs=api('job','logs',a.job);out=logs.get('log',{}).get('output','')
 assert prior['status']=='failed' and 'TimeoutError' in out and ('wall budget' in out or 'wall limit' in out), 'Resource-only completion requires the original wall-budget failure'
 art=prior['artifact'];assert art['state']=='ready';a.output.mkdir(parents=True,exist_ok=False)
 save(a.output/'prior-job.json',prior);save(a.output/'prior-logs.json',logs)
 archive=a.output/'completion-overlay.tar'
 with tarfile.open(archive,'w') as t:
  for name in ('scripts/complete_recovered_pilot.py','protocols/SCC_PILOT_RESOURCE_COMPLETION_V1.md'):
   data=(ROOT/name).read_bytes();info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;t.addfile(info,io.BytesIO(data))
 data=archive.read_bytes();sha=hashlib.sha256(data).hexdigest()
 context=api('api','post','/contexts','--data',json.dumps({'sha256':sha,'size_bytes':len(data)}))
 with urllib.request.urlopen(urllib.request.Request(context['upload']['url'],data=data,headers=context['upload']['headers'],method=context['upload']['method']),timeout=60) as response:response.read()
 api('api','post','/contexts/'+context['context_id']+'/finalize');ready=api('api','get','/contexts/'+context['context_id']);parent=api('api','get','/artifacts/'+art['artifact_id'])
 loader='''import pathlib,urllib.request,hashlib,tarfile,shutil,json,runpy,sys,os
p=pathlib.Path('/tmp/completion-overlay.tar');urllib.request.urlretrieve(os.environ['SCC_OVERLAY_URL'],p)
assert hashlib.sha256(p.read_bytes()).hexdigest()==os.environ['SCC_OVERLAY_SHA256']
with tarfile.open(p) as t:t.extractall('/workspace',filter='data')
p=pathlib.Path('/tmp/prior.tar');urllib.request.urlretrieve(os.environ['SCC_PRIOR_URL'],p);h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
assert h.hexdigest()==os.environ['SCC_PRIOR_SHA256']
with tarfile.open(p) as t:t.extractall('/tmp/prior',filter='data')
prior=pathlib.Path('/tmp/prior/recovered-pilot')
shutil.copytree(prior/'source','/workspace',dirs_exist_ok=True)
c=json.loads((prior/'configuration.json').read_text());target=pathlib.Path('/workspace/protocols')/c['protocol'];target.parent.mkdir(exist_ok=True);shutil.copyfile(prior/'protocol.md',target)
sys.argv=['complete_recovered_pilot.py','--prior',str(prior),'--output','/output/recovered-pilot','--data','/workspace/data']
runpy.run_path('/workspace/scripts/complete_recovered_pilot.py',run_name='__main__')
'''
 (a.output/'loader.py').write_text(loader)
 request={'chip':'h100','chip_count':1,'context_id':'ctx-983c199b','command':'python -c '+shlex.quote(loader),'env':{'SCC_OVERLAY_URL':ready['download']['url'],'SCC_OVERLAY_SHA256':sha,'SCC_PRIOR_URL':parent['download']['url'],'SCC_PRIOR_SHA256':art['sha256']},'max_duration_minutes':60,'build_timeout_minutes':30,'mission':'scc-developmental-coupling','idempotency_key':a.key,'label':'SCC resource-only completion of '+a.job,'resume':'none','max_restarts':0,'checks':{'success':{'check':{'type':'file_exists','path':'/output/recovered-pilot/result.json'}}}}
 redacted=json.loads(json.dumps(request));redacted['env']['SCC_OVERLAY_URL']='<signed source URL>';redacted['env']['SCC_PRIOR_URL']='<signed artifact URL>';save(a.output/'request.json',redacted)
 result=api('api','post','/jobs','--data',json.dumps(request));save(a.output/'submit.json',result);print(json.dumps({k:result.get(k) for k in ('job_id','status','max_cost_usd')}))
if __name__=='__main__':main()
