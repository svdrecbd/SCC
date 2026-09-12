"""Submit the frozen complete-replay probe against hash-identified GMAN parents."""
import argparse,hashlib,io,json,pathlib,shlex,subprocess,tarfile,urllib.request

ROOT=pathlib.Path(__file__).resolve().parents[1]
def api(*args):
    p=subprocess.run(['/Users/svdr/.local/bin/gman',*args,'--json'],capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stderr)
    return json.loads(p.stdout)
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parents',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--key',required=True);p.add_argument('--minutes',type=int,default=30)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    specs=json.loads(a.parents.read_text());save(a.output/'parents.json',specs)
    paths=sorted((ROOT/'scc').glob('*.py'))+[ROOT/n for n in ('pyproject.toml','uv.lock','.python-version','scripts/alignment_consolidated_probe.py','protocols/SCC_ALIGNMENT_CONSOLIDATED_REPLAY_V2.md')]
    archive=a.output/'source-overlay.tar';manifest={}
    with tarfile.open(archive,'w') as t:
        for path in paths:
            data=path.read_bytes();name=str(path.relative_to(ROOT));manifest[name]=hashlib.sha256(data).hexdigest()
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;t.addfile(info,io.BytesIO(data))
    save(a.output/'overlay-manifest.json',manifest)
    content=archive.read_bytes();digest=hashlib.sha256(content).hexdigest()
    context=api('api','post','/contexts','--data',json.dumps({'sha256':digest,'size_bytes':len(content)}))
    with urllib.request.urlopen(urllib.request.Request(context['upload']['url'],data=content,headers=context['upload']['headers'],method=context['upload']['method']),timeout=60) as response:response.read()
    api('api','post','/contexts/'+context['context_id']+'/finalize')
    ready=api('api','get','/contexts/'+context['context_id']);save(a.output/'overlay-context.json',{'id':context['context_id'],'sha256':digest,'bytes':len(content)})
    downloads=[]
    for spec in specs:
        job=api('job','get',spec['job_id']);art=job['artifact'];assert art['state']=='ready'
        metadata=api('api','get','/artifacts/'+art['artifact_id'])
        downloads.append({'url':metadata['download']['url'],'sha256':art['sha256'],'parents':spec['parents']})
    loader='''import pathlib,urllib.request,tarfile,os,hashlib,runpy,sys,json,shutil
p=pathlib.Path('/tmp/scc-source.tar')
urllib.request.urlretrieve(os.environ['SCC_OVERLAY_URL'],p)
assert hashlib.sha256(p.read_bytes()).hexdigest()==os.environ['SCC_OVERLAY_SHA256']
with tarfile.open(p) as t:t.extractall('/workspace',filter='data')
parents={}
for j,spec in enumerate(json.loads(os.environ['SCC_PARENTS'])):
 p=pathlib.Path('/tmp/parents-'+str(j)+'.tar');urllib.request.urlretrieve(spec['url'],p)
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 assert h.hexdigest()==spec['sha256']
 with tarfile.open(p) as t:
  for label,definition in spec['parents'].items():
   matches=[m for m in t.getmembers() if m.name.removeprefix('./')==definition['member']]
   assert len(matches)==1 and matches[0].isfile(),definition['member']
   target=pathlib.Path('/tmp/parent-'+label+'.pt')
   with t.extractfile(matches[0]) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
   assert hashlib.sha256(target.read_bytes()).hexdigest()==definition['sha256']
   assert label not in parents;parents[label]=str(target)
 p.unlink()
pathlib.Path('/tmp/parents.json').write_text(json.dumps(parents))
sys.argv=['alignment_consolidated_probe.py','--output','/output/alignment-consolidated','--data','/workspace/data','--parents','/tmp/parents.json','--device','cuda']
runpy.run_path('/workspace/scripts/alignment_consolidated_probe.py',run_name='__main__')
'''
    (a.output/'loader.py').write_text(loader)
    request={'chip':'h100','chip_count':1,'context_id':'ctx-983c199b','command':'python -c '+shlex.quote(loader),
      'env':{'SCC_OVERLAY_URL':ready['download']['url'],'SCC_OVERLAY_SHA256':digest,'SCC_PARENTS':json.dumps(downloads)},
      'max_duration_minutes':a.minutes,'build_timeout_minutes':30,'mission':'scc-developmental-coupling',
      'idempotency_key':a.key,'label':'SCC protected-rule and capability consolidation: '+', '.join(label for spec in specs for label in spec['parents']),
      'resume':'none','max_restarts':0,'checks':{'success':{'check':{'type':'file_exists','path':'/output/alignment-consolidated/result.json'}}}}
    redacted=json.loads(json.dumps(request));redacted['env']['SCC_OVERLAY_URL']='<signed source URL>';redacted['env']['SCC_PARENTS']=json.dumps([{**s,'url':'<signed artifact URL>'} for s in downloads]);save(a.output/'request.json',redacted)
    result=api('api','post','/jobs','--data',json.dumps(request));save(a.output/'submit.json',result)
    print(json.dumps({k:result.get(k) for k in ('job_id','status','max_cost_usd')}))

if __name__=='__main__':main()
