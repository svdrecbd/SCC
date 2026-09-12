"""Create a portable ZIP of pilot evidence without duplicating full optimizer states."""
import argparse,hashlib,io,json,pathlib,re,zipfile,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from scc.provenance import file_digest
ROOT=pathlib.Path(__file__).resolve().parents[1]


def tensor_digest(state):
 h=hashlib.sha256()
 for key,value in sorted(state.items()):
  h.update(json.dumps([key,str(value.dtype),list(value.shape)],separators=(',',':')).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser();p.add_argument('--index',type=pathlib.Path,required=True);p.add_argument('--report',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--guide',type=pathlib.Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 index=json.loads(a.index.read_text());files={};mapping={};weight_records={};model_sources={}
 def add(path,name):
  path=path.resolve();assert path.is_file();assert name not in files,(path,name)
  files[name]=path;mapping[path]=name
 for pth in sorted((ROOT/'scc').glob('*.py')):add(pth,'source/scc/'+pth.name)
 for pth in sorted((ROOT/'tests').glob('*.py')):add(pth,'source/tests/'+pth.name)
 for name in ('pyproject.toml','uv.lock','.python-version'):add(ROOT/name,'source/'+name)
 for name in index['scripts']:add(ROOT/'scripts'/name,'source/scripts/'+name)
 for pth in sorted((ROOT/'protocols').glob('SCC_*.md')):
  if any(s in pth.name for s in ('RECOVERED','SEAM','AUTHORIZED_REPLAY','PILOT_RESOURCE','CONSOLIDATED')):add(pth,'source/protocols/'+pth.name)
 for dataset in index['datasets']:
  folder=ROOT/dataset['path'];prefix='evidence/'+dataset['label']
  for pth in sorted(folder.rglob('*')):
   if not pth.is_file() or pth.suffix not in ('.json','.jsonl','.md','.py','.toml','.lock','.npz') and pth.name!='.python-version':continue
   if 'interrupted' in pth.relative_to(folder).parts or pth.suffix=='.pt':continue
   # Dataset roots contain experiment outputs, never submission secrets or job download URLs.
   add(pth,prefix+'/'+str(pth.relative_to(folder)))
 for audit in index['audits']:add(ROOT/audit,'audits/'+audit.replace('/','__'))
 for pth in sorted((ROOT/index['validation_inputs']).iterdir()):
  if pth.is_file():add(pth,'inputs/validation/'+pth.name)
 for pth in sorted((ROOT/index['figures']).iterdir()):
  if pth.suffix in ('.png','.svg','.pdf'):add(pth,'figures/'+pth.name)
 add(ROOT/index['summary'],'summary.json');add(ROOT/index['ledger'],'compute-ledger.json')
 for extra in index.get('extras',[]):add(ROOT/extra['path'],extra['name'])
 for label,relative in index['weights'].items():
  path=(ROOT/relative).resolve();model_sources[label]=path;mapping[path]='weights/'+label+'.pt'
 report=a.report.read_text()
 omissions=[]
 def link(match):
  label,target=match.groups()
  if target.startswith(('https://','http://','#')):return match.group(0)
  clean=target.strip('<>');anchor=''
  if '#' in clean:clean,anchor=clean.split('#',1);anchor='#'+anchor
  absolute=(a.report.parent/clean).resolve()
  if absolute in mapping:return '['+label+']('+mapping[absolute]+anchor+')'
  omissions.append({'label':label,'path':clean});return label+' (full lab archive; outside this compact package)'
 report=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,report)
 manifests={}
 def write(z,name,data):
  if isinstance(data,str):data=data.encode()
  if pathlib.Path(name).suffix in ('.json','.jsonl','.md','.py','.toml','.lock'):
   assert b'X-Amz-Signature=' not in data and b'X-Amz-Credential=' not in data, 'Scoped provider URL in '+name
  z.writestr(name,data);manifests[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(a.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for name,path in sorted(files.items()):write(z,name,path.read_bytes())
  for label,path in model_sources.items():
   state=torch.load(path,map_location='cpu',weights_only=True);config=state.get('configuration') or state['contract']['configuration'];original=file_digest(path);tensor_sha=tensor_digest(state['model'])
   export={'schema_version':1,'configuration':config,'model':state['model'],'original_checkpoint_sha256':original,'model_tensor_sha256':tensor_sha}
   stream=io.BytesIO();torch.save(export,stream);content=stream.getvalue()
   restored=torch.load(io.BytesIO(content),map_location='cpu',weights_only=True);assert tensor_digest(restored['model'])==tensor_sha
   write(z,'weights/'+label+'.pt',content);weight_records[label]={'original_checkpoint_sha256':original,'model_tensor_sha256':tensor_sha,'exported_checkpoint_sha256':hashlib.sha256(content).hexdigest(),'optimizer_included':False}
   del state,export,restored,content
  write(z,'REPORT.md',report);write(z,'README.md',a.guide.read_text())
  write(z,'weights/lineage.json',json.dumps(weight_records,indent=2)+'\n')
  write(z,'scope.json',json.dumps({'omitted_report_links':omissions,'full_training_corpus_included':False,'sealed_test_inputs_are_now_open':True,'optimizer_states_included':False,'originals_preserved':True},indent=2)+'\n')
  write(z,'MANIFEST.json',json.dumps(manifests,indent=2)+'\n')
 with zipfile.ZipFile(a.output) as z:
  assert z.testzip() is None
  expected=json.loads(z.read('MANIFEST.json'))
  for name,record in expected.items():assert hashlib.sha256(z.read(name)).hexdigest()==record['sha256']
  names=set(z.namelist());broken=[]
  for name in ('README.md','REPORT.md'):
   for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',z.read(name).decode()):
    if target.startswith(('http://','https://','#')):continue
    clean=target.strip('<>').split('#',1)[0]
    if clean not in names:broken.append((name,target))
  assert not broken,broken
 receipt={'zip_sha256':file_digest(a.output),'zip_bytes':a.output.stat().st_size,'files':len(names),'weights':len(weight_records),'central_document_links_verified':True,'all_member_hashes_verified':True,'omitted_background_links':len(omissions)}
 a.output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
