from pathlib import Path
import hashlib,importlib.util,json,platform,sys
import torch
root=Path(sys.argv[1])
records=[]
for path in sorted(root.glob('*.pt')):
 data=torch.load(path,map_location='cpu',weights_only=True)
 states=data['state_dict']
 records.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'keys':list(data),'node_dim':data.get('node_dim'),'args':data.get('args'),'tensor_count':len(states),'parameters':sum(value.numel() for value in states.values()),'tensors':{name:{'shape':list(value.shape),'dtype':str(value.dtype),'finite':bool(torch.isfinite(value).all())} for name,value in states.items()}})
result={'python':sys.version,'torch':torch.__version__,'platform':platform.platform(),'packages':{name:importlib.util.find_spec(name) is not None for name in ['torch_geometric','torch_scatter','networkx','numpy','scipy','pandas','xxhash','requests','tqdm','jinja2','psutil']},'checkpoints':records}
(root/'checkpoint_inspection.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({**result,'checkpoints':[{key:value for key,value in record.items() if key!='tensors'} for record in records]}))
