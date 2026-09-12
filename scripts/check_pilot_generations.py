"""Regenerate endpoint predictions and test internal confidence scaling on CPU."""
import argparse,copy,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from scc.developmental_run import configure,predictions
from scc.model import Transformer,ModelConfig
from scc.provenance import atomic_json,file_digest
from scc.recovered_capability import Reader
from scc.recovered_pilot_evaluation import Viewed


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=pathlib.Path,required=True)
    p.add_argument('--measurement',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    configure('cpu',2);state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    c=state.get('configuration') or state['contract']['configuration']
    model=Transformer(ModelConfig(**c['model'])).eval();model.load_state_dict(state['model']);del state
    scaled=copy.deepcopy(model)
    with torch.no_grad():scaled.norm.weight.mul_(.1);scaled.norm.bias.mul_(.1)
    measurement=json.loads(a.measurement.read_text());checks=[]
    # Identity is the predeclared primary reader, not chosen by endpoint utility.
    branch=measurement['branches'][0];assert branch['reader']['sign']==1 and branch['reader']['digit_sources']==list(range(10))
    view=Viewed(model,Reader(),branch['temperature']).eval();control=Viewed(scaled,Reader(),branch['temperature']).eval()
    for layout in ('original','reordered'):
        raw_path=a.measurement.parent/f'predictions-0-{layout}.json';raw=json.loads(raw_path.read_text())
        for rows_key,pred_key in (('rows','predictions'),('target_rows','target_predictions')):
            rows=raw[rows_key];computed=predictions(view,rows);confidence=predictions(control,rows)
            mismatches=[i for i,(x,y) in enumerate(zip(computed,raw[pred_key],strict=True)) if x!=y]
            changed=[i for i,(x,y) in enumerate(zip(computed,confidence,strict=True)) if x!=y]
            checks.append({'layout':layout,'rows':rows_key,'n':len(rows),'cpu_gpu_mismatch_indices':mismatches,
                           'internal_scale_mismatch_indices':changed,'raw_sha256':file_digest(raw_path)})
    passed=all(not c['cpu_gpu_mismatch_indices'] and not c['internal_scale_mismatch_indices'] for c in checks)
    atomic_json(a.output,{'passed':passed,'checkpoint_sha256':file_digest(a.checkpoint),'measurement_sha256':file_digest(a.measurement),
        'script_sha256':file_digest(__file__),'checks':checks,'reader':branch['reader'],'temperature':branch['temperature'],
        'scale':.1,'scope':'Exact CPU regeneration of primary-reader GPU outputs, including targeted rows; confidence control is not rule removal'})
    print(json.dumps({'passed':passed,'predictions':sum(c['n'] for c in checks),'checkpoint':str(a.checkpoint)}))
    if not passed:raise AssertionError('Generation audit discrepancy; see preserved report')

if __name__=='__main__':main()
