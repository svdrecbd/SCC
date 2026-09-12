"""Assemble all declared main and separately identified adaptive results."""
import argparse,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.summarize_recovered_pilot import pilot,probe,endpoint
from scc.provenance import atomic_json,file_digest

def read_holdout(folder):
 results={}
 for p in sorted(folder.glob('*-intact/result.json')):
  label=p.parent.name.removesuffix('-intact');before=json.loads(p.read_text());after=json.loads((folder/(label+'-repaired')/'result.json').read_text())
  results[label]={'intact':endpoint(before,before),'repaired':endpoint(after,before),'intact_parent_sha256':before['parent_sha256'],'repaired_parent_sha256':after['parent_sha256']}
 return results

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 result={'pilots':{},'probes':{},'heldout':{},'alignment_consolidation':{},'consolidated_heldout':{},'benign_consolidation_v1':{},'script_sha256':file_digest(__file__)}
 for seed in (17,23,41):
  stem=f'scc-pilot-seed{seed}'+('-completion' if seed==23 else '')+'-20260911-v1-result'
  folder=ROOT/'artifacts'/stem/'files/recovered-pilot';result['pilots'][str(folder.relative_to(ROOT))]=pilot(folder)
 for label,version in [('seed17','v1'),('seed23','v1'),('seed41','v1'),('seam','v2')]:
  folder=ROOT/f'artifacts/scc-authorized-replay-{label}-20260911-{version}-result/files/authorized-replay';result['probes'][str(folder.relative_to(ROOT))]=probe(folder)
  folder=ROOT/f'artifacts/scc-heldout-{label}-20260911-{version}-result/files/heldout';result['heldout'].update(read_holdout(folder))
 for label in ('seed17','seed23','seed41','seam'):
  folder=ROOT/f'artifacts/scc-alignment-consolidated-{label}-20260911-v2-result/files/alignment-consolidated';result['alignment_consolidation'].update(probe(folder))
  folder=ROOT/f'artifacts/scc-consolidated-heldout-{label}-20260911-v1-result/files/consolidated-heldout';result['consolidated_heldout'].update(read_holdout(folder))
 result['benign_consolidation_v1']=probe(ROOT/'artifacts/scc-consolidated-seed17-20260911-v1-result/files/consolidated-replay')
 assert len(result['pilots'])==3 and len(result['heldout'])==len(result['consolidated_heldout'])==13
 result['scope']='Original matched pilot, complete-replay follow-up, and protected-rule consolidation are separate stages. Same-model identity-reader joint scores; minima over correlated contexts/layouts.'
 atomic_json(a.output,result)
 for stage in ('heldout','consolidated_heldout'):
  print(stage)
  for label,r in result[stage].items():
   b=r['intact'];e=r['repaired'];print(label,b['intact_qualification'],round(100*b['benign_min_strict'],3),round(100*e['target_min_strict'],3),round(100*e['benign_min_strict'],3),round(100*e['other_refusal_min'],3),round(100*e['min_text_context_gain_retention'],3))
if __name__=='__main__':main()
