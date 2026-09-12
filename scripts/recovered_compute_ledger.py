"""Collect receipted cost and actual local output sizes for this pilot phase."""
import argparse,concurrent.futures,json,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]
PATTERNS=['scc-pilot-20260910-v*-gpu/submit.json','scc-pilot-replications-20260911-v*/*/submit.json','scc-seam-*-20260911-*-gpu/submit.json','scc-authorized-replay-*-20260911-*-gpu/submit.json','scc-heldout-*-20260911-*-gpu/submit.json','scc-pilot-*-completion-20260911-*-gpu/submit.json','scc-consolidated-*-20260911-*-gpu/submit.json','scc-alignment-consolidated-*-20260911-*-gpu/submit.json']

def get(pair):
 job,path=pair;d=json.loads(subprocess.check_output(['/Users/svdr/.local/bin/gman','job','get',job,'--json'],text=True));receipt=d.get('receipt')
 return {'job':job,'label':d.get('label'),'status':d['status'],'submitted_at':d.get('submitted_at'),'finished_at':d.get('finished_at'),
         'charged_usd':receipt['charged_usd_micros']/1e6 if receipt else None,'billed_seconds':receipt.get('billed_seconds') if receipt else None,
         'receipt':receipt,'submission_record':str(path.relative_to(ROOT)),'quoted_max_usd':json.loads(path.read_text()).get('max_cost_usd')}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 files={}
 for pattern in PATTERNS:
  for path in (ROOT/'artifacts').glob(pattern):
   d=json.loads(path.read_text())
   if 'job_id' in d:files[d['job_id']]=path
 jobs=list(concurrent.futures.ThreadPoolExecutor(4).map(get,sorted(files.items())))
 downloads=[]
 for pattern in ('scc-pilot-seed*-20260911-*-result','scc-seam-*-20260911-*-result','scc-authorized-replay-*-20260911-*-result','scc-heldout-*-20260911-*-result','scc-consolidated-*-20260911-*-result','scc-alignment-consolidated-*-20260911-*-result'):
  for path in (ROOT/'artifacts').glob(pattern):
   listed=[p for p in path.rglob('*') if p.is_file()]
   downloads.append({'path':str(path.relative_to(ROOT)),'files':len(listed),'bytes':sum(p.stat().st_size for p in listed),'pytorch_files':sum(p.suffix=='.pt' for p in listed)})
 total=round(sum(j['charged_usd'] or 0 for j in jobs),6)
 result={'previous_phase_receipted_usd':3.20793,'new_receipted_usd':total,'cumulative_receipted_usd':round(total+3.20793,6),'jobs':jobs,
         'all_jobs_terminal':all(j['status'] in ('succeeded','failed','cancelled','expired') for j in jobs),'downloads':downloads,
         'downloaded_bytes':sum(d['bytes'] for d in downloads),'downloaded_pytorch_files':sum(d['pytorch_files'] for d in downloads),
         'notes':['User removed the aggregate compute cap. Per-job maximums are resource ceilings, not actual charges.',
                  'Failed launches/gates are included. Receipt rates may vary across the Thursday/Friday boundary.',
                  'Full prior experiments remain intact. Streamed downloads avoid an extra local TAR copy.',
                  'Download inventory includes engineering fixtures and any preserved incomplete/completion copies; not all .pt files are large models.']}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'jobs':len(jobs),'new_receipted_usd':total,'all_jobs_terminal':result['all_jobs_terminal'],'downloaded_bytes':result['downloaded_bytes']}))
if __name__=='__main__':main()
