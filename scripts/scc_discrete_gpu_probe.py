"""Short CUDA check of the frozen discrete construction and finite-edit package."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scc.coupling import nll
from scc.developmental_run import configure, TextBank, Streams
from scc.discrete_models import DiscreteModel, discrete_config
from scc.discrete_objective import discrete_objective
from scc.discrete_train import guarded_construction_step
from scc.finite_edit import FiniteEditStepper
from scc.portfolio_objective import make_contraction_episode
from scc.provenance import atomic_json, source_manifest


def inline_summary(result):
    summary = {k:result[k] for k in ('ok','source_files_verified','data_files_verified',
               'elapsed_seconds','runtime_downloads','scope')}
    summary['checks'] = [{k:row[k] for k in ('bits','hard','scope','accepted',
                          'gradient_norm','peak_cuda_memory_bytes','finite_edit_checked')}
                         for row in result['checks']]
    summary['full_result_location'] = 'discrete-check/result.json in returned artifact'
    if len((json.dumps(summary, indent=2, sort_keys=True)+'\n').encode()) > 32*1024:
        raise ValueError('Compact probe result exceeds provider inline limit')
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--parents', type=Path, required=True)
    p.add_argument('--device', choices=['cuda'], required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    device = configure(a.device, 4)
    assert json.loads(a.parents.read_text()) == {}
    expected = json.loads((ROOT/'transport-source-manifest.json').read_text())
    for name, sha in expected.items():
        assert hashlib.file_digest((ROOT/name).open('rb'),'sha256').hexdigest() == sha, name
    assert source_manifest() == {n:v for n,v in expected.items()
        if n.startswith('scc/') or n in ('pyproject.toml','uv.lock','.python-version')}
    data = json.loads((ROOT/'transport-data-manifest.json').read_text())
    for name,sha in data.items():
        assert hashlib.file_digest((a.data/name).open('rb'),'sha256').hexdigest() == sha, name
    bank = TextBank(a.data, 2)
    results = []
    start = time.monotonic()
    for bits in (32,128):
        for hard in (False,True):
            for ordinal in (3,4):
                torch.manual_seed(23)
                model = DiscreteModel(discrete_config(bits,hard)).to(device).eval()
                fixed = {n:b.clone() for n,b in model.named_buffers()}
                torch.cuda.reset_peak_memory_stats()
                # Exercise the full ordinary batch size as well as the meta episode.
                ordinary = Streams(bank,101,device,64,.5).task('lookup','authorized')
                optimizer = torch.optim.AdamW(model.parameters(),lr=3e-6,foreach=False)
                loss = nll(model,dict(model.named_parameters()),ordinary)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                episode = make_contraction_episode(bank,ordinal,device,batch_size=2,inner_steps=8)
                record = guarded_construction_step(model,episode,ordinary)
                value,details = discrete_objective(model,episode,create_graph=False)
                assert torch.isfinite(value)
                assert abs(float(value.detach())-record['after']) <= 1e-6*max(1.,abs(record['after']))
                assert record['proposal_gradient_norm'] > 0
                if record['accepted']:
                    assert record['after'] < record['before']-1e-6*max(1.,abs(record['before']))
                    assert record['candidate_checks'][record['selected_candidate']]['ordinary_guard_passed']
                if hard: assert all(c['exact_binary'] for c in details['code_counts'].values())
                editor = FiniteEditStepper(model,bank,episode['configuration']['inner_scope'],16,4)
                before = {n:p.detach().clone() for n,p in model.named_parameters()}
                finite = editor.advance()
                assert finite['after'] <= finite['before']
                assert all(torch.equal(p,before[n]) for n,p in model.named_parameters() if n not in editor.names)
                assert all(torch.equal(b,fixed[n]) for n,b in model.named_buffers())
                assert all(torch.isfinite(p).all() for p in model.parameters())
                results.append({'bits':bits,'hard':hard,'scope':episode['configuration']['inner_scope'],
                    'accepted':record['accepted'],'gradient_norm':record['proposal_gradient_norm'],
                    'record':record,'verified_objective':float(value.detach()),'finite_edit':finite,
                    'finite_edit_checked':True,'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated()})
                atomic_json(a.output/'progress.json',results)
                print(json.dumps({k:results[-1][k] for k in ('bits','hard','scope','accepted')}),flush=True)
                del model,optimizer,loss,value,editor,before,ordinary,episode
                gc.collect(); torch.cuda.empty_cache()
    result = {'ok':True,'checks':results,'source_files_verified':len(expected),'data_files_verified':len(data),
              'elapsed_seconds':time.monotonic()-start,'runtime_downloads':False,
              'scope':'Full-sized CUDA ordinary update, guarded proposal with independent score rerun, and one finite edit per condition. Implementation validation, not trained-model evidence or SCC.'}
    atomic_json(a.output/'result.json',result)
    if os.environ.get('GMN_RESULT_PATH'): atomic_json(os.environ['GMN_RESULT_PATH'],inline_summary(result))


if __name__ == '__main__': main()
