"""LN-114 bounded CPU/CUDA timing and numerical validation; no scientific repair."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scc.device_benchmark import buffered_objective, buffered_step, selection_indices
from scc.provenance import atomic_json, file_digest, snapshot_sources
from scc.separated_binding import MEMORY_CONDITIONS, functional_window
from scc.sharded_repair import start_values
from scripts.train_separated_binding import step
from scripts.run_curriculum_repair import objective
from scripts.diagnose_separated_binding import evaluate
from scripts.diagnose_curriculum_persistence import correspondence


def move_indices(indices, device):
    return tuple([v.to(device) for v in group] for group in indices)


def cpu_trajectory_check(pack, dest):
    results=[]
    hidden=start_values(pack['origin']['state'])[1].repeat(8,1)
    for name,rules in MEMORY_CONDITIONS.items():
        parameters=[torch.nn.Parameter(pack['origin']['payload'].clone()) for _ in range(2)]
        optimizers=[torch.optim.Adam([q],lr=.003,foreach=False) for q in parameters]
        for i in range(4):
            ids,labels,target=[pack[k][i] for k in ('ids','labels','target')]
            a=step(parameters[0],optimizers[0],hidden,ids,labels,target,rules)
            b=buffered_step(parameters[1],optimizers[1],hidden,ids,labels,target,rules,selection_indices(target))
            assert a['loss_before_update']==float(b[0])
            torch.testing.assert_close(parameters[0],parameters[1],atol=0,rtol=0)
            for key in ('exp_avg','exp_avg_sq'):
                torch.testing.assert_close(optimizers[0].state[parameters[0]][key],
                                           optimizers[1].state[parameters[1]][key],atol=0,rtol=0)
        results.append({'condition':name,'updates':4,'payload_and_adam_moments_bitwise_equal':True})
    atomic_json(dest/'cpu-trajectory-equivalence.json',{'passed':True,'results':results})


def numerical(pack, devices, dest):
    results = []
    state = pack['origin']['state']
    hidden = start_values(state)[1]
    for name, rules in MEMORY_CONDITIONS.items():
        for endpoint in ('origin', 'repaired'):
            initial = pack['origin']['payload'] if endpoint == 'origin' else pack['repaired']
            for dtype in (torch.float32, torch.float64):
                baseline = None
                for device in devices:
                    q = initial.to(device=device, dtype=dtype).clone().requires_grad_()
                    ids, labels, target = [pack[k][0].to(device) for k in ('ids','labels','target')]
                    indices = move_indices(selection_indices(pack['target'][0]), device)
                    out, shards, h = functional_window(q, hidden.to(device=device,dtype=dtype).repeat(8,1), ids,128,*rules)
                    loss, _ = buffered_objective(out,labels,target,indices)
                    grad, = torch.autograd.grad(loss,q)
                    record = {**{k:v.detach().cpu() for k,v in out.items()}, 'shards':shards.detach().cpu(),
                              'hidden':h.detach().cpu(),'loss':loss.detach().cpu(),'gradient':grad.cpu()}
                    assert all(torch.isfinite(v).all() for v in record.values())
                    atol,rtol = (1e-4,1e-3) if dtype==torch.float32 else (1e-9,1e-7)
                    errors = {}
                    if baseline is None:
                        baseline = record
                        legacy, _ = objective(out,labels,target)
                        torch.testing.assert_close(loss,legacy,atol=0,rtol=0)
                    else:
                        for k,v in record.items():
                            torch.testing.assert_close(v,baseline[k],atol=atol,rtol=rtol)
                            if v.is_floating_point(): errors[k]=float((v-baseline[k]).abs().max())
                        assert torch.equal(record['logits'].argmax(-1),baseline['logits'].argmax(-1))
                        assert torch.equal(record['admitted'],baseline['admitted'])
                    filename=f'{name}-{endpoint}-{str(dtype).split(".")[-1]}-{device}.pt'
                    saved = {k:v for k,v in record.items() if k!='shards'}
                    saved['shards_first_stream'] = record['shards'][0].clone()
                    saved['all_shards_equal_first'] = torch.equal(record['shards'],record['shards'][0].expand_as(record['shards']))
                    torch.save(saved,dest/filename)
                    results.append({'condition':name,'endpoint':endpoint,'dtype':str(dtype),'device':device,
                                    'max_absolute_differences_from_cpu':errors,'passed':True})
                reduced,physical,_ = evaluate(state,initial.to(dtype),pack['ids'][0,:4],*rules)
                check = correspondence(reduced,physical,dtype)
                assert check['passed']
                torch.save({'reduced':reduced,'physical':physical},dest/f'{name}-{endpoint}-{str(dtype).split(".")[-1]}-physical.pt')
    return results


def timed(pack, device, rules, mode, warmup, measured):
    def sync():
        if device=='cuda': torch.cuda.synchronize()
    transfer_started=time.perf_counter()
    q=torch.nn.Parameter(pack['origin']['payload'].to(device).clone())
    h=start_values(pack['origin']['state'])[1].repeat(8,1).to(device)
    ids,labels,target=[pack[k][:warmup+measured].to(device) for k in ('ids','labels','target')]
    indices=[move_indices(selection_indices(t),device) for t in pack['target'][:warmup+measured]]
    opt=torch.optim.Adam([q],lr=.003,foreach=False)
    sync(); transfer_seconds=time.perf_counter()-transfer_started
    logs=[]; started=None
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    for i in range(warmup+measured):
        if i==warmup:
            sync(); started=time.perf_counter()
        if mode=='legacy':
            row=step(q,opt,h,ids[i],labels[i],target[i],rules)
        else:
            row=buffered_step(q,opt,h,ids[i],labels[i],target[i],rules,indices[i])
        if i>=warmup: logs.append(row)
    if mode=='buffered':
        values=torch.stack(logs).cpu()
        assert torch.isfinite(values).all()
        logs=values.tolist()
    sync(); elapsed=time.perf_counter()-started
    assert torch.isfinite(q).all()
    return {'seconds_per_update':elapsed/measured,'timed_seconds':elapsed,
            'setup_and_transfer_seconds':transfer_seconds,'logs':logs,
            'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cuda',action='store_true'); p.add_argument('--fixture',action='store_true')
    p.add_argument('--disable-triton-overrides',action='store_true',
                   help='Use eager CUDA instead of automatic Python-native Triton overrides (LN-120)')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    def timeout(*_): raise TimeoutError('LN-114 1500-second internal cap')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(1500)
    try:
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        import torch.backends.python_native as python_native
        if a.disable_triton_overrides:
            python_native.triton.enabled=False
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        if a.cuda and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
        devices=['cpu','cuda'] if a.cuda else ['cpu']
        config={'plan':'LN-114','fixture':a.fixture,'devices':devices,'threads':2,
            'warmup':1 if a.fixture else 16,'measured':2 if a.fixture else 64,'repetitions':1 if a.fixture else 3,
            'torch':str(torch.__version__),'python':sys.version,'platform':platform.platform(),
            'gpu':torch.cuda.get_device_name() if a.cuda else None,'cuda':torch.version.cuda,
            'tf32':False,'amp':False,'compile':False,'deterministic':True,'wall_seconds':1500,
            'python_native_triton_enabled':python_native.triton.enabled,
            'output_limit_bytes':256*1024**2,'scope':'Short disposable benchmark, not scientific training'}
        atomic_json(a.output/'configuration.json',config)
        sources=snapshot_sources(a.output/'source')
        for source in (ROOT/'scripts').glob('*.py'):
            relative=source.relative_to(ROOT); target=a.output/'source'/relative
            target.parent.mkdir(exist_ok=True,parents=True); shutil.copyfile(source,target)
            sources[str(relative)]=file_digest(target)
        atomic_json(a.output/'source-manifest.json',sources)
        shutil.copyfile(a.inputs.parent/'plan.md',a.output/'plan.md')
        shutil.copytree(a.inputs,a.output/'inputs')
        for name,expected in json.loads((a.inputs/'manifest.json').read_text()).items():
            assert file_digest(a.output/'inputs'/name)==expected
        pack=torch.load(a.output/'inputs/windows.pt',weights_only=True)
        cpu_trajectory_check(pack,a.output)
        validation=numerical(pack,devices,a.output)
        atomic_json(a.output/'numerical.json',{'passed':True,'results':validation})
        results=[]
        for rep in range(config['repetitions']):
            for name,rules in MEMORY_CONDITIONS.items():
                for device in (devices if rep%2==0 else devices[::-1]):
                    for mode in (('legacy','buffered') if rep%2==0 else ('buffered','legacy')):
                        row=timed(pack,device,rules,mode,config['warmup'],config['measured'])
                        row.update(repetition=rep,condition=name,device=device,mode=mode)
                        results.append(row); atomic_json(a.output/'timings.json',results)
                        print(json.dumps({k:v for k,v in row.items() if k!='logs'}),flush=True)
        for name,expected in sources.items(): assert file_digest(ROOT/name)==expected
        assert sum(v.stat().st_size for v in a.output.rglob('*') if v.is_file() and not v.name.startswith('._'))<config['output_limit_bytes']
        atomic_json(a.output/'result.json',{'status':'fixture-complete' if a.fixture else 'complete',
            'numerical_passed':True,'timing_rows':len(results),'seconds':time.monotonic()-started})
    except BaseException as exc:
        atomic_json(a.output/'failure.json',{'error':repr(exc),'seconds':time.monotonic()-started}); raise
    finally:
        signal.alarm(0)
        atomic_json(a.output/'artifact-manifest.json',{str(v.relative_to(a.output)):file_digest(v)
            for v in sorted(a.output.rglob('*')) if v.is_file() and v.name!='artifact-manifest.json' and not v.name.startswith('._')})


if __name__=='__main__': main()
