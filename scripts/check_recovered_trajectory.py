"""Check the candidate through eight removal/replay and two repair updates."""

import argparse
import json
from pathlib import Path
import random
import shutil
import sys
import time
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from scc.coupling import Batch, nll
from scc.developmental_run import TextBank, configure
from scc.developmental_tasks import TaskStream, FAMILIES, batch_rows
from scc.differentiable_modify import adam_unroll
from scc.gradient_diagnostics import norm
from scc.model import ModelConfig, Transformer
from scc.provenance import atomic_json, file_digest, snapshot_sources
from scc.recovered_capability import Reader, model_objective, recovery_envelope


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    configure('cpu', 2)
    source = snapshot_sources(args.output/'source')
    shutil.copyfile(__file__, args.output/'runner.py')
    protocol = root/'protocols/SCC_RECOVERED_CAPABILITY_TRAJECTORY_CHECK_V2.md'
    shutil.copyfile(protocol, args.output/'protocol.md')
    original = json.loads((root/'artifacts/scc-recovered-capability-20260910-v3-signal/contract.json').read_text())
    queries = {k: Batch(torch.tensor(v['tokens']), torch.tensor(v['targets']), k) for k,v in original['query_batches'].items()}
    unauthorized = [k for k in queries if k.endswith('/unauthorized')]
    bank = TextBank(root/'artifacts/retrieval-recovery/byte-prepared', blocks=4)
    stream = TaskStream(853946)
    stream.excluded = {r['latent_id'] for r in original['query_rows']+original['support_rows']}
    rng = random.Random(945023)
    losses = []
    batches = []
    for index in range(8):
        removal = batch_rows(stream.rows(2, FAMILIES[index%3], 'unauthorized'), disclose=True)
        if index%2:
            source_name = tuple(bank.indices)[(index//2)%4]
            replay = bank.batch(source_name, 2, rng, 'cpu')
            from scc.data import IGNORE
            scored = replay.targets[replay.targets != IGNORE]
            floor = float(-bank.log_unigrams[source_name][scored].mean())
        else:
            replay = batch_rows(stream.rows(2, FAMILIES[index%3], 'ungated'))
            floor = math.log(10)
        batches.append((removal,replay,floor))
    repairs = [batch_rows(stream.rows(2, f, 'ungated')) for f in ('lookup','arithmetic')]
    serialized = lambda b: {'tokens': b.tokens.tolist(), 'targets': b.targets.tolist(), 'role': b.role}
    atomic_json(args.output/'contract.json', {'source':source, 'protocol_sha256':file_digest(protocol),
        'runner_sha256':file_digest(__file__), 'original_query_contract_sha256':file_digest(root/'artifacts/scc-recovered-capability-20260910-v3-signal/contract.json'),
        'support_train_ids':sorted(stream.seen), 'support_query_disjoint':not bool(stream.seen & stream.excluded),
        'text_replay_normalization':'Train replay targets under training-data unigram probabilities; no query-label coefficients',
        'training_unigrams':{k:v.tolist() for k,v in bank.log_unigrams.items()},
        'removal_replay': [{'removal':serialized(a),'replay':serialized(b),'floor':f} for a,b,f in batches],
        'repair':[serialized(b) for b in repairs]})
    assert not stream.seen & stream.excluded
    path = root/'artifacts/scc-diagnostics-20260910-v1/continuation/downloaded/full-gradient-continuation/full_10/step-00004000.pt'
    parent_hash = file_digest(path)
    results = {}
    for name in ('initialization','full_gradient_candidate'):
        torch.manual_seed(17)
        model = Transformer(ModelConfig(**original['model'])).eval()
        if name != 'initialization':
            saved = torch.load(path,map_location='cpu',weights_only=True)
            model.load_state_dict(saved['model'])
            del saved
        parameters = dict(model.named_parameters())
        def meta(p, create_graph):
            functions = [lambda q,a=a,b=b,f=f: (nll(model,q,a)+.5*nll(model,q,b)/f)/1.5 for a,b,f in batches]
            changed = adam_unroll(p,functions,lr=.001,eps=1e-4,create_graph=create_graph)
            repaired = adam_unroll(changed,[lambda q,b=b:nll(model,q,b) for b in repairs],lr=1e-4,eps=1e-4,create_graph=create_graph)
            endpoints = [model_objective(model,q,queries,unauthorized,[Reader(),Reader(-1)]) for q in (changed,repaired)]
            return recovery_envelope(endpoints)
        with sdpa_kernel(SDPBackend.MATH):
            value = meta(parameters,True)
            gradient = torch.autograd.grad(value,tuple(parameters.values()))
            length = float(norm(gradient))
            assert math.isfinite(length) and length > 0
            directions = {k:g.detach()/length for k,g in zip(parameters,gradient)}
            outcomes = {}
            for sign in (-1,1):
                shifted = {k:(v.detach()+sign*.0001*directions[k]).requires_grad_() for k,v in parameters.items()}
                changed_value = meta(shifted,False)
                outcomes[str(sign)] = float(changed_value.detach())
                del shifted,changed_value
        numerical = (outcomes['1']-outcomes['-1'])/.0002
        record = {'value':float(value.detach()),'gradient_norm':length,'numerical':numerical,
                  'relative_error':abs(numerical-length)/length,'epsilon_l2':.0001,'outcomes':outcomes,
                  'descent_reduced_objective':outcomes['-1']<float(value.detach())}
        results[name] = record
        atomic_json(args.output/(name+'.json'),record)
        print(json.dumps({'stage':name,**record,'elapsed_seconds':time.monotonic()-started}),flush=True)
        del model,parameters,value,gradient,directions
        if time.monotonic()-started > 1200:
            raise RuntimeError('Calibration wall budget exhausted')
    assert file_digest(path) == parent_hash
    atomic_json(args.output/'summary.json',{'status':'complete','results':results,'parent_sha256_unchanged':parent_hash,
        'elapsed_seconds':time.monotonic()-started,'new_cloud_compute_usd':0,'new_model_files_written':0,
        'scope':'Eight-step replay-aware modification plus two-step repair; two stages, one tiny episode each'})


if __name__ == '__main__':
    main()
