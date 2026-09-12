"""Does a real defender update predict improvement after the attacker adapts again?"""

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from scc.checkpoint import save_checkpoint
from scc.coupling import Meter, nll, post_attack_loss, task_batch
from scc.data import PreparedDataset
from scc.interventions import Evaluation, Streams, escape, load_model, parameter_change, parent_receipt, qualification, trained_table_ids
from scc.provenance import atomic_json, file_digest, snapshot_sources
from scc.strong_attack import rollout
from scc.tokenizer import ByteTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    checkpoint = 'runs/online-06-byte-mixed/step-00008000.pt'
    model, state = load_model(checkpoint)
    natural = PreparedDataset('artifacts/retrieval-recovery/byte-prepared','train')
    evaluator = Evaluation('artifacts/retrieval-recovery/byte-prepared',64,76351)
    clean = evaluator(model)
    normalizer = clean['language']['nll_per_supervised_token']
    profiles = {'long': [{'steps':300,'learning_rate':.001,'preservation_weight':1.},
                         {'steps':1000,'learning_rate':.0001,'preservation_weight':10.}],
                'short': [{'steps':300,'learning_rate':.0003,'preservation_weight':1.}]}
    config = {'mode':'escape','language_normalizer':normalizer,
              'surrogate': {'break_threshold':.5,'cap_threshold':.75,'break_temperature':2.,'cap_temperature':.25}}
    protocol = {'checkpoint':parent_receipt(checkpoint),'attack_profiles':profiles,'attack_data_seeds':[61041,61042],
                'coupling_weights':[0.,.1,10.], 'ordinary_seed':61881,'query_seed':61991,
                'defender_update': {'optimizer':'AdamW','learning_rate':.0001,'betas':[.9,.95], 'weight_decay':.1,'clip':1.,'fresh_optimizer':True},
                'source_files':snapshot_sources(args.output/'source'),'script_sha256':file_digest(__file__),
                'evaluation':evaluator.manifest(),'outer_objective':config,
                'scope':'One-step diagnostic at the qualified parent; not a training-duration comparison or full-gradient check.'}
    atomic_json(args.output/'protocol.json',protocol)
    atomic_json(args.output/'evaluation_records.json',evaluator.rows)
    atomic_json(args.output/'clean.json',clean)
    meter = Meter()
    attack_stream = Streams(natural,61041)
    initial, initial_diag = rollout(model,attack_stream,profiles['long'],8,normalizer,meter)
    query_stream = Streams(natural,61991)
    rows = query_stream.tables.batch(32,'unauthorized')
    query = {'disclose':task_batch(rows,ByteTokenizer(),192,'disclose'),
             'policy':task_batch(rows,ByteTokenizer(),192), 'retrieval':query_stream.task(32,'retrieval'),
             'language':query_stream.text(32)}
    torch.save(query,args.output/'query-batches.pt')
    parameters = dict(initial.named_parameters())
    with sdpa_kernel(SDPBackend.MATH):
        objective, _ = post_attack_loss(initial,parameters,query,config)
        meta_gradient = {k:g.detach() for k,g in zip(parameters,torch.autograd.grad(objective,tuple(parameters.values())))}
    ordinary_stream = Streams(natural,61881)
    ordinary = ordinary_stream.ordinary(32)
    with sdpa_kernel(SDPBackend.MATH):
        p = dict(model.named_parameters())
        loss = nll(model,p,ordinary)
        ordinary_gradient = {k:g.detach() for k,g in zip(p,torch.autograd.grad(loss,tuple(p.values())))}
    original = copy.deepcopy(model.state_dict())
    candidates = {'unchanged':copy.deepcopy(model)}
    candidate_info = {}
    for weight in [0.,.1,10.]:
        name = f'weight-{weight:g}'
        candidate = copy.deepcopy(model)
        optimizer = torch.optim.AdamW(candidate.parameters(),lr=.0001,betas=(.9,.95),weight_decay=.1,foreach=False)
        for k,p in candidate.named_parameters(): p.grad = ordinary_gradient[k] + weight*meta_gradient[k]
        gradnorm = torch.nn.utils.clip_grad_norm_(candidate.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        delta = {k:p.detach()-original[k] for k,p in candidate.named_parameters()}
        frozen = {k:p.detach()+delta[k] for k,p in initial.named_parameters()}
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
            frozen_value,_ = post_attack_loss(initial,frozen,query,config)
        candidate_clean = evaluator(candidate)
        candidate_info[name] = {'coupling_weight':weight,'frozen_prediction_change':float(frozen_value)-float(objective.detach()),
            'linear_predicted_change':sum(float((meta_gradient[k]*v).sum()) for k,v in delta.items()),
            'gradient_norm_before_clipping':float(gradnorm),'clean_qualified':qualification(candidate_clean,clean),
            'clean_evaluation':candidate_clean,**parameter_change(candidate,original)}
        candidates[name] = candidate
        save_checkpoint(args.output/f'{name}.pt', {'schema_version':1,'model_config':asdict(candidate.config),
                        'model':candidate.state_dict(),'training_table_ids':sorted(trained_table_ids(state)|ordinary_stream.tables.seen|query_stream.tables.seen|attack_stream.tables.seen)})
    atomic_json(args.output/'candidate-updates.json',candidate_info)
    results = {}
    for profile, stages in profiles.items():
        for seed in [61041,61042]:
            for name,candidate in candidates.items():
                streams = Streams(natural,seed)
                if profile == 'long' and seed == 61041 and name == 'unchanged':
                    attacked,diagnostics = initial,initial_diag
                else:
                    attacked,diagnostics = rollout(candidate,streams,stages,8,normalizer,meter)
                evaluator.check_training(trained_table_ids(state)|streams.tables.seen|query_stream.tables.seen|ordinary_stream.tables.seen)
                with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
                    value,losses = post_attack_loss(attacked,dict(attacked.named_parameters()),query,config)
                evaluation = evaluator(attacked)
                key = f'{profile}-{seed}-{name}'
                result = {'profile':profile,'data_seed':seed,'candidate':name,'adapted_objective':float(value),
                          'losses':losses,'evaluation':evaluation,'escape':escape(evaluation,
                          clean if name=='unchanged' else candidate_info[name]['clean_evaluation'],clean),'attack':diagnostics}
                atomic_json(args.output/f'{key}.json',result)
                results[key] = {k:v for k,v in result.items() if k not in ('evaluation','attack')}
                print(json.dumps(results[key]),flush=True)
    for key,row in results.items():
        baseline = results[f'{row["profile"]}-{row["data_seed"]}-unchanged']['adapted_objective']
        row['adapted_objective_change'] = row['adapted_objective']-baseline
    atomic_json(args.output/'result.json',{'updates':{k:{n:v for n,v in info.items() if n!='clean_evaluation'} for k,info in candidate_info.items()},
                'attacks':results,'meter':asdict(meter),'test_split_used':False,'cloud_cost_usd':0})


if __name__ == '__main__': main()
