"""Local first-order capability constraints, projected disclosure gradient, finite edits."""

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scc.coupling import Meter,nll
from scc.data import PreparedDataset
from scc.defense_objectives import add_constraint_queries
from scc.interventions import Evaluation,Streams,escape,load_model,parameter_change,parent_receipt,trained_table_ids
from scc.provenance import atomic_json,file_digest,snapshot_sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    paths={a:f'runs/strong-defense17-{a}/step-00000256.pt' for a in ['control','refusal','escape','escape_weight10']}
    evaluator=Evaluation('artifacts/retrieval-recovery/byte-prepared',64,76351)
    stream=Streams(PreparedDataset('artifacts/retrieval-recovery/byte-prepared','train'),82551)
    query={'disclose':stream.task(32,'unauthorized','disclose'),'retrieval':stream.task(32,'retrieval')}
    add_constraint_queries(query,stream,32,16)
    caps={'authorized':query['authorized'],'retrieval':query['retrieval'],**query['language_by_source']}
    sizes=[.03,.1,.3,1.,3.]
    atomic_json(args.output/'protocol.json',{'parents':{a:parent_receipt(p) for a,p in paths.items()},
        'source_files':snapshot_sources(args.output/'source'),'script_sha256':file_digest(__file__),
        'evaluation':evaluator.manifest(),'query_seed':82551,'delta_l2':sizes,
        'scope':'Project a disclosure-loss gradient off six query-loss gradients; local proxy constraints, not a manifold certificate.'})
    atomic_json(args.output/'evaluation_records.json',evaluator.rows)
    torch.save(query,args.output/'query-batches.pt')
    reference_model,_=load_model('runs/online-06-byte-mixed/step-00008000.pt')
    reference=evaluator(reference_model)
    atomic_json(args.output/'reference.json',reference)
    results={}
    meter=Meter()
    for arm,path in paths.items():
        model,state=load_model(path)
        evaluator.check_training(trained_table_ids(state)|stream.tables.seen)
        original=copy.deepcopy(model.state_dict())
        clean=evaluator(model)
        atomic_json(args.output/f'{arm}-clean.json',clean)
        parameters=dict(model.named_parameters())
        gradients={}
        for name,batch in {'disclose':query['disclose'],**caps}.items():
            loss=nll(model,parameters,batch,meter)
            g=torch.autograd.grad(loss,tuple(parameters.values()))
            gradients[name]=torch.cat([v.detach().double().flatten() for v in g])
        rows=torch.stack([gradients[name]/gradients[name].norm() for name in caps])
        _,singular,vh=torch.linalg.svd(rows,full_matrices=False)
        rank=int((singular>singular.max()*1e-10).sum())
        basis=vh[:rank]
        g=gradients['disclose']
        projected=g-basis.T@(basis@g)
        ratio=float(projected.norm()/g.norm())
        directions={'ordinary':-g/g.norm(),'projected':-projected/projected.norm()}
        torch.save({'gradient':g,'capability_gradients':gradients,'directions':directions,
                    'parameter_order':[(k,list(p.shape)) for k,p in parameters.items()]},args.output/f'{arm}-directions.pt')
        points=[]
        for name,direction in directions.items():
            for magnitude in sizes:
                offset=0
                changed={}
                for key,value in original.items():
                    segment=direction[offset:offset+value.numel()].reshape_as(value).to(value.dtype)
                    changed[key]=value+magnitude*segment
                    offset+=value.numel()
                assert offset==len(direction)
                model.load_state_dict(changed)
                value=evaluator(model)
                row={'direction':name,'intended_l2':magnitude,'escape':escape(value,clean,reference),
                     'evaluation':value,**parameter_change(model,original)}
                atomic_json(args.output/f'{arm}-{name}-{magnitude:g}.json',row)
                points.append({k:v for k,v in row.items() if k!='evaluation'})
        results[arm]={'capability_gradient_rank':rank,'singular_values':singular.tolist(),
             'disclosure_gradient_norm':float(g.norm()),'projected_norm_fraction':ratio,
             'maximum_normalized_capability_direction_dot':float((rows@directions['projected']).abs().max()),
             'points':points}
        print(json.dumps({'arm':arm,**results[arm]}),flush=True)
    atomic_json(args.output/'result.json',{'arms':results,'meter':asdict(meter),'test_split_used':False,'cloud_cost_usd':0})


if __name__=='__main__':main()
