"""Trained-parent float64 derivative fixture with explicitly reduced batch size."""

import copy
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from scc.checkpoint import load_checkpoint
from scc.developmental_run import TextBank,configure
from scc.differentiable_modify import differentiable_modification
from scc.gradient_diagnostics import prepare_episode,evaluate_objectives,norm,query_batches,cosine
from scc.strong_attack import straight_through_parameters
from scc.model import ModelConfig,Transformer
from scc.provenance import atomic_json


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-batch-size",type=int,default=2)
    parser.add_argument("--epsilons",type=float,nargs="+",default=[1e-5,1e-6])
    parser.add_argument("--optimizer",choices=["adam","sgd"],default="adam")
    parser.add_argument("--inner-lr",type=float,default=.001)
    parser.add_argument("--inner-epsilon",type=float,default=1e-8)
    parser.add_argument("--arm",choices=["rule_only","early","late"],default="late")
    parser.add_argument("--output",default="artifacts/scc-diagnostics-20260910-v1/cpu-full-derivative-fixture.json")
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    configure("cpu")
    state=load_checkpoint(f"artifacts/developmental-gpu-20260910-v5/experiment-downloaded/development/comparison/{args.arm}/step-00018000.pt")
    config=copy.deepcopy(state["contract"]["configuration"]);config["meta_batch_size"]=args.meta_batch_size
    config.update(inner_optimizer=args.optimizer,inner_lr=args.inner_lr,inner_epsilon=args.inner_epsilon)
    model=Transformer(ModelConfig(**config["model"])).double().eval();model.load_state_dict(state["model"])
    bank=TextBank("artifacts/retrieval-recovery/byte-prepared");bank.floors=dict(state["contract"]["text"]["floors"])
    stock=None
    if args.optimizer=="adam" and args.inner_epsilon==1e-8:
        stock,_,_,_,_=prepare_episode(model,bank,config,1000,"cpu")
    with sdpa_kernel(SDPBackend.MATH):
        altered,stream=differentiable_modification(model,bank,config,1000,"cpu")
        query,refusals,floors=query_batches(stream,bank)
        values,_,gate=evaluate_objectives(model,altered,query,refusals,floors)
        gradient=torch.autograd.grad(values["average"],tuple(model.parameters()),retain_graph=True)
        linked=straight_through_parameters(dict(model.named_parameters()),altered)
        frozen,_,_=evaluate_objectives(model,linked,query,refusals,floors,gate)
        frozen_gradient=torch.autograd.grad(frozen["average"],tuple(model.parameters()))
    length=float(norm(gradient))
    record={"scope":"Trained-parent float64 derivative fixture", "meta_batch_size":args.meta_batch_size,
        "optimizer":args.optimizer,"inner_lr":args.inner_lr,"arm":args.arm,
        "inner_epsilon":args.inner_epsilon,
        "full_vs_frozen_cosine":cosine(gradient,frozen_gradient),"frozen_norm":float(norm(frozen_gradient)),
        "gradient_norm":length,"value":float(values["average"].detach()),"gate":float(gate),
        "stock_smooth_max_parameter_difference":max(float((altered[k].detach()-v.detach()).abs().max()) for k,v in stock.named_parameters()) if stock is not None else None,
        "finite_differences":[]}
    initial={k:v.detach().clone() for k,v in model.named_parameters()}
    direction={k:g.detach()/length for k,g in zip(initial,gradient)}
    del altered,values,stock
    for epsilon in args.epsilons:
        outputs=[]
        for sign in (-1,1):
            with torch.no_grad():
                for k,v in model.named_parameters():
                    v.copy_(initial[k]+sign*epsilon*direction[k])
            with sdpa_kernel(SDPBackend.MATH):
                moved,_=differentiable_modification(model,bank,config,1000,"cpu",create_graph=False)
                with torch.no_grad():
                    values,_,_=evaluate_objectives(model,moved,query,refusals,floors,gate)
                    outputs.append(float(values["average"]))
            del moved
        finite=(outputs[1]-outputs[0])/(2*epsilon)
        record["finite_differences"].append({"epsilon":epsilon,"numerical":finite,
            "relative_discrepancy":abs(finite-length)/length})
        print(record["finite_differences"][-1],flush=True)
    atomic_json(output,record)


if __name__=="__main__":
    main()
