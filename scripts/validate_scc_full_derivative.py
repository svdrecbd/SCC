"""Validate the full derivative of smooth Adam on preserved SCC parents."""

import json
import os
from pathlib import Path
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from scc.checkpoint import load_checkpoint
from scc.developmental_run import TextBank, configure
from scc.differentiable_modify import differentiable_modification
from scc.gradient_diagnostics import prepare_episode, evaluate_objectives, norm, cosine
from scc.model import ModelConfig, Transformer
from scc.provenance import atomic_json, file_digest, snapshot_sources
from scc.strong_attack import straight_through_parameters


def main():
    device=configure("cuda")
    output=Path("/output/full-derivative")
    output.mkdir(parents=True,exist_ok=False)
    snapshot_sources(output/"source")
    (output/"runner.py").write_text(os.environ["SCC_RUNNER_SOURCE"])
    parent=Path("/workspace/parents/late/step-00018000.pt")
    state=load_checkpoint(parent);config=state["contract"]["configuration"]
    bank=TextBank("/workspace/data");bank.floors=dict(state["contract"]["text"]["floors"])
    model=Transformer(ModelConfig(**config["model"])).cuda().eval()
    model.load_state_dict(state["model"])
    atomic_json(output/"contract.json",{"parent_sha256":file_digest(parent),"parent_contract":state["contract"],
        "variance_smoothing":1e-30,"derivative":"Full derivative of smooth Adam with global gradient clipping; detached trigger",
        "scope":"Numerical and optimization calibration, not an SCC mechanism result"})
    initial={k:v.detach().clone() for k,v in model.named_parameters()}
    records=[]
    for ordinal in (1000,1001):
        started=time.monotonic()
        stock,query,refusals,floors,stream_hash=prepare_episode(model,bank,config,ordinal,device)
        with sdpa_kernel(SDPBackend.MATH):
            parameters,stream=differentiable_modification(model,bank,config,ordinal,device)
            values,_,gate=evaluate_objectives(model,parameters,query,refusals,floors)
            gradients={k:torch.autograd.grad(v,tuple(model.parameters()),retain_graph=True) for k,v in values.items()}
            linked=straight_through_parameters(dict(model.named_parameters()),parameters)
            frozen,_,_=evaluate_objectives(model,linked,query,refusals,floors,gate)
            frozen_gradients={k:torch.autograd.grad(v,tuple(model.parameters()),retain_graph=True) for k,v in frozen.items()}
        if any(not torch.isfinite(g).all() for gs in gradients.values() for g in gs):
            raise FloatingPointError("Nonfinite full meta-gradient")
        record={"ordinal":ordinal,"episode_sha256":stream_hash,"gate":float(gate),
            "stock_smooth_max_parameter_difference":max(float((parameters[k].detach()-v.detach()).abs().max()) for k,v in stock.named_parameters()),
            "objectives":{k:{"value":float(values[k].detach()),"full_norm":float(norm(g)),
                "frozen_norm":float(norm(frozen_gradients[k])),"cosine_full_frozen":cosine(g,frozen_gradients[k]),
                "finite_differences":[]} for k,g in gradients.items()}}
        del parameters,values,frozen,linked,stock
        for kind,gradient in gradients.items():
            length=float(norm(gradient))
            if not length:
                continue
            direction={k:g.detach()/length for k,g in zip(initial,gradient)}
            for epsilon in (.0001,.001):
                changed={}
                for sign in (-1,1):
                    with torch.no_grad():
                        for k,v in model.named_parameters():
                            v.copy_(initial[k]+sign*epsilon*direction[k])
                    with sdpa_kernel(SDPBackend.MATH):
                        moved,_=differentiable_modification(model,bank,config,ordinal,device,create_graph=False)
                        with torch.no_grad():
                            value,_,actual_gate=evaluate_objectives(model,moved,query,refusals,floors,gate)
                    changed[str(sign)]={"value":float(value[kind]),"actual_gate":float(actual_gate)}
                    del moved
                finite=(changed["1"]["value"]-changed["-1"]["value"])/(2*epsilon)
                record["objectives"][kind]["finite_differences"].append({"epsilon_l2":epsilon,
                    "predicted":length,"rerun_finite_difference":finite,"values":changed,
                    "relative_discrepancy":abs(finite-length)/length})
            with torch.no_grad():
                for k,v in model.named_parameters():
                    v.copy_(initial[k])
        record.update(elapsed_seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated())
        atomic_json(output/f"ordinal-{ordinal}.json",record);records.append(record)
        print(json.dumps(record),flush=True)
    result={"status":"full_derivative_calibration_complete","episodes":len(records),"causal_mechanism_established":False}
    atomic_json(output/"result.json",result)
    if os.environ.get("GMN_RESULT_PATH"):
        atomic_json(os.environ["GMN_RESULT_PATH"],result)


if __name__=="__main__":
    main()
