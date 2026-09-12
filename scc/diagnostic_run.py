"""Run SCC gradient calibration and internal-intervention measurements."""

import argparse
import json
from pathlib import Path
import time

import torch

from .causal_interventions import sites, intervene, paired_rows, score, replaced_predictions
from .checkpoint import load_checkpoint
from .developmental_run import TextBank, configure, evaluate, predictions
from .developmental_tasks import evaluation_rows
from .gradient_diagnostics import audit_episode
from .model import ModelConfig, Transformer
from .provenance import atomic_json, file_digest, snapshot_sources


def refusal(evaluation):
    values=[v["refusal_exact"] for k,v in evaluation["tasks"].items() if k.endswith("/unauthorized")]
    return sum(values)/len(values)


def run(data, parents, output, device, ordinals=(1000,1001,1002,1003), screen_size=32):
    device=configure(device)
    output,parents=Path(output),Path(parents)
    output.mkdir(parents=True,exist_ok=False)
    source=snapshot_sources(output/"source")
    bank=TextBank(data)
    results={"scope":"Development calibration; no causal SCC mechanism claim", "arms":{}}
    for arm in ("rule_only","early","late"):
        started=time.monotonic()
        root=output/arm
        root.mkdir()
        parent=parents/arm/"step-00018000.pt"
        state=load_checkpoint(parent)
        config=state["contract"]["configuration"]
        if bank.manifest()["data_sha256"]!=state["contract"]["text"]["data_sha256"] or bank.manifest()["evaluation_indices"]!=state["contract"]["text"]["evaluation_indices"]:
            raise ValueError("Parent/evaluation data changed")
        bank.floors=dict(state["contract"]["text"]["floors"])
        model=Transformer(ModelConfig(**config["model"])).to(device).eval()
        model.load_state_dict(state["model"])
        atomic_json(root/"contract.json",{"parent_sha256":file_digest(parent),"parent_contract":state["contract"],
            "source":source,"ordinals":ordinals,"screen_size":screen_size,
            "changed_existing_source_files":[k for k,v in state["contract"]["source"].items() if source.get(k)!=v]})
        evaluations={}
        for name,reordered in (("original",False),("reordered",True)):
            rows=evaluation_rows(128,reordered=reordered)
            clean=evaluate(model,bank,rows)
            expected=json.loads((parents/arm/"result.json").read_text())["reordered_validation" if reordered else "validation"]
            if clean!=expected:
                raise AssertionError("Parent clean evaluation differs from original artifact")
            atomic_json(root/f"rows-{name}.json",rows)
            atomic_json(root/f"clean-{name}.json",clean)
            evaluations[name]=(rows,clean)
        gradient_results=[]
        for ordinal in ordinals:
            record=audit_episode(model,bank,config,ordinal,device)
            atomic_json(root/f"gradient-{ordinal}.json",record)
            gradient_results.append(record)
            print(json.dumps({"arm":arm,"gradient_ordinal":ordinal,
                              "norms":{k:v["gradient_norm"] for k,v in record["objectives"].items()}}),flush=True)
        # Screen on train-split problems; validation never chooses sites.
        screens={name:paired_rows(screen_size,reordered=reordered) for name,reordered in (("original",False),("reordered",True))}
        for name,rows in screens.items():
            atomic_json(root/f"screen-rows-{name}.json",rows)
        screen_records=[]
        for site in sites(model):
            record={"site":site,"layouts":{}}
            with intervene(model,site):
                for name,rows in screens.items():
                    record["layouts"][name]=score(rows,predictions(model,rows))
            record["mean_refusal"]=sum(refusal(v) for v in record["layouts"].values())/len(screens)
            atomic_json(root/("screen-"+site.replace("/","-")+".json"),record)
            screen_records.append(record)
        selected=sorted(screen_records,key=lambda r:r["mean_refusal"])[:3]
        atomic_json(root/"selection.json",{"criterion":"lowest train-screen refusal; capability excluded; stable site-order ties",
            "all_scores":[{"site":r["site"],"mean_refusal":r["mean_refusal"]} for r in screen_records],
            "selected":[r["site"] for r in selected]})
        print(json.dumps({"arm":arm,"selected_sites":[r["site"] for r in selected]}),flush=True)
        for record in selected:
            site=record["site"]
            prefix=root/site.replace("/","-")
            prefix.mkdir()
            for name,(rows,clean) in evaluations.items():
                with intervene(model,site):
                    lesion=evaluate(model,bank,rows)
                atomic_json(prefix/f"lesion-{name}.json",lesion)
                for mode in ("identity","opposite_permission","different_problem_same_permission"):
                    patched=replaced_predictions(model,rows,site,mode)
                    if mode=="identity" and patched!=clean["predictions"]:
                        raise AssertionError("Clean activation replacement was not identity")
                    atomic_json(prefix/f"{mode}-{name}.json",score(rows,patched))
            print(json.dumps({"arm":arm,"intervention_site_complete":site}),flush=True)
        for name,(rows,clean) in evaluations.items():
            if predictions(model,rows)!=clean["predictions"]:
                raise AssertionError("Intervention hook leaked into restored model")
        results["arms"][arm]={"selected_sites":[r["site"] for r in selected],"elapsed_seconds":time.monotonic()-started,
                              "parent_reproduced":True,"gradient_episodes":len(gradient_results),"hooks_restored":True}
        atomic_json(output/"progress.json",results)
        del model,state
    results.update(status="diagnostics_complete",causal_mechanism_established=False)
    atomic_json(output/"result.json",results)
    return results


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True)
    parser.add_argument("--parents",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=parser.parse_args()
    result=run(args.data,args.parents,args.output,args.device)
    import os
    if os.environ.get("GMN_RESULT_PATH"):
        atomic_json(os.environ["GMN_RESULT_PATH"],result)
    print(json.dumps(result),flush=True)
