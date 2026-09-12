"""Independently rescore SCC component interventions and gradient diagnostics."""

import argparse
import json
from pathlib import Path
import statistics

from audit_developmental_predictions import audit,retained_digits
from summarize_developmental_comparison import profile,read


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args();root,output=Path(args.directory),Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result={"scope":"Independent rescoring of development diagnostics", "causal_mechanism_established":False,"arms":{}}
    for arm in ("rule_only","early","late"):
        p=root/arm;selection=read(p/"selection.json");info={"selection":selection,"gradients":{},"sites":{}}
        gradients=[read(f) for f in sorted(p.glob("gradient-*.json"))]
        for kind in ("maximum","average"):
            values=[r["objectives"][kind] for r in gradients]
            differences=[d for v in values for d in v["finite_differences"]]
            info["gradients"][kind]={"episodes":len(values),"mean_value":statistics.mean(v["value"] for v in values),
                "gradient_norms":[v["gradient_norm"] for v in values],
                "gradient_ratios_to_ordinary":[v["gradient_ratio_to_ordinary"] for v in values],
                "rerun_direction_reversed_cases":sum(d["rerun_directional_derivative"]<0 for d in differences),
                "finite_difference_cases":len(differences)}
        for site in selection["selected"]:
            entry={}
            for layout in ("original","reordered"):
                rows,clean=read(p/f"rows-{layout}.json"),read(p/f"clean-{layout}.json")
                before=audit(rows,clean)
                q=p/site.replace("/","-");lesion=read(q/f"lesion-{layout}.json")
                item={"lesion":profile(lesion,clean),"partial_cognition":retained_digits(before,audit(rows,lesion)),"replacements":{}}
                for mode in ("identity","opposite_permission","different_problem_same_permission"):
                    altered=read(q/f"{mode}-{layout}.json");audit(rows,altered)
                    if mode=="identity" and altered["predictions"]!=clean["predictions"]:
                        raise ValueError("Identity replacement failed")
                    item["replacements"][mode]={"task_exact":{k:v["exact"] for k,v in altered["tasks"].items()},
                        "useful_unauthorized":{k.split('/')[0]:v["useful_answer_exact"] for k,v in altered["tasks"].items() if k.endswith('/unauthorized')}}
                entry[layout]=item
            info["sites"][site]=entry
        # All screen outputs are included in the audit, not only selected sites.
        for name in ("original","reordered"):
            rows=read(p/f"screen-rows-{name}.json")
            for f in p.glob("screen-head-*.json"):
                audit(rows,read(f)["layouts"][name])
            for f in p.glob("screen-mlp-*.json"):
                audit(rows,read(f)["layouts"][name])
        result["arms"][arm]=info
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({arm:{"sites":list(v["sites"]),"gradients":v["gradients"]} for arm,v in result["arms"].items()}))


if __name__=="__main__":
    main()
