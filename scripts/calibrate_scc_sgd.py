"""Small development screen of SGD modification strength, preserving each result."""

import copy
from pathlib import Path

from scc.checkpoint import load_checkpoint
from scc.developmental_run import TextBank,configure,evaluate
from scc.developmental_tasks import evaluation_rows
from scc.differentiable_modify import differentiable_modification
from scc.model import ModelConfig,Transformer
from scc.provenance import atomic_json,snapshot_sources


def main():
    output=Path('artifacts/scc-diagnostics-20260910-v1/sgd-calibration')
    output.mkdir(exist_ok=False)
    snapshot_sources(output/'source')
    configure('cpu')
    state=load_checkpoint('artifacts/developmental-gpu-20260910-v5/experiment-downloaded/development/comparison/rule_only/step-00018000.pt')
    config=copy.deepcopy(state['contract']['configuration']);config['inner_optimizer']='sgd'
    model=Transformer(ModelConfig(**config['model'])).eval();model.load_state_dict(state['model'])
    bank=TextBank('artifacts/retrieval-recovery/byte-prepared');bank.floors=dict(state['contract']['text']['floors'])
    rows=evaluation_rows(32);atomic_json(output/'rows.json',rows)
    atomic_json(output/'clean.json',evaluate(model,bank,rows))
    for lr in (.03,.1,.3,1.):
        config['inner_lr']=lr
        altered,stream=differentiable_modification(model,bank,config,1000,'cpu',create_graph=False)
        candidate=copy.deepcopy(model);candidate.load_state_dict(altered)
        result=evaluate(candidate,bank,rows)
        atomic_json(output/f'lr-{lr}.json',{'config':config,'evaluation':result,
            'scope':'32-problem development screen; unchanged eight-step removal/replay data'})
        print({'lr':lr,'useful_unauthorized':{k:v['useful_answer_exact'] for k,v in result['tasks'].items() if k.endswith('/unauthorized')}},flush=True)


if __name__=='__main__':
    main()
