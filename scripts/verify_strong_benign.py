"""Retain raw benign-task outputs and compare them with the larger evaluation."""

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scc.interventions import SPEC,generate_many,load_model,parent_receipt
from scc.online_tasks import evaluation_records
from scc.evaluate import score_predictions
from scc.provenance import atomic_json,file_digest,snapshot_sources

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config',type=Path,required=True)
parser.add_argument('--evaluation',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():raise FileExistsError(args.output)
args.output.mkdir(parents=True)
torch.set_num_threads(4)
config=json.loads(args.config.read_text())
evaluation=json.loads((args.evaluation/'result.json').read_text())
receipts={};scores={}
for arm in config['arms']:
    checkpoint=Path(config['run_roots'][arm])/arm/'benign/step-00001000.pt'
    if not checkpoint.exists():continue
    receipts[arm]=parent_receipt(checkpoint)
    model,_=load_model(checkpoint)
    for reordered in [False,True]:
        ordering='reordered' if reordered else 'original'
        rows=evaluation_records(SPEC,128,921591,('retrieval',),reordered)
        rows=[{**r,'prompt':'Uppercase. '+r['prompt'],'target':r['target'].upper()} for r in rows]
        predictions=generate_many(model,[r['prompt'] for r in rows])
        score=score_predictions(rows,predictions)['retrieval']
        expected=json.loads((Path(evaluation['artifact_roots'][arm])/f'{arm}-benign-{ordering}.json').read_text())['benign_uppercase']
        assert score==expected
        scores[arm+'/'+ordering]=score
        atomic_json(args.output/f'{arm}-{ordering}.json',{'records':rows,'predictions':predictions,'score':score})
atomic_json(args.output/'result.json',{'scores':scores,'all_match_broader_evaluation':True,'parents':receipts,
            'source_files':snapshot_sources(args.output/'source'),'script_sha256':file_digest(__file__)})
