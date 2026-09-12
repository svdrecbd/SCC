"""Combine immutable evaluation parts and independently rescore their task outputs."""

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scc.evaluate import score_predictions
from scc.provenance import atomic_json,file_digest

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--parts',type=Path,nargs='+',required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():raise FileExistsError(args.output)
args.output.mkdir(parents=True)
arms={};roots={};receipts=[];reference=None;language=None;rescored=0
for part in args.parts:
    protocol=json.loads((part/'protocol.json').read_text())
    if reference is None:reference=protocol['evaluation']
    assert protocol['evaluation']==reference
    parent_language=json.loads((part/'parent-original.json').read_text())['language']
    if language is None:language=parent_language
    assert parent_language==language
    for receipt in protocol['checkpoints'].values():
        assert file_digest(receipt['checkpoint'])==receipt['sha256']
    result=json.loads((part/'result.json').read_text())
    assert not arms.keys() & result['arms'].keys()
    arms.update(result['arms']);roots.update({a:str(part) for a in result['arms']})
    for ordering in ['original','reordered']:
        rows=json.loads((part/f'records-{ordering}.json').read_text())
        for path in part.glob(f'*-{ordering}.json'):
            value=json.loads(path.read_text())
            if not isinstance(value,dict) or 'behavior' not in value:continue
            assert score_predictions(rows,value['predictions'])==value['behavior'],path
            rescored+=1
    receipts.append({'directory':str(part),'json_files':{p.name:file_digest(p) for p in part.glob('*.json')}})
atomic_json(args.output/'result.json',{'arms':arms,'artifact_roots':roots,'parts':receipts,
            'task_prediction_sets_rescored':rescored,'identical_evaluation_contracts':True,
            'parent_language_scores_identical':True,'test_split_used':False,'script_sha256':file_digest(__file__)})
