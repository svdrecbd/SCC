"""Audit feedback-model records and replay the saved final live computation."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from audit_persistent_reference import rescore
from scc.persistent_matrix import MatrixConfig
from scc.persistent_tasks import input_code, initialize_matrix
from scc.persistent_feedback import feedback_wiring, feedback_window
from scc.provenance import atomic_json, digest, file_digest


def readlog(path): return [json.loads(line) for line in path.read_text().splitlines()]


def audit(folder, comparison_log, original_initial, expected_manifest=None):
    result = json.loads((folder/'result.json').read_text())
    manifest = json.loads((folder/'source/source_manifest.json').read_text())
    assert digest(manifest) == result['source_sha256']
    assert all(file_digest(folder/'source'/n) == h for n,h in manifest.items())
    assert file_digest(folder/'runner.py') == result['runner_sha256']
    assert file_digest(folder/'protocol.md') == result['protocol_sha256']
    if expected_manifest:
        frozen = json.loads(expected_manifest.read_text())
        expected = {n:h for n,h in frozen.items() if n.startswith('scc/') or n in ('pyproject.toml','uv.lock','.python-version')}
        assert manifest == expected
        assert result['runner_sha256'] == frozen['scripts/run_persistent_feedback.py']
        assert result['protocol_sha256'] == frozen['experiment.md']
        assert result['initial_conditions_sha256'] == frozen['initial-conditions.json']
    gate = json.loads((folder/'implementation-gate.json').read_text())
    assert gate['passed'] and gate['training_initial_unchanged']
    assert [g['ordinal'] for g in gate['checks']] == [5999,6000]
    assert all(max(g['maximum_errors'].values()) <= 1e-4 for g in gate['checks'])
    a = result['arguments']
    initial = torch.load(folder/'initial.pt',map_location='cpu',weights_only=True)
    config = MatrixConfig(**initial['configuration'])
    assert config == MatrixConfig(a['width'],4,'soft')
    frozen_values = json.loads((folder/'initial-conditions.json').read_text())
    assert file_digest(folder/'initial-conditions.json') == result['initial_conditions_sha256']
    assert torch.equal(initial['initial_weights'],torch.tensor(frozen_values['initial_weights']))
    old = torch.load(original_initial,map_location='cpu',weights_only=True)
    assert torch.equal(initial['initial_weights'],old['initial_weights'])
    wiring = torch.load(folder/'fixed-wiring.pt',map_location='cpu',weights_only=True)
    assert frozen_values['original_initial_file_sha256'] == file_digest(original_initial)
    assert torch.equal(wiring,torch.tensor(frozen_values['fixed_wiring']))
    assert torch.allclose(wiring,feedback_wiring(config),atol=1e-6,rtol=1e-6)
    assert file_digest(folder/'fixed-wiring.pt')==result['fixed_wiring_sha256']
    assert result['feedback_strength']==1. and result['feedback_seed']==130913
    assert file_digest(folder/'training.jsonl')==result['training_log_sha256']
    rows=readlog(folder/'training.jsonl');other=readlog(comparison_log)
    chain=digest('persistent-feedback/v1')
    for i,row in enumerate(rows):
        chain=digest({'previous':chain,'record':{k:v for k,v in row.items() if k!='chain'}})
        assert row['chain']==chain and row['step']==i+1
        assert row['learning_rate']==(.003 if i<6000 else .0003)
        assert row['sample_sha256']==other[i]['sample_sha256']
        assert math.isfinite(row['loss']) and math.isfinite(row['gradient_norm_before_clip'])
    assert len(rows)==result['completed_steps'] and chain==result['training_chain']
    scores={};count=0;order=None;first=None
    for mode,summary in result['evaluation'].items():
        path=folder/'evaluation'/(mode+'.jsonl')
        assert file_digest(path)==summary['prediction_sha256']
        records=readlog(path);length=len(records)//a['eval_streams']
        assert len(records)==18*a['eval_per_cell']
        assert [(r['request_index'],r['stream']) for r in records]==[(k,s) for k in range(length) for s in range(a['eval_streams'])]
        assert all(r['late_half']==(r['request_index']>=length//2) for r in records)
        this_order=[(r['tokens'],r['label'],r['stream'],r['request_index']) for r in records]
        if order is None: order=this_order
        assert this_order==order
        scores[mode]=rescore(records)
        for key,cell in scores[mode]['cells'].items():
            for metric,value in cell.items():assert math.isclose(value,summary['cells'][key][metric],abs_tol=1e-12)
        assert scores[mode]['qualified']==summary['qualified']
        assert summary['minimum_accuracy']==min(v['accuracy'] for v in scores[mode]['cells'].values())
        expected_interval={'continuous':length,'reset-1':1,'reset-4':4}[mode]
        assert summary['reset_interval']==expected_interval
        assert summary['clean_resets_within_stream']==(length-1)//expected_interval
        if mode=='continuous':first=[r for r in records if r['stream']==0]
        count+=len(records)
    declared=(not a['fixture'] and a['device']=='cuda' and a['width']==128 and a['seed']==17
              and a['data_seed']==24017 and a['steps']==12000 and a['batch']==32 and a['window']==4
              and a['eval_per_cell']==128 and a['eval_streams']==16
              and a['wall_seconds']==6600 and a['total_seconds']==6900)
    assert result['declared_screen_configuration']==declared
    assert result['training_complete']==(len(rows)==a['steps'])
    validated = result.get('evaluation_validation_passed', True)
    if result['schema'] == 'persistent-feedback/v2':
        continuous = result['evaluation']['continuous']
        assert validated == continuous['numerical_validation_passed']
        assert continuous['numerical_logit_tolerance'] == 1e-4
        assert validated == (continuous['single_stream_failure'] is None
            and continuous['single_stream_requests_completed'] == len(first)
            and continuous['single_stream_decision_mismatches'] == 0
            and continuous['single_stream_maximum_logit_error'] <= 1e-4)
        assert file_digest(folder/'evaluation/single-stream.jsonl') == continuous['single_stream_prediction_sha256']
    qualified=declared and result['training_complete'] and validated and scores['continuous']['qualified']
    assert qualified==result['qualified_learnability']
    saved=torch.load(folder/'trained.pt',map_location='cpu',weights_only=True)
    assert saved['step']==len(rows) and saved['configuration']==initial['configuration']
    with torch.no_grad():
        out,_,_=feedback_window(saved['weights'],torch.tensor([[r['tokens'] for r in first]]),config,
                                input_code(config.input_size,'anchor'),wiring)
    error=float((out[0]-torch.tensor([r['logits'] for r in first])).abs().max())
    mismatches=sum(p!=r['prediction'] for p,r in zip(out[0].argmax(-1).tolist(),first))
    assert error<=1e-4 and mismatches==0
    assert not any(result[k] for k in ('positive_scc_result','coupling_trained','protection_removal_tested'))
    return {'passed':True,'source_files_verified':len(manifest),'predictions_rescored':count,
            'training_updates_verified':len(rows),'sample_hashes_matched_reference':len(rows),
            'comparison_log_sha256':file_digest(comparison_log),'original_initial_exact_match':True,
            'fixed_wiring_reproduced':True,'cpu_replay_requests':len(first),
            'cpu_replay_maximum_logit_error':error,'cpu_replay_decision_mismatches':mismatches,
            'qualified_learnability':qualified,'scores':scores,'positive_scc_result':False,
            'source_scope':'Frozen source byte checks; optional submission manifest strengthens external provenance',
            'result_sha256':file_digest(folder/'result.json'),'auditor_sha256':file_digest(__file__)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path);p.add_argument('--comparison-log',type=Path,required=True)
    p.add_argument('--original-initial',type=Path,required=True)
    p.add_argument('--expected-manifest',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh audit path')
    torch.set_num_threads(2)
    result=audit(a.folder,a.comparison_log,a.original_initial,a.expected_manifest)
    atomic_json(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='scores'}))


if __name__=='__main__':main()
