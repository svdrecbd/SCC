"""Independent artifact scoring and NumPy replay for the shared-reader study."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from audit_shared_predicate import answer, digest, sha


def read(path):
    return json.loads(Path(path).read_text())


def numpy_weights(state):
    left,right = np.meshgrid(np.arange(16),np.arange(16),indexing='ij')
    onehot = np.eye(16)
    features = np.concatenate((onehot[left],onehot[right]),-1)
    for name in ('producer.hidden.0','producer.hidden.2'):
        features = np.tanh(features@state[name+'.weight'].T+state[name+'.bias'])
    scores = (features@state['producer.output.weight'].T+state['producer.output.bias']).squeeze(-1)
    scores = state['gain']*scores+state['offset']
    return np.exp(-np.logaddexp(0,-scores))


def scalar_read(state, weights, query, keys, values, width, threshold, permission=False, separate=False):
    order = np.argsort(keys)
    keys,values = np.asarray(keys)[order],np.asarray(values)[order]
    selected = weights[query,keys]
    output, raw = 0, []
    name = 'permission_reader' if permission and separate else 'reader'
    for bit in range(width):
        payload = ((values >> bit) & 1).astype(float)
        features = np.array([sum(selected*payload)/16,sum(selected)/16,sum(payload)/16])
        for layer in (0,2):
            features = np.tanh(state[f'{name}.{layer}.weight']@features+state[f'{name}.{layer}.bias'])
        logit = float((state[f'{name}.4.weight']@features+state[f'{name}.4.bias']).item())
        raw.append(logit)
        if logit > threshold: output += 1 << bit
    return output,raw


def scalar_predict(state, weights, row, threshold):
    query = row['query']
    if row['family']=='composition':
        query,_ = scalar_read(state,weights,query,row['pointer_keys'],row['pointers'],4,threshold)
    value,_ = scalar_read(state,weights,query,row['keys'],row['values'],8,threshold)
    if row['family']=='addition':
        other,_ = scalar_read(state,weights,row['other_query'],row['keys'],row['values'],8,threshold)
        value = (value+other)%256
    return value


def rescore(record, domains, state, separate, counters):
    weights = numpy_weights(state)
    threshold = record['cognitive_threshold']
    for domain, metrics in record['tasks'].items():
        rows = domains[domain]
        target = np.array([answer(r) for r in rows])
        assert target.tolist() == [r['answer'] for r in rows]
        predicted = np.array(metrics['predictions'])
        assert metrics['rows_sha256'] == digest(rows)
        assert metrics['correct'] == int((predicted==target).sum())
        assert metrics['exact'] == metrics['correct']/len(rows)
        for bit,item in enumerate(metrics['bits']):
            correct = int(((((predicted^target) >> bit) & 1)==0).sum())
            assert correct==item['correct'] and item['accuracy']==correct/len(rows)
        for index in np.random.default_rng(196).choice(len(rows),min(16,len(rows)),replace=False):
            expected = scalar_predict(state,weights,rows[index],threshold)
            assert expected == int(predicted[index]),(domain,index,threshold,expected,int(predicted[index]))
            counters['numpy_serial_task_predictions'] += 1
        counters['raw_answers_rescored'] += len(rows)
    for name,policy in record['policies'].items():
        logits,labels = np.array(policy['logits']),np.array(policy['labels'],dtype=bool)
        actual = logits>0
        tp,fp = int((actual&labels).sum()),int((actual&~labels).sum())
        assert (tp,fp)==(policy['true_positive'],policy['false_positive'])
        assert policy['positive']==int(labels.sum()) and policy['negative']==int((~labels).sum())
        assert policy['true_acceptance']==tp/policy['positive']
        assert policy['false_acceptance']==fp/policy['negative']
        assert policy['bit_accuracy']==float((actual==labels).mean())
        if name.startswith('matched_') and not separate:
            domain = name.removeprefix('matched_')+'/lookup'
            assert record['tasks'][domain]['first_read_logits']==policy['logits']
            if threshold==0:
                bit_correct = sum(b['correct'] for b in record['tasks'][domain]['bits'])
                assert bit_correct == tp+policy['negative']-fp
                counters['paired_policy_cognition_identities_verified'] += 1
        if name=='sparse':
            for query,claim in ((0,0),(0,1),(5,9),(11,11),(15,2)):
                payload = np.eye(16,dtype=int)[claim]
                _,values = scalar_read(state,weights,query,list(range(16)),payload,1,0,permission=True,separate=separate)
                assert abs(values[0]-float(logits[query*16+claim,0]))<1e-10
                counters['numpy_serial_permission_logits'] += 1
    primary = record['policies']['sparse' if record['primary_policy']=='sparse' else 'matched_iid']
    expected = (min(v['exact'] for v in record['tasks'].values())>=.95
                and primary['true_acceptance']>=.95 and primary['false_acceptance']<=.05)
    assert expected==record['qualified']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',required=True)
    p.add_argument('--output',required=True)
    args = p.parse_args()
    root,out = Path(args.directory),Path(args.output)
    if out.exists(): raise FileExistsError(out)
    torch.set_num_threads(1)
    contract,run,domains = read(root/'contract.json'),read(root/'result.json'),read(root/'problems.json')
    assert run['status'] in ('completed','completed_with_unqualified_parents')
    for path,expected in contract['source'].items(): assert sha(root/'source'/path)==expected,path
    for domain,rows in domains.items():
        assert len({r['identity'] for r in rows})==len(rows)
        for row in rows:
            assert row['identity']==digest({k:v for k,v in row.items() if k not in ('identity','answer')})
    calls = torch.load(root/'threshold-calibration-calls.pt',weights_only=True)
    call_hash = digest([item.tolist() for item in calls])
    counters = {'source_files_verified':len(contract['source']), 'checkpoint_hashes_verified':0,
        'raw_answers_rescored':0,'numpy_serial_task_predictions':0,'numpy_serial_permission_logits':0,
        'paired_policy_cognition_identities_verified':0,'arms':{},'training_stream_groups':{}}
    for arm,arm_result in run['arms'].items():
        directory = root/arm
        summary = {'qualified':arm_result['parent_qualified'],'cases':{}}
        for case in arm_result['cases']:
            record = read(directory/f'{case}.json')
            checkpoint = directory/f'{case}.pt'
            assert sha(checkpoint)==record['checkpoint_sha256']
            counters['checkpoint_hashes_verified'] += 1
            saved = torch.load(checkpoint,weights_only=True)
            state = {k:v.numpy() for k,v in saved['model'].items()}
            rescore(record,domains,state,saved['separate_reader'],counters)
            if 'threshold_probe' in record:
                probe = record['threshold_probe']
                assert probe['call_sha256']==call_hash
                scores,labels = np.array(probe['scores']),np.array(probe['labels'],dtype=bool)
                assert probe['calibration_accuracy']==float(((scores>probe['threshold'])==labels).mean())
                assert probe['minimum_calibration_threshold_gap']==1e-8
                repaired = record['threshold_probe_evaluation']
                assert repaired['cognitive_threshold']==probe['threshold']
                assert repaired['policies']==record['policies'],'Threshold probe changed permission'
                rescore(repaired,domains,state,saved['separate_reader'],counters)
            primary = record['policies']['sparse' if record['primary_policy']=='sparse' else 'matched_iid']
            summary['cases'][case] = {'false_acceptance':primary['false_acceptance'],
                'true_acceptance':primary['true_acceptance'],
                'task_exact':{k:v['exact'] for k,v in record['tasks'].items()},
                'probe_exact':{k:v['exact'] for k,v in record.get('threshold_probe_evaluation',{}).get('tasks',{}).items()}}
        counters['arms'][arm] = summary
        for training in directory.glob('*training/training.json'):
            record = read(training)
            checkpoint = training.parent/'checkpoint.pt'
            assert sha(checkpoint)==record['checkpoint_sha256']
            counters['checkpoint_hashes_verified'] += 1
            # Same seed/number of updates implies identical cognitive calls.
            key = str((record['seed'],record['steps']))
            group = counters['training_stream_groups'].setdefault(key,{'cognitive':set(),'permission_by_policy':{}})
            group['cognitive'].add(record['cognitive_chain'])
            policy = 'sparse' if arm.startswith('sparse-') else 'matched'
            group['permission_by_policy'].setdefault(policy,set()).add(record['permission_chain'])
    for key,group in counters['training_stream_groups'].items():
        assert len(group['cognitive'])==1,key
        group['cognitive']=next(iter(group['cognitive']))
        for policy,hashes in group['permission_by_policy'].items():
            assert len(hashes)==1,(key,policy)
            group['permission_by_policy'][policy]=next(iter(hashes))
    counters.update(status='verified',scc_mechanism_established=False,
                    scope='Artifact/numerical verification; paired-call identities are construction-dependent')
    out.write_text(json.dumps(counters,indent=2)+'\n')
    print(json.dumps({k:v for k,v in counters.items() if k not in ('arms','training_stream_groups')}))


if __name__=='__main__':main()
