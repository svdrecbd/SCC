"""Aggregate independent answerability labels and held-out scalar readers."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
from summarize_support_assessment import threshold_candidates
from evaluate_answerability import answer_scores, normalized


def main(directory, root):
    started=time.perf_counter()
    config=json.loads((directory/'config.json').read_text())
    assert normalized('The Cat, an animal!')=='cat animal'
    assert answer_scores('a computer',['the computer'])==(1,1.0)
    assert answer_scores('computer',['computer and alarm clock'])==(0,.4)
    assert answer_scores('wrong',['right','correct'])==(0,0.0)
    records=[]
    receipts=[]
    for index in range(1,5):
        source=root/f'evaluation{index:02d}'
        summary=json.loads((source/'summary.json').read_text())
        assert summary['status']=='complete' and summary['case_count']==16
        assert summary['parameter_digest']==config['expected_parameter_digest']
        block=[json.loads(line) for line in (source/'cases.jsonl').read_text().splitlines()]
        assert len(block)==16
        for row in block:row['block']=index
        records.extend(block)
        for filename in ['cases.jsonl','summary.json','selection.json','config.json','evaluate_answerability.py','evaluate_grounded_claims.py']:
            data=(source/filename).read_bytes()
            receipts.append({'path':str(source.relative_to(root)/filename),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    assert len({row['id'] for row in records})==64
    identities=list(dict.fromkeys(row['paragraph_id'] for row in records))
    assert len(identities)==32
    pairs=[[row for row in records if row['paragraph_id']==identity] for identity in identities]
    assert all(len(pair)==2 and {row['impossible'] for row in pair}=={False,True} for pair in pairs)
    calibration=[row for row in records if row['block']<=2]
    evaluation=[row for row in records if row['block']>2]
    test_pairs=[pair for pair in pairs if pair[0]['block']>2]
    assert len(test_pairs)==16
    generator=np.random.default_rng(config['bootstrap_seed'])
    resamples=generator.integers(0,len(pairs),(config['bootstrap_replicates'],len(pairs)))
    test_resamples=generator.integers(0,len(test_pairs),(config['bootstrap_replicates'],len(test_pairs)))
    answerable=[row for row in records if not row['impossible']]
    impossible=[row for row in records if row['impossible']]
    exact_pairs=np.array([np.mean([row['exact_match'] for row in pair]) for pair in pairs])
    answerability_pairs=np.array([np.mean([row['abstained']==row['impossible'] for row in pair]) for pair in pairs])
    calibrated={}
    test_vectors={}
    for name in records[0]['scores']:
        orientations=[1] if name.startswith('reader_') else [1,-1]
        for orientation in orientations:
            identifier=name if orientation==1 else name+'_reversed'
            candidates=threshold_candidates([orientation*row['scores'][name] for row in calibration])
            def correct(row,threshold):
                return int((orientation*row['scores'][name]>threshold)==row['impossible'])
            threshold=max(candidates,key=lambda value:(sum(correct(row,value) for row in calibration),-abs(value),-value))
            values=np.array([np.mean([correct(row,threshold) for row in pair]) for pair in test_pairs])
            test_vectors[identifier]=values
            positives=[orientation*row['scores'][name] for row in evaluation if row['impossible']]
            negatives=[orientation*row['scores'][name] for row in evaluation if not row['impossible']]
            calibrated[identifier]={'threshold':threshold,'threshold_count':len(candidates),
                                    'calibration_accuracy':np.mean([correct(row,threshold) for row in calibration]).item(),
                                    'evaluation_accuracy':values.mean().item(),
                                    'evaluation_interval':np.quantile(values[test_resamples].mean(axis=1),[.025,.975]).tolist(),
                                    'evaluation_auc':np.mean([int(left>right)+.5*int(left==right) for left in positives for right in negatives]).item()}
    public=[name for name in test_vectors if not name.startswith('reader_')]
    public_matrix=np.stack([test_vectors[name] for name in public]+[np.full(len(test_pairs),.5)],axis=1)
    advantages={}
    for name in ['reader_null_mean','reader_null_total']:
        sampled=test_vectors[name][test_resamples].mean(axis=1)-public_matrix[test_resamples].mean(axis=1).max(axis=1)
        advantages[name]=np.quantile(sampled,[.025,.975]).tolist()
    summary={'status':'complete','paragraph_pairs':32,'case_count':64,'normalization_checks':4,
             'answerable_exact_match':np.mean([row['exact_match'] for row in answerable]).item(),
             'answerable_token_f1':np.mean([row['token_f1'] for row in answerable]).item(),
             'null_recall':np.mean([row['abstained'] for row in impossible]).item(),
             'balanced_answerability':answerability_pairs.mean().item(),
             'balanced_answerability_interval':np.quantile(answerability_pairs[resamples].mean(axis=1),[.025,.975]).tolist(),
             'overall_exact_match':exact_pairs.mean().item(),
             'overall_token_f1':np.mean([row['token_f1'] for row in records]).item(),
             'exact_match_advantage_over_constant_null_interval':np.quantile(exact_pairs[resamples].mean(axis=1)-.5,[.025,.975]).tolist(),
             'calibrated_readers':calibrated,'calibrated_advantage_intervals':advantages,
             'nonterminated_count':sum(not row['terminated'] for row in records),
             'neural_training':False,'training_admitted':False,'source_files':receipts,'wall_seconds':time.perf_counter()-started}
    summary['provisional_useful_qualification_passed']=summary['overall_exact_match']>=.65 and summary['answerable_exact_match']>=.5 and summary['balanced_answerability']>=.65
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({key:value for key,value in summary.items() if key!='source_files'}))


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
