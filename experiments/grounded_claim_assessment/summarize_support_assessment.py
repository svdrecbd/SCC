"""Aggregate paired support assessment and evaluate fixed calibrated readers."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np


def threshold_candidates(values):
    unique = sorted(set(values))
    return [unique[0]-1, 0.0, unique[-1]+1]+[(left+right)/2 for left,right in zip(unique,unique[1:])]


def margin(row, name):
    scores = row['scores'][name]
    return max(scores[:3])-scores[3]


def classification(rows, name, threshold):
    return [int((margin(row,name)>threshold)==(row['arm']=='supported')) for row in rows]


def auc(rows, name):
    positive = [margin(row,name) for row in rows if row['arm']=='supported']
    negative = [margin(row,name) for row in rows if row['arm']=='unsupported']
    return float(np.mean([int(left>right)+.5*int(left==right) for left in positive for right in negative]))


def interval(values, generator, count):
    values = np.asarray(values)
    selected = generator.integers(0,len(values),(count,len(values)))
    return np.quantile(values[selected].mean(axis=1),[.025,.975]).tolist()


def main(directory, root):
    started = time.perf_counter()
    configuration = json.loads((directory/'config.json').read_text())
    records = []
    source_receipts = []
    for index in range(1,5):
        source = root/f'support_assessment{index:02d}'
        summary = json.loads((source/'summary.json').read_text())
        assert summary['status']=='complete'
        assert summary['parameter_digest']==configuration['expected_parameter_digest']
        block = [json.loads(line) for line in (source/'cases.jsonl').read_text().splitlines()]
        assert len(block)==summary['case_count']
        for row in block:
            row['block']=index
        records.extend(block)
        for filename in ['config.json','cases.jsonl','summary.json','selection.json','evaluate_support_assessment.py','evaluate_grounded_claims.py']:
            data=(source/filename).read_bytes()
            source_receipts.append({'path':str(source.relative_to(root)/filename),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    identifiers = list(dict.fromkeys(row['id'] for row in records))
    assert len(identifiers)==63 and len(records)==126
    pairs = [[row for row in records if row['id']==identity] for identity in identifiers]
    assert all(len(pair)==2 and {row['arm'] for row in pair}=={'supported','unsupported'} for pair in pairs)
    calibration = [row for row in records if row['block']<=2]
    evaluation = [row for row in records if row['block']>2]
    assert set(row['id'] for row in calibration).isdisjoint(row['id'] for row in evaluation)
    generator = np.random.default_rng(configuration['bootstrap_seed'])
    count = configuration['bootstrap_replicates']
    measures = {}
    calibrated = {}
    for name in records[0]['scores']:
        supported = [row for row in records if row['arm']=='supported']
        unsupported = [row for row in records if row['arm']=='unsupported']
        pair_scores = [np.mean([(row['predictions'][name]!=3)==(row['arm']=='supported') for row in pair]) for pair in pairs]
        measures[name] = {
            'supported_answer_accuracy':float(np.mean([row['predictions'][name]==row['gold'] for row in supported])),
            'unsupported_null_accuracy':float(np.mean([row['predictions'][name]==3 for row in unsupported])),
            'balanced_answerability':float(np.mean(pair_scores)),
            'balanced_answerability_interval':interval(pair_scores,generator,count),
            'overall_answer_accuracy':float(np.mean([row['predictions'][name]==row['gold'] for row in records])),
            'both_answers_correct':float(np.mean([all(row['predictions'][name]==row['gold'] for row in pair) for pair in pairs])),
            'answerability_auc':auc(records,name)}
        candidates=threshold_candidates([margin(row,name) for row in calibration])
        threshold=max(candidates,key=lambda value:(sum(classification(calibration,name,value)),-abs(value),-value))
        test_pairs=[[row for row in evaluation if row['id']==identity] for identity in dict.fromkeys(row['id'] for row in evaluation)]
        correct=[np.mean(classification(pair,name,threshold)) for pair in test_pairs]
        calibrated[name]={'threshold':threshold,'threshold_count':len(candidates),
                          'calibration_accuracy':float(np.mean(classification(calibration,name,threshold))),
                          'evaluation_accuracy':float(np.mean(correct)),
                          'evaluation_interval':interval(correct,generator,count),
                          'evaluation_auc':auc(evaluation,name)}
    primary='reader_mean_log_probability'
    public=[name for name in measures if not name.startswith('reader_')]
    primary_values=np.array([np.mean(classification(pair,primary,calibrated[primary]['threshold'])) for pair in test_pairs])
    public_values=np.array([[np.mean(classification(pair,name,calibrated[name]['threshold'])) for name in public] for pair in test_pairs])
    samples=generator.integers(0,len(test_pairs),(count,len(test_pairs)))
    differences=primary_values[samples].mean(axis=1)-public_values[samples].mean(axis=1).max(axis=1)
    result={'status':'complete','article_count':len(pairs),'calibration_article_count':len(calibration)//2,
            'evaluation_article_count':len(evaluation)//2,'measures':measures,'calibrated_readers':calibrated,
            'selection_aware_calibrated_advantage_interval':np.quantile(differences,[.025,.975]).tolist(),
            'diagnostic_screen_passed':measures[primary]['supported_answer_accuracy']>=.6 and measures[primary]['balanced_answerability']>=.6,
            'neural_training':False,'training_admitted':False,'source_files':source_receipts,'wall_seconds':time.perf_counter()-started}
    (directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='source_files'}))


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
