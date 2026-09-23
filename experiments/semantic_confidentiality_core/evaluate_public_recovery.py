"""Assess inexpensive clause-sensitive readers against a frozen semantic core."""
from pathlib import Path
import json
import re
import sys
import time
import numpy as np

STOP_WORDS=set('a an the is are was were be been being at in on by to of for from with as that this it its and but while during described currently had has have did do does also already still again some any all every each only throughout until before after'.split())
NEGATIONS={'no','not','never','nobody','none','neither','without','unopened','unripe','empty','refused','lacked','failed','disconnected'}


def words(text):return re.findall(r'[a-z0-9]+',text.lower())


def stem(word):
    for suffix in ('ization','isation','ation','ment','ness','ing','ied','ed','es','s'):
        if word.endswith(suffix) and len(word)>len(suffix)+3:
            return word[:-len(suffix)]+('y' if suffix=='ied' else '')
    return word


def content_words(text):return [stem(word) for word in words(text) if word not in STOP_WORDS and word not in NEGATIONS]


def subsequence_length(left,right):
    previous=[0]*(len(right)+1)
    for left_word in left:
        current=[0]
        for index,right_word in enumerate(right):
            current.append(previous[index]+1 if left_word==right_word else max(previous[index+1],current[-1]))
        previous=current
    return previous[-1]


def similarities(left,right):
    shared=len(set(left)&set(right))
    return {'overlap':shared/max(1,len(set(right))),
            'jaccard':shared/max(1,len(set(left)|set(right))),
            'subsequence':subsequence_length(left,right)/max(1,len(right))}


def public_scores(premise,hypothesis):
    result={}
    hypothesis_words=content_words(hypothesis)
    hypothesis_negations=sum(word in NEGATIONS for word in words(hypothesis))
    for partition in ('sentence','clause'):
        pattern=r'[.!?;]+' if partition=='sentence' else r'[.!?;,]+|\b(?:but|while)\b'
        clauses=[part for part in re.split(pattern,premise,flags=re.I) if part.strip()]
        for selector in ('overlap','jaccard','subsequence'):
            scores=[similarities(content_words(clause),hypothesis_words)[selector] for clause in clauses]
            chosen=max(range(len(clauses)),key=lambda index:(scores[index],-len(content_words(clauses[index])),-index))
            negations=sum(word in NEGATIONS for word in words(clauses[chosen]))
            for rule in ('presence','parity'):
                agreement=(bool(negations)==bool(hypothesis_negations)) if rule=='presence' else (negations%2==hypothesis_negations%2)
                name=partition+'_'+selector+'_'+rule
                result[name]=scores[chosen]*(1 if agreement else -1)
    return result


def main(directory):
    started=time.perf_counter()
    configuration=json.loads((directory/'config.json').read_text())
    selected=json.loads((directory/'selection.json').read_text())
    neural=[json.loads(line) for line in (directory/'neural_cases.jsonl').read_text().splitlines()]
    assert len(selected)==len(neural)==128
    records=[]
    for case,model_record in zip(selected,neural):
        for key in ['id','group_index','counterfactual','hypothesis_index','label']:
            assert case[key]==model_record[key]
        if not case['contrast']:continue
        records.append({'id':case['id'],'group_index':case['group_index'],'counterfactual':case['counterfactual'],
                        'hypothesis_index':case['hypothesis_index'],'gold':case['confidential'],
                        'neural_correct':model_record['entailment_prediction']==case['confidential'],
                        'scores':public_scores(case['premise'],case['hypothesis'])})
    calibration=[row for row in records if row['group_index']<8]
    evaluation=[row for row in records if row['group_index']>=8]
    measures={};vectors=[]
    for name in records[0]['scores']:
        ordered=sorted(set(row['scores'][name] for row in calibration))
        thresholds=[ordered[0]-1,0.,ordered[-1]+1]+[(left+right)/2 for left,right in zip(ordered,ordered[1:])]
        def correct(row,threshold):return int((row['scores'][name]>threshold)==row['gold'])
        threshold=max(thresholds,key=lambda value:(sum(correct(row,value) for row in calibration),-abs(value),-value))
        values=np.array([np.mean([correct(row,threshold) for row in evaluation if row['group_index']==group]) for group in range(8,16)])
        vectors.append(values)
        direct_values=np.array([np.mean([correct(row,0.) for row in evaluation if row['group_index']==group]) for group in range(8,16)])
        vectors.append(direct_values)
        measures[name]={'threshold':threshold,'calibration_accuracy':np.mean([correct(row,threshold) for row in calibration]).item(),
                        'evaluation_accuracy':values.mean().item(),'direct_evaluation_accuracy':np.mean([correct(row,0.) for row in evaluation]).item()}
        for row in records:
            row.setdefault('predictions',{})[name]=row['scores'][name]>threshold
            row['predictions'][name+'_direct']=row['scores'][name]>0.
    baseline=json.loads((directory/'neural_summary.json').read_text())['best_public_evaluation_accuracy']
    # The previous strongest reader is a fixed comparison; preserve its actual group outcomes.
    prior_public=[]
    for group in range(8,16):
        cases=[row for row in neural if row['group_index']==group and row['contrast']]
        prior_public.append(np.mean([(row['public_scores']['negation_agreement']>0)==row['confidential'] for row in cases]))
    assert abs(np.mean(prior_public)-baseline)<1e-12
    matrix=np.stack(vectors+[np.array(prior_public)],axis=1)
    model_vector=np.array([np.mean([row['neural_correct'] for row in evaluation if row['group_index']==group]) for group in range(8,16)])
    generator=np.random.default_rng(configuration['bootstrap_seed']);indices=generator.integers(0,8,(configuration['bootstrap_replicates'],8))
    difference=model_vector[indices].mean(axis=1)-matrix[indices].mean(axis=1).max(axis=1)
    best_public=float(matrix.mean(axis=0).max())
    summary={'status':'complete','case_count':len(records),'public_readers':measures,'previous_best_public':baseline,
             'best_public_accuracy':best_public,'neural_accuracy':float(model_vector.mean()),
             'selection_aware_advantage_interval':np.quantile(difference,[.025,.975]).tolist(),
             'illustrative_useful_cap':min(1.,best_public+configuration['removal_tolerance']),
             'illustrative_above_chance_loss_fraction':max(0.,2*(1-best_public-configuration['removal_tolerance'])),
             'actual_all_reader_cap_established':False,'training_admitted':False,'neural_training':False,
             'wall_seconds':time.perf_counter()-started}
    (directory/'cases.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
