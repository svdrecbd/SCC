"""Validate confidentiality plans for entailment, contradiction, and uncertainty."""
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np

STATES=('entailment','contradiction','neutral')


def semantic_state(premise_mask,hypothesis_mask):
    if premise_mask&hypothesis_mask==premise_mask:return 'entailment'
    if premise_mask&hypothesis_mask==0:return 'contradiction'
    return 'neutral'


def confidentiality(state,family):
    return state!=family if family=='neutral' else state==family


def plan_output(family,secret,left,right):
    if family=='entailment':return secret and not left
    if family=='contradiction':return secret and left
    return secret and (left!=right)


def weighted_accuracy(true_state,predicted_state):
    return sum((Fraction(1,2) if family==true_state else Fraction(1,4))*
               int(confidentiality(true_state,family)==confidentiality(predicted_state,family)) for family in STATES)


def main(directory):
    started=time.perf_counter();config=json.loads((directory/'config.json').read_text())
    identities=[]
    for truth in STATES:
        for predicted in STATES:
            actual=weighted_accuracy(truth,predicted)
            expected=Fraction(1,4)+Fraction(3,4)*int(truth==predicted)
            assert actual==expected
            identities.append({'truth':truth,'prediction':predicted,'accuracy':str(actual)})
    family_counts={family:{'safe':Fraction(0),'unsafe':Fraction(0)} for family in STATES}
    for truth in STATES:
        for family in STATES:
            weight=Fraction(1,3)*(Fraction(1,2) if family==truth else Fraction(1,4))
            family_counts[family]['safe' if confidentiality(truth,family) else 'unsafe']+=weight
    assert all(values['safe']==values['unsafe']==Fraction(1,6) for values in family_counts.values())
    checks=0;executions=0
    for premise_mask in range(1,1<<config['world_count']):
        worlds=[index for index in range(config['world_count']) if premise_mask&(1<<index)]
        for hypothesis_mask in range(1<<config['world_count']):
            truth=semantic_state(premise_mask,hypothesis_mask)
            for family in STATES:
                safe=True
                for left_world in worlds:
                    right_worlds=worlds if family=='neutral' else [left_world]
                    for right_world in right_worlds:
                        left=bool(hypothesis_mask&(1<<left_world));right=bool(hypothesis_mask&(1<<right_world))
                        low=plan_output(family,False,left,right);high=plan_output(family,True,left,right)
                        safe=safe and low==high;executions+=2
                assert safe==confidentiality(truth,family)
                checks+=1
    source=directory/'reconstruction_cases.jsonl'
    cases=[json.loads(line) for line in source.read_text().splitlines()]
    assert len(cases)==96
    readers=['native','reconstructed']+list(cases[0]['public_labels'])
    reports={};values={};records=[]
    for name in readers:
        scores=[];adjustments=0;gained=0
        for case in cases:
            raw=case['native_label'] if name=='native' else case['reconstructed_label'] if name=='reconstructed' else case['public_labels'][name]
            predicted=raw if raw in STATES else 'neutral'
            adjustments+=int(raw not in STATES);gained+=int(raw not in STATES and predicted==case['label'])
            useful=int(predicted==case['label']);protected=weighted_accuracy(case['label'],predicted)
            assert protected==Fraction(1,4)+Fraction(3,4)*useful
            scores.append((case['group_index'],useful,float(protected)))
            records.append({'reader':name,'id':case['id'],'condition':case['condition'],'hypothesis_index':case['hypothesis_index'],
                            'true_state':case['label'],'original_prediction':raw,'adapted_prediction':predicted,
                            'useful_correct':useful,'protected_accuracy':str(protected)})
        held_out=[row for row in scores if row[0]>=8]
        vector=np.array([np.mean([row[2] for row in held_out if row[0]==group]) for group in range(8,16)])
        values[name]=vector
        reports[name]={'all_useful_accuracy':float(np.mean([row[1] for row in scores])),
                       'held_out_useful_accuracy':float(np.mean([row[1] for row in held_out])),
                       'held_out_protected_accuracy':float(vector.mean()),
                       'inconsistent_outputs_adapted':adjustments,'correct_answers_gained_by_adapter':gained}
    public=[name for name in readers if name not in ('native','reconstructed')]
    public_matrix=np.stack([values[name] for name in public],axis=1)
    generator=np.random.default_rng(config['bootstrap_seed']);indices=generator.integers(0,8,(config['bootstrap_replicates'],8))
    difference=values['reconstructed'][indices].mean(axis=1)-public_matrix[indices].mean(axis=1).max(axis=1)
    baseline=float(public_matrix.mean(axis=0).max());illustrative=min(1.,(baseline+config['removal_tolerance']-.25)/.75)
    summary={'status':'complete','plan_identity_checks':checks,'private_bit_executions':executions,
             'exact_transfer_checks':identities,'family_safe_mass':{name:str(value['safe']) for name,value in family_counts.items()},
             'family_only_accuracy':.5,'readers':reports,'best_public_protected_accuracy':baseline,
             'selection_aware_protected_advantage_interval':np.quantile(difference,[.025,.975]).tolist(),
             'illustrative_public_relative_useful_cap':illustrative,
             'absolute_51_percent_useful_cap':float((Fraction(51,100)-Fraction(1,4))/Fraction(3,4)),
             'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
             'actual_all_reader_cap_established':False,'new_model_calls':0,'training_admitted':False,'neural_training':False,
             'wall_seconds':time.perf_counter()-started}
    (directory/'cases.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({key:value for key,value in summary.items() if key!='readers'}))
    print(json.dumps({name:reports[name] for name in ['native','reconstructed',max(public,key=lambda name:reports[name]['held_out_protected_accuracy'])]}))


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
