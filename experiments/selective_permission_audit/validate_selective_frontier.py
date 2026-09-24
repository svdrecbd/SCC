"""Validate the complete approximate selective-removal recognition frontier."""
from fractions import Fraction
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import sys
import time


def construct_frontier(record, cap, group_size):
    if not Fraction(1,2)<=cap<=1:
        raise ValueError('accuracy cap outside binary Bayes range')
    pairs=record['pairs']; count=2*len(pairs)
    ordered=sorted(pairs,key=lambda pair: (pair[0]//group_size==pair[1]//group_size,pair[0],pair[1]))
    remaining=2*cap-1
    protected=Fraction(); fine=Fraction(); coarse=Fraction(); total=Fraction(); conditional_difference=Fraction()
    probabilities=[]
    for left,right in ordered:
        pair_mass=Fraction(2,count)
        reveal_mass=min(pair_mass,remaining)
        remaining-=reveal_mass
        hidden_mass=pair_mass-reveal_mass
        # Hidden pair state: both true labels have mass hidden_mass/2.
        # Revealed singleton states: one true label each, mass reveal_mass/2.
        protected+=hidden_mass/2+reveal_mass
        fine+=hidden_mass/2+reveal_mass
        coarse+=hidden_mass*(1 if left//group_size==right//group_size else Fraction(1,2))+reveal_mass
        total+=hidden_mass+reveal_mass
        # Normalize each B-conditional state law by P(B)=1/2.
        # The hidden state masses agree. The two revealed states contribute
        # twice reveal_mass to the L1 distance, hence reveal_mass to TV.
        conditional_difference+=reveal_mass
        probabilities.append([left,right,str(reveal_mass/pair_mass),str(hidden_mass),str(reveal_mass/2)])
    assert remaining==0 and total==1
    assert protected==fine==cap
    difference=Fraction(record['conditional_group_total_variation'])
    coarse_bound=min(Fraction(1),1-difference/2+cap-Fraction(1,2))
    assert coarse==coarse_bound
    assert conditional_difference==2*cap-1
    return {'predicate':record['name'],'cap':str(cap),'protected_accuracy':str(protected),
            'fine_accuracy':str(fine),'coarse_accuracy':str(coarse),'coarse_upper_bound':str(coarse_bound),
            'conditional_retained_state_total_variation':str(conditional_difference),
            'encoder':probabilities}


def audit_encoder(record, outcome, group_size):
    count=len(record['permission'])
    joint=defaultdict(Fraction)
    for left,right,reveal,hidden,singleton in outcome['encoder']:
        probability=Fraction(reveal)
        assert 0<=probability<=1
        assert Fraction(hidden)==2*(1-probability)/count
        assert Fraction(singleton)==probability/count
        for label in (left,right):
            if probability<1:
                joint[(label,('pair',left,right))]+=(1-probability)/count
            if probability>0:
                joint[(label,('revealed',label))]+=probability/count
    assert sum(joint.values(),Fraction())==1
    protected=defaultdict(lambda:defaultdict(Fraction))
    fine=defaultdict(lambda:defaultdict(Fraction))
    coarse=defaultdict(lambda:defaultdict(Fraction))
    source_mass=defaultdict(Fraction)
    for (label,state),mass in joint.items():
        protected[state][record['permission'][label]]+=mass
        fine[state][label]+=mass
        coarse[state][label//group_size]+=mass
        source_mass[label]+=mass
    assert len(source_mass)==count and all(value==Fraction(1,count) for value in source_mass.values())
    for values,field in [(protected,'protected_accuracy'),(fine,'fine_accuracy'),(coarse,'coarse_accuracy')]:
        score=sum((max(distribution.values()) for distribution in values.values()),Fraction())
        assert score==Fraction(outcome[field])
    variation=sum((abs(distribution.get(0,Fraction())-distribution.get(1,Fraction())) for distribution in protected.values()),Fraction())
    assert variation==Fraction(outcome['conditional_retained_state_total_variation'])
    return len(joint)


def main(directory):
    started=time.perf_counter()
    configuration=json.loads((directory/'config.json').read_text())
    path=directory.parent/'exact01/results.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==configuration['input_sha256']
    records=json.loads(path.read_text())['records']
    outcomes=[construct_frontier(record,Fraction(cap),configuration['group_size'])
              for record in records for cap in configuration['accuracy_caps']]
    indexed_records={record['name']:record for record in records}
    joint_rows_checked=sum(audit_encoder(indexed_records[outcome['predicate']],outcome,configuration['group_size']) for outcome in outcomes)
    for cap in [Fraction(-1),Fraction(49,100),Fraction(101,100)]:
        try: construct_frontier(records[0],cap,configuration['group_size'])
        except ValueError: pass
        else: raise AssertionError('invalid cap accepted')
    selected=next(row for row in outcomes if row['predicate']=='balanced_within_groups' and row['cap']=='107/200')
    assert Fraction(selected['fine_accuracy'])==Fraction(107,200)
    assert Fraction(selected['coarse_accuracy'])==Fraction(187,200)
    summary={'status':'complete','predicates':len(records),'frontier_points':len(outcomes),
             'independently_checked_joint_rows':joint_rows_checked,
             'selected_frontier':{key:value for key,value in selected.items() if key!='encoder'},
             'retained_fine_advantage':str((Fraction(107,200)-Fraction(1,100))/Fraction(99,100)),
             'retained_coarse_advantage':str((Fraction(187,200)-Fraction(1,20))/Fraction(19,20)),
             'neural_training':False,'neural_removal_demonstrated':False,'wall_seconds':time.perf_counter()-started}
    payload=json.dumps({'summary':summary,'encoder_columns':['positive_class','negative_class','reveal_probability','hidden_state_mass','each_revealed_state_mass'],'outcomes':outcomes},separators=(',',':'))+'\n'
    assert len(payload.encode())<=configuration['maximum_result_bytes']
    (directory/'results.json').write_text(payload)
    print(json.dumps(summary))


if __name__=='__main__':
    main(Path(sys.argv[1]))
