"""Exact privacy and retained recognition under selective permission removal."""
from fractions import Fraction
from pathlib import Path
import json
import sys
import time
from finite_permission_design import permission_design


def construct_pairs(predicate, group_size):
    count=len(predicate)
    pairs=[]; positive_remaining=[]; negative_remaining=[]
    for start in range(0,count,group_size):
        positive=[label for label in range(start,start+group_size) if predicate[label]]
        negative=[label for label in range(start,start+group_size) if not predicate[label]]
        matched=min(len(positive),len(negative))
        pairs.extend(zip(positive[:matched],negative[:matched]))
        positive_remaining.extend(positive[matched:]); negative_remaining.extend(negative[matched:])
    assert len(positive_remaining)==len(negative_remaining)
    pairs.extend(zip(positive_remaining,negative_remaining))
    return pairs


def analyze_predicate(name, predicate, group_size):
    count=len(predicate)
    assert sum(predicate)==count//2
    pairs=construct_pairs(predicate,group_size)
    assert sorted(label for pair in pairs for label in pair)==list(range(count))
    assert all(predicate[left]!=predicate[right] for left,right in pairs)
    protected=Fraction(1,2)
    fine=Fraction(len(pairs),count)
    coarse=Fraction(sum(2 if left//group_size==right//group_size else 1 for left,right in pairs),count)
    imbalance=sum(abs(2*sum(predicate[start:start+group_size])-group_size) for start in range(0,count,group_size))
    conditional_tv=Fraction(imbalance,count)
    upper_bound=1-conditional_tv/2
    assert coarse==upper_bound and fine==Fraction(1,2)
    return {'name':name,'permission':list(map(int,predicate)), 'pairs':[list(pair) for pair in pairs],
            'protected_accuracy':str(protected),'fine_accuracy':str(fine),'coarse_accuracy':str(coarse),
            'conditional_group_total_variation':str(conditional_tv),'coarse_upper_bound':str(upper_bound),
            'all_retained_posteriors_balanced':True}


def family_accuracy(pairs,masks,group_size=1):
    total=Fraction()
    for left,right in pairs:
        for mask in masks:
            total+=1 if mask[left//group_size]==mask[right//group_size] else Fraction(1,2)
    return total/(len(pairs)*len(masks))


def main(directory):
    started=time.perf_counter()
    configuration=json.loads((directory/'config.json').read_text())
    count=configuration['fine_classes']; groups=configuration['coarse_classes']; size=configuration['classes_per_group']
    assert count==groups*size and groups%2==0 and size==5
    fine_masks=permission_design(count).tolist(); coarse_masks=permission_design(groups).tolist()
    mixed=[int(label%size<(3 if label//size<groups//2 else 2)) for label in range(count)]
    aligned=[int(label//size<groups//2) for label in range(count)]
    records=[analyze_predicate('balanced_within_groups',mixed,size),analyze_predicate('whole_group_permission',aligned,size)]
    records.extend(analyze_predicate('finite_permission_'+str(index),mask,size) for index,mask in enumerate(fine_masks))
    assert Fraction(records[0]['coarse_accuracy'])==Fraction(9,10)
    assert Fraction(records[1]['coarse_accuracy'])==Fraction(1,2)
    pairs=records[0]['pairs']
    family_fine=family_accuracy(pairs,fine_masks)
    family_coarse=family_accuracy(pairs,coarse_masks,size)
    family_mixture=(family_fine+family_coarse)/2
    assert family_fine==Fraction(3*count-4,4*(count-1))
    assert family_mixture>Fraction(107,200)
    coarse_advantage=(Fraction(9,10)-Fraction(1,groups))/(1-Fraction(1,groups))
    fine_advantage=(Fraction(1,2)-Fraction(1,count))/(1-Fraction(1,count))
    summary={'status':'complete','predicates_checked':len(records),'retained_states_checked':sum(len(record['pairs']) for record in records),
             'specified_predicate':{key:value for key,value in records[0].items() if key not in ('pairs','permission')},
             'retained_fine_advantage':str(fine_advantage),'retained_coarse_advantage':str(coarse_advantage),
             'full_fine_family_accuracy':str(family_fine),'full_coarse_family_accuracy':str(family_coarse),
             'full_mixture_accuracy':str(family_mixture),
             'finite_family_coarse_accuracy_range':[str(min(Fraction(row['coarse_accuracy']) for row in records[2:])),str(max(Fraction(row['coarse_accuracy']) for row in records[2:]))],
             'neural_removal_demonstrated':False,'neural_training':False,'wall_seconds':time.perf_counter()-started}
    (directory/'results.json').write_text(json.dumps({'summary':summary,'records':records},indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':
    main(Path(sys.argv[1]))
