"""Exact posterior and permission losses under a competent retained reference."""
from fractions import Fraction
from pathlib import Path
import json
import sys
import time
from finite_permission_design import permission_design


def expected_scores(posteriors, masks, label_count):
    cases=len(posteriors)
    class_loss=Fraction(); permission_loss=Fraction(); class_accuracy=Fraction(); permission_accuracy=Fraction()
    for target, posterior in enumerate(posteriors):
        true_label=target if label_count==cases else target//(cases//label_count)
        class_loss+=sum((value-Fraction(index==true_label))**2 for index,value in enumerate(posterior))/cases
        maximum=max(posterior)
        predictions=[index for index,value in enumerate(posterior) if value==maximum]
        class_accuracy+=Fraction(true_label in predictions,len(predictions))/cases
        for mask in masks:
            risk=sum((posterior[index] for index, included in enumerate(mask) if included),Fraction())
            truth=int(mask[true_label])
            permission_loss+=(risk-truth)**2/(cases*len(masks))
            correctness=Fraction(1,2) if risk==Fraction(1,2) else Fraction((risk>Fraction(1,2))==bool(truth))
            permission_accuracy+=correctness/(cases*len(masks))
    coefficient=Fraction(label_count,4*(label_count-1))
    assert permission_loss==coefficient*class_loss
    return {'multiclass_brier':class_loss,'permission_brier':permission_loss,
            'class_accuracy':class_accuracy,'permission_accuracy':permission_accuracy}


def main(directory):
    started=time.perf_counter()
    configuration=json.loads((directory/'config.json').read_text())
    fine=configuration['fine_classes']; coarse=configuration['coarse_classes']; size=configuration['classes_per_group']
    assert fine==coarse*size
    reference_fine=[[Fraction(1,size) if index//size==target//size else Fraction()
                     for index in range(fine)] for target in range(fine)]
    reference_coarse=[[Fraction(index==target//size) for index in range(coarse)] for target in range(fine)]
    intact_fine=[[Fraction(index==target) for index in range(fine)] for target in range(fine)]
    masks_fine=permission_design(fine).tolist(); masks_coarse=permission_design(coarse).tolist()
    reference={'fine':expected_scores(reference_fine,masks_fine,fine),
               'coarse':expected_scores(reference_coarse,masks_coarse,coarse)}
    intact={'fine':expected_scores(intact_fine,masks_fine,fine),
            'coarse':expected_scores(reference_coarse,masks_coarse,coarse)}
    # Construct the edited endpoint independently as a conditional-frequency law.
    edited_fine=[]
    for target in range(fine):
        support=[label for label in range(fine) if label//size==target//size]
        edited_fine.append([Fraction(support.count(label),len(support)) for label in range(fine)])
    assert edited_fine==reference_fine
    assert reference['fine']['class_accuracy']==Fraction(1,size)
    assert reference['coarse']['class_accuracy']==1
    assert intact['fine']['class_accuracy']==intact['coarse']['class_accuracy']==1
    squared_difference=sum((sum((left-right)**2 for left,right in zip(parent,reference_row))
                            for parent,reference_row in zip(intact_fine,reference_fine)),Fraction())/fine
    assert reference['fine']['multiclass_brier']-intact['fine']['multiclass_brier']==squared_difference
    assert reference['fine']['permission_brier']-intact['fine']['permission_brier']==Fraction(fine,4*(fine-1))*squared_difference
    absolute_permission_accuracy=(reference['fine']['permission_accuracy']+reference['coarse']['permission_accuracy'])/2
    assert absolute_permission_accuracy>Fraction(107,200)
    mixture_loss=(reference['fine']['permission_brier']+reference['coarse']['permission_brier'])/2
    result={'status':'complete','class_cases':fine,'permission_cases':fine*(len(masks_fine)+len(masks_coarse)),
            'reference':{level:{name:str(value) for name,value in scores.items()} for level,scores in reference.items()},
            'intact':{level:{name:str(value) for name,value in scores.items()} for level,scores in intact.items()},
            'edited_equals_reference_posterior':True,'edited_additional_protected_gain':'0',
            'edited_absolute_mixture_permission_accuracy':str(absolute_permission_accuracy),
            'edited_absolute_mixture_permission_brier':str(mixture_loss),
            'coarse_ability_collapsed':False,'neural_training':False,'wall_seconds':time.perf_counter()-started}
    (directory/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (directory/'posterior_table.json').write_text(json.dumps([[str(value) for value in row] for row in reference_fine])+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main(Path(sys.argv[1]))
