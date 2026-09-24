"""Exact finite joint-law checks for functional removal certificate transfer."""
from collections import defaultdict
from fractions import Fraction
from itertools import product
from pathlib import Path
import json
import sys
import time


def validate_law(law):
    assert sum(law.values(), Fraction()) == 1
    assert all(value >= 0 for value in law.values())


def total_variation(left, right):
    return sum((abs(left.get(key, 0)-right.get(key, 0)) for key in left.keys() | right.keys()), Fraction())/2


def marginal_state(law):
    result = defaultdict(Fraction)
    for (target, public, state), probability in law.items():
        result[(public, state)] += probability
    return dict(result)


def observation_values(law):
    groups = defaultdict(lambda: [Fraction(), Fraction()])
    for (target, public, state), probability in law.items():
        groups[(public, state)][target] += probability
    return {key: {'mass':sum(values), 'accuracy':max(values)/sum(values)}
            for key, values in groups.items() if sum(values)}


def bayes_accuracy(law):
    return sum((value['mass']*value['accuracy'] for value in observation_values(law).values()), Fraction())


def domination_remainder(left, right, factor):
    return sum((max(Fraction(), left.get(key, 0)-factor*right.get(key, 0))
                for key in left.keys() | right.keys()), Fraction())


def certificate_record(name, edited, reference, factor=Fraction(1)):
    validate_law(edited); validate_law(reference)
    distance = total_variation(edited, reference)
    remainder = domination_remainder(edited, reference, factor)
    observations = sorted({(public, state) for target, public, state in edited.keys() | reference.keys()})
    readers = 0
    best_edited, best_reference = Fraction(), Fraction()
    for answers in product((0, 1), repeat=len(observations)):
        decoder = dict(zip(observations, answers))
        scores = [sum((mass for (target, public, state), mass in law.items()
                       if decoder[(public,state)] == target), Fraction()) for law in (edited, reference)]
        assert abs(scores[0]-scores[1]) <= distance
        assert scores[0] <= factor*scores[1]+remainder
        best_edited, best_reference = max(best_edited, scores[0]), max(best_reference, scores[1])
        readers += 1
    assert best_edited == bayes_accuracy(edited) and best_reference == bayes_accuracy(reference)
    assert best_edited <= best_reference+distance
    assert best_edited <= factor*best_reference+remainder
    return {'name':name, 'joint_total_variation':str(distance),
            'marginal_state_total_variation':str(total_variation(marginal_state(edited), marginal_state(reference))),
            'domination_factor':str(factor), 'forward_domination_remainder':str(remainder),
            'reverse_domination_remainder':str(domination_remainder(reference, edited, factor)),
            'edited_bayes_accuracy':str(best_edited), 'reference_bayes_accuracy':str(best_reference),
            'enumerated_decoders':readers,
            'edited_joint_law': [{'target':target,'public':public,'state':state,'probability':str(mass)}
                                 for (target, public, state),mass in sorted(edited.items())],
            'reference_joint_law': [{'target':target,'public':public,'state':state,'probability':str(mass)}
                                    for (target, public, state),mass in sorted(reference.items())],
            'edited_conditional_values':[{'public':public,'state':state,**{key:str(item) for key,item in value.items()}}
                                         for (public,state),value in observation_values(edited).items()]}


def main(directory):
    started=time.perf_counter()
    settings=json.loads((directory/'config.json').read_text())
    results=[]
    # A retained source determines the rule theta XOR input after another source
    # has been removed exactly. This evaluates all inputs without any optimizer.
    retained={(theta^public, public, 'rule_'+str(theta)):Fraction(1,4)
              for theta in (0,1) for public in (0,1)}
    record=certificate_record('exact_retraining_retains_rule',retained,retained)
    assert record['joint_total_variation']=='0' and record['edited_bayes_accuracy']=='1'
    results.append(record)
    correlated={(target,0,'value_'+str(target)):Fraction(1,2) for target in (0,1)}
    independent={(target,0,'value_'+str(state)):Fraction(1,4) for target in (0,1) for state in (0,1)}
    record=certificate_record('marginal_state_equality',correlated,independent)
    assert record['marginal_state_total_variation']=='0' and record['joint_total_variation']=='1/2'
    assert record['edited_bayes_accuracy']=='1' and record['reference_bayes_accuracy']=='1/2'
    results.append(record)
    backup={(target,0,'empty_with_copy_'+str(target)):Fraction(1,2) for target in (0,1)}
    backup_reference={(target,0,'empty_with_copy_'+str(state)):Fraction(1,4) for target in (0,1) for state in (0,1)}
    record=certificate_record('auxiliary_copy_in_complete_state',backup,backup_reference)
    visible={(target,0,'empty'):Fraction(1,2) for target in (0,1)}
    assert total_variation(visible,visible)==0 and bayes_accuracy(backup)==1
    record['visible_only_joint_total_variation']='0'
    results.append(record)
    public={(target,target,'empty'):Fraction(1,2) for target in (0,1)}
    record=certificate_record('reference_public_input_solver',public,public)
    assert record['reference_bayes_accuracy']=='1'
    results.append(record)
    empty={(target,0,'empty'):Fraction(1,2) for target in (0,1)}
    for parameter in settings['retention_probabilities']:
        retention=Fraction(parameter)
        law={}
        for target in (0,1):
            if retention < 1:
                law[(target,0,'empty')]=(1-retention)/2
            law[(target,0,'retained_'+str(target))]=retention/2
        record=certificate_record('rare_retention_'+parameter,law,empty)
        assert Fraction(record['joint_total_variation'])==retention
        assert Fraction(record['edited_bayes_accuracy'])==Fraction(1,2)+retention/2
        intact_probability=sum((value['mass'] for value in observation_values(law).values() if value['accuracy']==1),Fraction())
        assert intact_probability==retention
        record['probability_of_fully_retained_endpoint']=str(intact_probability)
        results.append(record)
    for parameter in settings['noisy_correctness']:
        correctness=Fraction(parameter)
        law={(target,0,'value_'+str(state)):(correctness if target==state else 1-correctness)/2
             for target in (0,1) for state in (0,1)}
        factor=max(2*correctness, 1/(2*(1-correctness)))
        record=certificate_record('bounded_likelihood_ratio_'+parameter,law,independent,factor)
        assert record['forward_domination_remainder']=='0' and record['reverse_domination_remainder']=='0'
        cap=factor*factor/(1+factor*factor)
        assert all(value['accuracy']<=cap for value in observation_values(law).values())
        assert bayes_accuracy(law)==correctness
        record['pointwise_posterior_cap']=str(cap)
        results.append(record)
    output={'status':'complete','cases':len(results),'enumerated_decoders':sum(row['enumerated_decoders'] for row in results),
            'records':results,'neural_training':False,'functional_neural_removal_established':False,
            'wall_seconds':time.perf_counter()-started}
    (directory/'results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({key:value for key,value in output.items() if key!='records'}))


if __name__=='__main__':
    main(Path(sys.argv[1]))
