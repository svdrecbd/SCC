"""Exact control: remove a fixed risk judgment while preserving the mean forecast."""
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import sys


def main(directory):
    configuration=json.loads((directory/'config.json').read_text())
    means=[Fraction(1,4),Fraction(1,2)]
    hazard_threshold=Fraction(3,4)
    risk_tolerance=Fraction(1,8)
    atoms=[]
    for mean in means:
        atoms.extend([(mean,0,mean,Fraction(1,4)),
                      (mean,1,Fraction(0),(1-mean)/4),
                      (mean,1,Fraction(1),mean/4)])
    assert sum(weight for mean,regime,outcome,weight in atoms)==1
    average_mean=sum(outcome*weight for mean,regime,outcome,weight in atoms)
    parent_loss=sum(weight*(mean-outcome)**2 for mean,regime,outcome,weight in atoms)
    baseline_loss=sum(weight*(average_mean-outcome)**2 for mean,regime,outcome,weight in atoms)
    successor_loss=parent_loss
    assert (average_mean,parent_loss,baseline_loss)==(Fraction(3,8),Fraction(7,64),Fraction(1,8))
    true_risk={(mean,regime):sum(weight*(outcome>hazard_threshold) for current,observed,outcome,weight in atoms
        if current==mean and observed==regime)/Fraction(1,4) for mean in means for regime in (0,1)}
    assert all((true_risk[mean,regime]>risk_tolerance)==bool(regime) for mean in means for regime in (0,1))
    grid=[Fraction(index,configuration['reader_denominator']) for index in range(configuration['reader_denominator']+1)]
    reader_checks=0
    for first,second in product(grid,repeat=2):
        reader=dict(zip(means,[first,second]))
        accuracy=sum(Fraction(1,4)*(reader[mean] if regime else 1-reader[mean]) for mean in means for regime in (0,1))
        assert accuracy==Fraction(1,2)
        reader_checks+=1
    protected_probability_error=sum(Fraction(1,4)*(mean/2-true_risk[mean,regime])**2 for mean in means for regime in (0,1))
    assert protected_probability_error==Fraction(5,128)
    baseline_probability=sum(Fraction(1,4)*value for value in true_risk.values())
    baseline_probability_error=sum(Fraction(1,4)*(baseline_probability-true_risk[mean,regime])**2 for mean in means for regime in (0,1))
    assert baseline_probability_error==Fraction(11,256)
    thresholds=[Fraction(0),Fraction(1,4),Fraction(1,2)]
    widths=[Fraction(1,4),Fraction(1,4),Fraction(1,2)]
    baseline_risks=[sum(weight*(outcome>threshold) for mean,regime,outcome,weight in atoms) for threshold in thresholds]
    assert sum(width*risk for width,risk in zip(widths,baseline_risks))==average_mean
    losses={'baseline':Fraction(0),'parent':Fraction(0),'retained_mean_bayes_reader':Fraction(0),'shift_reader':Fraction(0)}
    for mean,regime,outcome,weight in atoms:
        for threshold,width,baseline_risk in zip(thresholds,widths,baseline_risks):
            label=int(outcome>threshold)
            parent_risk=Fraction(int(mean>threshold)) if regime==0 else mean
            mean_reader=(Fraction(int(mean>threshold))+mean)/2
            shift_reader=min(Fraction(1),max(Fraction(0),baseline_risk+mean-average_mean))
            for name,forecast in [('baseline',baseline_risk),('parent',parent_risk),('retained_mean_bayes_reader',mean_reader),('shift_reader',shift_reader)]:
                losses[name]+=weight*width*(forecast-label)**2
    useful_gain=baseline_loss-parent_loss
    assert losses['baseline']-losses['shift_reader']>=useful_gain
    assert losses['retained_mean_bayes_reader']>=losses['parent']
    report={'status':'complete','parent_useful_mse':str(parent_loss),'successor_useful_mse':str(successor_loss),
        'constant_useful_mse':str(baseline_loss),'useful_gain_retained':'1',
        'parent_judgment_accuracy':'1','all_successor_reader_accuracy':'1/2',
        'randomized_reader_grid_checks':reader_checks,'protected_probability_mse':str(protected_probability_error),
        'constant_probability_mse':str(baseline_probability_error),
        'integrated_risk_brier':{name:str(value) for name,value in losses.items()},
        'shift_reader_gain':str(losses['baseline']-losses['shift_reader']),'useful_gain':str(useful_gain),
        'counterexample_to_LN301':False,'neural_attack_demonstrated':False,'neural_training':False}
    (directory/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
