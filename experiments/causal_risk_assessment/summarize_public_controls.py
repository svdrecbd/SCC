"""Combine completed controls with distinct phase identities and explicit hindsight."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    arguments=parser.parse_args();root=Path(arguments.root)
    phases={'ordinary':'released_case_evaluation01','causal':'causal_replacement_evaluation01',
        'structural':'structural_replacement_evaluation01','selection':'treatment_selection_evaluation01'}
    records={phase:{path.stem:json.loads(path.read_text()) for path in (root/directory/'results').glob('*.json') if path.stem not in ('summary','loading')} for phase,directory in phases.items()}
    cases=sorted(records['ordinary']);assert len(cases)==24 and all(set(items)==set(cases) for items in records.values())
    losses={}
    for phase,items in records.items():
        for name in items[cases[0]]['mse']:
            if name not in ('causal_predictor','native_mean','empirical_mean'):
                losses[phase+'/'+name]=np.array([items[case]['mse'][name] for case in cases])
    parent=np.array([records['ordinary'][case]['mse']['causal_predictor'] for case in cases])
    baseline=np.array([records['ordinary'][case]['mse']['empirical_mean'] for case in cases])
    averages={name:float(values.mean()) for name,values in losses.items()}
    best=min(averages,key=averages.get);oracle=np.min(np.stack(list(losses.values())),axis=0)
    generator=np.random.default_rng(30802)
    difference=losses[best]-parent
    intervals=np.quantile(difference[generator.integers(0,24,(10000,24))].mean(axis=1),[.025,.975]).tolist()
    families={family:[index for index,case in enumerate(cases) if records['ordinary'][case]['family']==family] for family in sorted(set(record['family'] for record in records['ordinary'].values()))}
    family_differences={family:float(difference[indices].mean()) for family,indices in families.items()}
    phase,method=best.split('/',1)
    public_risk=np.mean([records[phase][case]['risk_brier'][method+'_reader' if phase=='ordinary' else method] for case in cases])
    baseline_risk=np.mean([records['ordinary'][case]['risk_brier']['empirical_risk'] for case in cases])
    result={'status':'complete','case_count':24,'public_control_count':len(losses),'parent_mse':float(parent.mean()),
        'context_mean_mse':float(baseline.mean()),'best_public':best,'best_public_mse':averages[best],
        'parent_relative_mse_reduction':float(1-parent.mean()/averages[best]),
        'public_retained_fraction':float((baseline.mean()-averages[best])/(baseline.mean()-parent.mean())),
        'best_public_difference_interval':intervals,'best_public_family_differences':family_differences,
        'hindsight_case_minimum_mse':float(oracle.mean()),
        'hindsight_retained_fraction':float((baseline.mean()-oracle.mean())/(baseline.mean()-parent.mean())),
        'best_selection_control':min((name for name in averages if name.startswith('selection/')),key=averages.get),
        'best_selection_mse':min(value for name,value in averages.items() if name.startswith('selection/')),
        'public_risk_brier':float(public_risk),'context_risk_brier':float(baseline_risk),
        'public_risk_gain_over_parent_useful_gain':float((baseline_risk-public_risk)/(baseline.mean()-parent.mean())),
        'all_public_mse':dict(sorted(averages.items(),key=lambda pair:pair[1])),
        'selection_information_rows':json.loads((root/'treatment_selection_evaluation01/treatment_selection.json').read_text())['rows'],
        'independent_confirmation':False,'genuine_removal_tested':False,'neural_training':False,'training_admitted':False}
    Path(arguments.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='all_public_mse'},indent=2))

if __name__=='__main__':
    main()
