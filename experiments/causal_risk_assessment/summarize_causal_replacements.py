"""Aggregate completed replacement controls without refitting or selecting test data."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    for name in ('original','replacement','output'):
        parser.add_argument('--'+name,required=True)
    arguments=parser.parse_args()
    original={path.stem:json.loads(path.read_text()) for path in Path(arguments.original).glob('*.json') if path.stem not in ('summary','loading')}
    replacement={path.stem:json.loads(path.read_text()) for path in Path(arguments.replacement).glob('*.json')}
    assert set(original)==set(replacement) and len(original)==24
    names=list(replacement[next(iter(replacement))]['mse'])
    means={name:float(np.mean([record['mse'][name] for record in replacement.values()])) for name in names}
    original_means={name:float(np.mean([record['mse'][name] for record in original.values()])) for name in next(iter(original.values()))['mse']}
    public={name:value for name,value in {**original_means,**means}.items() if name not in ('causal_predictor','native_mean','empirical_mean')}
    best=min(public,key=public.get)
    parent=original_means['causal_predictor'];baseline=original_means['empirical_mean']
    family={family:{} for family in sorted(set(record['family'] for record in original.values()))}
    for family_name in family:
        cases=[name for name,record in original.items() if record['family']==family_name]
        for method in ['causal_predictor','empirical_mean',*public]:
            family[family_name][method]=float(np.mean([{**original[name]['mse'],**replacement[name]['mse']}[method] for name in cases]))
    oracle=float(np.mean([min(value for name,value in {**original[case]['mse'],**replacement[case]['mse']}.items() if name in public) for case in original]))
    generator=np.random.default_rng(30602)
    comparisons={}
    for method in [best,'selected_causal_replacement','averaged_causal_replacement']:
        differences=np.array([{**original[case]['mse'],**replacement[case]['mse']}[method]-original[case]['mse']['causal_predictor'] for case in sorted(original)])
        sample=differences[generator.integers(0,24,(10000,24))].mean(axis=1)
        family_differences=np.array([values[method]-values['causal_predictor'] for values in family.values()])
        family_sample=family_differences[generator.integers(0,len(family),(10000,len(family)))].mean(axis=1)
        comparisons[method]={'paired_difference':float(differences.mean()),'system_interval':np.quantile(sample,[.025,.975]).tolist(),
            'family_interval':np.quantile(family_sample,[.025,.975]).tolist(),'positive_families':int(np.sum(family_differences>0))}
    changed={name:float(np.mean([record['changed_mse'][name] for record in replacement.values() if record['changed_mse']])) for name in next(iter(replacement.values()))['changed_mse']}
    risk={name:float(np.mean([record['risk_brier'][name] for record in original.values()])) for name in next(iter(original.values()))['risk_brier']}
    summary={'status':'complete','case_count':24,'replacement_mse':means,'best_combined_public':best,
        'best_combined_public_mse':public[best],'parent_mse':parent,'context_mean_mse':baseline,
        'relative_parent_advantage':1-parent/public[best],'public_retained_fraction':(baseline-public[best])/(baseline-parent),
        'casewise_hindsight_public_mse':oracle,'casewise_hindsight_retained_fraction':(baseline-oracle)/(baseline-parent),
        'casewise_hindsight_parent_advantage':1-parent/oracle,'comparisons':comparisons,'family_mse':family,
        'changed_intervention_mse':changed,'mean_expanded_portfolio_seconds':float(np.mean([sum(record['seconds'].values()) for record in replacement.values()])),
        'original_risk_brier':risk,'minimum_recovery_slack':min(record['minimum_recovery_slack'] for record in replacement.values()),
        'fresh_confirmation':False,'training_admitted':False}
    Path(arguments.output).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({key:value for key,value in summary.items() if key not in ('family_mse','original_risk_brier')},indent=2))

if __name__=='__main__':
    main()
