"""Fresh structural systems with independently randomized intervention queries."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

FAMILIES = ['backdoor', 'common_effect', 'frontdoor', 'observed_confounder',
    'observed_confounder_small', 'observed_mediator', 'mediator_and_confounder', 'unobserved_confounder']


def structural_value(specification, parents, noise):
    linear_value = np.column_stack(parents) @ np.asarray(specification['weights'])
    activation = specification['activation']
    if activation == 'square':
        value = linear_value ** 2
    elif activation == 'tanh':
        value = np.tanh(linear_value)
    elif activation == 'rectified_linear':
        value = np.maximum(linear_value, 0)
    elif activation == 'identity':
        value = linear_value
    else:
        raise ValueError(activation)
    return value + noise


def construct_system(family, generator):
    counts = {'treatment':2 if family == 'unobserved_confounder' else 1,
        'mediator':2 if family == 'mediator_and_confounder' else 1,
        'outcome':3 if family in ('mediator_and_confounder','unobserved_confounder') else
            (1 if family == 'backdoor' else 2), 'common_effect':2}
    nodes = {name:{'weights':generator.uniform(-1/np.sqrt(count),1/np.sqrt(count),count).tolist(),
        'activation':str(generator.choice(['square','tanh','rectified_linear']))} for name,count in counts.items()}
    specification = {'family':family,'nodes':nodes,'root_scale':float(generator.uniform(1,3)),
        'noise_scale':float(.3*generator.beta(1,5))}
    calibration = generator.normal(0,specification['root_scale'],(2048,2))
    parents = [calibration[:,0],calibration[:,1]] if family == 'unobserved_confounder' else [calibration[:,0]]
    signal = structural_value(nodes['treatment'],parents,generator.normal(0,specification['noise_scale'],2048))
    specification['treatment_threshold'] = float(np.median(signal))
    return specification


def draw_noise(specification, count, generator):
    return {'observed_root':generator.normal(0,specification['root_scale'],count),
        'hidden_root':generator.normal(0,specification['root_scale'],count),
        'treatment_noise':generator.normal(0,specification['noise_scale'],count),
        'mediator_noise':generator.normal(0,specification['noise_scale'],count),
        'outcome_noise':generator.normal(0,specification['noise_scale'],count),
        'effect_noise':generator.normal(0,specification['noise_scale'],count),
        'random_treatment':generator.integers(0,2,count).astype(float)}


def natural_treatment(specification, noise):
    family=specification['family']
    if family in ('common_effect','observed_mediator'):
        return noise['random_treatment']
    if family in ('backdoor','frontdoor'):
        parents=[noise['hidden_root']]
    elif family=='unobserved_confounder':
        parents=[noise['observed_root'],noise['hidden_root']]
    else:
        parents=[noise['observed_root']]
    return (structural_value(specification['nodes']['treatment'],parents,noise['treatment_noise'])>
        specification['treatment_threshold']).astype(float)


def execute_system(specification, noise, treatment):
    family=specification['family'];nodes=specification['nodes']
    observed=noise['observed_root'];hidden=noise['hidden_root']
    if family=='backdoor':
        mediator=structural_value(nodes['mediator'],[hidden],noise['mediator_noise'])
        outcome=structural_value(nodes['outcome'],[mediator],noise['outcome_noise'])
        covariates=[mediator]
    elif family=='common_effect':
        outcome=observed
        covariates=[structural_value(nodes['common_effect'],[treatment,outcome],noise['effect_noise'])]
    elif family=='frontdoor':
        mediator=structural_value(nodes['mediator'],[treatment],noise['mediator_noise'])
        outcome=structural_value(nodes['outcome'],[hidden,mediator],noise['outcome_noise'])
        covariates=[mediator]
    elif family in ('observed_confounder','observed_confounder_small'):
        outcome=structural_value(nodes['outcome'],[observed,treatment],noise['outcome_noise'])
        covariates=[observed]
    elif family=='observed_mediator':
        mediator=structural_value(nodes['mediator'],[treatment],noise['mediator_noise'])
        outcome=structural_value(nodes['outcome'],[mediator,treatment],noise['outcome_noise'])
        covariates=[mediator]
    elif family=='mediator_and_confounder':
        mediator=structural_value(nodes['mediator'],[observed,treatment],noise['mediator_noise'])
        outcome=structural_value(nodes['outcome'],[observed,treatment,mediator],noise['outcome_noise'])
        covariates=[observed,mediator]
    elif family=='unobserved_confounder':
        outcome=structural_value(nodes['outcome'],[observed,hidden,treatment],noise['outcome_noise'])
        covariates=[observed]
    else:
        raise ValueError(family)
    return np.column_stack([treatment,*covariates]),outcome


def validate_linear_mediator():
    treatment=np.array([0.,0.,1.,1.]);action=np.array([0.,1.,0.,1.])
    noise={'mediator_noise':np.array([.25,-.25,.5,-.5]),'outcome_noise':np.array([.125]*4),
        'observed_root':np.zeros(4),'hidden_root':np.zeros(4)}
    specification={'family':'observed_mediator','nodes':{'mediator':{'weights':[2.],'activation':'identity'},
        'outcome':{'weights':[1.,3.],'activation':'identity'}}}
    observed,outcome=execute_system(specification,noise,treatment)
    intervened,counterfactual=execute_system(specification,noise,action)
    assert np.array_equal(counterfactual-outcome,5*(action-treatment))
    incorrect=observed[:,1]+3*action+noise['outcome_noise']
    assert np.array_equal(counterfactual-incorrect,2*(action-treatment))
    assert np.array_equal(intervened[:,1]-observed[:,1],2*(action-treatment))
    return {'rows':4,'unchanged_covariate_control_rejected':int(np.sum(counterfactual!=incorrect))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    arguments=parser.parse_args();configuration=json.loads(Path(arguments.config).read_text())
    output=Path(arguments.output);output.mkdir(exist_ok=True)
    controls=validate_linear_mediator();records=[]
    for family_index,family in enumerate(FAMILIES):
        for repetition in range(configuration['systems_per_family']):
            seed=configuration['seed']+1000*family_index+repetition
            generator=np.random.default_rng(seed)
            specification=construct_system(family,generator)
            count=configuration['small_system_rows'] if family=='observed_confounder_small' else configuration['system_rows']
            noise=draw_noise(specification,count,generator)
            natural=natural_treatment(specification,noise)
            action=generator.integers(0,2,count).astype(float)
            observed,outcome=execute_system(specification,noise,natural)
            intervened,counterfactual=execute_system(specification,noise,action)
            factual,factual_outcome=execute_system(specification,noise,natural.copy())
            assert np.array_equal(observed,factual) and np.array_equal(outcome,factual_outcome)
            unchanged=action==natural
            assert np.array_equal(counterfactual[unchanged],outcome[unchanged])
            assert np.array_equal(intervened[unchanged],observed[unchanged])
            if family in ('backdoor','common_effect'):
                assert np.array_equal(counterfactual,outcome)
            assert all(np.all(np.isfinite(array)) for array in (observed,outcome,intervened,counterfactual))
            directory=output/'data/prior_sampling'/family;directory.mkdir(parents=True,exist_ok=True)
            identifier=f'{family}_{repetition+1}'
            path=directory/(identifier+'.npz')
            np.savez_compressed(path,x_obs=observed.astype(np.float32),y_obs=outcome.astype(np.float32),
                x_int=intervened.astype(np.float32),y_int=counterfactual.astype(np.float32))
            pairs,pair_counts=np.unique(np.column_stack([natural,action]),axis=0,return_counts=True)
            record={'case':identifier,'family':family,'seed':seed,'rows':count,'specification':specification,
                'pairs':pairs.tolist(),'pair_counts':pair_counts.tolist(),'changed_count':int(np.sum(~unchanged)),
                'mean_squared_intervention_effect':float(np.mean((counterfactual-outcome)**2)),
                'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            path.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n');records.append(record)
    summary={'status':'complete','system_count':len(records),'rows':sum(record['rows'] for record in records),
        'changed_count':sum(record['changed_count'] for record in records),'linear_mediator_control':controls,
        'factual_replay_checks':sum(record['rows'] for record in records),'model_scores_inspected':False,
        'neural_training':False,'systems':records}
    (output/'generation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({key:value for key,value in summary.items() if key!='systems'}))

if __name__=='__main__':
    main()
