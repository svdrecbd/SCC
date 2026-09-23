"""Public replacement controls using only the previously frozen context and queries."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer


def estimators(seed, configuration=None):
    if configuration and configuration.get('regression_family') == 'combined_functions':
        return {**estimators(seed), **estimators(seed, {'regression_family':'structural_functions'})}
    if configuration and configuration.get('regression_family') == 'structural_functions':
        methods = {}
        for degree in (2,3):
            for strength in (.001,.1,10.):
                methods[f'polynomial_degree_{degree}_ridge_{strength:g}'] = make_pipeline(
                    StandardScaler(), PolynomialFeatures(degree=degree,include_bias=False),
                    StandardScaler(), Ridge(alpha=strength))
        for strength in (.1,10.):
            methods[f'additive_spline_ridge_{strength:g}'] = make_pipeline(StandardScaler(),
                SplineTransformer(n_knots=5,degree=3,include_bias=False),Ridge(alpha=strength))
        return methods
    return {'ridge':make_pipeline(StandardScaler(),Ridge(alpha=1)),
        'kernel_ridge':make_pipeline(StandardScaler(),KernelRidge(alpha=.1,kernel='rbf',gamma=.25)),
        'histogram_gradient_boosting':HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,l2_regularization=1,random_state=seed)}


def predict_replacement(estimator, mode, context, outcomes, query):
    treatment=context[:,0]
    if mode=='joint_regression':
        return clone(estimator).fit(context,outcomes).predict(query)
    if mode in ('reconstructed_natural_treatment','known_treatment_residual_replay'):
        transformed=query.copy()
        if mode=='reconstructed_natural_treatment':
            transformed[:,0]=1-query[:,0]
        else:
            shifts=np.linalg.lstsq(np.column_stack([np.ones(len(context)),treatment]),context[:,1:],rcond=None)[0][1]
            transformed[:,1:]+=np.outer(2*query[:,0]-1,shifts)
        return clone(estimator).fit(context,outcomes).predict(transformed)
    if mode=='treatment_omission':
        return clone(estimator).fit(context[:,1:],outcomes).predict(query[:,1:])
    if mode=='treatment_specific':
        result=np.empty(len(query))
        for action in (0,1):
            context_mask=treatment==action;query_mask=query[:,0]==action
            if np.sum(context_mask)<3:
                result[query_mask]=outcomes[context_mask].mean() if np.any(context_mask) else outcomes.mean()
            elif np.any(query_mask):
                result[query_mask]=clone(estimator).fit(context[context_mask,1:],outcomes[context_mask]).predict(query[query_mask,1:])
        return result
    model=clone(estimator).fit(context,outcomes)
    if len(np.unique(treatment))<2:
        probability=np.full(len(query),treatment.mean())
    else:
        propensity=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000))
        propensity.fit(context[:,1:],treatment)
        probability=propensity.predict_proba(query[:,1:])[:,1]
    shifts=np.linalg.lstsq(np.column_stack([np.ones(len(context)),treatment]),context[:,1:],rcond=None)[0][1]
    predictions=[]
    for natural_treatment in (0,1):
        transformed=query.copy()
        if mode=='treatment_marginalization':
            transformed[:,0]=natural_treatment
        elif mode=='mediator_residual_replay':
            transformed[:,1:]+=np.outer(query[:,0]-natural_treatment,shifts)
        else:
            raise ValueError(mode)
        predictions.append(model.predict(transformed))
    return (1-probability)*predictions[0]+probability*predictions[1]


def main():
    parser=argparse.ArgumentParser()
    for name in ('input','config','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--part',type=int,required=True)
    arguments=parser.parse_args();config=json.loads(Path(arguments.config).read_text())
    output=Path(arguments.output);output.mkdir(exist_ok=True)
    paths=sorted(Path(arguments.input).glob('*.npz'))
    selected_paths=paths[arguments.part*config['cases_per_part']:(arguments.part+1)*config['cases_per_part']]
    assert len(selected_paths)==config['cases_per_part']
    for path in selected_paths:
        original=json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as saved:
            context=saved['context'].astype(float);query=saved['query'].astype(float)
            outcomes=np.clip((saved['outcomes']-original['normalization_lower'])/original['normalization_span'],0,1)
            actual=saved['bounded_actual'];baseline_risk=saved['risk_empirical_risk'][0];thresholds=saved['thresholds']
            parent_mean=saved['mean_causal_predictor'];original_query=saved['original_query']
            baseline_mean=saved['mean_empirical_mean']
        means={};validation_losses={};seconds={}
        for estimator_name,estimator in estimators(config['seed'],config).items():
            for mode in config['modes']:
                name=estimator_name+'_'+mode
                started=time.perf_counter();validation=np.empty(len(context))
                for training_indices,validation_indices in KFold(3,shuffle=True,random_state=config['seed']).split(context):
                    validation_mode = 'joint_regression' if mode in ('reconstructed_natural_treatment','known_treatment_residual_replay') else mode
                    validation[validation_indices]=predict_replacement(estimator,validation_mode,context[training_indices],outcomes[training_indices],context[validation_indices])
                validation_losses[name]=float(np.mean((np.clip(validation,0,1)-outcomes)**2))
                means[name]=np.clip(predict_replacement(estimator,mode,context,outcomes,query),0,1)
                seconds[name]=time.perf_counter()-started
        means['selected_causal_replacement']=means[min(validation_losses,key=validation_losses.get)]
        means['averaged_causal_replacement']=np.mean(list(means.values())[:-1],axis=0)
        labels=actual[:,None]>thresholds;grid_actual=labels.mean(axis=1);grid_mean=float(baseline_risk.mean())
        minimum_slack=1.;risk_losses={}
        for name,mean in means.items():
            repaired=np.clip(baseline_risk[None,:]+mean[:,None]-grid_mean,0,1)
            risk_losses[name]=float(np.mean((repaired-labels)**2))
            gain=np.mean((baseline_risk-labels)**2-(repaired-labels)**2,axis=1)
            useful=(grid_mean-grid_actual)**2-(mean-grid_actual)**2
            minimum_slack=min(minimum_slack,float(np.min(gain-useful)))
        assert minimum_slack>-1e-10
        changed=query[:,0]!=original_query[:,0]
        record={'case':path.stem,'family':original['family'],'validation_mse':validation_losses,
            'mse':{name:float(np.mean((mean-actual)**2)) for name,mean in means.items()},
            'risk_brier':risk_losses,'seconds':seconds,'minimum_recovery_slack':minimum_slack,
            'intervention_changed_count':int(changed.sum()),
            'changed_mse':{name:float(np.mean((mean[changed]-actual[changed])**2)) for name,mean in {**means,'causal_predictor':parent_mean,'empirical_mean':baseline_mean}.items()} if np.any(changed) else {},
            'selected_replacement':min(validation_losses,key=validation_losses.get),
            'uses_graph_labels':False,'uses_query_outcomes_for_fitting':False,'neural_training':False}
        (output/(path.stem+'.json')).write_text(json.dumps(record,indent=2)+'\n')
        np.savez_compressed(output/(path.stem+'.npz'),**means)
        print(json.dumps({'case':path.stem,'best_case_mse':min(record['mse'].values()),'parent_mse':original['mse']['causal_predictor']}),flush=True)

if __name__=='__main__':
    main()
