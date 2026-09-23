"""Fixed released-case screen with charged public replacements; no neural training."""
import argparse
import hashlib
import io
import json
import pickle
import pickletools
from pathlib import Path
import socket
import time
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from causal_model_interface import RestrictedModelUnpickler, load_predictor, model_state_digest
from distribution_readout import ConsequenceDistribution


class DatasetPrefixUnpickler(RestrictedModelUnpickler):
    def find_class(self, module, name):
        if (module, name) == ('datasets', 'InterventionalDataset'):
            return SimpleNamespace
        return super().find_class(module, name)


def extract_dataset(path):
    """Stop before the serialized causal model; never reconstruct embedded code."""
    data = path.read_bytes()
    operations = list(pickletools.genops(data))
    segments = []
    for index, (operation, argument, position) in enumerate(operations):
        if operation.name in ('SHORT_BINUNICODE','BINUNICODE') and argument == 'do_scm':
            break
        if operation.name != 'FRAME':
            end = operations[index+1][2] if index+1 < len(operations) else len(data)
            segments.append(data[position:end])
    else:
        raise ValueError('Expected causal-model boundary absent')
    # The prefix leaves the dataset state dictionary's key-value MARK open.
    record = DatasetPrefixUnpickler(io.BytesIO(b''.join(segments) + b'ub.')).load()
    assert set(vars(record)) == {'x_obs','x_int','y_obs','y_int','attribute_names'}
    arrays = {name:getattr(record,name).detach().cpu().numpy() for name in ('x_obs','x_int','y_obs','y_int')}
    arrays['y_obs'] = arrays['y_obs'].reshape(-1)
    arrays['y_int'] = arrays['y_int'].reshape(-1)
    count = len(arrays['y_obs'])
    assert all(len(value)==count and np.all(np.isfinite(value)) for value in arrays.values())
    assert arrays['x_obs'].shape == arrays['x_int'].shape
    assert all(set(np.unique(arrays[name][:,0])) <= {0,1} for name in ('x_obs','x_int'))
    return arrays, {'source_sha256':hashlib.sha256(data).hexdigest(), 'rows':count,
        'features':arrays['x_obs'].shape[1], 'attributes':record.attribute_names,
        'embedded_causal_code_loaded':False}


def public_estimators(seed):
    return {
        'ridge':make_pipeline(StandardScaler(),Ridge(alpha=1)),
        'kernel_ridge':make_pipeline(StandardScaler(),KernelRidge(alpha=.1,kernel='rbf',gamma=.25)),
        'random_forest':RandomForestRegressor(n_estimators=80,min_samples_leaf=3,random_state=seed,n_jobs=1),
        'extra_trees':ExtraTreesRegressor(n_estimators=80,min_samples_leaf=3,random_state=seed,n_jobs=1),
        'histogram_gradient_boosting':HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,l2_regularization=1,random_state=seed),
    }


def main():
    parser=argparse.ArgumentParser()
    for name in ('source','checkpoint','datasets','config','output'):
        parser.add_argument('--'+name,required=True)
    arguments=parser.parse_args()
    configuration=json.loads(Path(arguments.config).read_text())
    output=Path(arguments.output);output.mkdir(exist_ok=True)
    def reject_connection(*arguments,**keywords):
        raise RuntimeError('Network disabled during evaluation')
    socket.socket.connect=reject_connection;socket.create_connection=reject_connection
    torch.set_num_threads(1);torch.manual_seed(configuration['seed'])
    predictor,metadata=load_predictor(arguments.source,arguments.checkpoint,configuration['seed'])
    (output/'loading.json').write_text(json.dumps(metadata,indent=2)+'\n')
    results=[]
    thresholds=np.arange(configuration['threshold_count'])/configuration['threshold_count']
    paths=sorted(Path(arguments.datasets).glob('data/prior_sampling/*/*.pkl'))
    assert len(paths)==configuration['expected_cases']
    for case_index,path in enumerate(paths):
        arrays,receipt=extract_dataset(path)
        generator=np.random.default_rng(configuration['seed']+case_index)
        ordering=generator.permutation(receipt['rows'])
        context_count=min(configuration['maximum_context'],int(.8*receipt['rows']))
        context_indices=ordering[:context_count]
        query_indices=ordering[context_count:context_count+configuration['maximum_queries']]
        assert context_count>=6 and len(query_indices)>=2
        context=arrays['x_obs'][context_indices].astype(np.float32)
        outcomes=arrays['y_obs'][context_indices].astype(np.float32)
        query=arrays['x_obs'][query_indices].copy().astype(np.float32)
        query[:,0]=arrays['x_int'][query_indices,0]
        observed_query=arrays['x_obs'][query_indices].copy()
        actual=arrays['y_int'][query_indices]
        location=float(outcomes.mean());scale=max(float(outcomes.std()),1e-6)
        lower=location-2*scale;span=4*scale
        bounded_outcomes=np.clip((outcomes-lower)/span,0,1)
        bounded_actual=np.clip((actual-lower)/span,0,1)
        baseline_risk=np.mean(bounded_outcomes[:,None]>thresholds,axis=0)
        baseline_mean=float(bounded_outcomes.mean())
        started=time.perf_counter()
        predictor.fit(context,outcomes)
        state_before=model_state_digest(predictor.model)
        prediction=predictor.predict_full(torch.from_numpy(query.copy()))
        learned_seconds=time.perf_counter()-started
        assert state_before==model_state_digest(predictor.model)
        distribution=ConsequenceDistribution((prediction['criterion'].borders.numpy()-lower)/span,prediction['buckets'])
        learned_mean=distribution.bounded_mean()
        learned_risk=1-distribution.cdf(thresholds)
        means={'empirical_mean':np.full(len(query),baseline_mean),'causal_predictor':learned_mean,
            'native_mean':np.clip((prediction['mean']-lower)/span,0,1)}
        validation_losses={};public_seconds={}
        for name,estimator in public_estimators(configuration['seed']+case_index).items():
            started=time.perf_counter()
            validation=cross_val_predict(clone(estimator),context,bounded_outcomes,
                cv=KFold(3,shuffle=True,random_state=configuration['seed']),n_jobs=1)
            validation_losses[name]=float(np.mean((np.clip(validation,0,1)-bounded_outcomes)**2))
            estimator.fit(context,bounded_outcomes)
            means[name]=np.clip(estimator.predict(query),0,1)
            public_seconds[name]=time.perf_counter()-started
        selected=min(validation_losses,key=validation_losses.get)
        means['selected_public']=means[selected]
        means['averaged_public']=np.mean([means[name] for name in validation_losses],axis=0)
        labels=bounded_actual[:,None]>thresholds
        grid_baseline_mean=float(baseline_risk.mean())
        grid_actual=labels.mean(axis=1)
        risks={'empirical_risk':np.broadcast_to(baseline_risk,labels.shape),'causal_risk':learned_risk}
        slacks=[]
        for name,mean in means.items():
            repaired=np.clip(baseline_risk[None,:]+mean[:,None]-grid_baseline_mean,0,1)
            risks[name+'_reader']=repaired
            risk_gain=np.mean((baseline_risk-labels)**2-(repaired-labels)**2,axis=1)
            useful_gain=(grid_baseline_mean-grid_actual)**2-(mean-grid_actual)**2
            slacks.extend((risk_gain-useful_gain).tolist())
        assert min(slacks)>-1e-10
        record={'case':path.stem,'family':path.parent.name,**receipt,
            'context_count':context_count,'query_count':len(query),'normalization_lower':lower,'normalization_span':span,
            'selected_public':selected,'validation_mse':validation_losses,
            'mse':{name:float(np.mean((mean-bounded_actual)**2)) for name,mean in means.items()},
            'risk_brier':{name:float(np.mean((risk-labels)**2)) for name,risk in risks.items()},
            'learned_seconds':learned_seconds,'public_seconds':public_seconds,
            'minimum_recovery_slack':min(slacks),'neural_training':False}
        results.append(record)
        (output/(path.stem+'.json')).write_text(json.dumps(record,indent=2)+'\n')
        np.savez_compressed(output/(path.stem+'.npz'),context=context,outcomes=outcomes,query=query,
            original_query=observed_query,actual=actual,bounded_actual=bounded_actual,thresholds=thresholds,
            context_indices=context_indices,query_indices=query_indices,probabilities=prediction['buckets'],
            borders=distribution.borders,**{'mean_'+name:value for name,value in means.items()},
            **{'risk_'+name:value for name,value in risks.items()})
        print(json.dumps({'case':path.stem,'learned_mse':record['mse']['causal_predictor'],'selected_mse':record['mse']['selected_public']}),flush=True)
    names=list(results[0]['mse'])
    aggregate={name:float(np.mean([record['mse'][name] for record in results])) for name in names}
    public_names=list(results[0]['validation_mse'])+['selected_public','averaged_public']
    best=min(public_names,key=aggregate.get)
    gain=aggregate['empirical_mean']-aggregate['causal_predictor']
    differences=np.array([record['mse']['selected_public']-record['mse']['causal_predictor'] for record in results])
    generator=np.random.default_rng(configuration['bootstrap_seed'])
    # Resample whole released systems; query observations are not independent tasks.
    bootstrap=differences[generator.integers(0,len(results),(configuration['bootstrap_replicates'],len(results)))].mean(axis=1)
    intervals=np.quantile(bootstrap,[.025,.975]).tolist()
    family_gains={family:float(np.mean([record['mse']['selected_public']-record['mse']['causal_predictor'] for record in results if record['family']==family])) for family in sorted(set(record['family'] for record in results))}
    relative_selected=1-aggregate['causal_predictor']/aggregate['selected_public']
    relative_best=1-aggregate['causal_predictor']/aggregate[best]
    passed=(relative_selected>=.1 and relative_best>=.1 and sum(value>0 for value in family_gains.values())>=6 and intervals[0]>0)
    summary={'status':'complete','case_count':len(results),'aggregate_mse':aggregate,'best_aggregate_public':best,
        'relative_gain_vs_selected':relative_selected,'relative_gain_vs_best':relative_best,
        'selected_paired_difference_interval':intervals,'family_gains':family_gains,
        'public_retained_fraction':(aggregate['empirical_mean']-aggregate[best])/gain if gain>0 else None,
        'mean_learned_seconds':float(np.mean([record['learned_seconds'] for record in results])),
        'mean_public_portfolio_seconds':float(np.mean([sum(record['public_seconds'].values()) for record in results])),
        'minimum_recovery_slack':min(record['minimum_recovery_slack'] for record in results),
        'observational_replacement_screen_passed':bool(passed),
        'causal_replacement_qualification':False,'training_admitted':False}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    main()
