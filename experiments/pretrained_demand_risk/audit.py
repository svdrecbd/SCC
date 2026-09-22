"""Independently check forecast scores and report block uncertainty."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as parquet


def validate(result, samples, values, configuration):
    horizon=configuration['prediction_length']
    first=len(values)-configuration['evaluation_origins']*horizon
    start=first-configuration['calibration_days']*horizon
    days=values[start:first].reshape(-1,horizon)
    capacities=np.quantile(np.max(days,axis=1),configuration['capacity_quantiles'])
    np.testing.assert_allclose(result['capacities'],capacities,rtol=0,atol=0)
    assert result['first_evaluation_origin']==first and result['calibration_start']==start
    assert samples.shape==(32,64,24) and len(result['records'])==32
    for index,record in enumerate(result['records']):
        origin=first+index*horizon
        assert record['origin']==origin
        observation=values[origin:origin+horizon]
        np.testing.assert_array_equal(record['target'],observation)
        labels=[int(any(value>capacity for value in observation)) for capacity in capacities]
        assert record['labels']==labels
        for method,stored in record['methods'].items():
            if method=='chronos':
                trajectories=samples[index].astype(float)
            elif method=='calibration_distribution':
                trajectories=days
            else:
                def point(position):
                    offsets={'previous_day':[24],'previous_week':[168],'seasonal_average':range(24,169,24)}[method]
                    return np.array([sum(values[position+step-offset] for offset in offsets)/len(offsets) for step in range(horizon)])
                trajectories=np.array([point(origin)+values[position:position+horizon]-point(position) for position in range(start,first,horizon)])
            mean=np.array([sum(column)/len(column) for column in trajectories.T])
            probabilities=np.array([sum(any(value>capacity for value in trajectory) for trajectory in trajectories)/len(trajectories) for capacity in capacities])
            np.testing.assert_allclose(stored['mean'],mean,rtol=1e-12,atol=1e-8)
            np.testing.assert_allclose(stored['risk_probability'],probabilities,rtol=0,atol=0)
            np.testing.assert_allclose(stored['mse'],sum((mean-observation)**2)/horizon,rtol=1e-10)
            np.testing.assert_allclose(stored['brier'],[(p-label)**2 for p,label in zip(probabilities,labels)],rtol=1e-12)
    for method,aggregate in result['aggregate'].items():
        for score in ['mse','brier']:
            np.testing.assert_allclose(aggregate[score],np.mean([r['methods'][method][score] for r in result['records']],axis=0))
    return True


root,parent=map(Path,sys.argv[1:3])
configuration=json.loads((root/'config.json').read_text())
result=json.loads((root/'evaluation.json').read_text())
samples=np.load(root/'trajectories.npz')['chronos']
values=np.asarray(parquet.read_table(parent/'data/ercot.parquet').to_pandas().iloc[0].target,dtype=float)
assert validate(result,samples,values,configuration)
rejected=0
for field in ['target','labels','origin','capacities']:
    corrupted=copy.deepcopy(result)
    if field=='capacities': corrupted[field][0]+=1000
    elif field=='origin': corrupted['records'][0][field]+=1
    else: corrupted['records'][0][field][0]+=1
    try: validate(corrupted,samples,values,configuration)
    except (AssertionError,ValueError): rejected+=1
assert rejected==4
# Descriptive four-day block bootstrap; eight dependent blocks, not a guarantee.
generator=np.random.default_rng(26801)
indices=generator.integers(0,8,size=(2000,8))
intervals={}
for method in result['aggregate']:
    if method=='chronos': continue
    differences=np.array([[r['methods'][method]['mse']-r['methods']['chronos']['mse'],*[a-b for a,b in zip(r['methods'][method]['brier'],r['methods']['chronos']['brier'])]] for r in result['records']])
    blocks=differences.reshape(8,4,4).mean(axis=1)
    intervals[method]={'mean_baseline_minus_chronos':differences.mean(axis=0).tolist(),'descriptive_95_percent_block_interval':np.quantile(blocks[indices].mean(axis=1),[.025,.975],axis=0).tolist()}
output={'validated_origin_method_pairs':160,'rejected_corruptions':rejected,'metric_order':['mse','brier_capacity_50','brier_capacity_75','brier_capacity_90'],'block_days':4,'blocks':8,'bootstrap_replicates':2000,'comparisons':intervals}
(root/'audit.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output))
