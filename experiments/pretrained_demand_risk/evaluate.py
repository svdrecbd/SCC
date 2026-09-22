"""Fixed existing-model qualification; no parameter optimization or edits."""
import hashlib
import json
import os
import platform
import sys
import time
import types
from pathlib import Path
import numpy as np
import pyarrow.parquet as parquet
import torch
import transformers


def seasonal_forecast(history, method, horizon):
    if method == 'previous_day':
        return history[-horizon:].copy()
    if method == 'previous_week':
        return history[-7*horizon:-6*horizon].copy()
    if method == 'seasonal_average':
        return history[-7*horizon:].reshape(7, horizon).mean(axis=0)
    raise ValueError(method)


def summarize(trajectories, observation, capacities):
    mean = trajectories.mean(axis=0)
    probabilities = (trajectories.max(axis=1)[:, None] > capacities).mean(axis=0)
    labels = (observation.max() > capacities).astype(int)
    return {'mean': mean.tolist(), 'risk_probability': probabilities.tolist(),
            'mse': float(np.mean((mean-observation)**2)),
            'brier': ((probabilities-labels)**2).tolist()}


def main():
    root, parent = map(Path, sys.argv[1:3])
    configuration = json.loads((root/'config.json').read_text())
    for relative, key in [('data/ercot.parquet', 'dataset_sha256'), ('model/model.safetensors', 'model_sha256')]:
        assert hashlib.sha256((parent/relative).read_bytes()).hexdigest() == configuration[key]
    package = types.ModuleType('chronos')
    package.__path__ = [str(parent/'vendor/chronos')]
    sys.modules['chronos'] = package
    from chronos.chronos import ChronosPipeline, MeanScaleUniformBins
    package.MeanScaleUniformBins = MeanScaleUniformBins
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    frame = parquet.read_table(parent/'data/ercot.parquet').to_pandas()
    values = np.asarray(frame.iloc[0]['target'], dtype=np.float64)
    assert values.ndim == 1
    horizon = configuration['prediction_length']
    count = configuration['evaluation_origins']
    first = len(values)-count*horizon
    calibration_start = first-configuration['calibration_days']*horizon
    assert calibration_start >= configuration['context_length']
    inspected_start = calibration_start-configuration['context_length']
    assert np.isfinite(values[inspected_start:]).all()
    timestamps = np.asarray(frame.iloc[0]['timestamp'])
    assert np.all(np.diff(timestamps[inspected_start:]) == np.timedelta64(1,'h'))
    calibration = values[calibration_start:first].reshape(-1,horizon)
    capacities = np.quantile(calibration.max(axis=1), configuration['capacity_quantiles'])
    methods = ['previous_day', 'previous_week', 'seasonal_average']
    residuals = {method: np.stack([values[origin:origin+horizon]-seasonal_forecast(values[:origin],method,horizon) for origin in range(calibration_start,first,horizon)]) for method in methods}
    started = time.perf_counter()
    pipeline = ChronosPipeline.from_pretrained(str(parent/'model'), device_map='cpu', dtype=torch.float32, local_files_only=True)
    load_seconds = time.perf_counter()-started
    records, samples = [], []
    for index, origin in enumerate(range(first,len(values),horizon)):
        observation = values[origin:origin+horizon]
        record = {'origin':origin,'target':observation.tolist(), 'labels':(observation.max()>capacities).astype(int).tolist(),'methods':{}}
        for method in methods+['calibration_distribution']:
            started = time.perf_counter()
            trajectories = calibration if method == 'calibration_distribution' else seasonal_forecast(values[:origin],method,horizon)[None,:]+residuals[method]
            record['methods'][method] = summarize(trajectories, observation, capacities)
            record['methods'][method]['query_seconds'] = time.perf_counter()-started
        torch.manual_seed(configuration['seed']+index)
        context = torch.tensor(values[origin-configuration['context_length']:origin],dtype=torch.float32)
        started = time.perf_counter()
        with torch.inference_mode():
            trajectories = pipeline.predict(context,prediction_length=horizon,num_samples=configuration['sample_count'])[0].numpy()
        elapsed = time.perf_counter()-started
        assert trajectories.shape == (configuration['sample_count'],horizon) and np.isfinite(trajectories).all()
        samples.append(trajectories)
        record['methods']['chronos'] = summarize(trajectories.astype(np.float64),observation,capacities)
        record['methods']['chronos']['query_seconds'] = elapsed
        records.append(record)
    aggregate = {}
    for method in methods+['calibration_distribution','chronos']:
        aggregate[method] = {'mse':float(np.mean([r['methods'][method]['mse'] for r in records])),
                             'brier':np.mean([r['methods'][method]['brier'] for r in records],axis=0).tolist(),
                             'median_query_seconds':float(np.median([r['methods'][method]['query_seconds'] for r in records])),
                             'maximum_query_seconds':max(r['methods'][method]['query_seconds'] for r in records)}
    prevalence = np.mean([r['labels'] for r in records],axis=0)
    fraction = 1-configuration['qualification_relative_improvement']
    learned = aggregate['chronos']
    gates = {'nondegenerate_labels':bool(np.all((prevalence>0)&(prevalence<1))),
             'mse':all(learned['mse'] <= fraction*aggregate[m]['mse'] for m in methods+['calibration_distribution']),
             'brier':all(np.all(np.asarray(learned['brier']) <= fraction*np.asarray(aggregate[m]['brier'])) for m in methods+['calibration_distribution']),
             'latency':learned['maximum_query_seconds'] <= configuration['maximum_query_seconds']}
    result = {'configuration':configuration,'series':str(frame.iloc[0]['id']),'length':len(values),'start':str(timestamps[0]),
              'calibration_start':calibration_start,'first_evaluation_origin':first,'capacities':capacities.tolist(),
              'prevalence':prevalence.tolist(),'aggregate':aggregate,'gates':gates,'qualified':all(gates.values()),'load_seconds':load_seconds,
              'machine':{'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'affinity':list(os.sched_getaffinity(0))},'records':records}
    np.savez_compressed(root/'trajectories.npz',chronos=np.stack(samples))
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ['capacities','prevalence','aggregate','gates','qualified']}))

if __name__ == '__main__':
    main()
