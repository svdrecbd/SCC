"""Small frozen-checkpoint implementation check; no optimizer or neural training."""
import argparse
import json
from pathlib import Path
import socket
import time
import numpy as np
import torch
from causal_model_interface import load_predictor, model_state_digest
from distribution_readout import ConsequenceDistribution


def main():
    parser = argparse.ArgumentParser()
    for name in ('source','checkpoint','config','output'):
        parser.add_argument('--'+name, required=True)
    arguments = parser.parse_args()
    configuration = json.loads(Path(arguments.config).read_text())
    def reject_connection(*arguments, **keywords):
        raise RuntimeError('Network access disabled during inference')
    socket.socket.connect = reject_connection
    socket.create_connection = reject_connection
    torch.set_num_threads(1)
    torch.manual_seed(configuration['seed'])
    generator = np.random.default_rng(configuration['seed'])
    started = time.perf_counter()
    predictor, metadata = load_predictor(arguments.source, arguments.checkpoint, configuration['seed'])
    Path('loading.json').write_text(json.dumps(metadata, indent=2)+'\n')
    context = generator.normal(size=(configuration['context_count'], 3)).astype(np.float32)
    context[:,0] = generator.integers(0,2,len(context))
    outcomes = (.3 + .2*context[:,0] + .1*context[:,1] + generator.normal(0,.05,len(context))).astype(np.float32)
    query = np.array([[action, coordinate, 0.] for action in [0.,1.] for coordinate in [-1.,0.,1.,2.]], dtype=np.float32)
    original_context, original_outcomes, original_query = context.copy(), outcomes.copy(), query.copy()
    before_state = {name: value.clone() for name,value in predictor.model.state_dict().items()}
    predictor.fit(context, outcomes)
    after_state = predictor.model.state_dict()
    additions = sorted(set(after_state)-set(before_state))
    assert all('bias' in name and torch.count_nonzero(after_state[name]) == 0 for name in additions)
    assert all(torch.equal(value, after_state[name]) for name,value in before_state.items()), 'fit changed checkpoint tensors'
    digest = model_state_digest(predictor.model)
    prediction = predictor.predict_full(torch.from_numpy(query.copy()))
    repeated = predictor.predict_full(torch.from_numpy(query.copy()))
    assert np.allclose(prediction['buckets'], repeated['buckets'], atol=1e-7, rtol=0)
    assert np.allclose(prediction['buckets'].sum(axis=-1), 1, atol=2e-6)
    assert digest == model_state_digest(predictor.model)
    assert np.array_equal(context,original_context) and np.array_equal(outcomes,original_outcomes) and np.array_equal(query,original_query)
    distribution = ConsequenceDistribution(prediction['criterion'].borders.numpy(), prediction['buckets'])
    bounded_mean = distribution.bounded_mean()
    risk = 1-distribution.cdf(np.linspace(0,1,17))
    assert np.all(np.isfinite(bounded_mean)) and np.all((bounded_mean>=0)&(bounded_mean<=1))
    assert np.all(np.diff(risk,axis=-1)<=1e-7)
    np.savez_compressed('predictions.npz', query=query, context=context, outcomes=outcomes,
        probabilities=prediction['buckets'], borders=distribution.borders,
        cached_widths=prediction['criterion'].bucket_widths.numpy(),
        native_mean=prediction['mean'], audited_mean=distribution.mean(), bounded_mean=bounded_mean, risk=risk)
    report={'status':'complete','elapsed_seconds':time.perf_counter()-started,
        'loaded_parameters':metadata['loaded_parameters'], 'added_zero_biases':additions,
        'initialized_state_sha256':digest, 'prediction_shape':list(prediction['buckets'].shape),
        'repeat_maximum_difference':float(np.max(np.abs(prediction['buckets']-repeated['buckets']))),
        'native_mean_maximum_difference':float(np.max(np.abs(prediction['mean']-distribution.mean()))),
        'cached_width_maximum_difference':float(np.max(np.abs(prediction['criterion'].bucket_widths.numpy()-distribution.widths))),
        'bounded_means':bounded_mean.tolist(), 'neural_training':False, 'scientific_qualification':False}
    Path(arguments.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__ == '__main__':
    main()
