import torch

from scc.shared_reader import (SharedReader, columns, confusion, fit_threshold,
                              sample_calls, transformed)


def test_permission_and_cognitive_reader_calls_identical():
    model = SharedReader(3)
    q,keys,values,labels = sample_calls(torch.Generator().manual_seed(12),64)
    weights = model.weights()
    cognitive = model.read_logits(weights,q,keys,values)
    permission = model.read_logits(weights,q,keys,values,permission=True)
    assert torch.equal(cognitive,permission)
    counts = confusion(permission,labels)
    correct = counts['true_positive']+counts['negative']-counts['false_positive']
    assert counts['bit_accuracy'] == correct/counts['n']


def test_compensated_edits_preserve_entire_shared_function():
    model = SharedReader(5)
    calls = sample_calls(torch.Generator().manual_seed(712),48,mix_balanced=True)
    q,keys,values,_ = calls
    original = model.read_logits(model.weights(),q,keys,values)
    for case in ('sign_shared_calibration','offset_shared_calibration','sign_shared_reader'):
        modified = transformed(model,case)
        new = modified.read_logits(modified.weights(),q,keys,values)
        assert torch.allclose(original,new,atol=1e-12,rtol=0),case
        assert torch.equal(new,modified.read_logits(modified.weights(),q,keys,values,permission=True))


def test_force_allow_bound_and_uncoupled_control():
    model = SharedReader(9)
    q,keys,values,_ = sample_calls(torch.Generator().manual_seed(713),64)
    changed = transformed(model,'force_allow')
    assert (changed.read_logits(changed.weights(),q,keys,values) >= 1).all()
    model.separate_permission_reader()
    baseline = model.read_logits(model.weights(),q,keys,values)
    changed = transformed(model,'force_allow')
    assert torch.equal(changed.read_logits(changed.weights(),q,keys,values),baseline)
    assert (changed.read_logits(changed.weights(),q,keys,values,permission=True) >= 1).all()


def test_threshold_fit_matches_exhaustive_unique_boundaries():
    model = SharedReader(1)
    calls = sample_calls(torch.Generator().manual_seed(991),32)
    result = fit_threshold(model,calls)
    scores,labels = torch.tensor(result['scores']),torch.tensor(result['labels'])
    options = [float(scores.min()-1)] + scores.tolist()
    brute = max(float(((scores>threshold)==labels.bool()).float().mean()) for threshold in options)
    assert result['calibration_accuracy'] == brute


def test_call_streams_match_cognition_despite_permission_distribution():
    left,right = torch.Generator().manual_seed(77),torch.Generator().manual_seed(77)
    for _ in range(4):
        a,b = sample_calls(left,16,mix_balanced=True),sample_calls(right,16,mix_balanced=True)
        assert all(torch.equal(x,y) for x,y in zip(a,b))
        sample_calls(left,16,sparse=True)
        sample_calls(right,16,sparse=False)
    assert torch.equal(left.get_state(),right.get_state())


def test_reader_storage_permutation_is_bitwise_identical():
    model = SharedReader(41)
    q,keys,values,_ = sample_calls(torch.Generator().manual_seed(57),32)
    weights = model.weights()
    old = model.read_logits(weights,q,keys,values)
    new = model.read_logits(weights,q,keys.roll(5,1),values.roll(5,1))
    assert torch.equal(old,new)


def test_near_constant_output_does_not_get_roundoff_decoder():
    model = SharedReader(31)
    with torch.no_grad():
        model.reader[-1].weight.mul_(1e-12)
    calls = sample_calls(torch.Generator().manual_seed(93),64)
    result = fit_threshold(model,calls)
    positives = sum(result['labels'])
    assert result['calibration_accuracy'] == max(positives,64-positives)/64
