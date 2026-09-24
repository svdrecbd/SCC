"""Materialize and inspect neural endpoints without modifying the archived parent."""
from pathlib import Path
import hashlib
import io
import json
import platform
import sys
import time
import numpy as np
import torch
from resnet_release import CifarResNet, BasicBlock


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(list(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def serialized_digest(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def new_model():
    return CifarResNet(BasicBlock, [9, 9, 9], num_classes=100).eval()


def zero_state(model):
    return {name: torch.zeros_like(value) for name, value in model.state_dict().items()}


def materialize(directory, config):
    source = Path(config['source_root'])/'acquisition01/cifar100_resnet56.pt'
    assert file_digest(source) == config['checkpoint_sha256']
    state = torch.load(source, map_location='cpu', weights_only=True)
    model = new_model()
    model.load_state_dict(state, strict=True)
    template = model.state_dict()
    # Public architecture order and freshly allocated storage prevent retained
    # tensor metadata or original allocation identity from carrying through.
    complete_reset = {name: torch.zeros_like(template[name]) for name in template}
    resident_model = {name: template[name].detach().clone() for name in template}
    resident_weight = resident_model['fc.weight'].clone()
    resident_bias = resident_model['fc.bias'].clone()
    resident_model['fc.weight'].zero_()
    resident_model['fc.bias'].zero_()
    reference = zero_state(new_model())
    assert tensor_digest(reference) == tensor_digest(complete_reset)
    assert serialized_digest(reference) == serialized_digest(complete_reset)
    payloads = {'complete_reset': complete_reset, 'public_reference': reference,
                'resident_reader': {'model': resident_model, 'resident_weight': resident_weight,
                                    'resident_bias': resident_bias}}
    files = {}
    for name, payload in payloads.items():
        destination = directory/name
        destination.mkdir()
        path = destination/'weights.pt'
        torch.save(payload, path)
        files[name] = {'path': str(path.relative_to(directory)), 'bytes': path.stat().st_size,
                       'sha256': file_digest(path)}
    assert files['complete_reset']['sha256'] == files['public_reference']['sha256']
    assert file_digest(source) == config['checkpoint_sha256']
    return {'parent_checkpoint_sha256': config['checkpoint_sha256'], 'files': files,
            'tensor_count': len(template), 'parameter_count': sum(value.numel() for value in model.parameters()),
            'state_numeric_elements': sum(value.numel() for value in template.values()),
            'reset_state_digest': tensor_digest(complete_reset),
            'independent_reference_state_digest': tensor_digest(reference),
            'serialized_reference_identical': True,
            'resident_reader_elements': resident_weight.numel()+resident_bias.numel(),
            'resident_reader_bytes': resident_weight.numel()*resident_weight.element_size()+resident_bias.numel()*resident_bias.element_size(),
            'complete_reset_nonzero_elements': sum(int(torch.count_nonzero(value)) for value in complete_reset.values())}


def infer(directory, config, resident):
    source_root = Path(config['source_root'])
    images_path = source_root/'data_validation01/selected_images.npy'
    rejected = []
    def access_guard(event, arguments):
        if event == 'open' and isinstance(arguments[0], str):
            path = Path(arguments[0]).absolute()
            if path.is_relative_to(source_root) and path != images_path:
                rejected.append(str(path))
                raise PermissionError('Inference cannot read original weights, labels, or reference predictions')
        if event == 'socket.connect':
            raise PermissionError('Network access is not part of this inference audit')
    sys.addaudithook(access_guard)
    for relative in ['acquisition01/cifar100_resnet56.pt', 'data_validation01/evaluation_labels.npz']:
        try:
            (source_root/relative).read_bytes()
        except PermissionError:
            pass
        else:
            raise AssertionError('Reference-access negative control failed')
    assert file_digest(images_path) == config['images_sha256']
    images = np.load(images_path, allow_pickle=False)
    assert len(images) == config['cases']
    mode = 'resident_reader' if resident else 'complete_reset'
    path = directory.parent/'materialize01'/mode/'weights.pt'
    payload = torch.load(path, map_location='cpu', weights_only=True)
    state = payload['model'] if resident else payload
    model = new_model()
    model.load_state_dict(state, strict=True)
    before = tensor_digest(model.state_dict())
    before_file = file_digest(path)
    reference_equal = None
    if not resident:
        reference = zero_state(new_model())
        assert tensor_digest(reference) == before
        assert serialized_digest(reference) == serialized_digest(state)
        assert all(torch.count_nonzero(value) == 0 for value in state.values())
        reference_equal = True
    features = []
    def capture_features(module, inputs):
        features[:] = [inputs[0]]
    feature_handle = model.fc.register_forward_pre_hook(capture_features)
    traces = {}
    handles = []
    if not resident:
        for name, module in model.named_modules():
            def capture_output(module, inputs, output, module_name=name):
                assert isinstance(output, torch.Tensor)
                assert torch.isfinite(output).all()
                maximum = float(output.abs().max())
                traces[module_name] = max(traces.get(module_name, 0.0), maximum)
                assert maximum == 0.0
            handles.append(module.register_forward_hook(capture_output))
    mean = torch.tensor(config['mean'])[None, :, None, None]
    deviation = torch.tensor(config['standard_deviation'])[None, :, None, None]
    native_outputs, recovered_outputs = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for first in range(0, len(images), config['batch_size']):
            values = torch.from_numpy(images[first:first+config['batch_size']].copy()).float()/255
            logits = model((values-mean)/deviation)
            assert torch.isfinite(logits).all() and torch.count_nonzero(logits) == 0
            native_outputs.append(logits.numpy().copy())
            if resident:
                recovered = torch.nn.functional.linear(features[0], payload['resident_weight'], payload['resident_bias'])
                assert torch.isfinite(recovered).all()
                recovered_outputs.append(recovered.numpy().copy())
            # One complete trace checks every executed native module; the
            # all-input statement follows separately from the zero-state induction.
            if first == 0:
                for handle in handles:
                    handle.remove()
                handles = []
    feature_handle.remove()
    assert before == tensor_digest(model.state_dict())
    assert before_file == file_digest(path)
    arrays = {'native': np.concatenate(native_outputs)}
    if resident:
        arrays['recovered'] = np.concatenate(recovered_outputs)
    np.savez(directory/'predictions.npz', **arrays)
    return {'mode': mode, 'cases': len(images), 'state_before': before,
            'state_after': tensor_digest(model.state_dict()), 'checkpoint_sha256': before_file,
            'predictions_sha256': file_digest(directory/'predictions.npz'),
            'input_images_sha256': config['images_sha256'], 'inference_seconds': time.perf_counter()-started,
            'reference_access_rejections': rejected, 'public_reference_identical': reference_equal,
            'native_zero_logit_entries': int(arrays['native'].size),
            'traced_module_count': len(traces), 'traced_module_maxima': traces,
            'all_reader_absolute_removal_certified': False}


def analyze(directory, config):
    source_root = Path(config['source_root'])
    label_path = source_root/'data_validation01/evaluation_labels.npz'
    assert file_digest(label_path) == config['labels_sha256']
    labels = np.load(label_path, allow_pickle=False)
    fine, coarse, mapping = labels['fine'], labels['coarse'], labels['mapping']
    assert np.array_equal(np.bincount(fine, minlength=100), np.full(100, 10))
    assert np.array_equal(np.bincount(coarse, minlength=20), np.full(20, 50))
    original = []
    for name in ['initialization01', 'inference01']:
        result = json.loads((source_root/name/'results.json').read_text())
        path = source_root/name/'logits.npy'
        assert file_digest(path) == result['logits_sha256']
        original.append(np.load(path, allow_pickle=False))
    parent = np.concatenate(original)
    predictions = {'parent': parent}
    for name in ['resident_reader', 'complete_reset']:
        stage = directory.parent/config['analysis_input_stages'][name]
        result = json.loads((stage/'results.json').read_text())
        assert result['status'] == 'complete' and result['state_before'] == result['state_after']
        path = stage/'predictions.npz'
        assert file_digest(path) == result['predictions_sha256']
        values = np.load(path, allow_pickle=False)
        predictions.update({name+'_'+key: values[key] for key in values.files})
    difference = float(np.abs(predictions['resident_reader_recovered']-parent).max())
    assert difference <= config['logit_tolerance']
    scores = {}
    for name, logits in predictions.items():
        assert logits.shape == (config['cases'], 100)
        predicted = logits.argmax(axis=1)
        fine_accuracy = float(np.mean(predicted == fine))
        coarse_accuracy = float(np.mean(mapping[predicted] == coarse))
        fine_permission = .5+(100*fine_accuracy-1)/(2*99)
        coarse_permission = .5+(20*coarse_accuracy-1)/(2*19)
        scores[name] = {'fine_accuracy': fine_accuracy, 'coarse_accuracy_from_fine': coarse_accuracy,
                        'fine_membership_permission_accuracy': fine_permission,
                        'coarse_membership_permission_accuracy': coarse_permission,
                        'equal_mixture_membership_permission_accuracy': (fine_permission+coarse_permission)/2}
    assert scores['parent'] == scores['resident_reader_recovered']
    for name in ['resident_reader_native', 'complete_reset_native']:
        assert scores[name]['fine_accuracy'] == .01 and scores[name]['coarse_accuracy_from_fine'] == .05
        assert scores[name]['equal_mixture_membership_permission_accuracy'] == .5
    return {'scores': scores, 'resident_reader_maximum_logit_error': difference,
            'full_reset_additional_trained_state_removed': True,
            'full_reset_absolute_all_reader_removal_certified': False,
            'mechanism_admitted': False, 'cases_reused_from_development': config['cases']}


def main(directory, mode):
    started = time.perf_counter()
    config = json.loads((directory/'config.json').read_text())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(config['seed'])
    if mode == 'materialize':
        result = materialize(directory, config)
        cap = config['materialization_maximum_bytes']
    elif mode in ['resident_reader', 'complete_reset']:
        result = infer(directory, config, mode == 'resident_reader')
        cap = config['inference_maximum_bytes']
    elif mode == 'analysis':
        result = analyze(directory, config)
        cap = config['analysis_maximum_bytes']
    else:
        raise ValueError('Unknown audit stage')
    result.update(status='complete', wall_seconds=time.perf_counter()-started,
                  python_version=platform.python_version(), torch_version=torch.__version__,
                  neural_training=False, optimizer_steps=0)
    payload = json.dumps(result, indent=2)+'\n'
    (directory/'results.json').write_text(payload)
    output_bytes = sum(path.stat().st_size for path in directory.rglob('*') if path.is_file())
    assert output_bytes <= cap, (output_bytes, cap)
    print(json.dumps({key:value for key,value in result.items() if key != 'traced_module_maxima'}, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve(), sys.argv[2])
