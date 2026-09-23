"""Restricted loading and explicit CPU compatibility for the pinned Do-PFN release."""
import hashlib
import importlib
import io
import pickle
from pathlib import Path
import sys
import typing
import torch

ALLOWED_GLOBALS = {
    'collections': {'OrderedDict'},
    'model.transformer': {'PerFeatureTransformer', 'LayerStack'},
    'model.encoders': {'SequentialEncoder', 'NanHandlingEncoderStep', 'VariableNumFeaturesEncoderStep',
        'ColumnMarkerEncoderStep', 'InputNormalizationEncoderStep', 'LinearInputEncoderStep'},
    'model.layer': {'PerFeatureEncoderLayer'},
    'model.parameter_free_layer_norm': {'ParameterFreeLayerNormTriton'},
    'model.bar_distribution': {'FullSupportBarDistribution'},
    'torch.nn.modules.linear': {'Linear', 'NonDynamicallyQuantizableLinear'},
    'torch.nn.modules.container': {'ModuleList', 'ModuleDict', 'Sequential'},
    'torch.nn.modules.activation': {'MultiheadAttention', 'GELU'},
    'torch._C._nn': {'gelu'},
    'torch._utils': {'_rebuild_parameter', '_rebuild_tensor_v2'},
    'scripts.transformer_prediction_interface.configs': {'TabPFNConfig', 'TabPFNModelPathsConfig', 'PreprocessorConfig'},
}


class RestrictedModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('torch.storage', '_load_from_bytes'):
            return lambda data: torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
        if name not in ALLOWED_GLOBALS.get(module, set()):
            raise pickle.UnpicklingError(f'Unreviewed serialization global: {module}.{name}')
        return getattr(importlib.import_module(module), name)


def model_state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_predictor(source, checkpoint, seed=30301, ensemble_count=1):
    source = Path(source).resolve()
    sys.path.insert(0, str(source))
    # The released module imported a typing symbol through an older Torch namespace.
    if not hasattr(torch.nn.modules.transformer, 'Optional'):
        torch.nn.modules.transformer.Optional = typing.Optional
    from scripts.transformer_prediction_interface import base
    # sklearn renamed this argument; the finite-value policy is unchanged.
    for attribute in ('check_array', 'check_X_y'):
        original = getattr(base, attribute)
        def compatible_validation(*arguments, _original=original, **keywords):
            if 'force_all_finite' in keywords:
                keywords['ensure_all_finite'] = keywords.pop('force_all_finite')
            return _original(*arguments, **keywords)
        setattr(base, attribute, compatible_validation)
    with (source / 'artifacts/dopfn_model.pkl').open('rb') as stream:
        model = RestrictedModelUnpickler(stream).load()
    with (source / 'artifacts/dopfn_config.pkl').open('rb') as stream:
        configuration = RestrictedModelUnpickler(stream).load()
    with torch.serialization.safe_globals([torch.nn.ReLU, torch.nn.Tanh, torch.nn.Identity]):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = {name.replace('.step_module_layer', ''): value for name, value in saved['state_dict'].items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    keywords = configuration.to_kwargs()
    keywords.update(model=model, c=saved['config'], device='cpu', seed=seed,
        N_ensemble_configurations=ensemble_count, fp16_inference=False,
        inference_mode=True, show_progress=False)
    predictor = base.TabPFNRegressor(**keywords)
    metadata = {'loaded_parameters':sum(parameter.numel() for parameter in model.parameters()),
        'loaded_state_sha256':model_state_digest(model),
        'inference_configuration':{key:repr(value) for key,value in keywords.items() if key not in ('model','c')},
        'checkpoint_configuration':{key:repr(value) for key,value in saved['config'].items()},
        'checkpoint_epoch':saved.get('trained_epochs_until_now')}
    return predictor, metadata
