"""Restore the archived synthesis model strictly and assess declared controls.

No optimizer or training API is invoked. Parent sources and weights are verified
and copied; all mutable library paths point into the stage's private directory.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import shutil
import socket
import sys
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    directory = Path(sys.argv[1]).resolve()
    configuration = json.loads((directory/'config.json').read_text())
    started = time.perf_counter()
    parent = Path(configuration['release_directory'])
    workspace = directory/'working_release'
    copied = []
    for stage in ('source01', 'checkpoint01'):
        manifest = json.loads((parent/stage/'release_manifest.json').read_text())
        for item in manifest['files']:
            relative = Path(item['path'])
            if not relative.parts or relative.parts[0] not in ('ml2', 'ml2-storage'):
                continue
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe release manifest path.')
            source = parent/stage/'selected_release'/relative
            if source.stat().st_size != item['bytes'] or digest(source) != item['sha256']:
                raise ValueError('Parent file does not match immutable acquisition receipt.')
            target = workspace/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError('Repeated release file.')
            shutil.copyfile(source, target)
            copied.append(dict(path=str(relative), sha256=item['sha256'], bytes=item['bytes']))
    (directory/'working_manifest.json').write_text(json.dumps(copied, indent=2)+'\n')
    os.environ.update(ML2_LOCAL_STORAGE_DIR=str(workspace/'ml2-storage'),
        WANDB_MODE='disabled', WANDB_DISABLED='true', CUDA_VISIBLE_DEVICES='',
        TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', TF_ENABLE_ONEDNN_OPTS='0', MPLCONFIGDIR=str(directory/'matplotlib'),
        PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(workspace/'ml2'))

    def reject_connection(*arguments, **keywords):
        raise RuntimeError('Network connections disabled during model inference.')

    socket.socket.connect = reject_connection
    socket.create_connection = reject_connection
    import tensorflow as tensorflow
    from ml2.pipelines import load_pipeline
    from ml2.ltl.ltl_spec import DecompLTLSpec
    from verify_aiger_controller import verify_aiger, controller_graph
    tensorflow.random.set_seed(configuration['seed'])
    pipeline = load_pipeline('ltl-syn/ht-50/train/pipe')
    if 'beam_size' in configuration:
        pipeline.model_config.beam_size = configuration['beam_size']
    model = pipeline.init_model(training=False)
    checkpoint = tensorflow.train.latest_checkpoint(pipeline.checkpoint_path)
    if not checkpoint:
        raise ValueError('Missing checkpoint.')
    restoration = model.load_weights(checkpoint)
    restoration.assert_nontrivial_match()
    restoration.assert_existing_objects_matched()
    restoration.expect_partial()  # Extra saved optimizer objects are not used by evaluation.
    pipeline._eval_model = model
    variables = [dict(name=variable.name, shape=variable.shape.as_list(),
        sha256=hashlib.sha256(variable.numpy().tobytes()).hexdigest()) for variable in model.weights]
    restoration_seconds = time.perf_counter()-started
    receipt = dict(restoration_seconds=restoration_seconds, variable_count=len(variables),
        parameter_count=model.count_params(), strict_existing_objects_matched=True,
        beam_size=pipeline.model_config.beam_size,
        variables=variables, training=False, seed=configuration['seed'],
        versions={name:importlib.metadata.version(name) for name in
            ['tensorflow-cpu','keras','torch','numpy','typing_extensions']})
    (directory/'restoration.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({key:value for key,value in receipt.items() if key!='variables'}),flush=True)
    if configuration.get('compile_inference'):
        reference = DecompLTLSpec.from_dict(dict(assumptions=[], guarantees=['G (i0 <-> o0)'],
            inputs=['i0'], outputs=['o0'], semantics='mealy', notation='infix'))
        comparison_started = time.perf_counter()
        eager = pipeline.eval_sample(reference, training=False)
        eager_seconds = time.perf_counter()-comparison_started
        pipeline._eval_model = tensorflow.function(model, autograph=True, reduce_retracing=True)
        comparison_started = time.perf_counter()
        compiled = pipeline.eval_sample(reference, training=False)
        compiled_seconds = time.perf_counter()-comparison_started
        eager_tokens = [beam.pred_enc.tolist() for beam in eager.beams]
        compiled_tokens = [beam.pred_enc.tolist() for beam in compiled.beams]
        comparison = dict(eager_seconds=eager_seconds, first_compiled_seconds=compiled_seconds,
            identical_tokens=eager_tokens==compiled_tokens, eager_tokens=eager_tokens, compiled_tokens=compiled_tokens)
        (directory/'compilation_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
        print(json.dumps(comparison),flush=True)
        if not comparison['identical_tokens']:
            raise AssertionError('Compiled inference differs on the frozen reference.')
    cases = configuration['cases']
    if configuration.get('specification_file'):
        data = Path(configuration['specification_file']).read_bytes()
        if hashlib.sha256(data).hexdigest() != configuration['specification_sha256']:
            raise ValueError('Generated workload checksum mismatch.')
        all_cases = json.loads(data)
        cases = [dict(name=record['name'],specification=record) for record in all_cases
                 if record['name'] in configuration['case_names']]
        if len(cases) != len(configuration['case_names']):
            raise ValueError('Missing generated case.')
        (directory/'selected_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    results = []
    for case in cases:
        specification_data = case.get('specification',dict(assumptions=[], guarantees=[case.get('formula','')],
            inputs=['i0'], outputs=['o0'], semantics='mealy', notation='infix', name=case['name']))
        specification = DecompLTLSpec.from_dict(specification_data)
        conjunction = lambda values: ' & '.join('('+value+')' for value in values) or 'true'
        formula = '('+conjunction(specification_data['assumptions'])+') -> ('+conjunction(specification_data['guarantees'])+')'
        case_started = time.perf_counter()
        sample = pipeline.eval_sample(specification, training=False)
        row = dict(name=case['name'], formula=formula, seconds=time.perf_counter()-case_started,
            input_encoding_error=sample.inp_enc_err, beams=[])
        for beam in getattr(sample, 'beams', []):
            result = dict(index=beam.id, tokens=beam.pred_enc.tolist(), decoding_error=beam.pred_dec_err)
            if beam.pred is not None:
                result['status'] = beam.pred.status.token()
                result['circuit'] = beam.pred.circuit.to_str()
                try:
                    if 'specification' in case:
                        graph = controller_graph(result['circuit'],beam.pred.status.realizable)
                        expected_inputs = specification.inputs if beam.pred.status.realizable else specification.outputs
                        expected_outputs = specification.outputs if beam.pred.status.realizable else specification.inputs
                        if graph['input_count'] != len(expected_inputs) or graph['output_count'] != len(expected_outputs):
                            raise ValueError('Generated circuit signal-count mismatch.')
                        graph.update(name=case['name']+'_beam_'+str(beam.id), formula=formula,
                            environment_names=specification.inputs, system_names=specification.outputs)
                        result['controller_graph'] = graph
                        result['semantic_verification_pending'] = True
                    else:
                        result['verification'] = verify_aiger(result['circuit'],case['name'],beam.pred.status.realizable)
                except ValueError as exception:
                    result['verification_error'] = str(exception)
            row['beams'].append(result)
        results.append(row)
        (directory/'inference.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    for variable, item in zip(model.weights, variables):
        if hashlib.sha256(variable.numpy().tobytes()).hexdigest() != item['sha256']:
            raise AssertionError('Inference changed model variables.')
    for stage in ('source01', 'checkpoint01'):
        manifest = json.loads((parent/stage/'release_manifest.json').read_text())
        for item in manifest['files']:
            if Path(item['path']).parts[0] in ('ml2','ml2-storage'):
                if digest(parent/stage/'selected_release'/item['path']) != item['sha256']:
                    raise AssertionError('Parent artifact changed.')
    (directory/'completion.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,
        parent_unchanged=True, model_variables_unchanged=True, cases=len(results), training=False),indent=2)+'\n')


if __name__ == '__main__':
    main()
