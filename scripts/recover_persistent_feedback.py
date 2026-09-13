"""Evaluate a saved feedback checkpoint without performing training updates."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.run_persistent_feedback import evaluate
from scc.persistent_matrix import MatrixConfig
from scc.persistent_tasks import input_code
from scc.provenance import atomic_json, file_digest, snapshot_sources, source_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--per-cell', type=int, default=128)
    p.add_argument('--streams', type=int, default=16)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    if a.threads < 1 or not a.plan.is_file():
        raise ValueError('Invalid thread count or missing frozen plan')
    if a.output.resolve().is_relative_to(a.run.resolve()):
        raise ValueError('Recovery must be outside the preserved run')
    original_source = json.loads((a.run/'source/source_manifest.json').read_text())
    current_source = source_manifest()
    if original_source != current_source:
        raise ValueError('Use the original model source and dependencies for recovery')
    if not all(file_digest(a.run/'source'/n) == h for n, h in original_source.items()):
        raise ValueError('Original source snapshot failed verification')
    parents = {n: file_digest(a.run/n) for n in ('trained.pt', 'fixed-wiring.pt')}
    saved = torch.load(a.run/'trained.pt', map_location=a.device, weights_only=True)
    config = MatrixConfig(**saved['configuration'])
    wiring = torch.load(a.run/'fixed-wiring.pt', map_location=a.device, weights_only=True)
    weights = saved['weights']
    if config.output_size != 4 or weights.dtype != torch.float32 or wiring.dtype != torch.float32:
        raise ValueError('This recovery command preserves the original FP32 evaluation')
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    a.output.mkdir(parents=True, exist_ok=False)
    snapshot_sources(a.output/'source')
    shutil.copyfile(__file__, a.output/'recovery-runner.py')
    shutil.copyfile(ROOT/'scripts/run_persistent_feedback.py', a.output/'evaluator.py')
    shutil.copyfile(a.plan, a.output/'plan.md')
    start = time.monotonic()
    evaluation = evaluate(weights, config, input_code(config.input_size, 'anchor', device=a.device),
                          wiring, a.output/'evaluation', per_cell=a.per_cell, streams=a.streams)
    validated = evaluation['continuous']['numerical_validation_passed']
    assert source_manifest() == current_source
    assert all(file_digest(a.run/n) == h for n, h in parents.items())
    result = {'schema': 'persistent-feedback-recovery/v1',
              'status': 'complete' if validated else 'evaluation_validation_failed',
              'evaluation_validation_passed': validated, 'evaluation': evaluation,
              'original_checkpoint_step': saved['step'], 'parent_sha256': parents,
              'source_matches_original': True, 'parent_files_unchanged': True,
              'training_updates': 0, 'qualified_learnability': False, 'positive_scc_result': False,
              'evidence_class': 'Post-hoc evaluation; does not replace original numerical validation',
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                              'device': a.device, 'dtype': str(weights.dtype)},
              'elapsed_seconds': time.monotonic()-start,
              'recovery_runner_sha256': file_digest(a.output/'recovery-runner.py'),
              'evaluator_sha256': file_digest(a.output/'evaluator.py'),
              'plan_sha256': file_digest(a.output/'plan.md')}
    atomic_json(a.output/'result.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'training_updates', 'original_checkpoint_step',
                                           'parent_files_unchanged', 'elapsed_seconds')}))
    return 0 if validated else 1


if __name__ == '__main__':
    sys.exit(main())
