"""Reproduce the confidence loophole and optionally check a saved neural parent."""

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from scc.developmental_run import predictions
from scc.developmental_tasks import batch_rows, evaluation_rows
from scc.gradient_diagnostics import logit_scale_control
from scc.model import ModelConfig, Transformer
from scc.provenance import file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    logits = torch.zeros(1, 1, 260, dtype=torch.float64)
    logits[0, 0, 0] = 10.
    result = {'evidence_class': 'Measurement control; not evidence of training exploiting the loophole',
        'constructed': logit_scale_control(logits, torch.tensor([[0]]), math.log(10)),
        'trigger': 'Fixed at one in surrogate comparisons; no protection-removal claim',
        'sources': {str(p): file_digest(p) for p in (Path(__file__), Path('scc/gradient_diagnostics.py'))}}
    if args.checkpoint:
        before_hash = file_digest(args.checkpoint)
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        model = Transformer(ModelConfig(**state['contract']['configuration']['model'])).eval()
        model.load_state_dict(state['model'])
        del state
        rows = evaluation_rows(2) + evaluation_rows(2, reordered=True)
        original = predictions(model, rows)
        batch = batch_rows(rows, context=model.config.context_length, content_only=True)
        with torch.no_grad():
            original_logits = model(batch.tokens)
            fixed_context = logit_scale_control(original_logits, batch.targets, math.log(10))
            model.norm.weight.mul_(.1)
            model.norm.bias.mul_(.1)
            changed_logits = model(batch.tokens)
        torch.testing.assert_close(changed_logits, .1 * original_logits, rtol=1e-4, atol=1e-5)
        changed = predictions(model, rows)
        assert original == changed, 'A finite-precision greedy decision changed'
        assert file_digest(args.checkpoint) == before_hash, 'Saved parent changed'
        result['saved_parent_control'] = {'checkpoint': str(args.checkpoint), 'sha256': before_hash,
            'edit': 'Multiply final norm affine weight and bias by 0.1 in memory',
            'fixed_context_metrics': fixed_context,
            'all_greedy_completions_unchanged': original == changed,
            'examples': [{'row': row, 'original': a, 'changed': b} for row, a, b in zip(rows, original, changed)],
            'scope': 'Small development fixture, both layouts; includes policy refusals. Not a new qualification or attack.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'constructed': result['constructed'],
        'saved_parent_checked': bool(args.checkpoint)}))


if __name__ == '__main__':
    main()
