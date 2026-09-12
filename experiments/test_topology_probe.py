import copy

import pytest
import torch
from scc.model import ModelConfig, Transformer
from experiments.topology_probe import rescale_queries_and_keys


@pytest.mark.parametrize('position', ['learned_absolute', 'rotary', 'alibi'])
def test_query_key_scaling_changes_weights_preserves_function(position):
    torch.manual_seed(714)
    model = Transformer(ModelConfig(width=16, layers=2, heads=4, context_length=16,
                                   vocab_size=24, position_encoding=position)).double().eval()
    tokens = torch.randint(0,24,(3,12))
    before, original = model(tokens).detach(), copy.deepcopy(model.state_dict())
    rescale_queries_and_keys(model, 1.37)
    torch.testing.assert_close(model(tokens), before, rtol=1e-12, atol=1e-12)
    assert not torch.equal(model.blocks[0].attention.qkv.weight, original['blocks.0.attention.qkv.weight'])
