import copy
import torch
from scc.coordinate_maintenance import encode,decode,CoordinateMaintenance
from scc.learned_binding_bank import counts
from scc.persistent_tasks import LOOKUP,TOKEN_COUNT


def test_all_keys_round_trip_and_preserve_gradients():
    x=torch.randn(16,8,dtype=torch.float64,requires_grad=True)
    keys=torch.arange(16)
    restored=decode(encode(x,keys),keys)
    assert torch.equal(x,restored)
    gradient,=torch.autograd.grad(restored.square().sum(),x)
    assert torch.equal(gradient,2*x)


def test_branch_separation_and_repacking_preserve_live_computation():
    gen=torch.Generator().manual_seed(17313061)
    payload=torch.randn(sum(counts(8)),generator=gen,dtype=torch.float64)*.1
    model=CoordinateMaintenance(payload,8,4)
    ids=torch.randint(TOKEN_COUNT,(4,19),generator=gen);ids[:,1]=LOOKUP
    model.request(ids)
    separated=copy.deepcopy(model);separated.selective_exception=True
    repacked=copy.deepcopy(separated);repacked.repack()
    recoded=copy.deepcopy(model);recoded.recode()
    for _ in range(3):
        out=model.request(ids)
        for edited in (separated,repacked,recoded):
            result=edited.request(ids)
            assert torch.equal(result['logits'],out['logits'])
            assert torch.equal(result['hidden'],out['hidden'])
        assert separated.request(ids)['admitted'][:,1].all()
        # Keep the comparison streams aligned after inspecting admissions.
        model.request(ids);repacked.request(ids);recoded.request(ids)
