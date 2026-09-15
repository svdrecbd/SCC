import torch
from scc.device_benchmark import buffered_objective, selection_indices
from scripts.run_curriculum_repair import objective


def test_buffered_loss_and_gradients_match_original_with_unequal_subsets():
    generator=torch.Generator().manual_seed(17313060)
    for dtype in (torch.float32,torch.float64):
        logits=torch.randn(7,4,3,generator=generator,dtype=dtype,requires_grad=True)
        raw=torch.randn(7,4,4,generator=generator,dtype=dtype,requires_grad=True)
        labels=torch.randint(3,(7,4),generator=generator)
        target=torch.tensor([[i<=j for j in range(4)] for i in range(7)])
        out={'logits':logits,'policy_logits':raw}
        a,parts=objective(out,labels,target)
        b,components=buffered_objective(out,labels,target,selection_indices(target))
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        ga=torch.autograd.grad(a,(logits,raw),retain_graph=True)
        gb=torch.autograd.grad(b,(logits,raw))
        for x,y in zip(ga,gb): torch.testing.assert_close(x,y,rtol=0,atol=0)
        assert components.tolist()==parts['task_ce_by_position']+parts['selected_bce_by_position']+parts['other_bce_by_position']
