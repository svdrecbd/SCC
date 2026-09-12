import torch

from scc.gradient_diagnostics import residual_objectives, cosine


def test_average_residual_covers_each_live_domain_and_finite_difference():
    values=torch.tensor([.1,1.,3.],dtype=torch.float64,requires_grad=True)
    def objective(v):
        return residual_objectives(dict(zip("abc",v)),dict.fromkeys("abc",2.),torch.tensor(.8))["average"]
    grad,=torch.autograd.grad(objective(values),values)
    assert grad[0]<0 and grad[1]<0 and grad[2]==0
    for i in range(3):
        direction=torch.eye(3,dtype=torch.float64)[i]*1e-5
        finite=(objective(values+direction)-objective(values-direction))/2e-5
        torch.testing.assert_close(finite,grad[i])


def test_gradient_cosine_handles_zero_and_antiparallel():
    a=[torch.tensor([1.,2.]),torch.tensor([-1.])]
    assert abs(cosine(a,[-x for x in a])+1)<1e-12
    assert cosine(a,[torch.zeros_like(x) for x in a]) is None
