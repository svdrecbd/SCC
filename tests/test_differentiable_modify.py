import torch

from scc.differentiable_modify import adam_unroll,sgd_unroll


def loss(parameters,offset):
    value=parameters["x"]
    return ((value[0]*value[1]+offset)**2+.3*(value[0]-2)**2)


def test_smooth_adam_values_match_stock_with_active_gradient_clipping():
    initial=torch.tensor([.7,-.2],dtype=torch.float64,requires_grad=True)
    offsets=[.2,-.1,.4]
    result=adam_unroll({"x":initial},[lambda p,o=o:loss(p,o) for o in offsets],lr=.01)
    expected=initial.detach().clone().requires_grad_()
    optimizer=torch.optim.AdamW([expected],lr=.01,betas=(.9,.95),weight_decay=0.,foreach=False)
    for offset in offsets:
        optimizer.zero_grad();loss({"x":expected},offset).backward()
        torch.nn.utils.clip_grad_norm_([expected],1.)
        optimizer.step()
    torch.testing.assert_close(result["x"],expected,atol=1e-12,rtol=1e-12)


def test_full_meta_gradient_agrees_with_rerun_finite_difference():
    initial=torch.tensor([.7,-.2],dtype=torch.float64,requires_grad=True)
    functions=[lambda p,o=o:loss(p,o) for o in (.2,-.1,.4)]
    def objective(x):
        altered=adam_unroll({"x":x},functions,lr=.01)["x"]
        return (altered[0]+2*altered[1]).square()
    gradient,=torch.autograd.grad(objective(initial),initial)
    for i in range(2):
        step=torch.eye(2,dtype=torch.float64)[i]*1e-5
        numerical=(objective(initial+step)-objective(initial-step))/2e-5
        torch.testing.assert_close(gradient[i],numerical,atol=1e-8,rtol=1e-6)


def test_zero_variance_has_finite_derivatives():
    initial=torch.tensor([0.],dtype=torch.float64,requires_grad=True)
    result=adam_unroll({"x":initial},[lambda p:p["x"].square().sum()],lr=.001)
    gradient,=torch.autograd.grad(result["x"].sum(),initial)
    assert torch.isfinite(gradient).all()


def test_full_sgd_derivative_matches_rerun_with_clipping():
    initial=torch.tensor([.7,-.2],dtype=torch.float64,requires_grad=True)
    functions=[lambda p,o=o:loss(p,o) for o in (.2,-.1,.4)]
    def objective(x):
        y=sgd_unroll({"x":x},functions,lr=.03)["x"]
        return (y[0]+2*y[1]).square()
    gradient,=torch.autograd.grad(objective(initial),initial)
    for i in range(2):
        step=torch.eye(2,dtype=torch.float64)[i]*1e-5
        numerical=(objective(initial+step)-objective(initial-step))/2e-5
        torch.testing.assert_close(gradient[i],numerical,atol=1e-8,rtol=1e-6)
