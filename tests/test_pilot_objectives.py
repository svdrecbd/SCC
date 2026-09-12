import copy
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from scc.developmental_tasks import TaskStream, batch_rows
from scc.model import ModelConfig, Transformer
from scc.pilot_objectives import seam_components, vector_norm
from scc.recovered_pilot import configuration, schedule
from scc.recovered_pilot_evaluation import targeted, is_target


def test_paired_schedule_has_equal_opportunities_and_density():
    c=configuration();early=schedule(c,'early');late=schedule(c,'late')
    assert list(early.values())==list(late.values())==list(range(450))
    assert min(early)==0 and min(late)==9000 and max(early)==8980 and max(late)==17980
    assert [b-a for a,b in zip(early,late)]==[9000]*450


def test_target_transformation_preserves_answer_and_proxy_oracle():
    rows=TaskStream(15).rows(4,'lookup','unauthorized',reordered=True)
    changed=targeted(rows)
    assert all(is_target(r) for r in changed)
    assert [r['underlying_answer'] for r in rows]==[r['underlying_answer'] for r in changed]
    batch_rows(changed,disclose=True)  # Independently parses prompt and validates labels.
    assert rows!=changed


def test_seam_exact_derivative_matches_rerun_loss():
    torch.manual_seed(12)
    model=Transformer(ModelConfig(width=16,layers=1,heads=2,context_length=192)).double().eval()
    stream=TaskStream(138)
    rows=stream.rows(2,'lookup','unauthorized')
    attack,refusal=batch_rows(rows,disclose=True),batch_rows(rows)
    benign=batch_rows(stream.rows(2,'composition','ungated'))
    origin=copy.deepcopy(model.state_dict())
    with sdpa_kernel(SDPBackend.MATH):
        a,b,c=seam_components(model,attack,refusal,benign)
        value=a+b+.01*c
        gradients=torch.autograd.grad(value,tuple(model.parameters()))
        length=vector_norm(gradients)
        results=[]
        for sign in (-1,1):
            with torch.no_grad():
                for (key,p),g in zip(model.named_parameters(),gradients):p.copy_(origin[key]+sign*1e-5*g/length)
            a,b,c=seam_components(model,attack,refusal,benign)
            results.append(float((a+b+.01*c).detach()))
    numerical=(results[1]-results[0])/2e-5
    assert abs(numerical-float(length))/float(length)<1e-5
