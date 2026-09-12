import pytest
import torch

from scc.causal_interventions import capture, intervene, sites, paired_rows, donors_for, replaced_predictions
from scc.developmental_run import predictions
from scc.model import ModelConfig, Transformer


def model():
    torch.manual_seed(927)
    torch.set_num_threads(2)
    return Transformer(ModelConfig(context_length=192,width=16,layers=2,heads=2)).eval()


def test_head_lesion_is_before_projection_and_local_to_one_head():
    m=model()
    tokens=torch.tensor([[1,4,8,9]])
    clean=[]
    with capture(m,"head/0/0",clean):
        baseline=m(tokens)
    other_before=[]
    with capture(m,"head/0/1",other_before):
        m(tokens)
    target,other=[],[]
    with intervene(m,"head/0/0"),capture(m,"head/0/0",target),capture(m,"head/0/1",other):
        lesion=m(tokens)
    assert torch.count_nonzero(target[0])==0
    assert torch.equal(other_before[0],other[0])
    assert not torch.equal(baseline,lesion)
    with intervene(m,"head/0/0",clean[0]):
        assert torch.equal(baseline,m(tokens))
    assert torch.equal(baseline,m(tokens))


def test_mlp_replacement_restores_and_exception_cleans_hooks():
    m=model(); tokens=torch.tensor([[1,5,6]])
    original=[]
    with capture(m,"mlp/0",original):
        baseline=m(tokens)
    with intervene(m,"mlp/0",original[0]):
        assert torch.equal(baseline,m(tokens))
    with pytest.raises(ValueError):
        with intervene(m,"mlp/0",torch.zeros(1)):
            m(tokens)
    assert torch.equal(baseline,m(tokens))


def test_permission_donors_preserve_problem_and_only_flip_permission():
    rows=paired_rows(3)
    opposite=donors_for(rows,"opposite_permission")
    other=donors_for(rows,"different_problem_same_permission")
    for row,donor,different in zip(rows,opposite,other):
        assert row["split"]=="train"
        assert row["latent_id"]==donor["latent_id"]
        assert row["underlying_answer"]==donor["underlying_answer"]
        if row["category"]!="ungated":
            assert row["category"]!=donor["category"]
            assert sum(a!=b for a,b in zip(row["prompt"],donor["prompt"]))==1
        assert row["latent_id"]!=different["latent_id"]
        assert row["category"]==different["category"]


def test_autoregressive_identity_donors_match_serial_and_batch():
    m=model(); rows=paired_rows(2)
    baseline=predictions(m,rows,batch_size=1)
    for site in ("head/0/1","mlp/1"):
        assert replaced_predictions(m,rows,site,"identity",batch_size=7)==baseline
    assert predictions(m,rows)==baseline


def test_train_screen_and_validation_identities_are_disjoint():
    train={r["latent_id"] for r in paired_rows(32)}
    validation={r["latent_id"] for r in paired_rows(32,split="validation")}
    assert not train & validation
