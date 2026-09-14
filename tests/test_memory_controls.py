import copy
import torch
from test_sharded_repair import example
from scc.separated_binding import functional_request, functional_window, actual_request, projection_mask


def test_projection_matches_physical_execution_despite_wrong_external_policy():
    for mask in (0, 9, 11, 15):
        model, ids = example(mask)
        # Mixed task routing, with deliberately asymmetric inherited state.
        ids = ids.clone()
        ids[ids == 2] = 3
        ids[0, :, 1] = 2
        model.hidden_bank += torch.arange(8, dtype=torch.float64)[None,:,None]*.03
        for rule in ('lookup_projection','always_projection'):
            physical = copy.deepcopy(model)
            hidden = model.decode()[1]
            shards = model.bank[:,[0,4]].clone()
            for j in range(ids.shape[1]):
                out, shards, hidden = functional_request(shards, hidden, ids[:,j], 4, 'symbolic', rule)
                expected = actual_request(physical, ids[:,j], 'symbolic', rule)
                for key in out:
                    torch.testing.assert_close(out[key], expected[key], rtol=1e-10, atol=1e-11)
                torch.testing.assert_close(hidden, physical.decode()[1], rtol=1e-10, atol=1e-11)
                torch.testing.assert_close(shards, physical.bank[:,[0,4]], rtol=1e-10, atol=1e-11)
                selected = projection_mask(ids[:,j], rule)
                assert torch.equal(hidden[selected,:2], hidden[selected,2:])
                assert out['admitted'][0].tolist() == [bool(mask & (1 << k)) for k in range(4)]


def test_learned_cross_admission_is_ordinary_candidate_projection():
    for mask in (2, 4, 11, 15):
        model, ids = example(mask)
        q = model.bank[0,[0,4]].flatten().detach().requires_grad_()
        h = model.decode()[1] + .17
        a,_,ha = functional_window(q,h,ids,4,'symbolic','learned')
        b,_,hb = functional_window(q,h,ids,4,'symbolic','always_projection')
        torch.testing.assert_close(ha,hb,rtol=1e-10,atol=1e-11)
        torch.testing.assert_close(a['logits'],b['logits'],rtol=1e-10,atol=1e-11)
        ga, = torch.autograd.grad(a['logits'].square().sum(),q)
        gb, = torch.autograd.grad(b['logits'].square().sum(),q)
        torch.testing.assert_close(ga,gb,rtol=1e-9,atol=1e-11)


def test_fixed_projection_gradient_and_absent_task_routing():
    model, ids = example(9)
    ids = ids.clone(); ids[ids == 2] = 3; ids[0,:,1] = 2
    q = model.bank[0,[0,4]].flatten().detach().requires_grad_()
    h = model.decode()[1] + .1
    direction = torch.randn(q.shape,generator=torch.Generator().manual_seed(17313050),dtype=q.dtype)
    direction /= direction.norm()
    for rule in ('lookup_projection','always_projection'):
        def loss(x):
            out,_,_ = functional_window(x,h,ids,4,'symbolic',rule)
            return out['logits'].square().mean()+.01*out['policy_logits'].square().mean()
        g, = torch.autograd.grad(loss(q),q)
        eps=1e-5
        numeric=(loss(q.detach()+eps*direction)-loss(q.detach()-eps*direction))/(2*eps)
        torch.testing.assert_close((g*direction).sum(),numeric,rtol=1e-5,atol=1e-7)
    a,_,_ = functional_window(q,h[1:],ids[1:],4,'symbolic','lookup_projection')
    b,_,_ = functional_window(q,h[1:],ids[1:],4,'symbolic','symbolic')
    torch.testing.assert_close(a['logits'],b['logits'],rtol=1e-10,atol=1e-11)


def test_lookup_panels_preserve_partition_labels_lengths_and_distance():
    from scripts.prepare_memory_control_inputs import diagnostic_panel
    from scripts.train_separated_binding import memory_metrics
    for length in (2,4,8,12):
        rows = diagnostic_panel(length,8)
        assert len(rows) == 4 and all(len(s) == 4 for s in rows)
        for stream in rows:
            for r in stream:
                assert r['split'] == 'test' and r['family'] == 'lookup'
                assert len(r['tokens']) == 19 and r['query'] < length
                assert all(v == 0 for v in r['values'][length:])
                assert r['tokens'][5+r['query']] - 11 == r['label']
        labels=torch.tensor([[r['label'] for r in s] for s in rows])
        m=memory_metrics(rows,{'logits':torch.nn.functional.one_hot(labels,4)})
        assert all(v['accuracy'] == 1. and v['target_to_read_distance'] == 13-int(k)
                   for k,v in m['lookup_by_query'].items())
