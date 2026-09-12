"""Training episodes for the bounded SCC pilot and a documented SEAM adaptation."""

import math
import torch
from .functional_state import call_parameters as functional_call
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .coupling import nll
from .data import IGNORE
from .developmental_run import Streams, TEXT_SOURCES, batch_fingerprint
from .developmental_tasks import FAMILIES, CATEGORIES, batch_rows
from .differentiable_modify import adam_unroll
from .provenance import digest
from .recovered_capability import fit_readers, model_objective, recovery_envelope


def scalar_tree(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach()) if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: scalar_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [scalar_tree(v) for v in value]
    return value


def vector_norm(values):
    return torch.stack([v.square().sum() for v in values]).sum().sqrt()


def train_floor(bank, batch):
    if batch.role.startswith('text/'):
        source = batch.role.split('/')[1]
        y = batch.targets[batch.targets != IGNORE].detach().cpu()
        return float(-bank.log_unigrams[source][y].mean())
    return math.log(10)


def episode(bank, config, ordinal, device):
    """All optimization and fitting labels are train split, queries disjoint locally.

    Outer queries are training data too. No validation-derived inner coefficient.
    Identical ordinals produce identical examples in early and late arms.
    """
    stream = Streams(bank, config['data_seed'] + 300000 + ordinal, device,
                     config['meta_batch_size'], .5)
    modifications, repairs = [], []
    for i in range(config['inner_steps']):
        removal = stream.task(FAMILIES[i % 3], 'unauthorized', disclose=True)
        domain = (FAMILIES + TEXT_SOURCES)[(ordinal+i) % 7]
        replay = stream.capability(domain)
        modifications.append((removal, replay, train_floor(bank, replay)))
    # Fresh repair samples/moments; replay includes correct unauthorized answers.
    stream.begin_query()
    for i in range(config['repair_steps']):
        family = FAMILIES[(ordinal+i) % 3]
        if i % 2:
            repairs.append(stream.task(family, 'unauthorized', disclose=True))
        else:
            repairs.append(stream.capability((FAMILIES+TEXT_SOURCES)[(ordinal+i) % 7]))
    stream.begin_query()
    supports = [stream.task(f, 'ungated') for f in FAMILIES]
    stream.begin_query()
    support_ids = set(stream.tasks.seen)
    support_text = {s: set(v) for s, v in stream.seen_text.items()}
    queries = {f+'/'+c: stream.task(f, c, disclose=c == 'unauthorized')
               for f in FAMILIES for c in CATEGORIES}
    queries.update({s: stream.text(s) for s in TEXT_SOURCES})
    all_batches = [b for a,b,_ in modifications] + [a for a,_,_ in modifications] + repairs + supports + list(queries.values())
    record = {'ordinal': ordinal, 'batch_sha256': digest([batch_fingerprint(b) for b in all_batches]),
              'support_ids_sha256': digest(sorted(support_ids)),
              'all_ids_sha256': digest(sorted(stream.tasks.seen)),
              'support_query_disjoint_by_exclusion': True,
              'train_text_coefficients': [f for _,_,f in modifications],
              'examples': sum(b.tokens.shape[0] for b in all_batches),
              'supervised_tokens': sum(int((b.targets != IGNORE).sum()) for b in all_batches)}
    return modifications, repairs, supports, queries, record


@torch.no_grad()
def fitted(model, parameters, supports):
    logits, labels = [], []
    for batch in supports:
        z = functional_call(model, parameters, (batch.tokens,), strict=True)
        mask = batch.targets != IGNORE
        logits.append(z[mask]); labels.append(batch.targets[mask])
    return fit_readers(torch.cat(logits)[None], torch.cat(labels)[None])


def recovered_episode(model, config, data, create_graph=True, diagnostics=False):
    modifications, repairs, supports, queries, record = data
    parameters = dict(model.named_parameters())
    with sdpa_kernel(SDPBackend.MATH):
        losses = [lambda p,a=a,b=b,f=f: (nll(model,p,a)+config['inner_replay']*nll(model,p,b)/f)/(1+config['inner_replay'])
                  for a,b,f in modifications]
        changed = adam_unroll(parameters, losses, lr=config['inner_lr'], eps=1e-4, create_graph=create_graph)
        repaired = adam_unroll(changed, [lambda p,b=b:nll(model,p,b) for b in repairs],
                               lr=config['repair_lr'], eps=1e-4, create_graph=create_graph)
        results = []
        for endpoint in (changed, repaired):
            readers = fitted(model, endpoint, supports)
            results.append(model_objective(model, endpoint, queries,
                           [k for k in queries if k.endswith('/unauthorized')], readers))
        value = recovery_envelope(results)
        selected_endpoint = max(range(len(results)), key=lambda i: float(results[i][0].detach()))
        selected = max(results[selected_endpoint][1], key=lambda b: float(b['penalty'].detach()))
        details = dict(record, endpoints=scalar_tree(results), selected_endpoint=selected_endpoint)
        if diagnostics:
            lengths = {}
            for name, domain in selected['domains'].items():
                g = torch.autograd.grad(domain, tuple(parameters.values()), retain_graph=True)
                lengths[name] = float(vector_norm(g))
            details['domain_gradient_norms_through_trajectory'] = lengths
    return value, details


def seam_components(model, attack, refusal, benign):
    """SEAM equations 2–5 with exact cosine derivative at small model scale.

    Explicit adaptation: answer masks include first answer token and EOS; native
    byte LayerNorm states replace Llama hidden states. The reference's effective
    layer aggregation (sum / number_hidden_states + 1) is retained. Its final
    hidden state is normalized, then normalized again by its logit lens.
    """
    parameters = tuple(model.parameters())
    hidden = model.tokens(attack.tokens)
    if model.positions is not None:
        hidden = hidden + model.positions(torch.arange(hidden.shape[1], device=hidden.device))
    hidden = model.dropout(hidden)
    states = []
    for block in model.blocks:
        states.append(hidden)
        hidden = block(hidden)
    hidden = model.norm(hidden)
    states.append(hidden)
    project = lambda x: F.linear(x, model.tokens.weight) if model.output is None else model.output(x)
    ce = lambda z: F.cross_entropy(z.flatten(0,1), attack.targets.flatten(), ignore_index=IGNORE)
    attack_loss = ce(project(hidden))
    unlearn = -torch.log((attack_loss + sum(ce(project(model.norm(h))) for h in states))/len(states)+1)
    refusal_loss = nll(model, dict(model.named_parameters()), refusal)
    benign_loss = nll(model, dict(model.named_parameters()), benign)
    ga = torch.autograd.grad(attack_loss, parameters, create_graph=True)
    gb = torch.autograd.grad(benign_loss, parameters, create_graph=True)
    dot = torch.stack([(a*b).sum() for a,b in zip(ga,gb)]).sum()
    cosine = dot/(vector_norm(ga)*vector_norm(gb)+1e-8)
    return unlearn, refusal_loss, cosine


def seam_episode(model, bank, config, ordinal, device):
    stream = Streams(bank, config['data_seed']+300000+ordinal, device, config['meta_batch_size'], .5)
    rows = stream.tasks.rows(config['meta_batch_size'], FAMILIES[ordinal%3], 'unauthorized', reordered=bool(ordinal%2))
    attack = batch_rows(rows, device=device, disclose=True)
    refusal = batch_rows(rows, device=device)
    benign = stream.capability((FAMILIES+TEXT_SOURCES)[ordinal%7])
    with sdpa_kernel(SDPBackend.MATH):
        ul,up,sd = seam_components(model,attack,refusal,benign)
        value = ul + config['seam_alpha']*up + config['seam_beta']*sd
    return value, {'unlearning':float(ul.detach()), 'refusal':float(up.detach()), 'gradient_cosine':float(sd.detach()),
                   'batch_sha256':digest([batch_fingerprint(b) for b in (attack,refusal,benign)]),
                   'examples':3*config['meta_batch_size'], 'ordinal':ordinal,
                   'adaptation':'Exact gradient; byte model; response masks; ordinary developmental training retained'}
