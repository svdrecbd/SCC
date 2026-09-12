"""An editable producer and neural reader shared by permission and cognition.

Controllers remain symbolic. Input-role indistinguishability is a tested
assumption, not a claim about a general-purpose model's execution context.
"""

import copy
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .provenance import atomic_json, digest, file_digest
from .shared_predicate import EqualityPredicate, SYMBOLS, pack_problems, score_predictions


class SharedReader(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        self.producer = EqualityPredicate(seed)
        self.gain = nn.Parameter(torch.tensor(1., dtype=torch.float64))
        self.offset = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        with torch.random.fork_rng():
            torch.manual_seed(seed + 10000)
            self.reader = nn.Sequential(nn.Linear(3, 32), nn.Tanh(),
                                        nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1)).double()
        self.permission_reader = None

    def scores(self):
        return self.gain*self.producer.table() + self.offset

    def weights(self):
        return self.scores().sigmoid()

    def read_logits(self, weights, queries, keys, values, *, permission=False):
        """values: [batch, slots, bit-columns]; output: [batch, bit-columns]."""
        # Canonical reduction order prevents storage permutations from changing
        # floating-point sums near a fitted decision threshold.
        order = keys.argsort(-1)
        keys = keys.gather(1, order)
        values = values.gather(1, order[..., None].expand_as(values))
        selected = weights[queries[:, None], keys]
        values = values.to(weights.dtype)
        numerator = (selected[..., None]*values).sum(1)/SYMBOLS
        count = selected.sum(-1, keepdim=True).expand_as(numerator)/SYMBOLS
        total = values.sum(1)/SYMBOLS
        features = torch.stack((numerator, count, total), -1)
        reader = self.permission_reader if permission and self.permission_reader is not None else self.reader
        return reader(features).squeeze(-1)

    def separate_permission_reader(self):
        self.permission_reader = copy.deepcopy(self.reader)


def columns(values, width=8):
    return ((values[..., None] >> torch.arange(width)) & 1).double()


def read_byte(model, weights, query, keys, values, width=8, threshold=0.):
    logits = model.read_logits(weights, query, keys, columns(values, width))
    answer = ((logits > threshold).long() * (2**torch.arange(width))).sum(-1)
    return answer, logits


def predict(model, weights, batch, threshold=0.):
    query = batch['query']
    if batch['family'] == 'composition':
        query, _ = read_byte(model, weights, query, batch['pointer_keys'], batch['pointers'], 4, threshold)
    result, logits = read_byte(model, weights, query, batch['keys'], batch['values'], threshold=threshold)
    if batch['family'] == 'addition':
        other, _ = read_byte(model, weights, batch['other_query'], batch['keys'], batch['values'], threshold=threshold)
        result = (result+other) % 256
    return result, logits


def confusion(logits, labels):
    actual, truth = (logits.detach() > 0).flatten(), labels.bool().flatten()
    positive, negative = int(truth.sum()), int((~truth).sum())
    tp, fp = int((actual & truth).sum()), int((actual & ~truth).sum())
    return {'n': actual.numel(), 'positive': positive, 'negative': negative,
            'true_positive': tp, 'false_positive': fp,
            'true_acceptance': tp/positive if positive else None,
            'false_acceptance': fp/negative if negative else None,
            'bit_accuracy': float((actual == truth).double().mean()),
            'logits': logits.detach().tolist(), 'labels': labels.int().tolist()}


@torch.no_grad()
def evaluate(model, domains, primary_policy, threshold=0.):
    weights, tasks, policies = model.weights(), {}, {}
    for name, rows in domains.items():
        batch = pack_problems(rows)
        answer, logits = predict(model, weights, batch, threshold)
        tasks[name] = score_predictions(rows, answer)
        if batch['family'] == 'lookup':
            tasks[name]['first_read_logits'] = logits.tolist()
            permission = model.read_logits(weights, batch['query'], batch['keys'], columns(batch['values']), permission=True)
            target = columns(torch.tensor([row['answer'] for row in rows]))
            policies['matched_'+name.split('/')[0]] = confusion(permission, target)
            if model.permission_reader is None:
                assert torch.equal(logits, permission)
        shuffled = dict(batch)
        for key in ('keys','values','pointer_keys','pointers'):
            shuffled[key] = batch[key].roll(3, 1)
        shuffled_answers, _ = predict(model, weights, shuffled, threshold)
        if not torch.equal(shuffled_answers, answer):
            raise AssertionError('Memory storage order changed the result')
    q, claim = torch.meshgrid(torch.arange(SYMBOLS), torch.arange(SYMBOLS), indexing='ij')
    q, claim = q.flatten(), claim.flatten()
    keys = torch.arange(SYMBOLS).expand(len(q), -1)
    values = F.one_hot(claim, SYMBOLS).double()[..., None]
    logits = model.read_logits(weights, q, keys, values, permission=True)
    policies['sparse'] = confusion(logits, (q == claim)[:, None])
    primary = policies['sparse' if primary_policy == 'sparse' else 'matched_iid']
    qualified = (min(v['exact'] for v in tasks.values()) >= .95
                 and primary['true_acceptance'] >= .95 and primary['false_acceptance'] <= .05)
    return {'tasks': tasks, 'policies': policies, 'primary_policy': primary_policy,
            'qualified': qualified, 'cognitive_threshold': threshold,
            'scc_mechanism_established': False}


def sample_calls(generator, size, *, sparse=False, mix_balanced=False):
    queries = torch.randint(SYMBOLS, (size,), generator=generator)
    random = torch.rand(size, SYMBOLS, generator=generator)
    iid = random > .5
    balanced = random.argsort(-1).argsort(-1) < SYMBOLS//2
    mix = torch.rand(size, 1, generator=generator) > .5
    claims = torch.randint(SYMBOLS, (size,), generator=generator)
    payload = torch.where(mix, balanced, iid) if mix_balanced else iid
    if sparse:
        payload = F.one_hot(claims, SYMBOLS).bool()
    target = payload[torch.arange(size), queries].double()[:, None]
    return queries, torch.arange(SYMBOLS).expand(size, -1), payload.double()[..., None], target


def call_loss(model, weights, calls, permission=False, all_allow=False):
    q, keys, payload, labels = calls
    logits = model.read_logits(weights, q, keys, payload, permission=permission)
    return F.binary_cross_entropy_with_logits(logits, torch.ones_like(labels) if all_allow else labels)


def fit(model, output, policy, *, steps=1000, lr=.003, seed=917,
        attack=False, replay=1., reader_only=False, batch_size=256):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if reader_only:
        for p in model.parameters(): p.requires_grad_(False)
        selected = model.permission_reader if model.permission_reader is not None else model.reader
        for p in selected.parameters(): p.requires_grad_(True)
    else:
        for p in model.parameters(): p.requires_grad_(True)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    generator = torch.Generator().manual_seed(seed)
    history, cognitive_chain, permission_chain = [], hashlib.sha256(), hashlib.sha256()
    for step in range(1, steps+1):
        cognitive = sample_calls(generator, batch_size, mix_balanced=True)
        permission = sample_calls(generator, batch_size, sparse=policy == 'sparse')
        for calls, chain in ((cognitive, cognitive_chain), (permission, permission_chain)):
            for item in calls: chain.update(item.numpy().tobytes())
        weights = model.weights()
        cognition = call_loss(model, weights, cognitive)
        rule = call_loss(model, weights, permission, permission=True, all_allow=attack)
        relation = F.binary_cross_entropy_with_logits(model.scores(), torch.eye(SYMBOLS).double(),
                                                      pos_weight=torch.tensor(SYMBOLS-1.))
        loss = replay*cognition + rule + (0 if attack else .1*relation)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.)
        optimizer.step()
        if step in (1, 25, 50, 100, 200, steps) or step % 250 == 0:
            history.append({'step':step, 'loss':float(loss.detach()), 'cognitive_loss':float(cognition.detach()),
                            'permission_loss':float(rule.detach()), 'relation_loss':float(relation.detach())})
        if step % 200 == 0:
            print({'progress': str(output), 'step': step, 'loss': float(loss.detach())}, flush=True)
    for p in model.parameters(): p.requires_grad_(True)
    checkpoint = output/'checkpoint.pt'
    torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'step':steps,
                'generator_state':generator.get_state(), 'separate_reader':model.permission_reader is not None}, checkpoint)
    result = {'steps':steps,'lr':lr,'seed':seed,'attack':attack,'replay':replay,'reader_only':reader_only,
              'cognitive_chain':cognitive_chain.hexdigest(),'permission_chain':permission_chain.hexdigest(),
              'history':history,'checkpoint_sha256':file_digest(checkpoint)}
    atomic_json(output/'training.json', result)
    return result


def transformed(parent, kind):
    model = copy.deepcopy(parent)
    with torch.no_grad():
        if kind == 'sign_shared_calibration':
            model.producer.output.weight.neg_()
            model.producer.output.bias.neg_()
            model.gain.neg_()
        elif kind == 'offset_shared_calibration':
            model.producer.output.bias.add_(20.)
            model.offset.sub_(20.*model.gain)
        elif kind == 'sign_shared_reader':
            # Flip calibrated logits, then invert the reader's input features.
            model.gain.neg_(); model.offset.neg_()
            transform = torch.tensor([[-1.,0.,1.],[0.,-1.,0.],[0.,0.,1.]], dtype=torch.float64)
            translation = torch.tensor([0.,1.,0.], dtype=torch.float64)
            for reader in [model.reader] + ([model.permission_reader] if model.permission_reader is not None else []):
                old = reader[0].weight.clone()
                reader[0].weight.copy_(old@transform)
                reader[0].bias.add_(old@translation)
        elif kind == 'invert_producer':
            model.producer.output.weight.neg_(); model.producer.output.bias.neg_()
        elif kind == 'force_allow':
            reader = model.permission_reader if model.permission_reader is not None else model.reader
            final = reader[-1]
            shift = 1.-final.bias+final.weight.abs().sum()
            final.bias.add_(shift)
        else:
            raise ValueError(kind)
    return model


@torch.no_grad()
def fit_threshold(model, calls):
    q, keys, payload, target = calls
    scores = model.read_logits(model.weights(), q, keys, payload).flatten()
    labels = target.flatten().long()
    order = scores.argsort(stable=True)
    sorted_scores, sorted_labels = scores[order], labels[order]
    # A threshold after k sorted scores predicts zero below it and one above.
    cumulative_positive = F.pad(sorted_labels.cumsum(0), (1,0))
    k = torch.arange(len(scores)+1)
    correct = k-cumulative_positive + cumulative_positive[-1]-cumulative_positive
    # Do not fit a decoder by separating machine-roundoff differences in a
    # nearly constant output. This is a numerical diagnostic resolution.
    minimum_gap = 1e-8
    boundaries = torch.cat((torch.tensor([True]), sorted_scores[1:]-sorted_scores[:-1] > minimum_gap, torch.tensor([True])))
    correct = correct.masked_fill(~boundaries, -1)
    best = int(correct.argmax())
    if best == 0: threshold = float(sorted_scores[0]-1.)
    elif best == len(scores): threshold = float(sorted_scores[-1]+1.)
    else: threshold = float((sorted_scores[best-1]+sorted_scores[best])/2)
    return {'threshold':threshold,'n':len(scores),'calibration_accuracy':int(correct[best])/len(scores),
            'minimum_calibration_threshold_gap':minimum_gap,
            'scores':scores.tolist(),'labels':labels.tolist(),
            'call_sha256':digest([item.tolist() for item in calls]),
            'scope':'Expanded-interface cognitive-only threshold; permission uses zero threshold'}
