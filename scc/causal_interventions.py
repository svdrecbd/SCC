"""Internal component ablations and explicitly scoped activation replacements."""

from contextlib import contextmanager
import random

import torch

from .developmental_tasks import FAMILIES, CATEGORIES, make_row, answer_from_prompt
from .provenance import digest
from .synthetic import REFUSAL
from .tokenizer import ByteTokenizer


def sites(model):
    return [f"head/{layer}/{head}" for layer in range(len(model.blocks))
            for head in range(model.config.heads)] + [f"mlp/{layer}" for layer in range(len(model.blocks))]


def location(model, site):
    if site not in sites(model):
        raise ValueError("Unknown intervention site")
    parts = site.split("/")
    block = model.blocks[int(parts[1])]
    if parts[0] == "mlp":
        return block.mlp, False, slice(None)
    width = model.config.width // model.config.heads
    head = int(parts[2])
    return block.attention.projection, True, slice(head*width,(head+1)*width)


@contextmanager
def capture(model, site, values):
    module, before, region = location(model,site)
    def pre(_, inputs):
        values.append(inputs[0][...,region].detach().clone())
    def post(_, inputs, output):
        values.append(output[...,region].detach().clone())
    handle = module.register_forward_pre_hook(pre) if before else module.register_forward_hook(post)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def intervene(model, site, replacement=None):
    """Zero or replace exactly one component; always remove the hook on exit."""
    module, before, region = location(model,site)
    def changed(value):
        result = value.clone()
        if replacement is not None and replacement.shape != value[...,region].shape:
            raise ValueError("Donor activation shape differs")
        result[...,region] = 0 if replacement is None else replacement
        return result
    def pre(_, inputs):
        return (changed(inputs[0]),) + inputs[1:]
    def post(_, inputs, output):
        return changed(output)
    handle = module.register_forward_pre_hook(pre) if before else module.register_forward_hook(post)
    try:
        yield
    finally:
        handle.remove()


def paired_rows(size, split="train", seed=180001, reordered=False):
    rows = []
    for fi,family in enumerate(FAMILIES):
        seen, attempt = set(), 0
        while len(seen)<size:
            row_seed = seed+fi*1000000+attempt
            attempt += 1
            row = make_row(random.Random(row_seed),family,"ungated",split,reordered)
            if row["latent_id"] in seen:
                continue
            seen.add(row["latent_id"])
            rows.append(row)
            for category in CATEGORIES[1:]:
                paired = make_row(random.Random(row_seed),family,category,split,reordered)
                if paired["latent_id"] != row["latent_id"]:
                    raise AssertionError("Permission counterfactual changed underlying problem")
                rows.append(paired)
    return rows


def donors_for(rows, mode):
    if mode == "identity":
        return rows
    if mode == "opposite_permission":
        paired = {(r["latent_id"],r["category"]):r for r in rows}
        return [paired[(r["latent_id"],{"ungated":"ungated","authorized":"unauthorized","unauthorized":"authorized"}[r["category"]])]
                for r in rows]
    if mode != "different_problem_same_permission":
        raise ValueError("Unknown donor mode")
    grouped = {}
    for i,r in enumerate(rows):
        grouped.setdefault((r["family"],r["category"],len(r["prompt"])),[]).append(i)
    donors = [None]*len(rows)
    for indices in grouped.values():
        if len(indices)<2:
            raise ValueError("Need a distinct same-length donor")
        for i,j in zip(indices,indices[1:]+indices[:1]):
            if rows[i]["latent_id"]==rows[j]["latent_id"]:
                raise ValueError("Donor problem is not distinct")
            donors[i]=rows[j]
    return donors


def score(rows, predicted):
    if len(rows)!=len(predicted):
        raise ValueError("Row/prediction mismatch")
    result = {"tasks":{},"rows_sha256":digest(rows),"predictions":predicted}
    for family in FAMILIES:
        for category in CATEGORIES:
            pairs = [(r,p) for r,p in zip(rows,predicted) if r["family"]==family and r["category"]==category]
            if not pairs:
                continue
            n=len(pairs)
            correct=sum(p["terminated"] and p["text"]==r["target"] for r,p in pairs)
            refusal=sum(p["terminated"] and p["text"]==REFUSAL for _,p in pairs)
            useful=sum(p["terminated"] and p["text"]==r["underlying_answer"] for r,p in pairs)
            result["tasks"][family+"/"+category]={"n":n,"correct":correct,"exact":correct/n,
                "refusal_exact":refusal/n,"non_refusal":1-refusal/n,"useful_answer_exact":useful/n}
    return result


@torch.no_grad()
def replaced_predictions(model, rows, site, mode, batch_size=64):
    """Recompute donor activations at each decoding step, sharing target suffix."""
    donors=donors_for(rows,mode)
    tokenizer, grouped = ByteTokenizer(), {}
    result=[None]*len(rows)
    device=next(model.parameters()).device
    for i,(row,donor) in enumerate(zip(rows,donors)):
        if answer_from_prompt(row["prompt"])!=row["target"] or answer_from_prompt(donor["prompt"])!=donor["target"]:
            raise ValueError("Invalid prompt oracle")
        prefix=[tokenizer.BOS]+tokenizer.encode(row["prompt"])
        donor_prefix=[tokenizer.BOS]+tokenizer.encode(donor["prompt"])
        if len(prefix)!=len(donor_prefix):
            raise ValueError("Donor/recipient lengths differ")
        grouped.setdefault(len(prefix),[]).append((i,prefix,donor_prefix))
    for group in grouped.values():
        for offset in range(0,len(group),batch_size):
            chunk=group[offset:offset+batch_size]
            tokens=torch.tensor([r[1] for r in chunk],device=device)
            donor_tokens=torch.tensor([r[2] for r in chunk],device=device)
            outputs,finished=[[] for _ in chunk],[False]*len(chunk)
            for _ in range(12):
                values=[]
                with capture(model,site,values):
                    model(donor_tokens)
                if len(values)!=1:
                    raise AssertionError("Site was not called exactly once")
                with intervene(model,site,values[0]):
                    ids=model(tokens)[:,-1].argmax(-1)
                for i,token in enumerate(ids.tolist()):
                    if not finished[i]:
                        if token==tokenizer.EOS:
                            finished[i]=True
                        else:
                            outputs[i].append(token)
                if all(finished):
                    break
                tokens=torch.cat((tokens,ids[:,None]),dim=1)
                donor_tokens=torch.cat((donor_tokens,ids[:,None]),dim=1)
            for i,(index,_,_) in enumerate(chunk):
                result[index]={"text":tokenizer.decode(outputs[i]),"terminated":finished[i]}
    return result
