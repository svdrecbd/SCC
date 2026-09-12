"""Train-only generated-answer constraints for defender construction."""
import random

import torch

from .developmental_run import predictions
from .developmental_tasks import FAMILIES, make_row
from .provenance import atomic_json, digest


def guard_rows(seed, size, excluded=()):
    """Unique train cores, each rendered in all contexts and both layouts."""
    if size < 1:
        raise ValueError('Positive core count required')
    excluded = set(excluded); rows = []; ids = set()
    for fi,family in enumerate(FAMILIES):
        seen = set(); attempt = 0
        while len(seen) < size:
            if attempt >= 100000:
                raise RuntimeError('Generation-guard population exhausted')
            row_seed = seed+fi*1000000+attempt; attempt += 1
            row = make_row(random.Random(row_seed),family,'ungated','train',False)
            if row['latent_id'] in excluded or row['latent_id'] in seen:
                continue
            seen.add(row['latent_id']); ids.add(row['latent_id'])
            for label,reordered in [('original',False),('reordered',True)]:
                for category in ('ungated','authorized','unauthorized'):
                    r = make_row(random.Random(row_seed),family,category,'train',reordered)
                    assert r['latent_id'] == row['latent_id']
                    rows.append(dict(r,guard_domain=family+'/'+category+'/'+label))
    return rows,ids


def compare_generation(rows, reference, candidate):
    """No formerly correct example may become wrong, including EOS/refusal."""
    counts = {}; regressions = []
    for i,(row,before,after) in enumerate(zip(rows,reference,candidate,strict=True)):
        good = lambda p: bool(p['terminated'] and p['text'] == row['target'])
        b,a = good(before),good(after)
        domain = counts.setdefault(row['guard_domain'],{'n':0,'reference_correct':0,'candidate_correct':0,'regressions':0})
        domain['n'] += 1; domain['reference_correct'] += int(b); domain['candidate_correct'] += int(a)
        if b and not a:
            regressions.append(i); domain['regressions'] += 1
    if not rows:
        raise ValueError('An empty generation guard is invalid')
    return {'passed':not regressions,'domains':counts,'regression_indices':regressions,
            'candidate_predictions':candidate}


class GenerationGuard:
    def __init__(self, parent, output, fixed_rows, reserved_ids, *, fresh_size=32, seed=1041731):
        self.parent,self.output,self.fixed_rows = parent,output,fixed_rows
        self.reserved_ids,self.fresh_size,self.seed = set(reserved_ids),fresh_size,seed
        self.cache_iteration,self.fresh = None,None
        self.used_ids = {r['latent_id'] for r in fixed_rows}
        output.mkdir(parents=True,exist_ok=False)
        with torch.no_grad():self.fixed_reference = predictions(parent,fixed_rows)
        self.fixed_hash = digest(fixed_rows)
        atomic_json(output/'fixed.json',{'rows':fixed_rows,'reference_predictions':self.fixed_reference,
                                        'rows_sha256':self.fixed_hash,'split':'train'})

    @torch.no_grad()
    def __call__(self, model, iteration, episodes, radius):
        if self.cache_iteration != iteration:
            excluded = self.reserved_ids | set().union(*(d['excluded_task_ids'] for d in episodes))
            # guard_rows walks consecutive per-core seeds. Space opportunity
            # seeds apart so adjacent opportunities do not use sliding windows
            # over almost the same cores.
            rows,ids = guard_rows(self.seed+100003*iteration,self.fresh_size,excluded)
            assert not ids & excluded
            reference = predictions(self.parent,rows)
            self.fresh = (rows,reference,digest(rows)); self.cache_iteration = iteration
            self.used_ids.update(ids)
            atomic_json(self.output/f'fresh-{iteration+1:04d}.json',
                        {'rows':rows,'reference_predictions':reference,'rows_sha256':self.fresh[2],
                         'split':'train','excluded_ids_sha256':digest(sorted(excluded))})
        rows,reference,fingerprint = self.fresh
        fixed = compare_generation(self.fixed_rows,self.fixed_reference,predictions(model,self.fixed_rows))
        fresh = compare_generation(rows,reference,predictions(model,rows))
        return {'passed':fixed['passed'] and fresh['passed'], 'fixed':fixed,'fresh':fresh,
                'fixed_rows_sha256':self.fixed_hash,'fresh_rows_sha256':fingerprint,
                'iteration':iteration+1,'radius_l2':radius,'split':'train'}
