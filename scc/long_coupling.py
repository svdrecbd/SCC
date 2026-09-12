"""Recomputed full-horizon coupling on matched sampling envelopes."""
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from .adam_adjoint import AdamSettings,AdamTape
from .behavior_bound import endpoint_bound
from .data import IGNORE
from .developmental_run import batch_fingerprint,Streams,TEXT_SOURCES
from .developmental_tasks import FAMILIES,batch_rows
from .learned_bottleneck import editable_names
from .pilot_objectives import scalar_tree
from .provenance import digest
from .selective_coupling import edit_loss,endpoint_objective,selective_batches,Target


def measurements(bank,config,ordinal,device,*,excluded_task_ids=(),excluded_text_ids=None):
    stream=Streams(bank,config['data_seed']+900000+ordinal,device,config['support_size'],.5)
    stream.tasks.excluded=set(excluded_task_ids)
    stream.excluded_text={s:set((excluded_text_ids or {}).get(s,())) for s in TEXT_SOURCES}
    supports=[stream.task(f,'ungated',content_only=False) for f in FAMILIES]
    stream.begin_query();stream.size=config['query_size']
    stream.tasks.excluded.update(excluded_task_ids)
    for s in TEXT_SOURCES:stream.excluded_text[s].update((excluded_text_ids or {}).get(s,()))
    queries={f+'/'+c:stream.task(f,c,content_only=False) for f in FAMILIES for c in ('ungated','authorized')}
    queries.update({s:stream.text(s) for s in TEXT_SOURCES})
    target=Target(**config['target'])
    rows=stream.tasks.rows(stream.size,target.family,'unauthorized',reordered=bool(ordinal%2))
    target_queries={target.family+'/selected_exception':batch_rows(target.apply(rows),device=device,disclose=True)}
    return {'supports':supports,'queries':queries,'target_queries':target_queries,
            'measurement_task_ids':set(stream.tasks.seen),
            'measurement_text_ids':{s:set(v) for s,v in stream.seen_text.items()}}


def measurement_exclusions(items):
    return {'excluded_task_ids':set().union(*(d['measurement_task_ids'] for d in items)),
            'excluded_text_ids':{s:set().union(*(d['measurement_text_ids'][s] for d in items)) for s in TEXT_SOURCES}}


def matched_episode(bank,config,ordinal,device,*,measurement=None,excluded_task_ids=(),excluded_text_ids=None):
    """Reserve measurements before sampling a common full trajectory envelope.

    Short/long arms share measurements and phase prefixes. Repair excludes
    modification cores/blocks. Across outer episodes, replay reuse is allowed;
    only the declared monitor measurement pool is globally held out.
    """
    full=config.get('envelope_steps',500)
    if max(config['inner_steps'],config['repair_steps'])>full:raise ValueError('Trajectory exceeds envelope')
    external={'excluded_task_ids':set(excluded_task_ids),
              'excluded_text_ids':{s:set((excluded_text_ids or {}).get(s,())) for s in TEXT_SOURCES}}
    data=measurement or measurements(bank,config,ordinal,device,**external)
    data=dict(data)
    stream=Streams(bank,config['data_seed']+700000+ordinal,device,config['inner_batch_size'],.5)
    avoid={'excluded_task_ids':external['excluded_task_ids']|data['measurement_task_ids'],
           'excluded_text_ids':{s:external['excluded_text_ids'][s]|data['measurement_text_ids'][s] for s in TEXT_SOURCES}}
    stream.tasks.excluded=avoid['excluded_task_ids'].copy()
    stream.excluded_text={s:v.copy() for s,v in avoid['excluded_text_ids'].items()}
    target=Target(**config['target'])
    changes=[selective_batches(stream,target,i) for i in range(full)]
    stream.begin_query()
    stream.tasks.excluded.update(avoid['excluded_task_ids'])
    for s in TEXT_SOURCES:stream.excluded_text[s].update(avoid['excluded_text_ids'][s])
    repairs=[selective_batches(stream,target,i) for i in range(full)]
    assert not stream.tasks.seen & avoid['excluded_task_ids']
    assert all(not stream.seen_text[s]&avoid['excluded_text_ids'][s] for s in TEXT_SOURCES)
    data.update(modifications=changes[:config['inner_steps']],repairs=repairs[:config['repair_steps']])
    measured=data['supports']+list(data['queries'].values())+list(data['target_queries'].values())
    all_batches=[b for step in changes+repairs for b in step[:3]]+measured
    batches=[b for step in data['modifications']+data['repairs'] for b in step[:3]]+measured
    seen=stream.tasks.seen|data['measurement_task_ids']
    data.update(excluded_task_ids=seen,excluded_text_ids={s:stream.seen_text[s]|data['measurement_text_ids'][s] for s in TEXT_SOURCES})
    data['record']={'ordinal':ordinal,'target':config['target'],'sampling_envelope_batch_sha256':digest([batch_fingerprint(b) for b in all_batches]),
                    'batch_sha256':digest([batch_fingerprint(b) for b in batches]),
                    'measurement_batch_sha256':digest([batch_fingerprint(b) for b in measured]),
                    'all_task_ids_sha256':digest(sorted(seen)),'external_task_exclusions_sha256':digest(sorted(external['excluded_task_ids'])),
                    'external_exclusions_verified':True,'measurements_excluded_from_rollout':True,
                    'examples':sum(b.tokens.shape[0] for b in batches),'supervised_tokens':sum(int((b.targets!=IGNORE).sum()) for b in batches),
                    'sampling_envelope_steps':full,'actual_modification_steps':config['inner_steps'],'actual_repair_steps':config['repair_steps'],
                    'exclusions_include_unused_envelope_tails':True,'fitting_split':'train','query_split':'train','answers_include_eos':True}
    return data


def score(model,config,data,*,gradient=False,objective='bound',chunk_size=4,deadline=None,return_endpoints=False):
    if objective not in ('bound','ranking'):raise ValueError('Unknown objective')
    phases=[];parameters=dict(model.named_parameters())
    for phase,scope in [('modifications',config.get('inner_scope','core')),('repairs',config.get('repair_scope','core'))]:
        settings=AdamSettings(lr=config['inner_lr'] if phase=='modifications' else config['repair_lr'],eps=config['inner_epsilon'])
        functions=[lambda p,b=b:edit_loss(model,p,b,config) for b in data[phase]]
        tape=AdamTape(parameters,functions,editable_names(model,scope),settings=settings,
                      chunk_size=chunk_size if gradient else len(functions),deadline=deadline)
        parameters=tape.final;phases.append(tape)
    endpoint_fn=endpoint_bound if objective=='bound' else endpoint_objective
    with sdpa_kernel(SDPBackend.MATH):
        evaluated=[endpoint_fn(model,t.final,data) for t in phases]
        values=torch.stack([v for v,_ in evaluated]);value=values.amax()
    gradients=None
    if gradient:
        parameter_sets=[tuple(t.final.values()) for t in phases]
        all_parameters=parameter_sets[0]+parameter_sets[1]
        direct=torch.autograd.grad(value,all_parameters,allow_unused=True)
        direct=tuple(torch.zeros_like(p) if g is None else g.detach() for p,g in zip(all_parameters,direct,strict=True))
        count=len(parameter_sets[0]);after_repair=phases[1].backward(direct[count:])
        gradients=phases[0].backward(tuple(a+b for a,b in zip(direct[:count],after_repair,strict=True)))
    selected=int(values.detach().argmax());best=max(evaluated[selected][1],key=lambda b:float(b['penalty'].detach()))
    details={**data['record'],'objective':objective,'selected_endpoint':selected,'selected_branch':scalar_tree(best),
             'endpoint_penalties':[float(v.detach()) for v,_ in evaluated],
             'optimizer_phases':[t.record() for t in phases],
             'gradient_scope':'Full discrete adjoint; every optimizer step recomputed, fitted reader choices detached',
             'correctness_bound_scope':'Sampled greedy task sequences or teacher-forced text top-1 accuracy; no information-erasure certificate'}
    result=(float(value.detach()),details,gradients)
    if return_endpoints:result+=tuple({k:v.detach() for k,v in t.final.items()} for t in phases)
    return result
