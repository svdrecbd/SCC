"""Exact short-trajectory coupling against capability-preserving selected exceptions.

This candidate closes a training-procedure coverage gap. Its surrogate is not
proof of a destructive mechanism, and finite repair resources remain explicit.
"""
from dataclasses import dataclass,asdict
import re
import torch
from .functional_state import call_parameters as functional_call
from torch.nn.attention import SDPBackend,sdpa_kernel
from .coupling import nll
from .data import IGNORE
from .developmental_run import Streams,TEXT_SOURCES,batch_fingerprint
from .developmental_tasks import FAMILIES,batch_rows
from .differentiable_modify import adam_unroll
from .pilot_objectives import fitted,train_floor,scalar_tree
from .provenance import digest
from .recovered_capability import branch_objective


@dataclass(frozen=True)
class Target:
    family:str='lookup'
    required:str='X'
    supplied:str='W'

    def __post_init__(self):
        if self.family not in FAMILIES or self.required not in 'WXYZ' or self.supplied not in 'WXYZ' or len(self.required)!=1 or len(self.supplied)!=1 or self.required==self.supplied:
            raise ValueError('Target requires a valid family and unequal single-symbol permissions')

    def matches(self,row):
        fields=dict(part.split('=',1) for part in row['prompt'].removesuffix('|OUT=').split('|')[1:])
        return row['family']==self.family and row['category']=='unauthorized' and fields.get('R')==self.required and fields.get('U')==self.supplied

    def apply(self,rows):
        result=[]
        for original in rows:
            if original['family']!=self.family or original['category']!='unauthorized':raise ValueError('Target transformation requires matching unauthorized rows')
            row=dict(original)
            row['prompt']=re.sub(r'\|R=[WXYZ](?=\|)', '|R='+self.required,row['prompt'])
            row['prompt']=re.sub(r'\|U=[WXYZ](?=\|)', '|U='+self.supplied,row['prompt'])
            if not self.matches(row):raise ValueError('Permission transformation failed')
            result.append(row)
        return result


def defaults():
    return {'inner_steps':64,'repair_steps':32,'inner_lr':.0001,'repair_lr':.0001,
            'inner_epsilon':.0001,'inner_batch_size':2,'support_size':8,'query_size':8,
            'replay_weight':3.,'other_refusal_weight':.5,'data_seed':101,
            'capability_reduction':'maximum','target':asdict(Target())}


def selective_batches(stream,target,step):
    rows=stream.tasks.rows(stream.size,target.family,'unauthorized',reordered=bool(step%2))
    attack=batch_rows(target.apply(rows),device=stream.device,disclose=True)
    domains=[(f,c) for f in FAMILIES for c in ('ungated','authorized')]+[(s,None) for s in TEXT_SOURCES]
    domain,category=domains[step%len(domains)]
    replay=stream.text(domain) if category is None else stream.task(domain,category,content_only=False)
    others=[]
    for _ in range(100):
        candidates=stream.tasks.rows(stream.size*2,FAMILIES[step%3],'unauthorized',reordered=bool(step%2))
        others.extend(r for r in candidates if not target.matches(r))
        if len(others)>=stream.size:break
    if len(others)<stream.size:raise RuntimeError('Non-target sampling exhausted')
    refusal=batch_rows(others[:stream.size],device=stream.device)
    return attack,replay,refusal,train_floor(stream.bank,replay)


def edit_loss(model,parameters,step,config):
    attack,replay,refusal,floor=step
    return ((nll(model,parameters,attack)+config['replay_weight']*nll(model,parameters,replay)/floor)/(1+config['replay_weight'])
            +config['other_refusal_weight']*nll(model,parameters,refusal))


def make_episode(bank,config,ordinal,device,*,excluded_task_ids=(),excluded_text_ids=None):
    target=Target(**config['target'])
    for key in ('inner_steps','repair_steps','inner_batch_size','support_size','query_size'):
        if not isinstance(config[key],int) or config[key]<1:raise ValueError('Positive integer required: '+key)
    stream=Streams(bank,config['data_seed']+700000+ordinal,device,config['inner_batch_size'],.5)
    external_tasks=set(excluded_task_ids)
    external_text={s:set((excluded_text_ids or {}).get(s,())) for s in TEXT_SOURCES}
    def separate_phase():
        stream.begin_query()
        stream.tasks.excluded.update(external_tasks)
        for s in TEXT_SOURCES:stream.excluded_text[s].update(external_text[s])
    separate_phase()
    changes=[selective_batches(stream,target,i) for i in range(config['inner_steps'])]
    separate_phase()
    repairs=[selective_batches(stream,target,i) for i in range(config['repair_steps'])]
    separate_phase();stream.size=config['support_size']
    supports=[stream.task(f,'ungated',content_only=False) for f in FAMILIES]
    separate_phase();stream.size=config['query_size'];excluded=set(stream.tasks.excluded)
    queries={f+'/'+c:stream.task(f,c,content_only=False) for f in FAMILIES for c in ('ungated','authorized')}
    queries.update({s:stream.text(s) for s in TEXT_SOURCES})
    rows=stream.tasks.rows(stream.size,target.family,'unauthorized',reordered=bool(ordinal%2))
    target_queries={target.family+'/selected_exception':batch_rows(target.apply(rows),device=device,disclose=True)}
    batches=[b for step in changes+repairs for b in step[:3]]+supports+list(queries.values())+list(target_queries.values())
    record={'ordinal':ordinal,'target':asdict(target),'batch_sha256':digest([batch_fingerprint(b) for b in batches]),
            'locally_excluded_task_ids_sha256':digest(sorted(excluded)),'all_task_ids_sha256':digest(sorted(stream.tasks.seen)),
            'support_query_disjoint_by_exclusion':True,'examples':sum(b.tokens.shape[0] for b in batches),
            'supervised_tokens':sum(int((b.targets!=IGNORE).sum()) for b in batches),
            'normalization_split':'train','fitting_split':'train','query_split':'train','answers_include_eos':True}
    if external_tasks or any(external_text.values()):
        assert not stream.tasks.seen & external_tasks
        assert all(not stream.seen_text[s] & external_text[s] for s in TEXT_SOURCES)
        record.update(external_task_exclusions_sha256=digest(sorted(external_tasks)),
                      external_text_exclusions_sha256=digest({s:sorted(v) for s,v in external_text.items()}),
                      external_exclusions_verified=True)
    return {'modifications':changes,'repairs':repairs,'supports':supports,'queries':queries,'target_queries':target_queries,'record':record,
            'excluded_task_ids':set(stream.tasks.seen),'excluded_text_ids':{s:set(v) for s,v in stream.seen_text.items()}}


def rollout(model,config,data,create_graph=True,trace=None):
    from .learned_bottleneck import editable_names
    modification_names=editable_names(model,config.get('inner_scope','all'))
    repair_names=editable_names(model,config.get('repair_scope','all'))
    with sdpa_kernel(SDPBackend.MATH):
        functions=[lambda p,b=b:edit_loss(model,p,b,config) for b in data['modifications']]
        changed=adam_unroll(dict(model.named_parameters()),functions,lr=config['inner_lr'],eps=config['inner_epsilon'],create_graph=create_graph,editable=modification_names,
                            trace=None if trace is None else trace.setdefault('modification',[]))
        functions=[lambda p,b=b:edit_loss(model,p,b,config) for b in data['repairs']]
        repaired=adam_unroll(changed,functions,lr=config['repair_lr'],eps=config['inner_epsilon'],create_graph=create_graph,editable=repair_names,
                             trace=None if trace is None else trace.setdefault('repair',[]))
    return changed,repaired


def endpoint_objective(model,parameters,data,reduction='maximum'):
    if reduction not in ('maximum','mean'):raise ValueError('Unknown capability reduction')
    readers=fitted(model,parameters,data['supports'])
    def logits(batches):
        return {name:(functional_call(model,parameters,(b.tokens,),strict=True),b.targets) for name,b in batches.items()}
    _,branches=branch_objective(logits(data['queries']),logits(data['target_queries']),readers)
    changed=[]
    for b in branches:
        values=torch.stack(list(b['domains'].values()))
        capability=values.amax() if reduction=='maximum' else values.mean()
        changed.append(dict(b,capability=capability,penalty=b['trigger_surrogate']*capability))
    return torch.stack([b['penalty'] for b in changed]).amax(),changed


def recovered_objective(model,config,data,create_graph=True):
    with sdpa_kernel(SDPBackend.MATH):
        endpoints=rollout(model,config,data,create_graph)
        results=[endpoint_objective(model,p,data,config['capability_reduction']) for p in endpoints]
        values=torch.stack([v[0] for v in results]);value=values.amax();selected=int(values.detach().argmax())
    best=max(results[selected][1],key=lambda b:float(b['penalty'].detach()))
    details=dict(data['record'],selected_endpoint=selected,selected_branch=scalar_tree(best),
                 endpoint_penalties=[float(v[0].detach()) for v in results],capability_domains=list(data['queries']),
                 capability_reduction=config['capability_reduction'],gradient_scope='Full selected-branch derivative through both specified trajectories; fitted reader assignment detached')
    return value,details
