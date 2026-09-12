"""Recovered-answer bound plus finite-code disagreement after declared edits."""
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from .behavior_bound import bound_branches
from .data import IGNORE
from .differentiable_modify import adam_unroll
from .functional_state import call_parameters
from .learned_bottleneck import editable_names
from .pilot_objectives import fitted
from .portfolio_objective import make_contraction_episode
from .selective_coupling import edit_loss


def code_disagreement(codes, mask):
    rows=codes[mask]
    if rows.ndim!=2 or len(rows)<2: raise ValueError('Multiple code observations required')
    # For exact binary codes, zero iff every observed code is identical.
    # Maximum over bits prevents averaging away a surviving varying bit.
    variance=(rows-rows.mean(0,keepdim=True)).square().mean(0)
    return variance.amax()


def mean_code_variance(codes, mask):
    rows=codes[mask]
    if len(rows)<2: raise ValueError('Multiple code observations required')
    # For hard codes this equals twice the mean normalized Hamming distance
    # between two uniformly sampled rows. A small mean can retain useful bits.
    return (rows-rows.mean(0,keepdim=True)).square().mean()


def discrete_objective(model, episode, *, create_graph=True):
    config=episode['configuration']
    with sdpa_kernel(SDPBackend.MATH):
        parameters=dict(model.named_parameters())
        trace=[]
        changed=adam_unroll(parameters,
            [lambda p,b=b:edit_loss(model,p,b,config) for b in episode['changes']],
            lr=config['inner_lr'],eps=config['inner_epsilon'],create_graph=create_graph,
            editable=editable_names(model,config['inner_scope']),trace=trace)
        scores,maximum_bits,capabilities,counts={},{},{},{}
        for domain,batch in episode['queries'].items():
            logits,codes=call_parameters(model,changed,(batch.tokens,),{'return_codes':True},strict=True)
            capabilities[domain]=(logits,batch.targets)
            for layer,code in enumerate(codes):
                name=f'{domain}/pass-{layer}'
                mask=batch.targets!=IGNORE
                scores[name]=mean_code_variance(code,mask)
                maximum_bits[name]=code_disagreement(code,mask)
                with torch.no_grad():
                    rows=code[mask]
                    counts[name]={'rows':len(rows),'coordinates':rows.shape[1],
                                  'unique_codes':len(torch.unique(rows.detach(),dim=0)),
                                  'exact_binary':bool(((rows==1)|(rows==-1)).all())}
        spread=torch.stack(list(scores.values())).amax()
        readers=fitted(model,changed,episode['supports'])
        targets={n:(call_parameters(model,changed,(b.tokens,),strict=True),b.targets)
                 for n,b in episode['target_queries'].items()}
        bound,branches=bound_branches(capabilities,targets,readers)
        value=bound+.25*spread
    record={**episode['record'],'value':float(value.detach()),'behavior_bound':float(bound.detach()),
            'code_disagreement':float(spread.detach()),'code_disagreements':{n:float(v.detach()) for n,v in scores.items()},
            'maximum_bit_variances':{n:float(v.detach()) for n,v in maximum_bits.items()},
            'spread_definition':'Maximum across domain/pass means of per-coordinate variance; twice average normalized Hamming distance for exact binary codes',
            'code_counts':counts,'inner_trace':trace,'hard_forward':model.config.hard,
            'reader_branches':[{'reader':b['reader'],'stop_rule':b['stop_rule'],
                               'trigger_bound':float(b['trigger_surrogate'].detach()),
                               'capability_bound':float(b['capability'].detach()),
                               'domains':{n:float(v.detach()) for n,v in b['domains'].items()}} for b in branches],
            'derivative':('Full unrolled coarse/straight-through chain through hard forward operations; NOT the derivative of the hard objective'
                          if model.config.hard else 'Full selected-branch derivative through the declared smooth inner procedure; discrete reader assignments detached'),
            'limitation':'Identical observed binary codes concern only the sampled bottleneck interface. Earlier activations, weights, unobserved inputs and alternative graph interpretations may retain cognition.'}
    return value,record
