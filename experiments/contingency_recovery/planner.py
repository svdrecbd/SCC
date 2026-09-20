"""Finite-horizon plans with explicit nondeterministic outcomes."""


def reduce_outcomes(values, mode):
    if mode == 'robust':
        return all(values)
    if mode == 'optimistic':
        return any(values)
    if mode == 'first':
        return values[0]
    if mode == 'repaired':
        # Generic replacement, requiring neither labels nor a parent checkpoint.
        return not any(not value for value in values)
    raise ValueError(mode)


def plan(edges, terminal, horizon, objective, mode):
    n=len(edges); terminal=set(terminal)
    states=[[int(s in terminal if objective=='reach' else s not in terminal) for s in range(n)]]
    policies=[[-1]*n]
    meter=dict(outcome_reads=0,reducer_calls=0,state_writes=n,policy_writes=n)
    for depth in range(1,horizon+1):
        previous=states[-1];current=[];actions=[]
        for state in range(n):
            if state in terminal:
                current.append(int(objective=='reach'));actions.append(-1)
            else:
                winning=[]
                for successors in edges[state]:
                    selected=successors[:1] if mode=='first' else successors
                    values=[previous[t] for t in selected]
                    meter['outcome_reads']+=len(values);meter['reducer_calls']+=1
                    winning.append(reduce_outcomes(values,mode))
                action=next((i for i,ok in enumerate(winning) if ok),-1)
                current.append(int(action>=0));actions.append(action)
            meter['state_writes']+=1;meter['policy_writes']+=1
        states.append(current);policies.append(actions)
    return dict(states=states,policies=policies,meter=meter)
