"""Shortcut planner: no SAT solver, protected answer or original witness input."""


def common_literal(clauses):
    candidates=list(clauses[0]);comparisons=0
    for clause in clauses[1:]:
        kept=[]
        for candidate in candidates:
            for literal in clause:
                comparisons+=1
                if candidate==literal:
                    kept.append(candidate)
                    break
        candidates=kept
    if len(candidates)!=1:
        raise ValueError('fixture must have exactly one common literal')
    return candidates[0],comparisons


def plan(n,common,goal=None):
    if goal==-common:
        return None  # Unsupported infeasibility claim: deliberately tested.
    values={abs(common):common>0}
    if goal is not None:
        values[abs(goal)]=goal>0
    return [i if values.get(i,False) else -i for i in range(1,n+1)]
