"""Finite literal-goal cache. No search backend or parent oracle at runtime."""


def canonical(formula):
    return sorted(sorted(clause) for clause in formula)


def query(cache, formula, goal):
    meter = dict(model_goal_checks=0, formula_literal_checks=0)
    for model in cache['models']:
        meter['model_goal_checks'] += 1
        if model[abs(goal)-1] != goal:
            continue
        valid = True
        for clause in formula:
            satisfied = False
            for literal in clause:
                meter['formula_literal_checks'] += 1
                if model[abs(literal)-1] == literal:
                    satisfied = True
                    break
            if not satisfied:
                valid = False
                break
        if valid:
            return dict(status='SAT', model=model, meter=meter)
    if canonical(formula) == cache['formula'] and goal in cache['unsat_goals']:
        return dict(status='UNSAT', model=None, meter=meter)
    return dict(status='UNKNOWN', model=None, meter=meter)
