"""Conjunction reader, executable under Python -S with only retained cache inputs."""
import json
import sys
from cache import canonical


def query(cache, formula, goals):
    checks=0
    for model in cache['models']:
        if not all(g in model for g in goals):
            continue
        valid=True
        for clause in formula:
            satisfied=False
            for literal in clause:
                checks+=1
                if literal in model:
                    satisfied=True
                    break
            if not satisfied:
                valid=False
                break
        if valid:
            return dict(status='SAT',model=model,literal_checks=checks)
    if canonical(formula)==cache['formula'] and any(g in cache['unsat_goals'] for g in goals):
        return dict(status='UNSAT',model=None,literal_checks=checks)
    return dict(status='UNKNOWN',model=None,literal_checks=checks)


if __name__=='__main__':
    tasks=json.load(sys.stdin)
    print(json.dumps([[query(t['cache'],t['formula'],g) for g in t['goals']] for t in tasks]))
