"""Reconstruct finite-horizon hazard probability from retained one-step predictions."""


def recover(n, hazards, horizon, denominator, predict):
    hazards=set(hazards);rows={};entries=0
    for state in range(n):
        if state not in hazards:
            row=predict(state)
            assert sum(weight for target,weight in row)==denominator
            assert all(0<=target<n and weight>=0 for target,weight in row)
            rows[state]=row;entries+=len(row)
    previous=[int(s in hazards) for s in range(n)];scale=1
    for _ in range(horizon):
        scale*=denominator
        current=[scale if s in hazards else sum(w*previous[t] for t,w in rows[s]) for s in range(n)]
        previous=current
    return dict(numerators=previous,denominator=scale,
                meter=dict(predictor_calls=len(rows),returned_entries=entries,
                           multiply_adds=horizon*entries,state_writes=n*(horizon+1),
                           cached_rows=len(rows),peak_value_entries=2*n))
