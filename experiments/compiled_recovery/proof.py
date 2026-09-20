"""Small independent DRUP checker; retains deleted, already justified clauses."""


def masks(clause):
    pos=neg=0
    for literal in clause:
        if literal>0: pos |= 1 << (literal-1)
        else: neg |= 1 << (-literal-1)
    return pos,neg


def conflict(clauses,assumptions,meter):
    positive=negative=0
    for literal in assumptions:
        if literal>0: positive |= 1 << (literal-1)
        else: negative |= 1 << (-literal-1)
    if positive & negative: return True
    changed=True
    while changed:
        changed=False
        for pos,neg in clauses:
            meter['clause_visits']+=1
            if pos & neg: continue  # A tautological clause cannot force a value.
            if pos & positive or neg & negative: continue
            remaining=(pos|neg) & ~(positive|negative)
            if not remaining: return True
            if remaining & (remaining-1)==0:
                if remaining & pos: positive |= remaining
                else: negative |= remaining
                changed=True
    return False


def check_unsat(formula,lines):
    clauses=[masks(c) for c in formula]
    meter=dict(additions=0,ignored_deletions=0,clause_visits=0)
    for line in lines:
        fields=line.split()
        if not fields: continue
        if fields[0]=='d':
            meter['ignored_deletions']+=1
            continue
        values=list(map(int,fields))
        assert values[-1]==0 and all(v!=0 for v in values[:-1])
        clause=values[:-1]
        assert conflict(clauses,[-v for v in clause],meter),'non-RUP addition'
        clauses.append(masks(clause));meter['additions']+=1
    assert conflict(clauses,[],meter),'no certified contradiction'
    return meter
