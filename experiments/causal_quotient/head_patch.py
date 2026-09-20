"""Executable output-head removal control; the predictor remains editable/intact."""


def answer(q, state, word, remove_head):
    start = state
    for a in word:
        row = q['rows'][state][a]
        assert len(row) == 1 and row[0][1] == q['denominator']
        state = row[0][0]
    useful = q['labels'][state][:3]
    protected = 0 if remove_head else q['labels'][start][3]
    return [*useful, protected]
