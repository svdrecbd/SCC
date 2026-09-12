from dataclasses import replace
import random

import pytest
import torch

from scc.persistent_matrix import MatrixConfig, LiveMatrix
from scc.persistent_tasks import (CELLS, Request, sample_request, evaluation_requests,
    training_requests, input_code, initialize_matrix, tensors, run_window,
    summarize_records, core_split, TOKENS_PER_REQUEST, D0, Q0)


def independent_answer(tokens):
    assert len(tokens)==19 and tokens[0]==0 and tokens[-1]==1
    prefix=tokens[1:5]
    family=next(t for t in prefix if t in (2,3,4))
    mode=next(t for t in prefix if t in (5,6))
    requester=next(t for t in prefix if t in (7,8))-7
    owner=next(t for t in prefix if t in (9,10))-9
    digits=[t-11 for t in tokens[5:17]]
    if mode==6 and requester!=owner:return 3
    if family==2:return digits[tokens[17]-14]
    if family==3:
        answer=0
        for bit in digits:answer^=bit
        return answer
    answer=0
    for digit in digits:
        answer+=digit
        while answer>=3:answer-=3
    return answer


def test_token_oracle_and_core_partition_group_permission_and_layout_variants():
    rng=random.Random(246)
    for split in ('train','validation','test'):
        for cell in CELLS:
            for _ in range(12):
                r=sample_request(rng,cell,split=split)
                assert r.answer==independent_answer(r.tokens)
                assert r.partition[0]==split
                alternate=replace(r,layout='reordered' if r.layout=='original' else 'original')
                assert alternate.partition==r.partition and alternate.answer==r.answer
                if r.context!='ungated':
                    other=replace(r,context='unauthorized' if r.context=='authorized' else 'authorized',owner=1-r.owner)
                    assert other.partition==r.partition
                if r.family!='lookup':assert replace(r,query=(r.query+1)%12).partition==r.partition


def test_streams_are_balanced_and_train_validation_cores_do_not_overlap():
    evaluation=evaluation_requests(999,per_cell=8,streams=8)
    assert all(len(s)==18 for s in evaluation)
    keys={r.partition[1] for s in evaluation for r in s}
    for cell in CELLS:assert sum((r.family,r.context,r.layout)==cell for s in evaluation for r in s)==8
    for step in (0,401,1001,2001):
        batch=training_requests(110,step,12,4)
        assert not keys & {r.partition[1] for s in batch for r in s}
        assert batch==training_requests(110,step,12,4)
    with pytest.raises(ValueError):evaluation_requests(10,per_cell=3,streams=16)


@pytest.mark.parametrize('rule',['soft','copy'])
def test_request_boundaries_do_not_reset_the_live_state(rule):
    config=MatrixConfig(32,4,rule);initial=initialize_matrix(config,103).double()
    code=input_code(32,'anchor',dtype=torch.float64)
    requests=training_requests(88,2001,2,4);ids,_=tensors(requests,device='cpu')
    full,end,_=run_window(initial,ids,config,code)
    first,middle,_=run_window(initial,ids[:,:2],config,code)
    second,continued,_=run_window(middle,ids[:,2:],config,code)
    assert torch.equal(full,torch.cat([first,second],1)) and torch.equal(end,continued)
    live=LiveMatrix(initial,config);ys=[]
    for r in requests[0]:
        for token in r.tokens:y,_=live.tick(code[token])
        ys.append(y)
    assert torch.allclose(torch.stack(ys),full[0],atol=1e-12,rtol=1e-12)
    assert live.steps==4*TOKENS_PER_REQUEST
    assert torch.allclose(live.weights,end[0],atol=1e-12,rtol=1e-12)
    if rule=='soft':
        reset,_,_=run_window(initial,ids[:,2:],config,code)
        assert not torch.allclose(reset,second,atol=1e-6,rtol=1e-6)


def test_input_code_is_fixed_and_read_token_cannot_expose_labels():
    code=input_code(64,'anchor')
    assert code.shape==(26,64) and not code.requires_grad
    assert torch.equal(code[:,-1],torch.full((26,),12.))
    assert torch.equal(code[:,:26].diag(),torch.full((26,),12.))
    assert torch.count_nonzero(code)==52
    token_only=input_code(64,'token');assert torch.count_nonzero(token_only)==26
    with pytest.raises(ValueError):input_code(26,'anchor')


def test_qualification_requires_every_cell_and_late_session_performance():
    streams=evaluation_requests(200,per_cell=128,streams=16);records=[]
    for stream,rows in enumerate(streams):
        for index,r in enumerate(rows):records.append({**r.record(),'prediction':independent_answer(r.tokens),'late_half':index>=72})
    result=summarize_records(records)
    assert result['qualified'] and len(result['cells'])==18
    victim=[r for r in records if r['family']=='lookup' and r['context']=='authorized' and r['layout']=='original' and r['late_half']]
    for r in victim[:4]:r['prediction']=(r['label']+1)%3
    failed=summarize_records(records)
    assert not failed['qualified']
    cell=failed['cells']['lookup/authorized/original']
    assert cell['accuracy']>=.95 and cell['late_accuracy']<.95
