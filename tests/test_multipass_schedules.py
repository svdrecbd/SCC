"""Checks of compiled schedules and independent finite-word semantics."""
import json
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import screen_multipass_schedules as s
from scripts import audit_multipass_schedules as a


@pytest.mark.parametrize('n,passes',[(2,1),(2,2),(4,1),(4,4),(8,8)])
def test_concrete_schedules_preserve_honest_future(n,passes):
    C=s.constants(8,4); rng=random.Random(10)
    for strategy in ('honest','benign','honest_loop','destructive','copy','copy_loop','slice','recompute'):
        try: p=s.compile_program(n,8,passes,C,strategy)
        except s.CompileLimit:
            assert strategy=='recompute'; continue
        for _ in range(4):
            B=[rng.randrange(256) for _ in range(n)]; regs=B+[0]*(p['registers']-n)
            # Authorized first request must not consume the one-shot attack.
            for t,r in enumerate((1,0,0,1)):
                x=rng.randrange(4); before=list(regs)
                regs,out,steps=s.execute(p,regs,x,r)
                other,out2,steps2=a.machine(p,before,x,r)
                assert (regs,out,steps)==(other,out2,steps2)
                true=a.reference(B,x,r,passes,8,C)
                owner=a.reference(B,x,1,passes,8,C)[0]
                if strategy!='destructive':
                    assert regs[:n]==true
                    assert out==(owner if strategy not in ('honest','benign','honest_loop') and t==1 else true[0])
                B=true


@pytest.mark.parametrize('n,passes',[(2,1),(2,2),(4,1)])
def test_exact_schedule_minima_and_generated_programs(n,passes):
    C=s.constants(4,2)
    for capacity in range(n,2*n+1):
        result=s.joint_search(n,passes,capacity,seconds=10)
        expected=a.schedule_optimum(n,passes,capacity)
        assert result['status']!='unknown_cap'
        assert (result['status']=='unreachable_in_grammar')==(expected is None)
        if expected is None: continue
        assert result['T_calls']==expected
        p=s.compile_program(n,4,passes,C,'joint',result)
        for initial in ([0]*n,list(range(n)),[15]*n):
            regs,out,_=s.execute(p,initial+[0]*(p['registers']-n),2,0)
            assert out==a.reference(initial,2,1,passes,4,C)[0]
            assert regs[:n]==a.reference(initial,2,0,passes,4,C)


def test_one_pass_bypass_and_workspace_are_visible():
    p=s.compile_program(4,8,1,s.constants(8,4),'slice')
    baseline=s.compile_program(4,8,1,s.constants(8,4),'honest')
    assert p['common_cipher_workspace_bits']==baseline['common_cipher_workspace_bits']==48
    assert p['registers']==baseline['registers']+3  # flag, mode, one owner result
    graph,_,final=s.graph(4,1,(1,))
    assert {v for v in s.ancestors(graph,final[1][0]) if v<4}=={0,1}


def test_caps_are_unknown_not_unreachable():
    result=s.joint_search(2,2,3,max_states=1)
    assert result['status']=='unknown_cap'


def test_instruction_overflow_and_rotations():
    p=s.Program(2,4,1,[1]); p.emit('CONST',2,imm=15); p.emit('MULI',2,2,imm=3)
    p.emit('EMIT',2); p.emit('HALT'); p=p.finish()
    initial=[0]*p['registers']
    assert s.execute(p,initial,0,0)[1]==13
    assert a.machine(p,initial,0,0)[1]==13


def test_empty_evidence_fails_closed(tmp_path):
    (tmp_path/'sha256.json').write_text('{}')
    assert a.audit(tmp_path)['passed'] is False


def test_resource_ledger_cannot_omit_common_workspace():
    C=s.constants(8,4); p=s.compile_program(4,8,2,C,'slice')
    case={'n':4,'w':8,'passes':2,'constants':C}
    a.verify_program(p,case)
    p['common_cipher_workspace_bits']=0
    with pytest.raises(ValueError,match='resource ledger'): a.verify_program(p,case)


def test_unencodable_instruction_is_rejected():
    C=s.constants(8,4); p=s.compile_program(2,8,1,C,'copy')
    p['code'][0][4]=65536
    with pytest.raises(ValueError,match='instruction encoding'):
        a.verify_program(p,{'n':2,'w':8,'passes':1,'constants':C})
