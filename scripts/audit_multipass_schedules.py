"""Independent LN-147 primitive execution, scalar scoring and schedule checks.

No screen import. Exactness concerns saved programs and the specified dependency
schedule grammar, never arbitrary machine rewrites or learned cognition.
"""
import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import signal
import time


def require(ok, message):
    if not ok: raise ValueError(message)


def rotate(x, shift, width):
    shift%=width; mask=(1<<width)-1; x &= mask
    for _ in range(shift): x=((x<<1)|(x>>(width-1)))&mask
    return x


def reference(bank,x,role,passes,w,C):
    bank=list(bank); h=w//2; half=(1<<h)-1
    for _ in range(passes):
        for i in range(len(bank)):
            a,b=bank[i],bank[(i+1)%len(bank)]
            left,right=a% (1<<h),a//(1<<h)
            for j,c in enumerate(C):
                key=(rotate(b&half,j+1,h)^rotate(b>>h,2*j+1,h)^((2*x+role)*29)^c)&half
                left=((rotate(left,-(h//2+1),h)+right)% (1<<h))^key
                right=rotate(right,2,h)^left
            bank[i]=((right<<h)+left)^a
    return bank


def machine(program,regs,x,r):
    regs=list(regs); mask=(1<<program['w'])-1
    regs[program['n']]=x; regs[program['n']+1]=r
    pc=0; steps=0; out=[]
    while steps<32001:
        require(0<=pc<len(program['code']),'bad PC')
        op,a,b,c,i,h=program['code'][pc]; pc+=1; steps+=1
        if op in ('JZ','JNZ'):
            if (regs[a]==0)==(op=='JZ'): pc=i
        elif op=='JMP': pc=i
        elif op=='EMIT': out.append(regs[a])
        elif op=='HALT':
            require(len(out)==1,'wrong emission count')
            return regs,out[0],steps
        else:
            if op=='MOV': value=regs[b]
            elif op=='CONST': value=i
            elif op=='ANDI': value=regs[b]&i
            elif op=='SHR': value=regs[b]//(1<<i)
            elif op=='SHL': value=regs[b]*(1<<i)
            elif op=='ROL': value=rotate(regs[b],i,h)
            elif op=='ROR': value=rotate(regs[b],-i,h)
            elif op=='ADDH': value=(regs[b]+regs[c])%(1<<h)
            elif op=='XOR': value=regs[b]^regs[c]
            elif op=='XORI': value=regs[b]^i
            elif op=='OR': value=regs[b]|regs[c]
            elif op=='ORI': value=regs[b]|i
            elif op=='MULI': value=regs[b]*i
            else: raise ValueError('unknown opcode')
            regs[a]=value&mask
    raise ValueError('execution cap')


def dependency_graph(n,passes):
    edges={i:() for i in range(n)}; final={}
    for role in (1,0):
        current=list(range(n))
        for _ in range(passes):
            for i in range(n):
                edges[len(edges)]=(current[i],current[(i+1)%n]); current[i]=len(edges)-1
        final[role]=current
    needed=set()
    def visit(v):
        if v in needed: return
        needed.add(v)
        for p in edges[v]: visit(p)
    visit(final[1][0]); sources=sorted(i for i in needed if i<n)
    for v in final[0]: visit(v)
    return edges,final,needed,sources


def schedule_optimum(n,passes,capacity):
    edges,final,needed,_=dependency_graph(n,passes)
    goal=set(final[0]); owner=final[1][0]
    heap=[(0,tuple(range(n)),False)]; visited=set()
    while heap:
        cost,values,emitted=heapq.heappop(heap)
        state=(values,emitted)
        if state in visited: continue
        visited.add(state); held=set(values)
        if emitted and goal<=held: return cost
        for v in held:
            nxt=(tuple(sorted(held-{v})),emitted)
            if nxt not in visited: heapq.heappush(heap,(cost,*nxt))
        for v in sorted(needed-held):
            if v<n or not set(edges[v])<=held: continue
            choices=[held|{v}] if len(held)<capacity else []
            choices += [(held-{old})|{v} for old in set(edges[v])]
            for choice in choices:
                nxt=(tuple(sorted(choice)),emitted or v==owner)
                if nxt not in visited: heapq.heappush(heap,(cost+1,*nxt))
    return None


def verify_program(p,case):
    n,w=case['n'],case['w']
    require((p['n'],p['w'],p['passes'],p['constants'])==(n,w,case['passes'],case['constants']),'program context')
    require(0<len(p['code'])<=16000 and n+8<=p['registers']<=256,'program bounds')
    for ins in p['code']:
        require(len(ins)==6,'instruction width')
        op,a,b,c,imm,aux=ins
        require(op in ('MOV','CONST','ANDI','SHR','SHL','ROL','ROR','ADDH','XOR','XORI','OR','ORI','MULI','EMIT','JNZ','JZ','JMP','HALT'),'opcode')
        require(all(type(q)is int and 0<=q<p['registers'] for q in (a,b,c)),'register range')
        require(type(imm)is int and type(aux)is int and 0<=imm<65536 and 0<=aux<65536,'instruction encoding')
        if op in ('ROL','ROR','ADDH'): require(0<aux<=w,'arithmetic width')
        if op in ('SHR','SHL'): require(imm<=w,'shift bound')
        if op in ('JZ','JNZ','JMP'): require(imm<len(p['code']),'jump range')
    expected={'data_bits':p['registers']*w,'code_bits':64*len(p['code']),'pc_bits':32,
              'output_latch_bits':w,'total_bits':p['registers']*w+64*len(p['code'])+32+w,
              'common_cipher_workspace_bits':6*w}
    require(all(p[k]==v for k,v in expected.items()),'resource ledger')


def audit(directory):
    directory=Path(directory); report={'passed':False,'errors':[],'cases':[],'searches':[]}
    try:
        manifest=json.loads((directory/'sha256.json').read_text())
        required={'config.json','results.json','receipt.json','plan.md','source/screen_multipass_schedules.py','source/audit_multipass_schedules.py'}
        require(required<=manifest.keys(),'required evidence absent')
        for name,digest in manifest.items():
            path=directory/name
            require(path.resolve().is_relative_to(directory.resolve()),'manifest path')
            require(hashlib.sha256(path.read_bytes()).hexdigest()==digest,'hash '+name)
        config=json.loads((directory/'config.json').read_text()); receipt=json.loads((directory/'receipt.json').read_text())
        cases=json.loads((directory/'results.json').read_text())
        expected={(n,8,k) for n in (2,4,8) for k in (1,2,4,8)} | {(2,4,1),(2,4,2)}
        require(receipt['completed'] is True and receipt['cases']==len(cases)==len(expected),'completion/inventory')
        require({(c['n'],c['w'],c['passes']) for c in cases}==expected,'case inventory')
        require(config['schema']==1 and config['samples']==16 and config['future']==4,'config contract')
        checked_searches={}
        for case in cases:
            n,w,k,C=case['n'],case['w'],case['passes'],case['constants']
            require(case['R']==len(C)==(2 if w==4 else 4),'cipher rounds')
            require(all(type(c)is int and 0<=c<1<<(w//2) for c in C),'constants')
            _,_,_,sources=dependency_graph(n,k)
            require(sources==case['source_dependencies'],'dependency set')
            searches=case['searches']
            require({s['capacity'] for s in searches}==(set(range(n,2*n+1)) if (n,k) in ((2,1),(2,2),(4,1)) else set()),'search inventory')
            for s in searches:
                key=(n,k,s['capacity'])
                if s['status']=='unknown_cap': continue
                require(s['status'] in ('optimal_in_grammar','unreachable_in_grammar'),'search status')
                if key not in checked_searches:
                    checked_searches[key]=schedule_optimum(*key)
                    report['searches'].append({'n':n,'passes':k,'capacity':s['capacity'],'independent_T_calls':checked_searches[key]})
                expected_calls=checked_searches[key]
                require((s['status']=='unreachable_in_grammar')==(expected_calls is None),'reachability claim')
                if expected_calls is not None: require(s['T_calls']==expected_calls,'optimality in grammar')
            required_programs={'honest','benign','honest_loop','destructive','copy','copy_loop','slice','recompute'}
            required_programs |= {'joint_'+str(s['capacity']) for s in searches if s['status']=='optimal_in_grammar'}
            require(set(case['programs'])|set(case['skipped'])==required_programs,'program inventory')
            require(not(set(case['programs']) & set(case['skipped'])) and set(case['skipped'])<={'recompute'},'unexpected omitted program')
            for p in case['programs'].values(): verify_program(p,case)
            rows=case['rows']; tiny=w==4
            require(case['exhaustive_initial_caller_inputs'] is tiny,'coverage label')
            require(len(rows)==(1024 if tiny else 16),'sample inventory')
            if tiny: require({(tuple(row['initial']),row['requests'][0][0]) for row in rows}=={((a,b),x) for a in range(16) for b in range(16) for x in range(4)},'exhaustive coverage')
            stats={name:{'owner_correct':0,'caller_bank_after_attack':0,'future_answers':0,'future_banks':0,'first_steps':[],'future_steps':[]} for name in case['programs']}
            for row in rows:
                B=row['initial']; requests=row['requests']
                require(len(B)==n and all(type(v)is int and 0<=v<1<<w for v in B),'bank')
                require(len(requests)==(1 if tiny else 5) and requests[0][1]==0,'request count/trigger')
                require(all(len(q)==2 and q[0] in range(4) and q[1] in (0,1) for q in requests),'requests')
                require(set(row['traces'])==set(case['programs']),'trace inventory')
                owner=reference(B,requests[0][0],1,k,w,C)[0]
                reference_states=[]; current=B
                for x,r in requests:
                    current=reference(current,x,r,k,w,C); reference_states.append(current)
                for name,p in case['programs'].items():
                    regs=B+[0]*(p['registers']-n); traces=row['traces'][name]
                    require(len(traces)==len(requests),'trace length')
                    for t,((x,r),observed,ref) in enumerate(zip(requests,traces,reference_states)):
                        regs,output,steps=machine(p,regs,x,r)
                        require(observed=={'output':output,'bank':regs[:n],'steps':steps},'independent VM disagreement')
                        if t==0:
                            stats[name]['owner_correct']+=output==owner
                            stats[name]['caller_bank_after_attack']+=regs[:n]==ref
                            stats[name]['first_steps'].append(steps)
                        else:
                            stats[name]['future_answers']+=output==ref[0]
                            stats[name]['future_banks']+=regs[:n]==ref
                            stats[name]['future_steps'].append(steps)
                        if name in ('honest','benign','honest_loop'): require(regs[:n]==ref and output==ref[0],'intact qualification')
                        elif name!='destructive': require(regs[:n]==ref and output==(owner if t==0 else ref[0]),'useful bypass failed')
            summary={}
            for name,s in stats.items():
                p=case['programs'][name]; first=sum(s['first_steps'])/len(rows)
                summary[name]={'owner_answer_fraction':s['owner_correct']/len(rows),'caller_state_fraction':s['caller_bank_after_attack']/len(rows),
                               'future_answer_fraction':s['future_answers']/(len(rows)*4) if not tiny else None,
                               'future_state_fraction':s['future_banks']/(len(rows)*4) if not tiny else None,
                               'mean_first_steps':first,'max_first_steps':max(s['first_steps']),
                               'registers':p['registers'],'data_bits':p['data_bits'],'code_bits':p['code_bits'],'total_bits':p['total_bits']}
            if tiny:
                diag=case['repair_diagnostic']; curve=[]
                for x in range(4):
                    groups={}
                    for a in range(16):
                        for b in range(16):
                            src=tuple(reference([a,b],x,1,k,w,C)); dst=tuple(reference([a,b],x,0,k,w,C))
                            groups.setdefault(src,Counter())[dst]+=1
                    curve.append([sum(sum(sorted(g.values(),reverse=True)[:2**bits]) for g in groups.values())/256 for bits in range(9)])
                require(curve==diag['free_side_information_success_by_x_bits'],'repair curve')
                require(diag['public_full_lookup_data_bits']==4*256*3*4,'lookup storage')
            report['cases'].append({'n':n,'w':w,'passes':k,'source_dependencies':sources,'samples':len(rows),'chance':1/(1<<w),'programs':summary,'skipped':case['skipped']})
        report['passed']=True
    except (OSError,ValueError,KeyError,TypeError,IndexError) as exc: report['errors'].append(str(exc))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path); parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--wall-seconds',type=int,default=180); args=parser.parse_args()
    def timeout(*_): raise TimeoutError('audit cap')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(args.wall_seconds); start=time.monotonic()
    report=audit(args.directory); report['seconds']=time.monotonic()-start
    with args.out.open('x') as f: json.dump(report,f,indent=2); f.write('\n')
    print(json.dumps({'passed':report['passed'],'seconds':report['seconds'],'errors':report['errors']}))
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__': main()
