"""LN-147: explicit primitive programs for multi-pass trajectory attacks.

Exact search is restricted to serial, fixed-T dependency schedules. Executable
program costs include T internals; no general program lower bound is claimed.
"""
import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shutil
import signal
import time

SEED = 20260917
MAX_CODE = 16000
MAX_REGS = 256
OPS = ('MOV','CONST','ANDI','SHR','SHL','ROL','ROR','ADDH','XOR','XORI','OR','ORI',
       'MULI','EMIT','JNZ','JZ','JMP','HALT')


def constants(w, rounds):
    rng = random.Random(f'wide-{w//2}-{rounds}-20260916')
    return [rng.randrange(1 << (w//2)) for _ in range(rounds)]


def rol(v, s, h):
    s %= h
    return ((v << s) | (v >> ((h-s) % h))) & ((1 << h)-1)


def transition(a, b, x, role, w, C):
    h = w//2; mask = (1 << h)-1
    L, H = a & mask, a >> h
    for j, c in enumerate(C):
        key = (rol(b & mask,j+1,h) ^ rol(b >> h,2*j+1,h) ^ (((x<<1)|role)*29) ^ c) & mask
        L = ((rol(L,-(h//2+1),h)+H) & mask) ^ key
        H = rol(H,2,h) ^ L
    return ((H << h)|L) ^ a


def direct(B, x, role, passes, w, C):
    B = list(B)
    for _ in range(passes):
        for i in range(len(B)):
            B[i] = transition(B[i],B[(i+1)%len(B)],x,role,w,C)
    return B


class CompileLimit(Exception):
    pass


class Program:
    def __init__(self, n, w, passes, C):
        self.n, self.w, self.passes, self.C = n,w,passes,C
        self.code = []
        self.regs = n+8  # bank, x/role and six cipher workspace registers

    def emit(self, op, a=0, b=0, c=0, imm=0, aux=0):
        if len(self.code) >= MAX_CODE or max(a,b,c) >= MAX_REGS:
            raise CompileLimit('instruction/register compilation cap')
        assert op in OPS and 0 <= imm < 65536 and 0 <= aux < 65536
        self.regs = max(self.regs,a+1,b+1,c+1)
        self.code.append([op,a,b,c,imm,aux])
        return len(self.code)-1

    def patch(self, address, target):
        self.code[address][4] = target

    def T(self, a, b, dest, role):
        # Input words survive until the final MOV. Keys are recomputed on the fly.
        h = self.w//2; mask = (1<<h)-1
        L,H,bl,bh,k,t = range(self.n+2,self.n+8)
        self.emit('ANDI',L,a,imm=mask); self.emit('SHR',H,a,imm=h)
        self.emit('ANDI',bl,b,imm=mask); self.emit('SHR',bh,b,imm=h)
        for j, const in enumerate(self.C):
            self.emit('ROL',k,bl,imm=(j+1)%h,aux=h)
            self.emit('ROL',t,bh,imm=(2*j+1)%h,aux=h)
            self.emit('XOR',k,k,t)
            self.emit('SHL',t,self.n,imm=1)
            if role == 'true': self.emit('OR',t,t,self.n+1)
            else: self.emit('ORI',t,t,imm=role)
            self.emit('MULI',t,t,imm=29)
            self.emit('XOR',k,k,t); self.emit('XORI',k,k,imm=const)
            self.emit('ANDI',k,k,imm=mask)
            self.emit('ROR',L,L,imm=h//2+1,aux=h)
            self.emit('ADDH',L,L,H,aux=h); self.emit('XOR',L,L,k)
            self.emit('ROL',H,H,imm=2%h,aux=h); self.emit('XOR',H,H,L)
        self.emit('SHL',k,H,imm=h); self.emit('OR',k,k,L)
        self.emit('XOR',k,k,a); self.emit('MOV',dest,k)

    def honest_core(self):
        for _ in range(self.passes):
            for i in range(self.n): self.T(i,(i+1)%self.n,i,'true')

    def loop_core(self, bank, role, repetitions, counter):
        if repetitions == 0: return
        self.emit('CONST',counter,imm=repetitions)
        start=len(self.code)
        for i in range(self.n): self.T(bank[i],bank[(i+1)%self.n],bank[i],role)
        self.emit('CONST',self.n+6,imm=(1<<self.w)-1)
        self.emit('ADDH',counter,counter,self.n+6,aux=self.w)
        self.emit('JNZ',counter,imm=start)

    def finish(self):
        return {'code':self.code,'registers':self.regs,'w':self.w,'n':self.n,
                'passes':self.passes,'constants':self.C,
                'data_bits':self.regs*self.w,'code_bits':64*len(self.code),
                'pc_bits':32,'output_latch_bits':self.w,
                'total_bits':self.regs*self.w+64*len(self.code)+32+self.w,
                'common_cipher_workspace_bits':6*self.w}


def execute(program, registers, x, role):
    """Registers persist across requests; only x/role are replaced externally."""
    regs = list(registers); n,w = program['n'],program['w']; mask=(1<<w)-1
    regs[n],regs[n+1] = x,role
    code=program['code']; pc=0; steps=0; outputs=[]
    while steps <= 2*MAX_CODE:
        if not 0 <= pc < len(code): raise ValueError('PC out of program')
        op,a,b,c,imm,aux=code[pc]; steps+=1; pc+=1
        if op=='MOV': regs[a]=regs[b]
        elif op=='CONST': regs[a]=imm & mask
        elif op=='ANDI': regs[a]=regs[b]&imm
        elif op=='SHR': regs[a]=regs[b]>>imm
        elif op=='SHL': regs[a]=(regs[b]<<imm)&mask
        elif op=='ROL': regs[a]=rol(regs[b],imm,aux)
        elif op=='ROR': regs[a]=rol(regs[b],-imm,aux)
        elif op=='ADDH': regs[a]=(regs[b]+regs[c])&((1<<aux)-1)
        elif op=='XOR': regs[a]=regs[b]^regs[c]
        elif op=='XORI': regs[a]=(regs[b]^imm)&mask
        elif op=='OR': regs[a]=regs[b]|regs[c]
        elif op=='ORI': regs[a]=(regs[b]|imm)&mask
        elif op=='MULI': regs[a]=(regs[b]*imm)&mask
        elif op=='EMIT': outputs.append(regs[a])
        elif op=='JNZ':
            if regs[a]: pc=imm
        elif op=='JZ':
            if not regs[a]: pc=imm
        elif op=='JMP': pc=imm
        elif op=='HALT':
            if len(outputs)!=1: raise ValueError('one output required')
            return regs,outputs[0],steps
        else: raise ValueError(op)
    raise ValueError('step limit')


def graph(n, passes, roles=(1,0)):
    preds={i:() for i in range(n)}; labels={}; finals={}
    for r in roles:
        bank=list(range(n))
        for _ in range(passes):
            for i in range(n):
                node=len(preds); preds[node]=(bank[i],bank[(i+1)%n]); labels[node]=r
                bank[i]=node
        finals[r]=bank
    return preds,labels,finals


def ancestors(preds, node):
    found=set(); stack=[node]
    while stack:
        a=stack.pop()
        if a not in found: found.add(a); stack.extend(preds[a])
    return found


def forward_prefix(prog, strategy):
    n=prog.n; scratch=n+10
    if strategy=='copy':
        for i in range(n): prog.emit('MOV',scratch+i,i)
        for p in range(prog.passes):
            for i in range(n):
                prog.T(scratch+i,scratch+(i+1)%n,scratch+i,1)
                if p==prog.passes-1 and i==0:
                    prog.emit('EMIT',scratch); return
    preds,_,finals=graph(n,prog.passes,(1,)); output=finals[1][0]
    if strategy=='slice':
        needed=ancestors(preds,output); use={v:0 for v in needed}
        for v in needed:
            for q in set(preds[v]): use[q]+=1
        use[output]+=1
        mapping={i:i for i in range(n)}; free=[]; nextreg=scratch
        for v in sorted(needed):
            if v<n: continue
            a,b=preds[v]
            reusable=[q for q in set((a,b)) if q>=n and use[q]==1]
            if reusable: dest=mapping[reusable[0]]
            elif free: dest=free.pop()
            else: dest=nextreg; nextreg+=1
            prog.T(mapping[a],mapping[b],dest,1)
            for q in set((a,b)):
                use[q]-=1
                if q>=n and use[q]==0:
                    old=mapping.pop(q)
                    if old!=dest: free.append(old)
            mapping[v]=dest
        prog.emit('EMIT',mapping[output]); return
    if strategy=='recompute':
        free=[]; nextreg=scratch
        def visit(v):
            nonlocal nextreg
            if v<n: return v
            a,b=preds[v]; ar=visit(a); br=visit(b)
            if ar>=scratch: dest=ar
            elif br>=scratch: dest=br
            elif free: dest=free.pop()
            else: dest=nextreg; nextreg+=1
            prog.T(ar,br,dest,1)
            for q in {ar,br}:
                if q>=scratch and q!=dest: free.append(q)
            return dest
        prog.emit('EMIT',visit(output)); return
    raise ValueError(strategy)


def compile_program(n,w,passes,C,strategy,search=None):
    p=Program(n,w,passes,C)
    if strategy in ('honest','benign','honest_loop'):
        if strategy=='honest_loop': p.loop_core(list(range(n)),'true',passes,n+8)
        else: p.honest_core()
        p.emit('EMIT',0); p.emit('HALT')
        if strategy=='benign': p.emit('CONST',n+2,imm=1)  # unreachable
        return p.finish()
    flag,mode=n+8,n+9
    branchflag=p.emit('JNZ',flag); branchrole=p.emit('JNZ',n+1)
    if strategy=='joint':
        preds,labels,finals=graph(n,passes)
        mapping={i:i for i in range(n)}; free=list(range(n+10,n+10+search['capacity']-n))
        for action in search['actions']:
            if action[0]=='drop': free.append(mapping.pop(action[1])); continue
            _,v,reuse,emit=action
            a,b=preds[v]
            dest=mapping[reuse] if reuse is not None else free.pop()
            p.T(mapping[a],mapping[b],dest,labels[v])
            if reuse is not None: del mapping[reuse]
            mapping[v]=dest
            if emit: p.emit('EMIT',dest)
        # Parallel register move into the canonical bank, using free T workspace for cycles.
        pending={i:mapping[v] for i,v in enumerate(finals[0]) if i!=mapping[v]}
        temp=n+6
        while pending:
            dest=next((d for d in pending if d not in pending.values()),None)
            if dest is not None:
                p.emit('MOV',dest,pending.pop(dest)); continue
            dest=next(iter(pending)); p.emit('MOV',temp,dest)
            pending={d:(temp if s==dest else s) for d,s in pending.items()}
        p.emit('CONST',flag,imm=1); p.emit('HALT')
    elif strategy=='destructive':
        for _ in range(passes):
            for i in range(n): p.T(i,(i+1)%n,i,1)
        p.emit('EMIT',0); p.emit('CONST',flag,imm=1); p.emit('HALT')
    else:
        if strategy=='copy_loop':
            scratch=n+10
            for i in range(n): p.emit('MOV',scratch+i,i)
            p.loop_core(list(range(scratch,scratch+n)),1,passes-1,scratch+n)
            p.T(scratch,scratch+1,scratch,1); p.emit('EMIT',scratch)
        else: forward_prefix(p,strategy)
        p.emit('CONST',flag,imm=1); p.emit('CONST',mode,imm=0)
        corejump=p.emit('JMP')
    normal=len(p.code); p.patch(branchflag,normal); p.patch(branchrole,normal)
    if strategy not in ('joint','destructive'):
        p.emit('CONST',mode,imm=1); p.patch(corejump,len(p.code))
    if strategy=='copy_loop': p.loop_core(list(range(n)),'true',passes,2*n+10)
    else: p.honest_core()
    if strategy not in ('joint','destructive'): skip=p.emit('JZ',mode)
    p.emit('EMIT',0)
    if strategy not in ('joint','destructive'): p.patch(skip,len(p.code))
    p.emit('HALT')
    return p.finish()


def joint_search(n,passes,capacity,max_states=100000,seconds=3):
    """0-1 shortest path; cost is number of T evaluations in a fixed-node grammar."""
    preds,_,finals=graph(n,passes); owner=finals[1][0]
    keep=ancestors(preds,owner)
    for v in finals[0]: keep |= ancestors(preds,v)
    computed=sorted(v for v in keep if v>=n)
    masks={v:sum(1<<q for q in set(preds[v])) for v in computed}
    goal=sum(1<<v for v in finals[0]); start=((1<<n)-1,False)
    dist={start:0}; previous={}; queue=deque([(start,0)]); began=time.monotonic(); examined=0
    while queue:
        if len(dist)>max_states or time.monotonic()-began>seconds:
            return {'status':'unknown_cap','capacity':capacity,'states':len(dist),'seconds':time.monotonic()-began}
        state,cost=queue.popleft()
        if dist[state]!=cost: continue
        held,emitted=state; examined+=1
        if emitted and held & goal == goal:
            actions=[]; current=state
            while current!=start:
                parent,action=previous[current]; actions.append(action); current=parent
            return {'status':'optimal_in_grammar','capacity':capacity,'T_calls':cost,
                    'actions':list(reversed(actions)),'states':len(dist),'seconds':time.monotonic()-began}
        def push(nextstate, action, extra):
            newcost=cost+extra
            if newcost<dist.get(nextstate,float('inf')):
                dist[nextstate]=newcost; previous[nextstate]=(state,action)
                item=(nextstate,newcost)
                if extra: queue.append(item)
                else: queue.appendleft(item)
        present=[v for v in keep if held & (1<<v)]
        for v in present: push((held & ~(1<<v),emitted),['drop',v],0)
        for v in computed:
            if held & (1<<v) or held & masks[v]!=masks[v]: continue
            options=([None] if len(present)<capacity else [])+list(set(preds[v]))
            for old in options:
                nxt=(held if old is None else held & ~(1<<old)) | (1<<v)
                emit=not emitted and v==owner
                push((nxt,emitted or emit),['calc',v,old,emit],1)
    return {'status':'unreachable_in_grammar','capacity':capacity,'states':len(dist),'seconds':time.monotonic()-began}


def repair_curve(n,w,passes,C):
    curves=[]
    for x in range(4):
        fibers={}
        for s in range(1<<(n*w)):
            B=[(s>>(w*i))&((1<<w)-1) for i in range(n)]
            source=tuple(direct(B,x,1,passes,w,C)); target=tuple(direct(B,x,0,passes,w,C))
            g=fibers.setdefault(source,{}); g[target]=g.get(target,0)+1
        curves.append([sum(sum(sorted(g.values(),reverse=True)[:1<<bits]) for g in fibers.values())/(1<<(n*w))
                       for bits in range(n*w+1)])
    return {'free_side_information_success_by_x_bits':curves,
            'public_full_lookup_data_bits':4*(1<<(n*w))*(n+1)*w,
            'scope':'uniform finite prior; oracle witness optimization, not an executed metered repair program'}


def save(path,obj): path.write_text(json.dumps(obj,separators=(',',':'))+'\n')


def run_case(n,w,passes,R,samples,future,exhaustive=False,searches=None):
    C=constants(w,R); rng=random.Random(SEED+1000*n+100*passes+w)
    programs={}; skipped={}
    strategies=['honest','benign','honest_loop','destructive','copy','copy_loop','slice','recompute']
    for strategy in strategies:
        try: programs[strategy]=compile_program(n,w,passes,C,strategy)
        except CompileLimit as e: skipped[strategy]=str(e)
    for search in searches or []:
        if search['status']=='optimal_in_grammar':
            label='joint_'+str(search['capacity'])
            programs[label]=compile_program(n,w,passes,C,'joint',search)
    seeds=[]
    if exhaustive:
        for state in range(1<<(n*w)):
            for x in range(4): seeds.append(([state>>(i*w)&((1<<w)-1) for i in range(n)],x))
    else:
        for _ in range(samples): seeds.append(([rng.randrange(1<<w) for i in range(n)],rng.randrange(4)))
    rows=[]
    for B,x in seeds:
        requests=[[x,0]]+[[rng.randrange(4),rng.randrange(2)] for _ in range(future)]
        traces={}
        for name,program in programs.items():
            registers=B+[0]*(program['registers']-n); trace=[]
            for xx,role in requests:
                registers,out,steps=execute(program,registers,xx,role)
                trace.append({'output':out,'bank':registers[:n],'steps':steps})
            traces[name]=trace
        rows.append({'initial':B,'requests':requests,'traces':traces})
    preds,_,finals=graph(n,passes,(1,)); dependencies=sorted(v for v in ancestors(preds,finals[1][0]) if v<n)
    return {'n':n,'w':w,'passes':passes,'R':R,'constants':C,'programs':programs,'skipped':skipped,
            'source_dependencies':dependencies,'searches':searches or [],'rows':rows,
            'exhaustive_initial_caller_inputs':exhaustive,
            'repair_diagnostic':repair_curve(n,w,passes,C) if exhaustive else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True); parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--wall-seconds',type=int,default=180)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    def timeout(*_): raise TimeoutError('run wall cap')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(args.wall_seconds)
    started=time.monotonic()
    here=Path(__file__).resolve(); source=args.out/'source'; source.mkdir()
    for name in ('screen_multipass_schedules.py','audit_multipass_schedules.py'):
        shutil.copyfile(here.with_name(name),source/name)
    shutil.copyfile(args.plan,args.out/'plan.md')
    config={'schema':1,'seed':SEED,'main_n':[2,4,8],'main_passes':[1,2,4,8],'w':8,'R':4,
            'samples':16,'future':4,'tiny_n':2,'tiny_w':4,'tiny_R':2,'tiny_passes':[1,2],
            'max_code':MAX_CODE,'max_registers':MAX_REGS,'search_states':100000,'search_seconds':3,
            'wall_seconds':args.wall_seconds,'output_bytes_cap':32*1024**2,
            'python':platform.python_version(),'platform':platform.platform()}
    save(args.out/'config.json',config)
    search_results={}
    for n,passes in ((2,1),(2,2),(4,1)):
        search_results[n,passes]=[joint_search(n,passes,cap) for cap in range(n,2*n+1)]
        print(json.dumps({'search':[n,passes],'results':[(r['capacity'],r['status'],r.get('T_calls')) for r in search_results[n,passes]]}),flush=True)
    cases=[]
    for n in config['main_n']:
        for passes in config['main_passes']:
            case=run_case(n,8,passes,4,16,4,searches=search_results.get((n,passes)))
            cases.append(case); save(args.out/'results.json',cases)
            print(json.dumps({'case':[n,8,passes],'programs':list(case['programs']),'skipped':case['skipped']}),flush=True)
    for passes in (1,2):
        cases.append(run_case(2,4,passes,2,0,0,True,search_results[2,passes]))
        save(args.out/'results.json',cases)
    save(args.out/'receipt.json',{'completed':True,'cases':len(cases),'seconds':time.monotonic()-started})
    files=[p for p in args.out.rglob('*') if p.is_file() and not p.name.startswith('._')]
    assert sum(p.stat().st_size for p in files)<config['output_bytes_cap']
    save(args.out/'sha256.json',{str(p.relative_to(args.out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    signal.alarm(0)
    print(json.dumps({'completed':True,'seconds':time.monotonic()-started}),flush=True)


if __name__=='__main__': main()
