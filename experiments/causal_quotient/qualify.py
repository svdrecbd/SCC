"""Bounded Charon qualification with preserved commands and independent checks."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from check import best_class_accuracy, check, forward, reachability
from models import inherited, structured, transition, words
from quotient import construct, forecast

HERE = Path(__file__).resolve().parent


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def regenerate(cfg):
    spec=importlib.util.spec_from_file_location('parent_generator',HERE/'parent_generator.py')
    parent=importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    parent_cfg=json.loads((HERE/'parent_config.json').read_text())
    parents=parent.worlds(parent_cfg)
    assert len(parents)==72
    return structured(cfg)+inherited(parents)


def command(args, out, name, expected=0, stdin=None):
    started = time.perf_counter()
    with (out / (name + '.stdout')).open('w') as stdout, (out / (name + '.stderr')).open('w') as stderr:
        p = subprocess.run(list(map(str, args)), input=stdin, text=True, stdout=stdout, stderr=stderr,
                           timeout=120, env={**os.environ, 'BEND_NO_TELEMETRY': '1'})
    dump(out / (name + '.receipt.json'), dict(args=list(map(str, args)), returncode=p.returncode,
         seconds=time.perf_counter()-started, timeout_seconds=120))
    assert p.returncode == expected, (name, p.returncode)


def verify(models, result, cfg):
    assert models == regenerate(cfg), 'input lineage or coverage changed'
    assert len(result['answers']) == len(models)
    summary = []; forecast_rows = []; total_values = 0; corruptions = []
    for model, ans in zip(models, result['answers']):
        assert ans['id'] == model['id']
        check(model, ans['useful'], False); check(model, ans['joint'], True)
        alternate=copy.deepcopy(model); alternate['hazard']=[1-h for h in model['hazard']]
        blind=construct(alternate,False)
        for key in ('classes','trace','quotient'):
            assert blind[key]==ans['useful'][key], 'useful quotient depends on hazard labels'
        u = ans['useful']; j = ans['joint']; n = len(model['rows'])
        us = {}; js = {}; rs = {}
        wordlist = words(len(model['rows'][0]), cfg['horizon'])
        for word in wordlist:
            key = tuple(word)
            us[key] = [forecast(u['quotient'], word, b) for b in range(3)]
            js[key] = [forecast(j['quotient'], word, b) for b in range(4)]
            rs[key] = reachability(j['quotient'], word)
            for s in range(n):
                useful, terminal, hit = forward(model, s, word)
                assert [v[u['classes'][s]] for v in us[key]] == useful
                assert [v[j['classes'][s]] for v in js[key]] == useful + [terminal]
                assert rs[key][j['classes'][s]] == hit
                total_values += 8
            forecast_rows.append(dict(id=model['id'], word=word, useful=us[key], joint=js[key], reach=rs[key]))
        row = dict(id=model['id'], family=model['family'], n=n,
            useful_classes=len(u['quotient']['rows']), joint_classes=len(j['quotient']['rows']),
            class_only_best_hazard_accuracy=str(best_class_accuracy(model, u['classes'])),
            useful_meter=u['meter'], joint_meter=j['meter'])
        if model['family'] == 'structured':
            linked=model['linked']; codes=model['codes']; index={c:i for i,c in enumerate(codes)}
            assert row['useful_classes'] == (16 if linked else 8)
            assert row['joint_classes'] == 16
            assert row['class_only_best_hazard_accuracy'] == ('1' if linked else '1/2')
            # Exact causal prediction: only visits to phase0 with action1 use h.
            effects = dict(interchanges=0, changed_useful_bit=0, zero_h_useful_correct=0,
                           zero_h_scalar_correct=0, zero_h_vector_correct=0,
                           matched_u_changed=0, matched_phase_changed=0, head_removed_useful_correct=0)
            assert len(ans['head_patch']) == n * len(wordlist)
            by_horizon = {}
            for word in wordlist:
                key=tuple(word); stats=by_horizon.setdefault(str(len(word)), dict(cases=0, correct_u=0, correct_vector=0))
                for s, code in enumerate(codes):
                    original=[v[u['classes'][s]] for v in us[key]]
                    swap=index[code ^ 8]; swapped=[v[u['classes'][swap]] for v in us[key]]
                    phase=code%4; parity=0
                    for a in word:
                        parity ^= int(linked and phase==0 and a==1); phase=(phase+1)%4
                    assert swapped == [original[0] ^ parity, *original[1:]]
                    zero=index[code & ~8]; zeroed=[v[u['classes'][zero]] for v in us[key]]
                    change_u=index[code ^ 4]; changed=[v[u['classes'][change_u]] for v in us[key]]
                    assert changed == [original[0] ^ 1, *original[1:]]
                    phase_swap=index[code ^ 1]
                    phase_changed=[v[u['classes'][phase_swap]] for v in us[key]]
                    assert phase_changed == forward(model,phase_swap,word)[0]
                    patch=ans['head_patch'][wordlist.index(word)*n+s]
                    assert patch == [*original, 0]
                    alias=index[(code%16)+16*((code//16+1)%model['aliases'])]
                    assert [v[u['classes'][alias]] for v in us[key]] == original
                    effects['interchanges']+=1
                    effects['changed_useful_bit']+=int(swapped[0]!=original[0])
                    effects['zero_h_useful_correct']+=int(zeroed[0]==original[0])
                    effects['zero_h_scalar_correct']+=sum(x==y for x,y in zip(zeroed,original))
                    effects['zero_h_vector_correct']+=int(zeroed==original)
                    effects['matched_u_changed']+=int(changed[0]!=original[0])
                    effects['matched_phase_changed']+=int(phase_changed!=original)
                    effects['head_removed_useful_correct']+=int(patch[:3]==original)
                    stats['cases']+=1; stats['correct_u']+=int(zeroed[0]==original[0]); stats['correct_vector']+=int(zeroed==original)
            assert len(ans['repairs']) == n
            repair_correct=0
            for s, repair in enumerate(ans['repairs']):
                p=codes[s]%4; expected_word=[0]*((-p)%4)+[1]
                useful_before=model['useful'][s][0]
                useful_after=forward(model,s,expected_word)[0][0]
                assert repair == dict(answer=useful_before ^ 1 ^ useful_after,
                    useful_calls=2, scalar_outputs=4, transition_reads=len(expected_word), word=expected_word)
                repair_correct += int(repair['answer']==model['hazard'][s])
            assert repair_correct == (n if linked else n//2)
            row.update(linked=linked, aliases=model['aliases'], seed=model['seed'], effects=effects,
                by_horizon=by_horizon, repair_correct=repair_correct, repair_total=n,
                repair_transition_reads=sum(r['transition_reads'] for r in ans['repairs']),
                repair_max_transition_reads=max(r['transition_reads'] for r in ans['repairs']),
                repair_seconds=ans['repair_seconds'], head_removed_hazard_correct=n//2)
        summary.append(row)
    # Corruption controls: require the checker to reject actual changed evidence.
    m=models[0]; a=result['answers'][0]['useful']
    for kind in ('initial_partition','final_partition','weight','calls','map_bits'):
        bad=copy.deepcopy(a)
        if kind=='initial_partition':bad['trace'][0]=list(range(len(m['rows'])))
        if kind=='final_partition':bad['classes'][0]=(bad['classes'][0]+1)%len(bad['quotient']['rows'])
        if kind=='weight':bad['quotient']['rows'][0][0][0][1]+=1
        if kind=='calls':bad['meter']['predictor_calls']+=1
        if kind=='map_bits':bad['meter']['map_bits']+=1
        try:check(m,bad,False)
        except AssertionError:corruptions.append(kind)
        else:raise AssertionError('corruption accepted: '+kind)
    original=forward(m,0,[1])[0]; bad=forecast(a['quotient'],[1],0); bad[a['classes'][0]]+=1
    assert bad[a['classes'][0]] != original[0]; corruptions.append('forecast')
    return dict(summary=summary, forecast_values_checked=total_values, corruptions=corruptions), forecast_rows


def verify_bend(records):
    expected={(mode,state,tuple(word)) for mode in (0,1) for state in range(16) for word in words(2,4)}
    observed=set()
    for row in records:
        key=(row['mode'],row['state'],tuple(row['word']))
        assert key in expected and key not in observed; observed.add(key)
        state=row['state']
        for action in row['word']:
            state=transition(state,action,bool(row['mode']))
        assert row['result']==state
    assert observed==expected
    return len(observed)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--node',type=Path,required=True); parser.add_argument('--bend-root',type=Path,required=True)
    args=parser.parse_args(); out=args.out.resolve(); out.mkdir(exist_ok=False)
    cfg=json.loads((HERE/'config.json').read_text()); start=time.perf_counter()
    assert platform.node()=='charon' and (HERE/'plan-frozen.md').exists()
    dump(out/'machine.json',dict(host=platform.node(),platform=platform.platform(),python=sys.version,
        affinity=sorted(os.sched_getaffinity(0)),bend_commit=cfg['bend_commit'],node=str(args.node)))
    try:
        command([args.node,'--version'],out,'node-version')
        assert (out/'node-version.stdout').read_text().strip()==cfg['node_version']
        command([args.node,HERE/'build.mjs',args.bend_root,HERE/'kernel.bend',out/'kernel.mjs'],out,'compile')
        command([args.node,HERE/'build.mjs',args.bend_root,HERE/'false_claim.bend'],out,'false-proof',1)
        negative=json.loads((out/'false-proof.stdout').read_text())
        assert negative['stage']=='typecheck' and 'false_hazard_erased' in negative['error']
        command([args.node,HERE/'evaluate.mjs',out/'kernel.mjs',out/'bend-records.json'],out,'bend-evaluate')
        bend_count=verify_bend(json.loads((out/'bend-records.json').read_text()))
        models=regenerate(cfg); dump(out/'models.json',models)
        request=json.dumps(models)
        command([sys.executable,'-S',HERE/'runner.py'],out,'worker',stdin=request)
        worker=json.loads((out/'worker.stdout').read_text())
        verified,forecasts=verify(models,worker,cfg)
        dump(out/'summary.json',verified); dump(out/'forecasts.json',forecasts)
        codebytes=sum((HERE/name).stat().st_size for name in ('quotient.py','runner.py','head_patch.py'))
        size=sum(p.stat().st_size for p in out.iterdir() if p.is_file()); assert size<=cfg['output_limit_bytes']
        receipt=dict(validation='PASS',models=len(models),bend_cases=bend_count,
            forecast_values_checked=verified['forecast_values_checked'],corruptions=verified['corruptions'],
            worker_code_bytes=codebytes,worker_input_bytes=len(request.encode()),
            worker_cpu_seconds_before_output=worker['cpu_seconds_before_output'],
            worker_peak_rss_kib=worker['peak_rss_kib'],elapsed_seconds=time.perf_counter()-start,output_bytes=size)
        dump(out/'receipt.json',receipt)
        paths=[p for d in (HERE,out) for p in d.iterdir() if p.is_file()]
        paths += [args.node]+[args.bend_root/'bend2'/name for name in ('bend.ts','comp.ts','base.bend')]
        dump(out/'sha256.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        print(json.dumps(receipt))
    except BaseException as error:
        dump(out/'failure.json',dict(error=repr(error)));raise


if __name__=='__main__':main()
