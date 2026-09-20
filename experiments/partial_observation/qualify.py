import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from parents import population, terminal
from checks import check_parent, check_transcript, replay, evaluate, corruption_checks

HERE = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def discover(parent, cfg, stderr_path):
    transcript = []; start = time.perf_counter()
    worker_cfg = {k: cfg[k] for k in ('middle_depth', 'round_cap', 'query_cap', 'step_cap',
                                    'regime', 'probe_lengths', 'probe_patterns')}
    protocol_input_bytes = len((json.dumps(worker_cfg)+'\n').encode()); protocol_output_bytes = 0
    with stderr_path.open('w') as stderr:
        proc = subprocess.Popen(['timeout', str(cfg['worker_timeout']), sys.executable, '-S', str(HERE/'worker.py')],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1)
        try:
            proc.stdin.write(json.dumps(worker_cfg)+'\n')
            proc.stdin.flush()
            while True:
                line = proc.stdout.readline()
                assert line, 'worker ended before returning a model'
                protocol_output_bytes += len(line.encode())
                message = json.loads(line)
                if 'result' in message:
                    break
                assert set(message) == {'query'} and set(message['query']) <= {'0', '1'}
                word = message['query']; value = terminal(parent, word)
                transcript.append([word, value])
                response = json.dumps(dict(value=value))+'\n'
                protocol_input_bytes += len(response.encode())
                proc.stdin.write(response); proc.stdin.flush()
            proc.stdin.close(); assert proc.wait(timeout=5) == 0
        finally:
            if proc.poll() is None:
                proc.kill(); proc.wait()
    return dict(transcript=transcript, answer=message,
                protocol_input_bytes=protocol_input_bytes, protocol_output_bytes=protocol_output_bytes,
                wall_seconds=time.perf_counter()-start)


def verify_case(parent, case, cfg):
    check_parent(parent)
    result = case['answer']['result']
    check_transcript(parent, case['transcript'], result)
    replay(case['transcript'], result, cfg)
    expected = evaluate(parent, result, cfg)
    assert expected == case['evaluation']
    if parent['kind'] == 'delayed_control' and cfg['regime'] == 'short':
        assert result['status'] == 'bounded_conformance_pass'
        assert not expected['certificate']['exact']
        assert expected['certificate']['witness'] == '1'*(len(parent['outputs'])-1)
    return expected


def aggregate(parents, cases):
    rows = []
    for p, c in zip(parents, cases):
        r = c['answer']['result']; e = c['evaluation']; m = r['model']
        row = dict(id=p['id'], kind=p['kind'], physical_states=len(p['outputs']),
                   regime=p['regime'],
                   learned_states=len(m['outputs']) if m else None, status=r['status'],
                   exact=e.get('certificate', {}).get('exact', False),
                   queries=r['meter']['resets'], parent_steps=r['meter']['parent_steps'],
                   rounds=len(r['hypotheses']), wall_seconds=c['wall_seconds'],
                   cpu_seconds=c['answer']['cpu_seconds'], peak_rss_kib=c['answer']['peak_rss_kib'])
        row['model_json_bytes'] = len(json.dumps(m).encode())
        row['transcript_json_bytes'] = len(json.dumps(c['transcript']).encode())
        row['maximum_query_length'] = max(len(w) for w, _ in c['transcript'])
        row['protocol_input_bytes'] = c['protocol_input_bytes']
        row['protocol_output_bytes'] = c['protocol_output_bytes']
        if m:
            states = len(m['outputs']); width = (states-1).bit_length()
            row['operational_state_bits'] = width
            row['transition_output_initial_payload_bits'] = 2*states*width+states+width
        if 'tasks' in e:
            row['tasks'] = e['tasks']['counts']
            row['protected_homogeneous'] = e['tasks']['protected_factorization']['homogeneous']
        if 'witness' in e.get('certificate', {}):
            row['witness'] = e['certificate']['witness']
        rows.append(row)
    return rows


def cases_population(cfg):
    return [dict(p, id=p['id']+'-'+regime, regime=regime)
            for regime in cfg['regimes'] for p in population(cfg)]


def check_aliases(parents, cases):
    lookup = {p['id']: c for p, c in zip(parents, cases)}
    for p, c in zip(parents, cases):
        if p.get('aliases') == 4:
            other = lookup[p['id'].replace('-a4-', '-a1-')]
            assert c['transcript'] == other['transcript']
            assert c['answer']['result'] == other['answer']['result']


def audit(out):
    cfg = json.loads((HERE/'config.json').read_text())
    parents = json.loads((out/'parents.json').read_text()); assert parents == cases_population(cfg)
    cases = [json.loads((out/(p['id']+'.json')).read_text()) for p in parents]
    for p, c in zip(parents, cases):
        verify_case(p, c, dict(cfg, regime=p['regime']))
    check_aliases(parents, cases)
    rows = aggregate(parents, cases); assert rows == json.loads((out/'summary.json').read_text())
    hashes = json.loads((out/'sha256.json').read_text())
    for path, digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    first = next((p, c) for p, c in zip(parents, cases) if c['evaluation'].get('certificate', {}).get('exact'))
    p, c = first
    corruptions = corruption_checks(p, c['transcript'], c['answer']['result'], c['evaluation'])
    assert corruptions == json.loads((out/'receipt.json').read_text())['corruptions']
    return dict(audit='PASS', models=len(cases), hashes=len(hashes))


def main():
    out = Path(sys.argv[1]).resolve(); out.mkdir(exist_ok=False)
    start = time.perf_counter(); assert platform.node() == 'charon'
    assert (HERE/'plan-frozen.md').exists()
    cfg = json.loads((HERE/'config.json').read_text())
    save(out/'machine.json', dict(host=platform.node(), platform=platform.platform(),
                                 python=sys.version, affinity=sorted(os.sched_getaffinity(0))))
    parents = cases_population(cfg); save(out/'parents.json', parents); cases = []
    try:
        for p in parents:
            effective = dict(cfg, regime=p['regime'])
            case = discover(p, effective, out/(p['id']+'.stderr'))
            case['evaluation'] = evaluate(p, case['answer']['result'], effective)
            save(out/(p['id']+'.json'), case)
            verify_case(p, case, effective)
            cases.append(case)
            print(json.dumps(aggregate([p], [case])[0]), flush=True)
        check_aliases(parents, cases)
        p, c = next((p, c) for p, c in zip(parents, cases) if c['evaluation'].get('certificate', {}).get('exact'))
        corruptions = corruption_checks(p, c['transcript'], c['answer']['result'], c['evaluation'])
        rows = aggregate(parents, cases); save(out/'summary.json', rows)
        size = sum(p.stat().st_size for p in out.iterdir() if p.is_file()); assert size <= cfg['output_cap']
        receipt = dict(qualification='PASS', models=len(cases), exact=sum(r['exact'] for r in rows),
                       delayed_misses=sum(r['kind']=='delayed_control' and not r['exact'] for r in rows),
                       corruptions=corruptions, total_seconds=time.perf_counter()-start, output_bytes=size,
                       learner_source_bytes=sum((HERE/f).stat().st_size for f in ('learner.py','worker.py')))
        save(out/'receipt.json', receipt)
        save(out/'sha256.json', {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                for folder in (HERE, out) for p in folder.iterdir() if p.is_file()})
        print(json.dumps(receipt), flush=True)
    except BaseException as error:
        save(out/'failure.json', dict(error=repr(error)))
        raise


if __name__ == '__main__':
    main()
