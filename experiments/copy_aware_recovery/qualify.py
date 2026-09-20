import hashlib
import json
from pathlib import Path
import platform
import os
import resource
import sys
import time
from fractions import Fraction
from replay import Oracle, replay
from check import check_result, masses, check_mass, corruptions, fraction

HERE = Path(__file__).resolve().parent


def dump(path, value):
    path.write_text(json.dumps(value, separators=(',', ':'))+'\n')


def load_inputs():
    inherited = HERE.parent/'inherited'
    manifest = json.loads((inherited/'selected-sha256.json').read_text())
    for filename, digest in manifest.items():
        assert hashlib.sha256((inherited/filename).read_bytes()).hexdigest() == digest
    parents = json.loads((inherited/'parents.json').read_text())
    cases = []
    for parent in parents:
        original = json.loads((inherited/(parent['id']+'.json')).read_text())
        transcript = original['transcript']
        counts = original['answer']['result']['meter']
        assert len(transcript) == counts['resets']
        assert sum(len(w) for w, _ in transcript) == counts['parent_steps']
        cases.append(dict(parent=parent, transcript=transcript))
    assert len(cases) == 30
    assert sum(len(c['transcript']) for c in cases) == 81627
    assert sum(len(w) for c in cases for w, _ in c['transcript']) == 1209616
    return cases, manifest


def key(parent, capacity, policy='all_prefixes'):
    return parent['id']+'-'+policy+'-cache'+('all' if capacity is None else str(capacity))


def summary(parent, transcript, capacity, result, elapsed, policy='all_prefixes'):
    m = result['meter']; baseline = sum(len(w) for w, _ in transcript)
    copies = m['saves']+m['restores']
    return dict(id=parent['id'], policy=policy, capacity=capacity, full_replay_steps=baseline,
                **m, break_even_copy_cost=fraction(Fraction(baseline-m['transitions'], copies)) if copies else None,
                parent_json_bytes=len(json.dumps(parent).encode()),
                query_word_bits=baseline, wall_seconds=elapsed)


def audit(out):
    cfg = json.loads((HERE/'config.json').read_text()); cases, inherited = load_inputs()
    rows = json.loads((out/'summary.json').read_text()); index = 0
    for case in cases:
        parent, transcript = case['parent'], case['transcript']
        for policy in cfg['policies']:
            for capacity in cfg['capacities']:
                result = json.loads((out/(key(parent, capacity, policy)+'.json')).read_text())
                check_result(parent, transcript, capacity, result, policy)
                row = rows[index]; index += 1
                assert row == summary(parent, transcript, capacity, result, row['wall_seconds'], policy)
    delay_parents = [c['parent'] for c in cases if c['parent']['kind']=='delayed_control'
                     and c['parent']['regime']=='short']
    records = [r for p in delay_parents for r in masses(p, cfg)]
    assert records == json.loads((out/'mass-certificates.json').read_text())
    check_mass(records)
    parent, transcript = cases[0]['parent'], cases[0]['transcript']
    result = json.loads((out/(key(parent, 4)+'.json')).read_text())
    rejected = corruptions(parent, transcript, result, records)
    assert rejected == json.loads((out/'receipt.json').read_text())['corruptions']
    hashes = json.loads((out/'sha256.json').read_text())
    for path, digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    return dict(audit='PASS', cases=len(rows), inherited_hashes=len(inherited), hashes=len(hashes))


def main():
    assert platform.node() == 'charon' and (HERE/'plan-frozen.md').exists()
    out = Path(sys.argv[1]).resolve(); out.mkdir(exist_ok=False); start = time.perf_counter()
    cfg = json.loads((HERE/'config.json').read_text()); cases, inherited = load_inputs(); rows = []
    dump(out/'machine.json', dict(host=platform.node(), platform=platform.platform(), python=sys.version,
                                 affinity=sorted(os.sched_getaffinity(0))))
    try:
        for case in cases:
            parent, transcript = case['parent'], case['transcript']
            for policy in cfg['policies']:
                for capacity in cfg['capacities']:
                    before = time.perf_counter()
                    oracle = Oracle(parent)
                    result = replay(oracle, [w for w, _ in transcript], capacity,
                                    (len(parent['outputs'])-1).bit_length(), policy)
                    elapsed = time.perf_counter()-before
                    dump(out/(key(parent, capacity, policy)+'.json'), result)
                    check_result(parent, transcript, capacity, result, policy)
                    rows.append(summary(parent, transcript, capacity, result, elapsed, policy))
            print(json.dumps(dict(id=parent['id'], full_steps=rows[-1]['full_replay_steps'],
                                  snapshot_steps=rows[-1]['transitions'])), flush=True)
        records = [r for c in cases if c['parent']['kind']=='delayed_control'
                   and c['parent']['regime']=='short' for r in masses(c['parent'], cfg)]
        check_mass(records); dump(out/'mass-certificates.json', records); dump(out/'summary.json', rows)
        parent, transcript = cases[0]['parent'], cases[0]['transcript']
        result = json.loads((out/(key(parent, 4)+'.json')).read_text())
        rejected = corruptions(parent, transcript, result, records)
        size = sum(p.stat().st_size for p in out.iterdir() if p.is_file()); assert size <= cfg['output_cap']
        receipt = dict(qualification='PASS', replays=len(rows), inherited_cases=len(cases),
                       inherited_hashes=len(inherited), replayed_answers=sum(r['outputs'] for r in rows),
                       mass_rows=len(records), missed_mass_checks=sum(len(r['missed']) for r in records),
                       corruptions=rejected, total_seconds=time.perf_counter()-start,
                       qualifier_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       policy_source_bytes=(HERE/'replay.py').stat().st_size, output_bytes=size)
        dump(out/'receipt.json', receipt)
        dump(out/'sha256.json', {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                for directory in (HERE, out) for p in directory.iterdir() if p.is_file()})
        print(json.dumps(receipt), flush=True)
    except BaseException as error:
        dump(out/'failure.json', dict(error=repr(error))); raise


if __name__ == '__main__':
    main()
