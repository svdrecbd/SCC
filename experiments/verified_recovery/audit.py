"""Independent scalar arithmetic and adversarial record validation."""
import copy
import math
from fractions import Fraction


def scalar_product(rows, word):
    bits = [(word >> j) % 2 for j in range(len(rows))]
    return sum((sum(((row >> j) % 2)*bits[j] for j in range(len(rows))) % 2)*2**i
               for i, row in enumerate(rows))


def verify(records, cfg):
    expected = [('exact', n, seed, bits) for n in cfg['exhaustive_sizes']
                for seed in cfg['seeds'] for bits in cfg['support_bits']]
    expected += [('meter', n, seed, None) for n in cfg['meter_sizes'] for seed in cfg['seeds']]
    actual = [(r['kind'], r['n'], r['seed'], r.get('bits')) for r in records[:-1]]
    assert actual == expected
    exhaustive_checks = 0
    summaries = []
    for r in records[:-1]:
        n, rows = r['n'], r['rows']
        assert len(rows) == n and all(0 <= x < 2**n for x in rows)
        if r['kind'] == 'exact':
            inverse = {scalar_product(rows, x): x for x in range(2**n)}
            assert len(inverse) == 2**n
            count = 2**(n-r['bits'])
            assert r['success_by_rhs'] == [count]*2**n
            for b in range(2**n):
                hits = 0
                for mask in range(2**n):
                    shifted = b ^ scalar_product(rows, mask)
                    if shifted % 2**r['bits'] == 0:
                        assert inverse[shifted] ^ mask == inverse[b]
                        hits += 1
                    exhaustive_checks += 1
                assert hits == r['success_by_rhs'][b]
            delta = Fraction(count, 2**n)
            assert set(r['accuracies']) == set(map(str, cfg['retry_counts']))
            for k, value in r['accuracies'].items():
                assert Fraction(value) == 1-(1-delta)**int(k)/2
            total = 2**(2*n)
            assert r['meter'] == dict(oracle_calls=total, candidate_checks=total//2**r['bits'],
                row_parities=n*(2*total+total//2**r['bits']), pivot_tests=0,
                elimination_tests=0, row_xors=0)
        else:
            assert r['plain'] is None and r['repaired'] == r['direct'] == r['target']
            assert scalar_product(rows, r['target']) == r['rhs']
            shifted = r['rhs'] ^ scalar_product(rows, r['mask'])
            error_bit = shifted % n
            assert r['native_exact_accuracy'] == '0'
            assert Fraction(r['native_coordinate_accuracy']) == Fraction(n-1, n)
            for key, checks in [('plain_meter',1), ('repair_meter',error_bit+2)]:
                assert r[key] == dict(oracle_calls=1, candidate_checks=checks,
                    row_parities=n*(1+checks), pivot_tests=0, elimination_tests=0, row_xors=0)
            dm = r['direct_meter']
            assert dm['elimination_tests'] == n*n and 0 <= dm['row_xors'] <= n*(n-1)
            assert n <= dm['pivot_tests'] <= n*(n+1)//2
            assert dm['candidate_checks'] == dm['oracle_calls'] == dm['row_parities'] == 0
            assert r['public_matrix_bits'] == n*n and r['elimination_row_bits'] == n+1
            assert r['oracle_row_updates_per_call'] == r['oracle_fixture_operations'] == 12*n
            summaries.append({key:r[key] for key in ['n','seed','native_coordinate_accuracy',
                'repair_meter','direct_meter','public_matrix_bits','oracle_row_updates_per_call']})
    c = records[-1]
    assert c['kind'] == 'controls' and c['malformed_or_wrong_rejected'] == 6
    ds = list(map(Fraction,c['concentrated_success']))
    assert ds == [1,0] and sum(ds)/2 == Fraction(c['global_success'])
    acc = [1-(1-d)**64/2 for d in ds]
    assert acc == list(map(Fraction,c['concentrated_protected_accuracy']))
    assert sum(acc)/2 == Fraction(c['global_protected_accuracy'])
    return dict(exhaustive_target_mask_checks=exhaustive_checks,
                metered=summaries,
                illustrative_accuracy_delta_001_k64=1-0.99**64/2,
                exact_success_ceiling_eta_005_k64=-math.expm1(math.log(0.9)/64),
                retry_curves={str(d):{str(k):1-(1-d)**k/2 for k in cfg['retry_counts']}
                              for d in [0.01,0.0625,0.25,0.5,1.0]})


def corruptions(records, cfg):
    edits = [
        ('coverage', lambda r:r.pop(0)),
        ('matrix', lambda r:r[0]['rows'].__setitem__(0,0)),
        ('success', lambda r:r[0]['success_by_rhs'].__setitem__(0,0)),
        ('amplification', lambda r:r[0]['accuracies'].__setitem__('64','1/2')),
        ('meter', lambda r:r[0]['meter'].__setitem__('oracle_calls',0)),
        ('restored', lambda r:next(x for x in r if x['kind']=='meter').__setitem__('repaired',-1)),
        ('coordinate', lambda r:next(x for x in r if x['kind']=='meter').__setitem__('native_coordinate_accuracy','0')),
        ('repair_cost', lambda r:next(x for x in r if x['kind']=='meter')['repair_meter'].__setitem__('candidate_checks',0)),
        ('coverage_average', lambda r:r[-1].__setitem__('global_protected_accuracy','1')),
    ]
    passed = []
    for name, edit in edits:
        bad = copy.deepcopy(records)
        edit(bad)
        try:
            verify(bad, cfg)
        except AssertionError:
            passed.append(name)
        else:
            raise AssertionError('unrejected corruption '+name)
    return passed
