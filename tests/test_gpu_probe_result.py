import json
from scripts.scc_projected_gpu_probe import inline_summary


def test_provider_inline_result_excludes_large_traces_but_keeps_gate_evidence():
    full = {'ok':True,'source_files_verified':64,'data_files_verified':7,'elapsed_seconds':13.,
            'runtime_downloads':False,'scope':'fixture',
            'checks':[{'variant':v,'scope':s,'gradient_norm':1.,'peak_cuda_memory_bytes':1024,
                       'record':{'large_trace':'x'*20000}}
                      for v in ('standard','narrow32','tied') for s in ('core','all')]}
    assert len(json.dumps(full).encode()) > 32*1024
    compact = inline_summary(full)
    assert len(json.dumps(compact).encode()) < 2048
    assert compact['ok'] and len(compact['checks'])==6
    assert 'record' not in compact['checks'][0]
    assert full['checks'][0]['record']['large_trace']=='x'*20000


def test_discrete_inline_result_keeps_all_eight_gate_checks():
    from scripts.scc_discrete_gpu_probe import inline_summary as compact_discrete
    full = {'ok':True,'source_files_verified':68,'data_files_verified':7,'elapsed_seconds':30.,
            'runtime_downloads':False,'scope':'fixture',
            'checks':[{'bits':bits,'hard':hard,'scope':scope,'accepted':True,
                       'gradient_norm':1.,'peak_cuda_memory_bytes':1024,'finite_edit_checked':True,
                       'record':{'large_trace':'x'*20000}}
                      for bits in (32,128) for hard in (False,True) for scope in ('core','all')]}
    assert len(json.dumps(full).encode()) > 32*1024
    compact = compact_discrete(full)
    assert len(json.dumps(compact).encode()) < 4096
    assert compact['ok'] and len(compact['checks'])==8
    assert all(c['finite_edit_checked'] and 'record' not in c for c in compact['checks'])
