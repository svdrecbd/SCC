"""Exact-rational and numerical substrate checks, with no SCC success claim."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scc.persistent_matrix import MatrixConfig, LiveMatrix, matrix_step, exact_column_copy
from scc.provenance import atomic_json,snapshot_sources,source_manifest,file_digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    source=snapshot_sources(a.output/'source');shutil.copyfile(__file__,a.output/'runner.py')
    torch.set_num_threads(2);torch.manual_seed(8809)
    simplex=[tuple(Fraction(x,4) for x in (i,j,4-i-j)) for i in range(5) for j in range(5-i)]
    determinants=[]
    for key in simplex:
        for query in simplex:
            for beta in (Fraction(1,4),Fraction(1,2),Fraction(3,4)):
                determinant=1+beta*sum(k*(q-k) for k,q in zip(key,query))
                assert determinant>=1-beta>0
                determinants.append(determinant)
    rational={'cases':len(determinants),'minimum_determinant':str(min(determinants)),
              'scope':'Exact finite rational check of a general analytically derived bound; not an exhaustive proof over real inputs.'}
    rows=[]
    for width in (4,8,32):
        config=MatrixConfig(width,3)
        for dtype in (torch.float32,torch.float64):
            initial=torch.randn(config.rows,width,dtype=dtype)
            # Deliberately constructed erasing program, not a learned defender:
            # novel hot inputs choose their own column and copy column zero.
            initial[3:3+width]=20*torch.eye(width,dtype=dtype)
            initial[3+width:3+2*width]=-20
            initial[3+width]=20
            initial[-1]=20
            initial[:,0]=0
            live=LiveMatrix(initial,config);trajectory=[]
            for target in range(1,width):
                logits=torch.zeros(width,dtype=dtype);logits[target]=40
                before=live.weights.clone();_,control=live.tick(logits)
                assert int(control['keys'][0])==target and int(control['queries'][0])==0 and bool(control['enabled'][0])
                assert torch.equal(live.weights[:,target],torch.zeros(config.rows,dtype=dtype))
                assert all(any(torch.equal(col,old) for old in before.T) for col in live.weights.T)
                trajectory.append({'step':live.steps,'distinct_columns':torch.unique(live.weights.T,dim=0).shape[0]})
            assert torch.count_nonzero(live.weights)==0
            zero_snapshot=live.snapshot()
            outputs=[]
            for _ in range(128):
                y,_=live.tick(torch.randn(width,dtype=dtype)*20)
                assert torch.count_nonzero(y)==0 and torch.count_nonzero(live.weights)==0
                outputs.append(y.tolist())
            restored=LiveMatrix.from_snapshot(zero_snapshot)
            assert torch.count_nonzero(restored.tick(torch.randn(width,dtype=dtype))[0])==0
            soft=LiveMatrix(live.weights,MatrixConfig(width,3,'soft'))
            assert torch.count_nonzero(soft.tick(torch.randn(width,dtype=dtype))[0])==0
            # External knowledge injection is deliberately outside ordinary ticks.
            restored.external_write(initial)
            assert torch.equal(restored.weights,initial) and torch.count_nonzero(restored.weights)>0
            rows.append({'width':width,'dtype':str(dtype),'erasing_program':'constructed, not trained',
                         'ticks_to_zero':width-1,'trajectory':trajectory,'fresh_input_checks':128,
                         'snapshot_of_current_state_does_not_recover':True,'soft_rule_substitution_does_not_recover_zero':True,
                         'clean_external_write_restores_state':True})
    soft_rows=[]
    for width in (4,8,32):
        config=MatrixConfig(width,3,'soft');state=torch.randn(1,config.rows,width,dtype=torch.float64)
        for step in range(64):
            inputs=torch.randn(1,width,dtype=torch.float64)
            _,changed,control=matrix_step(state,inputs,config)
            key=control['key_probabilities'][0];query=control['query_probabilities'][0];beta=control['beta'][0]
            transform=torch.eye(width,dtype=state.dtype)+beta*torch.outer(query-key,key)
            determinant=1+beta*key.dot(query-key)
            inverse_reconstruction=torch.linalg.solve(transform.T,changed[0].T).T
            error=float((inverse_reconstruction-state[0]).abs().max())
            assert error<1e-12 and determinant>=1-beta-1e-15
            soft_rows.append({'width':width,'step':step+1,'determinant':float(determinant),'reconstruction_error':error})
            state=changed
    assert source_manifest()==source
    result={'status':'complete','rational_determinant_checks':rational,'hard_copy_cases':rows,'smooth_checks':soft_rows,
            'source_sha256':hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),
            'runner_sha256':file_digest(a.output/'runner.py'),'qualified_learned_models':0,'positive_scc_result':False,
            'protection_trigger_tested':False,'scope':'Engineering/mathematical calibration. Smooth affine-span invariant assumes exact arithmetic and sigmoid beta strictly below one; known-control inverse is not a global inverse of the nonlinear state map. Hard-copy support monotonicity applies to ordinary ticks, not arbitrary external parameter writes.'}
    atomic_json(a.output/'result.json',result)
    print(json.dumps({'status':'complete','rational_checks':len(determinants),'constructed_erasure_cases':len(rows),
                      'fresh_input_checks':sum(x['fresh_input_checks'] for x in rows),'smooth_local_inverse_checks':len(soft_rows),
                      'positive_scc_result':False}))


if __name__=='__main__':main()
