"""Rescore saved compensation outputs and verify provenance without refitting."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scc.provenance import atomic_json, file_digest
from scripts.localize_persistent_learning import independent_check


def audit(folder):
    hashes=json.loads((folder/'output-hashes.json').read_text())
    assert all(file_digest(folder/p)==h for p,h in hashes.items())
    parents=json.loads((folder/'parents.json').read_text())
    assert all(file_digest(p)==h for p,h in parents.items())
    manifest=json.loads((folder/'source-manifest.json').read_text())
    assert all(file_digest(folder/'source'/p)==h for p,h in manifest.items())
    rows=json.loads((folder/'requests.json').read_text())
    for stream in rows:
        for row in stream:
            independent_check(row); assert row['split']=='validation'
    config=json.loads((folder/'configuration.json').read_text())
    result=json.loads((folder/'result.json').read_text())
    assert result['status']=='complete'
    expected=(1 if config['fixture'] else 4)*3*7
    assert result['conditions']==len(result['results'])==expected
    groups={}; count=0; identities=0; state_checks=0
    for record in result['results']:
        mode_folder=folder/record['state']/record['mode']
        detail=json.loads((mode_folder/(record['edit']+'.json')).read_text())
        assert {k:v for k,v in detail.items() if k!='measurements'}==record
        raw=torch.load(mode_folder/(record['edit']+'.pt'), weights_only=True)
        baseline=torch.load(mode_folder/'baseline.pt',weights_only=True)
        logits, original=raw['logits'],baseline['logits']
        assert logits.shape==original.shape==(2,config['requests_per_stream'],4)
        assert torch.isfinite(logits).all() and torch.isfinite(raw['final']).all()
        name=record['edit'].split('-')[0]
        matrix=torch.tensor(config['transforms'][name],dtype=logits.dtype)
        # Independently solve L z = edited y, rather than applying the runner's inverse.
        decoded=torch.linalg.solve(matrix,logits.reshape(-1,4).T).T.reshape_as(logits)
        pred,raw_pred,base_pred=decoded.argmax(-1),logits.argmax(-1),original.argmax(-1)
        assert int((pred!=base_pred).sum())==record['behavior']['inverse_disagreements']
        assert int((raw_pred!=base_pred).sum())==record['behavior']['raw_disagreements']
        for context in ('ungated','authorized','unauthorized'):
            selection=[(i,j,r) for i,stream in enumerate(rows) for j,r in enumerate(stream) if r['context']==context]
            saved=record['behavior']['contexts'][context]
            assert saved['n']==len(selection)
            for label,values in (('baseline',base_pred),('raw',raw_pred),('inverse_readout',pred)):
                assert sum(int(values[i,j])==r['label'] for i,j,r in selection)==saved[label+'_correct']
            assert sum(int(pred[i,j])!=int(base_pred[i,j]) for i,j,_ in selection)==saved['inverse_disagreements']
        assert len(detail['measurements'])==config['requests_per_stream']
        for j,measurement in enumerate(detail['measurements']):
            expected_output=original[:,j] @ matrix.T
            error=float((logits[:,j]-expected_output).abs().max())
            scale=max(1.,float(expected_output.abs().max()))
            assert error==measurement['output']['absolute']
            assert error/scale==measurement['output']['scaled']
        for metric in ('H','state','output'):
            for kind in ('absolute','scaled'):
                assert max(m[metric][kind] for m in detail['measurements'])==record['rollout_max'][metric][kind]
        wiring_path=next(p for p in parents if p.endswith('/fixed-wiring.pt'))
        wiring=torch.load(wiring_path,weights_only=True).to(logits.dtype)
        original_state=baseline['final']
        transformed=original_state.clone()
        transformed[:,:4]=matrix @ original_state[:,:4]
        if record['edit'].endswith('-compensated'):
            transformed[:,4:]=original_state[:,4:]+wiring @ (original_state[:,:4]-transformed[:,:4])
        state_error=float((raw['final']-transformed).abs().max())
        h=raw['final'][:,4:]+wiring @ raw['final'][:,:4]
        h0=original_state[:,4:]+wiring @ original_state[:,:4]
        assert state_error==detail['measurements'][-1]['state']['absolute']
        assert float((h-h0).abs().max())==detail['measurements'][-1]['H']['absolute']
        state_checks+=1
        if name=='identity':
            assert torch.equal(logits,original) and torch.equal(raw['final'],original_state)
            identities+=1
        groups[record['state'],record['mode']]=baseline['logits']
        count+=logits.shape[0]*logits.shape[1]
    for record in result['unedited_numerical_comparisons']:
        base=groups[record['state'],'fp32-b2'].double()
        other=groups[record['state'],record['mode']].double()
        assert int((base.argmax(-1)!=other.argmax(-1)).sum())==record['decision_disagreements']
        assert float((base-other).abs().max())==record['logits']['absolute']
    return {'status':'passed','conditions':expected,'edited_predictions_rescored':count,
            'identity_replays':identities,'final_state_checks':state_checks,
            'parent_files_verified':len(parents),'frozen_source_files_verified':len(manifest),
            'artifact_files_verified':len(hashes),'auditor_sha256':file_digest(__file__),
            'limits':'Raw logits, context scores, output residuals and final states independently rescored; intermediate state-residual traces checked for internal consistency, not replayed.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.folder);atomic_json(args.output,result);print(json.dumps(result))
