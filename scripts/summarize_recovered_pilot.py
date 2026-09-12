"""Export consistent per-seed, per-arm endpoints without selecting the best reader."""
import argparse,json,pathlib,statistics,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from scc.provenance import atomic_json,file_digest


def endpoint(measurement,intact=None):
    branch=measurement['branches'][0]
    assert branch['reader']['sign']==1 and branch['reader']['digit_sources']==list(range(10))
    values=list(branch['layouts'].values())
    result={'reader':'identity','temperature':branch['temperature'],
        'benign_min_strict':min(s['useful'] for v in values for k,s in v['strict'].items() if not k.endswith('/unauthorized')),
        'benign_min_four_digits':min(s['useful'] for v in values for k,s in v['four_digits'].items() if not k.endswith('/unauthorized')),
        'unauthorized_min_strict':min(s['useful'] for v in values for k,s in v['strict'].items() if k.endswith('/unauthorized')),
        'unauthorized_min_four_digits':min(s['useful'] for v in values for k,s in v['four_digits'].items() if k.endswith('/unauthorized')),
        'target_min_strict':min(v['target_strict']['lookup/unauthorized']['useful'] for v in values),
        'target_min_four_digits':min(v['target_four_digits']['lookup/unauthorized']['useful'] for v in values),
        'other_refusal_min':min(v['other_unauthorized_refusal'] for v in values),
        'intact_qualification':all(v['qualification']['passed'] for v in values),
        'text':values[0]['calibrated_text']}
    if intact:
        before=intact['branches'][0]['layouts']['original']['calibrated_text']
        result['min_text_context_gain_retention']=min((v['unigram_nll']-v['nll'])/(before[k]['unigram_nll']-before[k]['nll']) for k,v in result['text'].items())
    return result

def pilot(root):
    aggregate=json.loads((root/'result.json').read_text());arms={}
    for arm in ('rule_only','early','late','seam_late'):
        trained=json.loads((root/arm/'result.json').read_text());logs=[json.loads(l) for l in (root/arm/'steps.jsonl').read_text().splitlines()]
        metas=[l for l in logs if l['meta'] is not None];challenge=root/('challenges-'+arm)
        intact=json.loads((challenge/'intact/result.json').read_text())
        record={'qualified':trained['qualified_both_layouts'],'training_seconds':trained['elapsed_seconds'],'ordinary_steps':trained['completed_steps'],
            'meta_episodes':len(metas),'endpoints':{}}
        if metas:
            ratios=[.1*l['meta']['meta_gradient_norm']/l['meta']['ordinary_gradient_norm'] for l in metas]
            record['signal']={'nonzero_meta_episodes':sum(l['meta']['meta_gradient_norm']>0 for l in metas),
                'median_weighted_meta_to_ordinary_gradient_ratio':statistics.median(ratios),
                'first_meta_step':metas[0]['step'],'last_meta_step':metas[-1]['step'],
                'first_loss':metas[0]['meta_loss'],'last_loss':metas[-1]['meta_loss'],
                'first_meta_details':metas[0]['meta'],'last_meta_details':metas[-1]['meta']}
        for p in sorted(challenge.glob('*/result.json')):
            m=json.loads(p.read_text())
            if 'branches' in m:record['endpoints'][p.parent.name]=endpoint(m,intact)
        arms[arm]=record
    assert len({aggregate['arms'][a]['ordinary_chain'] for a in arms})==1
    assert aggregate['arms']['early']['meta_chain']==aggregate['arms']['late']['meta_chain']
    return {'configuration':json.loads((root/'configuration.json').read_text()),'arms':arms,'ordinary_streams_matched':True,'early_late_episode_streams_matched':True}

def probe(root):
    result={}
    for p in sorted(root.glob('*/result.json')):
        parent=json.loads(p.read_text());intact=parent['measurements']['intact']
        result[p.parent.name]={'parent_sha256':parent['parent_sha256'],'endpoints':{k:endpoint(v,intact) for k,v in parent['measurements'].items()}}
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilots',nargs='*',type=pathlib.Path,default=[]);p.add_argument('--probes',nargs='*',type=pathlib.Path,default=[])
    p.add_argument('--heldouts',nargs='*',type=pathlib.Path,default=[])
    p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    heldout={}
    for root in a.heldouts:
        for path in sorted(root.glob('*-intact/result.json')):
            label=path.parent.name.removesuffix('-intact');before=json.loads(path.read_text())
            after=json.loads((root/(label+'-repaired')/'result.json').read_text())
            heldout[label]={'intact':endpoint(before,before),'repaired':endpoint(after,before),
                'intact_parent_sha256':before['parent_sha256'],'repaired_parent_sha256':after['parent_sha256']}
    report={'pilots':{str(r):pilot(r) for r in a.pilots},'probes':{str(r):probe(r) for r in a.probes},'heldout':heldout,'script_sha256':file_digest(__file__),
            'interpretation':'Identity reader, one jointly scored endpoint; minima over declared domains and layouts, not independent samples or inferred complete cognition.'}
    atomic_json(a.output,report);print(json.dumps({'pilots':len(report['pilots']),'probes':len(report['probes'])}))
if __name__=='__main__':main()
