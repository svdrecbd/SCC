"""Independently verify matched streams and rescore the full-gradient pilot."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from audit_developmental_predictions import audit,retained_digits
from summarize_developmental_comparison import profile,read


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True)
    parser.add_argument('--output',required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--parent')
    group.add_argument('--parent-directory')
    parser.add_argument('--arms',nargs='+',default=['rule_only','full_01','full_10'])
    parser.add_argument('--expected-meta-episodes',type=int,default=100)
    args=parser.parse_args();root,output=Path(args.directory),Path(args.output)
    if output.exists():raise FileExistsError(output)
    import sys
    sys.path.insert(0,str((root/'source').resolve()))
    from scc.developmental_tasks import evaluation_rows
    from scc.provenance import source_manifest
    source=source_manifest()
    parent_hashes={arm:hashlib.sha256((Path(args.parent) if args.parent else Path(args.parent_directory)/arm/'step-00001000.pt').read_bytes()).hexdigest() for arm in args.arms}
    result={'status':read(root/'result.json')['status'],'causal_mechanism_established':False,
            'derivative_validation':read(root/'derivative-validation.json') if (root/'derivative-validation.json').exists() else None,
            'resume_validation':read(root/'resume-validation.json') if (root/'resume-validation.json').exists() else None,'arms':{}}
    if result['status'] not in ('full_gradient_pilot_complete','full_gradient_continuation_complete'):raise ValueError('Run did not complete')
    control=read(root/'rule_only/result.json')
    chains,episode_hashes=[],{}
    for arm in args.arms:
        p=root/arm;contract=read(p/'contract.json');run=read(p/'result.json')
        if contract['parent_sha256']!=parent_hashes[arm] or contract['source']!=source:raise ValueError('Parent/source mismatch')
        records=[json.loads(line) for line in (p/'steps.jsonl').read_text().splitlines()]
        episodes=[r for r in records if r['meta'] is not None]
        episode_hashes[arm]=[r['meta']['support_query_ids_sha256'] for r in episodes]
        chains.append(run['ordinary_chain'])
        info={'clean':profile(run['validation'],control['validation']),
            'clean_reordered':profile(run['reordered_validation'],control['reordered_validation']),
            'qualification':run['validation']['qualification'],'reordered_qualification':run['reordered_validation']['qualification'],
            'meta_episodes':len(episodes),'positive_meta_episodes':sum(r['meta_loss']>0 for r in episodes),
            'first_twenty_meta_loss_mean':statistics.mean(r['meta_loss'] for r in episodes[:20]) if episodes else None,
            'last_twenty_meta_loss_mean':statistics.mean(r['meta_loss'] for r in episodes[-20:]) if episodes else None,
            'probes':[],'causal':{}}
        rows=read(p/'evaluation_rows.json');reordered=evaluation_rows(128,reordered=True)
        before=audit(rows,run['validation']);before_reordered=audit(reordered,run['reordered_validation'])
        for file in sorted((root/('probes-'+arm)).glob('stage-*.json')):
            record=read(file)
            info['probes'].append({'stage':record['stage'],'normal':profile(record['evaluation'],run['validation']),
                'reordered':profile(record['reordered_evaluation'],run['reordered_validation']),
                'partial_cognition':retained_digits(before,audit(rows,record['evaluation'])),
                'partial_cognition_reordered':retained_digits(before_reordered,audit(reordered,record['reordered_evaluation'])),
                'reported_collapse_in_both_renderings':record['collapse_in_both_renderings']})
        for site in ('head-0-1','mlp-0'):
            item={}
            for label,r,clean in (('original',rows,run['validation']),('reordered',reordered,run['reordered_validation'])):
                q=root/('causal-'+arm)/site
                lesion=read(q/f'lesion-{label}.json');audit(r,lesion)
                swap=read(q/f'opposite_permission-{label}.json');audit(r,swap)
                item[label]={'lesion':profile(lesion,clean),'opposite_permission':{
                    'task_exact':{k:v['exact'] for k,v in swap['tasks'].items()},
                    'useful_unauthorized':{k.split('/')[0]:v['useful_answer_exact'] for k,v in swap['tasks'].items() if k.endswith('/unauthorized')}}}
            info['causal'][site]=item
        result['arms'][arm]=info
        monitors=[]
        import math
        for file in sorted((root/('monitor-'+arm)).glob('step-*.json')):
            monitor=read(file);audit(evaluation_rows(32),monitor['quick_intact'])
            rows=monitor['held_meta_episodes'];ungated=[]
            for row in rows:
                residuals=[max(0,1-v/(contract['text']['floors'][k] if k in contract['text']['floors'] else math.log(10)))**2 for k,v in row['capability_nll'].items()]
                value=sum(residuals)/len(residuals)
                if abs(value*row['gate']-row['value'])>1e-6:raise ValueError('Held-episode objective does not reconstruct')
                ungated.append(value)
            monitors.append({'step':int(file.stem.split('-')[1]),'held_objective_mean':statistics.mean(r['value'] for r in rows),
                'held_gate_mean':statistics.mean(r['gate'] for r in rows),'held_residual_without_gate':statistics.mean(ungated),
                'quick_minimum_task_exact':min(v['exact'] for v in monitor['quick_intact']['tasks'].values())})
        info['monitors']=monitors
    candidates=[k for k in args.arms if k!='rule_only']
    if len(set(chains))!=1 or any(episode_hashes[k]!=episode_hashes[candidates[0]] for k in candidates):
        raise ValueError('Pilot data streams differ')
    if any(len(episode_hashes[k])!=args.expected_meta_episodes for k in candidates) or episode_hashes['rule_only']:
        raise ValueError('Pilot episode count changed')
    result.update(matched_ordinary_chain=chains[0],meta_episodes_verified=args.expected_meta_episodes,
                  meta_streams_paired_between_candidates=len(candidates)>1,parent_sha256_by_arm=parent_hashes)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{'qualification':v['qualification'],'reordered':v['reordered_qualification'],
                        'meta_first':v['first_twenty_meta_loss_mean'],'meta_last':v['last_twenty_meta_loss_mean']}
                       for k,v in result['arms'].items()}))


if __name__=='__main__':main()
