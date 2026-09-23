"""Reconstruct entailment, contradiction, and uncertainty from paired support queries."""
from pathlib import Path
import json
import sys
import time
import numpy as np
from evaluate_core import scores_for
from evaluate_public_recovery import public_scores


def decode(positive,negative):
    return 'inconsistent' if positive and negative else 'entailment' if positive else 'contradiction' if negative else 'neutral'


def public_predictions(premise,hypothesis,global_readers,local_readers):
    global_scores=scores_for(premise,hypothesis);local_scores=public_scores(premise,hypothesis)
    result={}
    for name,reader in global_readers.items():
        reverse=name.endswith('_reversed');base=name.removesuffix('_reversed')
        result['global_'+name]=(-1 if reverse else 1)*global_scores[base]>reader['threshold']
    for name,reader in local_readers.items():
        result['local_'+name]=local_scores[name]>reader['threshold']
        result['local_'+name+'_direct']=local_scores[name]>0.
    return result


def main(directory):
    started=time.perf_counter();configuration=json.loads((directory/'config.json').read_text())
    original=json.loads((directory/'prior_selection.json').read_text())
    original_predictions=[json.loads(line) for line in (directory/'prior_cases.jsonl').read_text().splitlines()]
    partial=[json.loads(line) for line in (directory/'partial_cases.jsonl').read_text().splitlines()]
    global_readers=json.loads((directory/'prior_summary.json').read_text())['public_readers']
    local_readers=json.loads((directory/'public_summary.json').read_text())['public_readers']
    records=[]
    for case,prediction in zip(original,original_predictions):
        assert (case['id'],case['hypothesis_index'],case['counterfactual'])==(prediction['id'],prediction['hypothesis_index'],prediction['counterfactual'])
        if not case['contrast']:continue
        records.append(case|{'condition':'counterfactual' if case['counterfactual'] else 'original',
                            'native_label':prediction['predicted_label'],'support':prediction['entailment_prediction']})
    for case in partial:
        records.append(case|{'condition':'partial','native_label':['contradiction','entailment','neutral'][case['predicted_class']],
                             'support':case['neural_prediction']})
    assert len(records)==96
    for row in records:
        row['public']=public_predictions(row['premise'],row['hypothesis'],global_readers,local_readers)
    output=[];inconsistent=[]
    for group in range(16):
        for condition in ('original','counterfactual','partial'):
            pair=[row for row in records if row['group_index']==group and row['condition']==condition]
            assert len(pair)==2 and {row['hypothesis_index'] for row in pair}=={1,2}
            positive=next(row for row in pair if row['hypothesis_index']==2)
            negative=next(row for row in pair if row['hypothesis_index']==1)
            if positive['support'] and negative['support']:inconsistent.append({'id':positive['id'],'condition':condition})
            for row,other in [(positive,negative),(negative,positive)]:
                reconstructed=decode(row['support'],other['support'])
                public={name:decode(row['public'][name],other['public'][name]) for name in row['public']}
                public['always_unresolved']='neutral'
                output.append({'id':row['id'],'group_index':group,'condition':condition,'hypothesis_index':row['hypothesis_index'],
                               'label':row['label'],'native_label':row['native_label'],'reconstructed_label':reconstructed,'public_labels':public})
    evaluation=[row for row in output if row['group_index']>=8]
    def group_vector(field,reader=None):
        return np.array([np.mean([(row[field][reader] if reader else row[field])==row['label'] for row in evaluation if row['group_index']==group]) for group in range(8,16)])
    native=group_vector('native_label');reconstructed=group_vector('reconstructed_label')
    public_matrix=np.stack([group_vector('public_labels',name) for name in output[0]['public_labels']],axis=1)
    generator=np.random.default_rng(configuration['bootstrap_seed']);indices=generator.integers(0,8,(configuration['bootstrap_replicates'],8))
    public_difference=reconstructed[indices].mean(axis=1)-public_matrix[indices].mean(axis=1).max(axis=1)
    native_difference=reconstructed[indices].mean(axis=1)-native[indices].mean(axis=1)
    summaries={}
    for field in ('native_label','reconstructed_label'):
        summaries[field]={'all_accuracy':float(np.mean([row[field]==row['label'] for row in output])),
                          'held_out_accuracy':float(np.mean([row[field]==row['label'] for row in evaluation])),
                          'all_uncertainty_recall':float(np.mean([row[field]=='neutral' for row in output if row['label']=='neutral'])),
                          'held_out_uncertainty_recall':float(np.mean([row[field]=='neutral' for row in evaluation if row['label']=='neutral']))}
    summary={'status':'complete','oriented_judgments':len(output),'distinct_contexts':48,'group_count':16,'methods':summaries,
             'inconsistent_contexts':inconsistent,
             'public_held_out_accuracy':{name:float(group_vector('public_labels',name).mean()) for name in output[0]['public_labels']},
             'best_public_held_out_accuracy':float(public_matrix.mean(axis=0).max()),
             'reconstruction_minus_native_interval':np.quantile(native_difference,[.025,.975]).tolist(),
             'selection_aware_public_advantage_interval':np.quantile(public_difference,[.025,.975]).tolist(),
             'model_calls_per_reconstructed_judgment':2,'new_model_calls':0,'neural_training':False,'training_admitted':False,
             'wall_seconds':time.perf_counter()-started}
    (directory/'cases.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in output))
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
