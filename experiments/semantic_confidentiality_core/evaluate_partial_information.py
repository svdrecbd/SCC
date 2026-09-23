"""Compare support and uncertainty for identical hypotheses with reduced evidence."""
from pathlib import Path
import json
import socket
import sys
import time
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2ForSequenceClassification
from evaluate_core import parameter_digest, reject_network, scores_for
from evaluate_public_recovery import public_scores


def main(directory,acquisition):
    started=time.perf_counter()
    socket.create_connection=reject_network;socket.socket.connect=reject_network
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    config=json.loads((directory/'config.json').read_text())
    descriptions=json.loads((directory/'development_descriptions.json').read_text())
    prior_cases=json.loads((directory/'prior_selection.json').read_text())
    prior_predictions=[json.loads(line) for line in (directory/'prior_cases.jsonl').read_text().splitlines()]
    global_readers=json.loads((directory/'prior_summary.json').read_text())['public_readers']
    local_readers=json.loads((directory/'public_summary.json').read_text())['public_readers']
    cases=[]
    for group,description in enumerate(descriptions):
        for index in (1,2):
            cases.append({'id':description['id'],'group_index':group,'hypothesis_index':index,'premise':description['atoms']['p'],
                          'hypothesis':description['hypotheses'][index]['text'],'label':'neutral','supported':False})
    assert len(cases)==32
    (directory/'selection.json').write_text(json.dumps(cases,indent=2)+'\n')
    model_directory=acquisition/'model'
    tokenizer=AutoTokenizer.from_pretrained(model_directory,local_files_only=True,trust_remote_code=False,use_fast=True)
    model=DebertaV2ForSequenceClassification(DebertaV2Config.from_pretrained(model_directory,local_files_only=True))
    state=load_file(model_directory/'model.safetensors',device='cpu')
    assert all(torch.isfinite(value).all() for value in state.values())
    embeddings=model.deberta.embeddings
    assert torch.equal(embeddings.position_ids,state['deberta.embeddings.position_ids'])
    embeddings.register_buffer('position_ids',embeddings.position_ids.clone(),persistent=True)
    model.load_state_dict(state,strict=True);del state
    model.eval();digest=parameter_digest(model)
    assert digest==config['expected_parameter_digest']
    records=[]
    for offset in range(0,len(cases),config['batch_size']):
        batch=cases[offset:offset+config['batch_size']]
        encoded=tokenizer([row['premise'] for row in batch],[row['hypothesis'] for row in batch],padding=True,truncation=False,return_tensors='pt')
        assert encoded['input_ids'].shape[1]<=config['maximum_input_tokens']
        with torch.inference_mode():logits=model(**encoded).logits
        assert torch.isfinite(logits).all()
        for case,values in zip(batch,logits.tolist()):
            predicted=max(range(3),key=lambda index:values[index])
            record=case|{'logits':values,'predicted_class':predicted,'neural_prediction':predicted==1,'condition':'partial'}
            records.append(record)
            with (directory/'partial_cases.jsonl').open('a') as output:output.write(json.dumps(record)+'\n')
    assert parameter_digest(model)==digest
    partial=list(records)
    for case,prediction in zip(prior_cases,prior_predictions):
        assert (case['id'],case['counterfactual'],case['hypothesis_index'])==(prediction['id'],prediction['counterfactual'],prediction['hypothesis_index'])
        if not case['contrast'] or case['label']!='entailment':continue
        records.append({'id':case['id'],'group_index':case['group_index'],'hypothesis_index':case['hypothesis_index'],
                        'premise':case['premise'],'hypothesis':case['hypothesis'],'supported':True,'condition':'supported',
                        'neural_prediction':prediction['entailment_prediction']})
    assert len(records)==64
    for description in descriptions:
        for index in (1,2):
            pair=[row for row in records if row['id']==description['id'] and row['hypothesis_index']==index]
            assert len(pair)==2 and pair[0]['hypothesis']==pair[1]['hypothesis'] and {row['supported'] for row in pair}=={True,False}
    for row in records:
        global_scores=scores_for(row['premise'],row['hypothesis']);local_scores=public_scores(row['premise'],row['hypothesis'])
        row['public_predictions']={}
        for name,reader in global_readers.items():
            reversed_score=name.endswith('_reversed');base=name.removesuffix('_reversed')
            row['public_predictions']['global_'+name]=((-1 if reversed_score else 1)*global_scores[base])>reader['threshold']
        for name,reader in local_readers.items():
            row['public_predictions']['local_'+name]=local_scores[name]>reader['threshold']
            row['public_predictions']['local_'+name+'_direct']=local_scores[name]>0
    evaluation=[row for row in records if row['group_index']>=8]
    model_vector=np.array([np.mean([row['neural_prediction']==row['supported'] for row in evaluation if row['group_index']==group]) for group in range(8,16)])
    public_measures={};public_vectors=[]
    for name in records[0]['public_predictions']:
        values=np.array([np.mean([row['public_predictions'][name]==row['supported'] for row in evaluation if row['group_index']==group]) for group in range(8,16)])
        public_vectors.append(values);public_measures[name]=float(values.mean())
    matrix=np.stack(public_vectors,axis=1)
    generator=np.random.default_rng(config['bootstrap_seed']);indices=generator.integers(0,8,(config['bootstrap_replicates'],8))
    differences=model_vector[indices].mean(axis=1)-matrix[indices].mean(axis=1).max(axis=1)
    summary={'status':'complete','new_case_count':32,'paired_case_count':64,'parameter_digest':digest,
             'partial_neutral_accuracy':np.mean([row['predicted_class']==2 for row in partial]).item(),
             'partial_support_rejection':np.mean([not row['neural_prediction'] for row in partial]).item(),
             'all_paired_support_accuracy':np.mean([row['neural_prediction']==row['supported'] for row in records]).item(),
             'held_out_paired_support_accuracy':model_vector.mean().item(),
             'held_out_public_accuracies':public_measures,'best_held_out_public_accuracy':float(matrix.mean(axis=0).max()),
             'selection_aware_advantage_interval':np.quantile(differences,[.025,.975]).tolist(),
             'hypothesis_only_accuracy':.5,'neural_training':False,'training_admitted':False,'wall_seconds':time.perf_counter()-started}
    (directory/'paired_cases.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
