"""Evaluate a shared semantic consequence core and confidentiality recovery."""
from pathlib import Path
import hashlib
import itertools
import json
import re
import socket
import sys
import time
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2ForSequenceClassification


def reject_network(*arguments,**keywords):
    raise RuntimeError('Network access is disabled during inference')


def expression_value(expression, world):
    if isinstance(expression,str):return world[expression]
    if expression[0]=='not':return not expression_value(expression[1],world)
    if expression[0]=='and':return all(expression_value(part,world) for part in expression[1:])
    raise ValueError('Unsupported Boolean operator')


def parameter_digest(model):
    result=hashlib.sha256()
    for name,value in model.state_dict().items():
        result.update(name.encode());result.update(value.detach().contiguous().numpy().tobytes())
    return result.hexdigest()


def words(text):
    return set(re.findall(r'[a-z0-9]+',text.lower()))


def has_negation(text):
    return bool(words(text)&{'not','no','nobody','never','none','neither','without','unopened','unripe','empty','refused','lacked'})


def scores_for(premise,hypothesis):
    left=words(premise);right=words(hypothesis)
    shared=len(left&right)/max(1,len(right))
    agreement=int(has_negation(premise)==has_negation(hypothesis))
    return {'constant':0,'exact_inclusion':int(hypothesis.lower().rstrip('.') in premise.lower()),
            'lexical_containment':int(right<=left),'word_overlap':shared,
            'hypothesis_length':len(hypothesis),'hypothesis_negation':int(has_negation(hypothesis)),
            'negation_agreement':agreement,'negation_qualified_overlap':agreement*shared}


def candidate_thresholds(values):
    ordered=sorted(set(values))
    return [ordered[0]-1,0.,ordered[-1]+1]+[(left+right)/2 for left,right in zip(ordered,ordered[1:])]


def main(directory, acquisition):
    started=time.perf_counter()
    socket.create_connection=reject_network;socket.socket.connect=reject_network
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    configuration=json.loads((directory/'config.json').read_text())
    descriptions=json.loads((directory/'development_descriptions.json').read_text())
    # Two independent finite calculations: logical inclusion and private-bit executions.
    structural_checks=0
    for premise_mask in range(1,256):
        for hypothesis_mask in range(256):
            entails=(premise_mask&hypothesis_mask)==premise_mask
            confidential=True
            for world in range(8):
                if not (premise_mask>>world)&1:continue
                hypothesis_true=(hypothesis_mask>>world)&1
                output_zero=0 if hypothesis_true else 0
                output_one=0 if hypothesis_true else 1
                confidential=confidential and output_zero==output_one
            assert entails==confidential
            structural_checks+=1
    cases=[]
    worlds=[dict(zip(('p','q','r'),assignment)) for assignment in itertools.product([False,True],repeat=3)]
    for group_index,description in enumerate(descriptions):
        for counterfactual in (False,True):
            premise_expression=['and','p','q' if counterfactual else ['not','q']]
            compatible=[world for world in worlds if expression_value(premise_expression,world)]
            assert len(compatible)==2
            for hypothesis_index,hypothesis in enumerate(description['hypotheses']):
                values=[expression_value(hypothesis['expression'],world) for world in compatible]
                label='entailment' if all(values) else 'contradiction' if not any(values) else 'neutral'
                private_runs=[(0,0 if value else 1) for value in values]
                confidential=all(left==right for left,right in private_runs)
                assert confidential==(label=='entailment')
                if not counterfactual:assert label==hypothesis['label']
                cases.append({'id':description['id'],'group_index':group_index,'counterfactual':counterfactual,
                              'hypothesis_index':hypothesis_index,'contrast':hypothesis_index in (1,2),
                              'premise':description['counterfactual_premise'] if counterfactual else description['premise'],
                              'hypothesis':hypothesis['text'],'label':label,'confidential':confidential,
                              'compatible_worlds':compatible,'private_runs':private_runs})
    assert len(cases)==configuration['case_count']
    (directory/'selection.json').write_text(json.dumps(cases,indent=2)+'\n')
    model_directory=acquisition/'model'
    tokenizer=AutoTokenizer.from_pretrained(model_directory,local_files_only=True,trust_remote_code=False,use_fast=True)
    model_config=DebertaV2Config.from_pretrained(model_directory,local_files_only=True)
    assert [model_config.id2label[index] for index in range(3)]==configuration['label_order']
    state=load_file(model_directory/'model.safetensors',device='cpu')
    assert all(torch.isfinite(value).all() for value in state.values())
    model=DebertaV2ForSequenceClassification(model_config)
    embeddings=model.deberta.embeddings
    position_key='deberta.embeddings.position_ids'
    assert torch.equal(embeddings.position_ids,state[position_key])
    embeddings.register_buffer('position_ids',embeddings.position_ids.clone(),persistent=True)
    model.load_state_dict(state,strict=True)
    del state
    model.eval()
    digest=parameter_digest(model)
    preparation_seconds=time.perf_counter()-started
    results=[]
    for offset in range(0,len(cases),configuration['batch_size']):
        batch=cases[offset:offset+configuration['batch_size']]
        encoded=tokenizer([row['premise'] for row in batch],[row['hypothesis'] for row in batch],padding=True,truncation=False,return_tensors='pt')
        assert encoded['input_ids'].shape[1]<=configuration['maximum_input_tokens']
        inference_started=time.perf_counter()
        with torch.inference_mode():logits=model(**encoded).logits
        assert torch.isfinite(logits).all()
        elapsed=time.perf_counter()-inference_started
        for case,values in zip(batch,logits.tolist()):
            predicted=max(range(3),key=lambda index:values[index])
            entailed=predicted==1
            record={key:case[key] for key in ['id','group_index','counterfactual','hypothesis_index','contrast','label','confidential']}
            record.update({'logits':values,'predicted_label':configuration['label_order'][predicted],
                           'entailment_prediction':entailed,'recovered_confidentiality_prediction':entailed,
                           'public_scores':scores_for(case['premise'],case['hypothesis']),'model_seconds':elapsed/len(batch)})
            results.append(record)
            with (directory/'cases.jsonl').open('a') as output:output.write(json.dumps(record)+'\n')
    assert parameter_digest(model)==digest
    calibration=[row for row in results if row['group_index']<8 and row['contrast']]
    evaluation=[row for row in results if row['group_index']>=8 and row['contrast']]
    public_readers={}
    vectors={}
    for name in calibration[0]['public_scores']:
        for orientation in (1,-1):
            identity=name if orientation==1 else name+'_reversed'
            thresholds=candidate_thresholds([orientation*row['public_scores'][name] for row in calibration])
            def correct(row,threshold):
                return int((orientation*row['public_scores'][name]>threshold)==row['confidential'])
            threshold=max(thresholds,key=lambda value:(sum(correct(row,value) for row in calibration),-abs(value),-value))
            values=np.array([np.mean([correct(row,threshold) for row in evaluation if row['group_index']==group]) for group in range(8,16)])
            vectors[identity]=values
            public_readers[identity]={'threshold':threshold,'calibration_accuracy':np.mean([correct(row,threshold) for row in calibration]).item(),'evaluation_accuracy':values.mean().item()}
    neural=np.array([np.mean([row['entailment_prediction']==row['confidential'] for row in evaluation if row['group_index']==group]) for group in range(8,16)])
    public=np.stack(list(vectors.values()),axis=1)
    generator=np.random.default_rng(configuration['bootstrap_seed'])
    indices=generator.integers(0,8,(configuration['bootstrap_replicates'],8))
    differences=neural[indices].mean(axis=1)-public[indices].mean(axis=1).max(axis=1)
    interval=np.quantile(differences,[.025,.975]).tolist()
    def accuracy(selected):return float(np.mean([row['entailment_prediction']==row['confidential'] for row in selected]))
    summary={'status':'complete','case_count':len(results),'structural_identity_checks':structural_checks,
             'parameter_count':sum(parameter.numel() for parameter in model.parameters()),'parameter_digest':digest,
             'overall_entailment_accuracy':accuracy(results),'overall_three_class_accuracy':float(np.mean([row['predicted_label']==row['label'] for row in results])),
             'contrast_accuracy':accuracy([row for row in results if row['contrast']]),
             'calibration_contrast_accuracy':accuracy(calibration),'evaluation_contrast_accuracy':accuracy(evaluation),
             'reference_representation_accuracy':1.,'hypothesis_only_contrast_accuracy':.5,
             'public_readers':public_readers,'best_public_evaluation_accuracy':float(public.mean(axis=0).max()),
             'selection_aware_advantage_interval':interval,
             'computational_development_screen_passed':bool(neural.mean()>=configuration['minimum_accuracy'] and neural.mean()-public.mean(axis=0).max()>=configuration['minimum_advantage'] and interval[0]>0),
             'preparation_seconds':preparation_seconds,'model_seconds':sum(row['model_seconds'] for row in results),
             'wall_seconds':time.perf_counter()-started,'neural_training':False,'training_admitted':False}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
