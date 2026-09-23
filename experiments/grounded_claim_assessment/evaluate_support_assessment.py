"""Paired assessment of answer support when the valid option may be absent."""
from pathlib import Path
import hashlib
import json
import socket
import sys
import time
import torch
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from evaluate_grounded_claims import parameter_digest, public_scores, reject_network


def ordered_indices(indices, identity, seed, arm):
    return sorted(indices, key=lambda index: hashlib.sha256(
        f'{seed}\0{identity}\0{arm}\0{index}'.encode()).digest())


def main(directory, acquisition):
    started = time.perf_counter()
    socket.create_connection = reject_network
    socket.socket.connect = reject_network
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    configuration = json.loads((directory/'config.json').read_text())
    model_directory = acquisition/'model'
    tokenizer = T5Tokenizer.from_pretrained(model_directory, local_files_only=True, legacy=True)
    records = []
    for filename in ('initial_selection.json', 'replication_selection.json'):
        records.extend(json.loads((directory/filename).read_text())['selected'])
    assert len(records) == 64 and len({row['example_id'] for row in records}) == 64
    offset = configuration['article_offset']
    records = records[offset:offset+configuration['article_count']]
    cases = []
    excluded = []
    for original in records:
        if original['example_id'] in configuration['excluded_article_ids']:
            excluded.append({'id':original['example_id'],'reason':'menu-relative answer option: both A and B'})
            continue
        gold = ord(original['answer'])-ord('A')
        identity = original['example_id']+'\0'+original['question']
        wrong = ordered_indices([index for index in range(4) if index != gold], identity,
                                configuration['selection_seed'], 'replacement')[0]
        for arm, removed in [('supported',wrong),('unsupported',gold)]:
            indices = ordered_indices([index for index in range(4) if index != removed], identity,
                                      configuration['selection_seed'], arm)
            options = [original['options'][index] for index in indices] + [configuration['null_option']]
            encoded_text = (original['question']+' \\n '+' '.join('('+chr(97+index)+') '+option
                            for index,option in enumerate(options))+' \\n '+original['article']).lower()
            assert len(tokenizer.encode(encoded_text)) <= configuration['maximum_input_tokens']
            assert max(len(tokenizer.encode(option.lower())) for option in options) <= configuration['maximum_option_tokens']
            cases.append({'id':original['example_id'],'arm':arm,'article':original['article'],
                          'question':original['question'],'options':options,'original_indices':indices,
                          'gold':indices.index(gold) if arm == 'supported' else 3,'encoded_text':encoded_text})
    (directory/'selection.json').write_text(json.dumps({'selected':cases,'excluded':excluded},indent=2)+'\n')
    state = torch.load(model_directory/'pytorch_model.bin',map_location='cpu',weights_only=True)
    assert all(torch.isfinite(value).all() for value in state.values())
    model = T5ForConditionalGeneration(T5Config.from_pretrained(model_directory,local_files_only=True))
    attention = model.decoder.block[0].layer[1].EncDecAttention
    attention.has_relative_attention_bias = True
    attention.relative_attention_bias = torch.nn.Embedding(32,8)
    model.load_state_dict(state,strict=True)
    del state
    model.config.use_cache = False
    model.eval()
    digest = parameter_digest(model)
    assert digest == configuration['expected_parameter_digest']
    preparation_seconds = time.perf_counter()-started
    results = []
    for case in cases:
        scores = public_scores(case['article'],case['question'],case['options'])
        encoded = tokenizer([case['encoded_text']]*4,return_tensors='pt',padding=True)
        labels = tokenizer([option.lower() for option in case['options']],return_tensors='pt',padding=True)['input_ids']
        mask = labels != tokenizer.pad_token_id
        labels = labels.masked_fill(~mask,-100)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            output = model(**encoded,labels=labels)
            assert torch.isfinite(output.logits).all()
            losses = torch.nn.functional.cross_entropy(output.logits.transpose(1,2),labels,ignore_index=-100,reduction='none')
            totals = -losses.sum(dim=1)
            means = totals/mask.sum(dim=1)
        scores['reader_mean_log_probability'] = means.tolist()
        scores['reader_total_log_probability'] = totals.tolist()
        predictions = {name:max(range(4),key=lambda index:values[index]) for name,values in scores.items()}
        result = {'id':case['id'],'arm':case['arm'],'gold':case['gold'],'scores':scores,
                  'predictions':predictions,'model_seconds':time.perf_counter()-inference_started}
        results.append(result)
        with (directory/'cases.jsonl').open('a') as output_file:
            output_file.write(json.dumps(result)+'\n')
    assert parameter_digest(model) == digest
    measures = {}
    for name in results[0]['predictions']:
        selected = [row for row in results if row['arm']=='supported']
        unsupported = [row for row in results if row['arm']=='unsupported']
        measures[name] = {
            'supported_accuracy':sum(row['predictions'][name]==row['gold'] for row in selected)/len(selected),
            'unsupported_null_accuracy':sum(row['predictions'][name]==3 for row in unsupported)/len(unsupported),
            'balanced_answerability':sum((row['predictions'][name]!=3)==(row['arm']=='supported') for row in results)/len(results),
            'overall_answer_accuracy':sum(row['predictions'][name]==row['gold'] for row in results)/len(results)}
    summary = {'status':'complete','case_count':len(results),'article_count':len(results)//2,
               'excluded':excluded,'parameter_digest':digest,'measures':measures,
               'preparation_seconds':preparation_seconds,'wall_seconds':time.perf_counter()-started,
               'model_seconds':sum(row['model_seconds'] for row in results),'neural_training':False,'training_admitted':False}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
