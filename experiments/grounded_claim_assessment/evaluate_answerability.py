"""Free-response evaluation on independently annotated answerability pairs."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import re
import socket
import string
import sys
import time
import torch
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from evaluate_grounded_claims import parameter_digest, reject_network, words, overlap


def normalized(text):
    text=text.lower()
    text=''.join(character for character in text if character not in string.punctuation)
    text=re.sub(r'\b(a|an|the)\b',' ',text)
    return ' '.join(text.split())


def answer_scores(prediction, references):
    predicted=normalized(prediction)
    exact=0
    f1=0.0
    for reference in references:
        expected=normalized(reference)
        exact=max(exact,int(predicted==expected))
        predicted_words=predicted.split()
        expected_words=expected.split()
        shared=sum((Counter(predicted_words)&Counter(expected_words)).values())
        value=2*shared/(len(predicted_words)+len(expected_words)) if predicted_words and expected_words else float(predicted_words==expected_words)
        f1=max(f1,value)
    return exact,f1


def stable_hash(seed, text):
    return hashlib.sha256((str(seed)+'\0'+text).encode()).hexdigest()


def main(directory, data_directory, model_directory):
    started=time.perf_counter()
    socket.create_connection=reject_network
    socket.socket.connect=reject_network
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    configuration=json.loads((directory/'config.json').read_text())
    tokenizer=T5Tokenizer.from_pretrained(model_directory,local_files_only=True,legacy=True)
    dataset=json.loads((data_directory/'dev-v2.0.json').read_text())
    paragraphs=[]
    for article in dataset['data']:
        for paragraph in article['paragraphs']:
            identity=stable_hash(configuration['selection_seed'],article['title']+'\0'+paragraph['context'])
            paragraphs.append((identity,article['title'],paragraph))
    selected=[]
    excluded=[]
    for identity,title,paragraph in sorted(paragraphs,key=lambda value:value[0]):
        pair=[]
        for impossible in (False,True):
            questions=sorted([row for row in paragraph['qas'] if row['is_impossible']==impossible],key=lambda row:stable_hash(configuration['selection_seed'],row['id']))
            for question in questions:
                text=(question['question']+' \\n ('+title+') '+paragraph['context']).lower()
                tokens=tokenizer.encode(text)
                if len(tokens)>configuration['maximum_input_tokens']:
                    excluded.append({'id':question['id'],'reason':'input_token_limit','tokens':len(tokens)})
                    continue
                references=[answer['text'] for answer in question['answers']]
                assert impossible or references
                assert not impossible or not references
                pair.append({'paragraph_id':identity,'id':question['id'],'title':title,
                             'context':paragraph['context'],'question':question['question'],
                             'impossible':impossible,'references':references,'text':text,'input_tokens':len(tokens)})
                break
        if len(pair)==2:
            selected.append(pair)
        else:
            excluded.append({'paragraph_id':identity,'reason':'no_eligible_balanced_pair'})
        if len(selected)==configuration['paragraph_pairs']:
            break
    assert len(selected)==configuration['paragraph_pairs']
    assert len({pair[0]['paragraph_id'] for pair in selected})==configuration['paragraph_pairs']
    offset=configuration['paragraph_offset']
    cases=[row for pair in selected[offset:offset+configuration['block_paragraph_pairs']] for row in pair]
    (directory/'selection.json').write_text(json.dumps({'selected':cases,'excluded':excluded},indent=2)+'\n')
    state=torch.load(model_directory/'pytorch_model.bin',map_location='cpu',weights_only=True)
    assert all(torch.isfinite(value).all() for value in state.values())
    model=T5ForConditionalGeneration(T5Config.from_pretrained(model_directory,local_files_only=True))
    attention=model.decoder.block[0].layer[1].EncDecAttention
    attention.has_relative_attention_bias=True
    attention.relative_attention_bias=torch.nn.Embedding(32,8)
    model.load_state_dict(state,strict=True)
    del state
    model.config.use_cache=False
    model.eval()
    digest=parameter_digest(model)
    assert digest==configuration['expected_parameter_digest']
    null_labels=tokenizer(configuration['null_phrase'],return_tensors='pt')['input_ids']
    null_spellings={normalized(configuration['null_phrase']),normalized(tokenizer.decode(null_labels[0],skip_special_tokens=True))}
    preparation_seconds=time.perf_counter()-started
    results=[]
    for case in cases:
        encoded=tokenizer(case['text'],return_tensors='pt')
        inference_started=time.perf_counter()
        with torch.inference_mode():
            encoder_output=model.get_encoder()(**encoded,return_dict=True)
            output=model(encoder_outputs=encoder_output,attention_mask=encoded['attention_mask'],labels=null_labels)
            assert torch.isfinite(output.logits).all()
            losses=torch.nn.functional.cross_entropy(output.logits.transpose(1,2),null_labels,reduction='none')
            generation=model.generate(encoder_outputs=encoder_output,attention_mask=encoded['attention_mask'],
                                      max_new_tokens=configuration['maximum_new_tokens'],do_sample=False,num_beams=1,use_cache=False)
        tokens=generation[0].tolist()
        text=tokenizer.decode(tokens,skip_special_tokens=True)
        terminated=tokenizer.eos_token_id in tokens
        abstained=terminated and normalized(text) in null_spellings
        if case['impossible']:
            exact=f1=float(abstained)
        else:
            exact,f1=answer_scores(text,case['references'])
            if not terminated:
                exact=0
        question_words=words(case['question'])
        passage_words=words(case['context'])
        sentences=re.split(r'[.!?]+',case['context'])
        scores={'reader_null_mean':-float(losses.mean()),'reader_null_total':-float(losses.sum()),
                'question_length':len(case['question']),'passage_length':len(case['context']),
                'passage_overlap':overlap(question_words,passage_words),
                'sentence_overlap':max(overlap(question_words,words(sentence)) for sentence in sentences)}
        result={'id':case['id'],'paragraph_id':case['paragraph_id'],'impossible':case['impossible'],
                'generated_tokens':tokens,'generated_text':text,'terminated':terminated,'abstained':abstained,
                'exact_match':exact,'token_f1':f1,'scores':scores,'model_seconds':time.perf_counter()-inference_started}
        results.append(result)
        with (directory/'cases.jsonl').open('a') as output_file:output_file.write(json.dumps(result)+'\n')
    assert parameter_digest(model)==digest
    answerable=[row for row in results if not row['impossible']]
    impossible=[row for row in results if row['impossible']]
    summary={'status':'complete','case_count':len(results),'parameter_digest':digest,
             'answerable_exact_match':sum(row['exact_match'] for row in answerable)/len(answerable),
             'answerable_token_f1':sum(row['token_f1'] for row in answerable)/len(answerable),
             'null_recall':sum(row['abstained'] for row in impossible)/len(impossible),
             'balanced_answerability':sum(row['abstained']==row['impossible'] for row in results)/len(results),
             'overall_exact_match':sum(row['exact_match'] for row in results)/len(results),
             'overall_token_f1':sum(row['token_f1'] for row in results)/len(results),
             'nonterminated_count':sum(not row['terminated'] for row in results),'null_spellings':sorted(null_spellings),
             'preparation_seconds':preparation_seconds,'model_seconds':sum(row['model_seconds'] for row in results),
             'wall_seconds':time.perf_counter()-started,'neural_training':False,'training_admitted':False}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve(),Path(sys.argv[3]).resolve())
