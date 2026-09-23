"""Fixed candidate-scoring screen for reading and evidence-grounded claim recovery."""
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import re
import socket
import sys
import time
import numpy as np
import pandas as pd
import sentencepiece
import torch
import transformers
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer


def reject_network(*arguments, **keywords):
    raise RuntimeError('Network access is disabled during evaluation')


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def claim_accuracy(ranking, gold, accepted_count=1):
    accepted = set(ranking[:accepted_count])
    return Fraction(int(gold in accepted), 2) + sum(
        Fraction(int(index not in accepted), 6) for index in range(4) if index != gold)


def words(text):
    return set(re.findall(r'[a-z0-9]+', text.lower()))


def overlap(candidate, reference):
    return len(candidate & reference) / max(1, len(candidate))


def public_scores(article, question, options):
    passage_words = words(article)
    question_words = words(question)
    sentences = re.split(r'[.!?]+', article)
    closest = max(sentences, key=lambda value: overlap(question_words, words(value)))
    option_words = [words(value) for value in options]
    scores = {
        'longest_option': [len(value) for value in options],
        'shortest_option': [-len(value) for value in options],
        'passage_overlap': [overlap(value, passage_words) for value in option_words],
        'question_overlap': [overlap(value, question_words) for value in option_words],
        'selected_sentence_overlap': [overlap(value, words(closest)) for value in option_words],
    }
    for index in range(4):
        scores['fixed_position_' + str(index + 1)] = [int(index == other) for other in range(4)]
    return scores


def main(directory, acquisition):
    socket.create_connection = reject_network
    socket.socket.connect = reject_network
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    configuration = json.loads((directory / 'config.json').read_text())
    identity_checks = 0
    for count in range(2, 9):
        for gold in range(count):
            for predicted in range(count):
                accuracy = Fraction(int(predicted == gold), 2) + sum(
                    Fraction(int(predicted != proposed), 2 * (count-1))
                    for proposed in range(count) if proposed != gold)
                assert accuracy == Fraction(count-2+count*int(predicted == gold), 2*(count-1))
                identity_checks += 1
    model_directory = acquisition / 'model'
    started = time.perf_counter()
    tokenizer = T5Tokenizer.from_pretrained(model_directory, local_files_only=True, legacy=True)
    raw_tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(model_directory / 'spiece.model'))
    sample = 'which option is supported? \\n (a) café (b) nothing \\n the café is open.'
    assert tokenizer.encode(sample) == raw_tokenizer.encode(sample) + [tokenizer.eos_token_id]
    state = torch.load(model_directory / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    assert all(isinstance(value, torch.Tensor) and torch.isfinite(value).all() for value in state.values())
    model = T5ForConditionalGeneration(T5Config.from_pretrained(model_directory, local_files_only=True))
    attention = model.decoder.block[0].layer[1].EncDecAttention
    legacy_key = 'decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight'
    assert tuple(state[legacy_key].shape) == (32, 8)
    attention.has_relative_attention_bias = True
    attention.relative_attention_bias = torch.nn.Embedding(32, 8)
    model.load_state_dict(state, strict=True)
    model.config.use_cache = False
    # Legacy T5 decoder bias: nonpositive relative positions are bucketed by distance.
    relative_positions = torch.arange(-512, 513)
    reference = []
    for relative in relative_positions.tolist():
        distance = max(-relative, 0)
        bucket = distance if distance < 16 else min(31, 16 + int(math.log(distance/16) / math.log(128/16) * 16))
        reference.append(bucket)
    actual_buckets = attention._relative_position_bucket(relative_positions, bidirectional=False, num_buckets=32, max_distance=128)
    assert actual_buckets.tolist() == reference
    assert torch.equal(attention.relative_attention_bias.weight, state[legacy_key])
    del state
    model.eval()
    original_digest = parameter_digest(model)
    preparation_seconds = time.perf_counter()-started
    table = pd.read_parquet(acquisition / 'dataset/middle/test-00000-of-00001.parquet')
    ordered = sorted(table.to_dict('records'), key=lambda row: hashlib.sha256(
        (str(configuration['selection_seed'])+'\0'+row['example_id']+'\0'+row['question']).encode()).hexdigest())
    selected = []
    excluded = []
    identifiers = set()
    for row in ordered:
        if row['example_id'] in identifiers:
            continue
        options = list(row['options'])
        text = (row['question']+' \\n '+' '.join('('+chr(97+index)+') '+option
                for index, option in enumerate(options))+' \\n '+row['article']).lower()
        tokens = tokenizer.encode(text)
        lengths = [len(tokenizer.encode(option.lower())) for option in options]
        if len(set(option.lower().strip() for option in options)) != 4 or len(tokens) > configuration['maximum_input_tokens'] or max(lengths) > configuration['maximum_option_tokens']:
            excluded.append({'id': row['example_id'], 'question': row['question'], 'input_tokens':len(tokens), 'option_tokens':lengths})
            continue
        row['options'] = options
        row['encoded_text'] = text
        row['input_tokens'] = len(tokens)
        selected.append(row)
        identifiers.add(row['example_id'])
        if len(selected) == configuration['case_count'] + configuration.get('case_offset', 0):
            break
    assert len(selected) == configuration['case_count'] + configuration.get('case_offset', 0)
    selected = selected[configuration.get('case_offset', 0):]
    (directory / 'selection.json').write_text(json.dumps({'selected':selected,'excluded':excluded},indent=2)+'\n')
    results = []
    for row in selected:
        started = time.perf_counter()
        scores = public_scores(row['article'], row['question'], row['options'])
        public_seconds = time.perf_counter()-started
        operative_text = row['encoded_text']
        if configuration.get('omit_passage', False):
            operative_text = (row['question']+' \\n '+' '.join('('+chr(97+index)+') '+option
                for index, option in enumerate(row['options']))).lower()
        encoded = tokenizer([operative_text]*4, return_tensors='pt', padding=True)
        labels = tokenizer([option.lower() for option in row['options']], return_tensors='pt', padding=True)['input_ids']
        mask = labels != tokenizer.pad_token_id
        labels = labels.masked_fill(~mask, -100)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(**encoded, labels=labels)
            assert torch.isfinite(output.logits).all()
            losses = torch.nn.functional.cross_entropy(output.logits.transpose(1,2), labels, ignore_index=-100, reduction='none')
            totals = -losses.sum(dim=1)
            means = totals / mask.sum(dim=1)
        model_seconds = time.perf_counter()-started
        scores['reader_mean_log_probability'] = means.tolist()
        scores['reader_total_log_probability'] = totals.tolist()
        gold = ord(row['answer'])-ord('A')
        rankings = {name: sorted(range(4), key=lambda index:(-values[index],index)) for name,values in scores.items()}
        useful = {name:int(ranking[0]==gold) for name,ranking in rankings.items()}
        claim = {name+'_top_'+str(count):float(claim_accuracy(ranking,gold,count))
                 for name,ranking in rankings.items() for count in (1,2,3)}
        for name, success in useful.items():
            assert abs(claim[name+'_top_1']-(1/3+2*success/3)) < 1e-12
        result = {'id':row['example_id'],'question':row['question'],'gold':gold,'input_tokens':row['input_tokens'],
                  'scores':scores,'rankings':rankings,'useful':useful,'claims':claim,
                  'public_seconds':public_seconds,'model_seconds':model_seconds,'operative_tokens':int(encoded['input_ids'].shape[1])}
        results.append(result)
        with (directory/'cases.jsonl').open('a') as output_file:
            output_file.write(json.dumps(result)+'\n')
    assert parameter_digest(model) == original_digest
    useful = {name:float(np.mean([row['useful'][name] for row in results])) for name in results[0]['useful']}
    claims = {name:float(np.mean([row['claims'][name] for row in results])) for name in results[0]['claims']}
    public_names = [name for name in useful if not name.startswith('reader_')]
    best_public = max(public_names, key=lambda name:useful[name])
    primary = 'reader_mean_log_probability'
    differences = np.array([row['useful'][primary]-row['useful'][best_public] for row in results])
    generator = np.random.default_rng(configuration['bootstrap_seed'])
    resamples = generator.integers(0,len(results),(configuration['bootstrap_replicates'],len(results)))
    interval = np.quantile(differences[resamples].mean(axis=1),[.025,.975]).tolist()
    # Conservative diagnostic also selects the strongest public control within every bootstrap sample.
    public_matrix = np.array([[row['useful'][name] for name in public_names] for row in results])
    neural_vector = np.array([row['useful'][primary] for row in results])
    adaptive = neural_vector[resamples].mean(axis=1)-public_matrix[resamples].mean(axis=1).max(axis=1)
    adaptive_interval = np.quantile(adaptive,[.025,.975]).tolist()
    best_claim = max(.5,max(value for name,value in claims.items() if not name.startswith('reader_')))
    summary = {'status':'complete','case_count':len(results),'case_offset':configuration.get('case_offset',0),'parameter_count':sum(value.numel() for value in model.parameters()),
               'parameter_digest':original_digest,'preparation_seconds':preparation_seconds,
               'model_seconds':sum(row['model_seconds'] for row in results),'public_seconds':sum(row['public_seconds'] for row in results),
               'useful_accuracy':useful,'claim_accuracy':claims,'uniform_choice_accuracy':.25,
               'best_public_answer':best_public,'best_public_claim_accuracy':best_claim,
               'paired_interval':interval,'selection_aware_interval':adaptive_interval,
               'identity_checks':identity_checks,'deleted_native_claim_accuracy':.5,
               'recovered_after_native_deletion':claims[primary+'_top_1'],
               'computational_screen_passed':bool(useful[primary]>=configuration['minimum_accuracy'] and useful[primary]-useful[best_public]>=configuration['minimum_advantage'] and adaptive_interval[0]>0),
               'training_admitted':False,'neural_training':False,'omit_passage':configuration.get('omit_passage',False),
               'versions':{'torch':torch.__version__,'transformers':transformers.__version__}}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
