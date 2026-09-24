"""Generate fresh pattern compositions without consulting outcomes or source names."""
from pathlib import Path
import hashlib
import json
import random
import re
import sys

TOKEN = re.compile(r'<->|->|[()!&|]|[A-Za-z_][A-Za-z_0-9]*|[01]')
OPERATORS = {'G','F','X','U','R','W','M','true','false','0','1','!','&','|','->','<->','(',')'}


def main():
    root=Path(sys.argv[1]); configuration=json.loads((root/'config.json').read_text())
    source=Path(configuration['source_directory'])
    manifest=json.loads(Path(configuration['source_manifest']).read_text())
    hashes={Path(item['path']).name:item['sha256'] for item in manifest['files'] if item['path'].endswith('.json')}
    library={'assumptions':{},'guarantees':{}}
    for path in sorted(source.glob('*.json')):
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=hashes[path.name]: raise ValueError('Source checksum mismatch.')
        specification=json.loads(data)
        if not all(key in specification for key in ('inputs','outputs','assumptions','guarantees')): continue
        for category in library:
            for formula in specification[category]:
                tokens=TOKEN.findall(formula)
                if ''.join(tokens)!=re.sub(r'\s+','',formula): continue
                if len([token for token in tokens if token not in ('(',')')])>configuration['maximum_pattern_tokens']: continue
                mapping={}; counts={'i':0,'o':0}; normalized=[]
                for token in tokens:
                    if token in specification['inputs'] or token in specification['outputs']:
                        role='i' if token in specification['inputs'] else 'o'
                        if token not in mapping:
                            mapping[token]=role+str(counts[role]);counts[role]+=1
                        normalized.append(mapping[token])
                    elif token in OPERATORS: normalized.append(token)
                    else: break
                else:
                    if counts['i']>3 or counts['o']>3: continue
                    if category=='assumptions' and (counts['i']==0 or counts['o']!=0): continue
                    if category=='guarantees' and counts['o']==0: continue
                    canonical=' '.join(normalized)
                    library[category].setdefault(canonical,dict(formula=canonical,inputs=counts['i'],outputs=counts['o'],source_sha256=hashes[path.name]))
    pools={category:[library[category][key] for key in sorted(library[category])] for category in library}
    (root/'pattern_library.json').write_text(json.dumps(pools,indent=2)+'\n')
    generator=random.Random(configuration['seed']); records=[]
    signal_counts={field:configuration.get(field[:-1]+'_count',3) for field in ('inputs','outputs')}
    if any(count<3 or count>5 for count in signal_counts.values()):
        raise ValueError('Signal counts must be between three and five per role.')
    for index in range(configuration['instance_count']):
        record=dict(name=f'specification_{index:04d}',inputs=['i'+str(index) for index in range(signal_counts['inputs'])],outputs=['o'+str(index) for index in range(signal_counts['outputs'])],semantics='mealy',assumptions=[],guarantees=[],generation=[])
        for category,count in [('assumptions',configuration['assumption_count']),('guarantees',configuration['guarantee_count'])]:
            for position in generator.sample(range(len(pools[category])),count):
                pattern=pools[category][position]
                mapping={}
                for prefix,field in [('i','inputs'),('o','outputs')]:
                    for original,replacement in enumerate(generator.sample(range(signal_counts[field]),pattern[field])):
                        mapping[prefix+str(original)]=prefix+str(replacement)
                formula=' '.join(mapping.get(token,token) for token in pattern['formula'].split())
                record[category].append(formula)
                record['generation'].append(dict(category=category,pattern_index=position,mapping=mapping))
        records.append(record)
    (root/'specifications.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(dict(patterns={key:len(value) for key,value in pools.items()},instances=len(records),seed=configuration['seed'],outcome_selection=False)))


if __name__=='__main__': main()
