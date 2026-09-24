"""Translate selected contradiction certificates into complete independent LTL checks."""
from pathlib import Path
import json
import sys
from verify_aiger_controller import controller_graph
root=Path(sys.argv[1])
configuration=json.loads((root/'config.json').read_text())
assessment=Path(configuration['assessment_directory'])
source=Path(configuration['specification_directory'])
results={row['name']:row for row in (json.loads(line) for line in (assessment/'results.jsonl').read_text().splitlines())}
records=[]
for name in configuration['names']:
    item=json.loads((source/(name+'.json')).read_text())
    assert results[name]['result']=='UNREALIZABLE'
    assumptions=' & '.join('('+formula+')' for formula in item['assumptions']) or 'true'
    guarantees=' & '.join('('+formula+')' for formula in item['guarantees']) or 'true'
    formula='('+assumptions+') -> ('+guarantees+')'
    for valid in (True,False):
        choice=set(results[name]['constant_environment_true_indices']) if valid else set()
        inputs=len(item['outputs']);outputs=len(item['inputs'])
        circuit='\n'.join([f'aag {inputs} {inputs} 0 {outputs} 0']+
            [str(2*(index+1)) for index in range(inputs)]+[str(int(index in choice)) for index in range(outputs)])+'\n'
        graph=controller_graph(circuit,False)
        graph.update(name=name+('_certificate' if valid else '_negative_control'),formula=formula,
            environment_names=item['inputs'],system_names=item['outputs'],expected=valid,circuit=circuit)
        records.append(graph)
(root/'checker_inputs.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(dict(controls=len(records))))
