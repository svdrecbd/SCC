"""Prepare independent semantic controls for the pinned general LTL checker."""
from pathlib import Path
import json
import sys
from verify_aiger_controller import controller_graph, verify_aiger


def main():
    directory = Path(sys.argv[1])
    records = []
    fixtures = json.loads((directory/'validation.json').read_text())
    formulas = dict(current_response='G (i0 <-> o0)', delayed_response='G (i0 <-> X o0)',
        future_input_prediction='G (o0 <-> X i0)', unforceable_input='G i0')
    for item in fixtures:
        try:
            graph = controller_graph(item['circuit'],item['system_wins'])
        except ValueError:
            continue
        graph.update(name=f"fixture_{item['index']:02d}",formula=formulas[item['case']],
            environment_names=['i0'],system_names=['o0'],expected=item['expected'])
        records.append(graph)
    controls = [
        ('alternating_liveness','G F o0','aag 2 1 1 1 0\n2\n4 5\n4\n', True, True, ['i0'],['o0']),
        ('alternating_invariance','G o0','aag 2 1 1 1 0\n2\n4 5\n4\n', True, False, ['i0'],['o0']),
        ('environment_liveness','G F i0','aag 1 1 0 1 0\n2\n0\n', False, True, ['i0'],['o0']),
        ('invalid_environment_liveness','G F i0','aag 1 1 0 1 0\n2\n1\n', False, False, ['i0'],['o0']),
        ('two_signal_copy','G ((i0 <-> o0) & (i1 <-> o1))','aag 2 2 0 2 0\n2\n4\n2\n4\n', True, True,['i0','i1'],['o0','o1']),
        ('two_signal_mismatch','G ((i0 <-> o0) & (i1 <-> o1))','aag 2 2 0 2 0\n2\n4\n2\n2\n', True, False,['i0','i1'],['o0','o1']),
        ('until_violation','(!o0) U o0','aag 1 1 0 1 0\n2\n0\n',True,False,['i0'],['o0']),
        ('until_satisfied','(!o0) U o0','aag 1 1 0 1 0\n2\n1\n',True,True,['i0'],['o0']),
    ]
    for name,formula,circuit,system_wins,expected,environment,system in controls:
        graph=controller_graph(circuit,system_wins)
        graph.update(name=name,formula=formula,environment_names=environment,system_names=system,expected=expected)
        records.append(graph)
    (directory/'checker_inputs.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(dict(prepared=len(records))))


if __name__ == '__main__':
    main()
