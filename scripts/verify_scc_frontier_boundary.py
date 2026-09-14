"""Independent finite boundary checks; imports the frozen supplied toy only."""
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frozen_toy', ROOT/'verify_scc_constructive_toy.py')
toy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = toy
spec.loader.exec_module(toy)

# A fixed two-bit XOR is input-independent, needs no snapshot, and leaves every
# task triplet untouched. It is outside the supplied radius-one/constant union.
fixed_masks = [sum(1 << b for b in bits) for bits in itertools.combinations(range(6, 9), 2)]
minimal=[]
for task in range(4):
    original=toy.encode(toy.Message(task,0))
    for mask in fixed_masks:
        tampered=original ^ mask
        decoded=toy.decode(tampered)
        committed=toy.atomic_commit(decoded)
        after=toy.decode(committed)
        assert after.task == task and after.policy == 1
        assert toy.hamming(original,tampered) == 2
        minimal.append({'task':task,'mask':mask,'committed':committed})

# The checker computes posterior counts using the actual full committed W.
# Baseline is decoupled; a deliberately leaky commit must fail the same test.
def posterior_guess(commit):
    from collections import defaultdict,Counter
    counts=defaultdict(Counter)
    for z in range(4):
        c=toy.encode(toy.Message(z,0))
        w=commit(c,toy.Message(0,1))
        counts[w][z]+=1
    return sum(max(row.values()) for row in counts.values())/4
pure_guess=posterior_guess(lambda c,m: toy.atomic_commit(m))
leaky_guess=posterior_guess(lambda c,m: (toy.atomic_commit(m),c))
assert pure_guess == .25 and leaky_guess == 1.

# Repeating honest atomic commits does not cover message-dependent selection of
# the next tampering function. A retained decoded observation chooses which
# constant to use; erasing that observation later does not erase the task already
# embedded in the unsafe replacement. This breaks the single-shot fixed-f premise.
adaptive=[]
for z in range(4):
    round1=toy.atomic_commit(toy.decode(toy.encode(toy.Message(z,0))))
    observed=toy.decode(round1).task
    selected_constant=toy.encode(toy.Message(observed,1))
    round2=toy.atomic_commit(toy.decode(selected_constant))
    after=toy.decode(round2)
    assert after.task==z and after.policy==1
    adaptive.append(after.task)

# A finite raw query alphabet is insufficient to tabulate a stateful interface.
# The sole query toggles and reports a bit: a one-answer table cannot reproduce
# both calls, although tabulating complete bounded histories could do so.
state=0
answers=[]
for _ in range(2):
    state ^= 1
    answers.append(state)
assert answers == [1,0]
assert max(sum(pred == truth for truth in answers) for pred in (0,1)) == 1

result={'two_policy_bit_flip':{'cases':len(minimal),'policy_removed_in_all':True,'task_accuracy':1.,'snapshots':False,'decoder_replacement':False,'trusted_commit_retained':True,'outside_declared_family':True},
        'complete_state_posterior_audit':{'honest_commit_guess_accuracy':pure_guess,'leaky_commit_guess_accuracy':leaky_guess},
        'adaptive_constant_selection':{'cases':len(adaptive),'task_accuracy':1.,'scope':'Counterexample to extending fixed-f security to history-dependent f selection without a joint leakage guarantee'},
        'finite_query_table_scope':{'query_alphabet_size':1,'stateful_outputs':answers,'single_value_table_best_accuracy':.5,'required_correction':'Assume a deterministic memoryless response or tabulate sufficient histories/state'},
        'passed':True}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
