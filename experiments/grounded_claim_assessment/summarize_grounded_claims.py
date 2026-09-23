"""Audit the fixed reading blocks and quantify recovery, evidence dependence and severity."""
from pathlib import Path
import json
import sys
import numpy as np


def read_run(root, name):
    directory = root / name
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['status'] == 'complete'
    cases = [json.loads(line) for line in (directory / 'cases.jsonl').read_text().splitlines()]
    assert len(cases) == summary['case_count']
    assert len({row['id'] for row in cases}) == len(cases)
    assert all(row['input_tokens'] <= 512 for row in cases)
    return summary, cases


def main(directory, root):
    configuration = json.loads((directory/'config.json').read_text())
    first, first_cases = read_run(root, 'evaluation03')
    second, second_cases = read_run(root, 'replication01')
    omission, omission_cases = read_run(root, 'evidence_omission01')
    assert first['parameter_digest'] == second['parameter_digest'] == omission['parameter_digest']
    assert not ({row['id'] for row in first_cases} & {row['id'] for row in second_cases})
    cases = first_cases + second_cases
    assert [(row['id'],row['question'],row['gold']) for row in cases] == [(row['id'],row['question'],row['gold']) for row in omission_cases]
    primary = 'reader_mean_log_probability'
    names = list(cases[0]['useful'])
    public_names = [name for name in names if not name.startswith('reader_')]
    for row in cases + omission_cases:
        for name in names:
            assert row['useful'][name] == int(row['rankings'][name][0] == row['gold'])
            assert abs(row['claims'][name+'_top_1'] - (1/3 + 2*row['useful'][name]/3)) < 1e-12
    useful = {name:float(np.mean([row['useful'][name] for row in cases])) for name in names}
    claim = {name:float(np.mean([row['claims'][name] for row in cases])) for name in cases[0]['claims']}
    best_public = max(public_names, key=lambda name:useful[name])
    best_public_claim = max(.5,max(value for name,value in claim.items() if not name.startswith('reader_')))
    actual = np.array([row['useful'][primary] for row in cases])
    public = np.array([[row['useful'][name] for name in public_names] for row in cases])
    omitted = np.array([row['useful'][primary] for row in omission_cases])
    generator = np.random.default_rng(configuration['bootstrap_seed'])
    indices = generator.integers(0,len(cases),(configuration['bootstrap_replicates'],len(cases)))
    differences = actual[indices].mean(axis=1) - public[indices].mean(axis=1).max(axis=1)
    omission_differences = (actual-omitted)[indices].mean(axis=1)
    public_retained = (useful[best_public]-.25)/(useful[primary]-.25)
    utility_cap = min(1.,1.5*(best_public_claim+configuration['removal_tolerance'])-.5)
    loss_bound = max(0.,(useful[primary]-utility_cap)/(useful[primary]-.25))
    summary = {'status':'complete','case_count':len(cases),'model_digest':first['parameter_digest'],
               'useful_accuracy':useful,'claim_accuracy':claim,'best_public_answer':best_public,
               'best_public_claim_accuracy':best_public_claim,
               'selection_aware_paired_interval':np.quantile(differences,[.025,.975]).tolist(),
               'public_retained_above_uniform_advantage':public_retained,
               'illustrative_claim_cap':best_public_claim+configuration['removal_tolerance'],
               'conditional_useful_accuracy_cap':utility_cap,
               'conditional_above_uniform_loss_fraction':loss_bound,
               'public_baseline_optimality_proved':False,
               'omitted_evidence_accuracy':float(omitted.mean()),
               'evidence_accuracy_difference':float((actual-omitted).mean()),
               'evidence_difference_interval':np.quantile(omission_differences,[.025,.975]).tolist(),
               'omission_lost_correct_answers':int(((actual==1)&(omitted==0)).sum()),
               'omission_gained_correct_answers':int(((actual==0)&(omitted==1)).sum()),
               'neural_training':False,'training_admitted':False}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({key:value for key,value in summary.items() if key not in ('useful_accuracy','claim_accuracy')}))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
