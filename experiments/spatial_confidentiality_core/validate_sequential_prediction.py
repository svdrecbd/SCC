"""Check every prefix of the sequential filter against direct conditioning."""

from pathlib import Path
import json
import sys
import time
import torch
from conditional_scene_distribution import conditional_scene, marginal_bin_log_probabilities
from joint_scene_mixture import JointSceneMixture
from sequential_scene_prediction import sequential_marginals, sequential_bin_log_probabilities


def main(directory):
    started = time.perf_counter()
    configuration=json.loads((directory/'config.json').read_text())
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    generator=torch.Generator().manual_seed(configuration['seed'])
    boundaries=torch.linspace(-3.,3.,63)
    records=[]
    for case in range(configuration['validation_cases']):
        mixture=JointSceneMixture(torch.randn(4,generator=generator),torch.randn(4,8,generator=generator),
                                  torch.randn(4,8,3,generator=generator)*.4,torch.rand(4,8,generator=generator)+.2)
        observations=torch.randn(8,generator=generator)
        sequential=sequential_marginals(mixture,observations)
        probabilities=sequential_bin_log_probabilities(sequential,boundaries)
        errors=[]
        for position in range(8):
            direct,density=conditional_scene(mixture,list(range(position)),observations[:position],[position])
            errors.extend([float((direct.locations[:,0]-sequential['locations'][position]).abs().max()),
                           float((direct.diagonal[:,0]+direct.factors[:,0].square().sum(-1)-sequential['variances'][position]).abs().max()),
                           float((direct.logits.softmax(-1)-sequential['log_weights'][position].exp()).abs().max()),
                           float((marginal_bin_log_probabilities(direct,boundaries)[0]-probabilities[position]).abs().max()),
                           float((density-sequential['log_densities'][:position].sum()).abs())])
        full_covariance=mixture.factors@mixture.factors.transpose(-1,-2)+torch.diag_embed(mixture.diagonal)
        dense=torch.distributions.MultivariateNormal(mixture.locations,covariance_matrix=full_covariance)
        dense_density=torch.logsumexp(mixture.logits.log_softmax(-1)+dense.log_prob(observations),-1)
        errors.append(float((dense_density-sequential['log_densities'].sum()).abs()))
        assert max(errors)<1e-9, errors
        # A future outcome cannot influence any earlier marginal.
        changed=observations.clone();changed[-1]+=10
        alternative=sequential_marginals(mixture,changed)
        for key in ['locations','variances','log_weights']:
            assert torch.equal(alternative[key],sequential[key])
        records.append(max(errors))
    summary={'status':'complete','prefixes_checked':8*len(records),'maximum_errors':records,
             'future_label_independence_checks':len(records)*3,'neural_training':False,
             'wall_seconds':time.perf_counter()-started}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main(Path(sys.argv[1]))
