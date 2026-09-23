"""Hold the public spatial conditioner fixed and substitute the neural mean."""

from pathlib import Path
import hashlib
import json
import math
import sys
import time
import numpy as np
import torch
from conditional_scene_distribution import disclosure_path
from joint_scene_mixture import JointSceneMixture
from sequential_scene_prediction import clipped_depth_mean, sequential_bin_log_probabilities, sequential_marginals


def main(directory,source_root,conditional_root):
    started=time.perf_counter()
    configuration=json.loads((directory/'config.json').read_text())
    manifest=json.loads((directory/'source_manifest.json').read_text())
    source_records={record['path']:record for record in manifest['files']}
    checked=[]
    def verified(relative):
        path=source_root/relative;content=path.read_bytes()
        record=source_records[relative]
        assert len(content)==record['bytes'] and hashlib.sha256(content).hexdigest()==record['sha256']
        checked.append(relative)
        return path
    source=np.load(verified('analysis01/per_scene_predictions.npz'),allow_pickle=False)
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    coordinates=source['coordinates'];names=source['reader_names'].tolist()
    distances=torch.tensor(((coordinates[:,None]-coordinates[None])**2).sum(-1))
    correlation=torch.exp(-distances/(2*configuration['length_scale_pixels']**2))
    factor=torch.linalg.cholesky(correlation+torch.eye(64)*configuration['correlation_diagonal_stabilizer'])*configuration['log_depth_standard_deviation']
    boundaries=torch.linspace(math.log(.1),math.log(10),63)
    assert torch.equal(torch.bucketize(boundaries,boundaries,right=False),torch.arange(63))
    log_losses=[];squared_errors=[];means=[];risks=[];labels=[];maximum_error=0.
    with torch.no_grad():
        for scene in range(16):
            saved=np.load(verified(f'inference{scene//4+1:02d}/scene_{scene+16:03d}.npz'),allow_pickle=False)
            assert np.array_equal(saved['coordinates'],coordinates)
            assert np.array_equal(saved['locations'],np.broadcast_to(saved['locations'][0],saved['locations'].shape))
            distribution=JointSceneMixture(torch.zeros(1),torch.tensor(saved['locations'][:1],dtype=torch.float64),
                                           factor[None],torch.full((1,64),configuration['diagonal_standard_deviation']**2))
            depths=torch.tensor(saved['depths']);observations=depths.log()
            marginals=sequential_marginals(distribution,observations)
            probabilities=sequential_bin_log_probabilities(marginals,boundaries)
            assert torch.isfinite(probabilities).all() and torch.logsumexp(probabilities,-1).abs().max()<1e-10
            outcome=torch.bucketize(observations,boundaries,right=False)
            paths=disclosure_path(probabilities,outcome)
            selected=probabilities[torch.arange(64),outcome]
            maximum_error=max(maximum_error,float((paths['selected_log_probabilities'].sum(-1)-selected).abs().max()))
            prediction=clipped_depth_mean(marginals)
            log_losses.append(-selected.numpy());squared_errors.append((prediction-depths.clamp(.1,10)).square().numpy())
            means.append(prediction.numpy());risks.append(paths['upper_probabilities'].numpy());labels.append(paths['upper_outcomes'].numpy())
    log_losses=np.stack(log_losses);squared_errors=np.stack(squared_errors)
    differences=source['log_losses'][:,1:].mean(-1)-log_losses.mean(-1)[:,None]
    mean_advantages=differences.mean(0)
    resamples=source['bootstrap_indices']
    expected=np.random.default_rng(configuration['seed']).integers(0,16,size=resamples.shape)
    assert np.array_equal(resamples,expected)
    maximum_errors=(mean_advantages[None]-differences[resamples].mean(1)).max(1)
    critical=float(np.quantile(maximum_errors,.95))
    best_index=int(np.argmin(source['log_losses'][:,1:].mean((0,2))))+1
    # The old integration's bin convention only changes labels at exact ties.
    old=np.load(conditional_root/'integration01'/'conditional_predictions.npz',allow_pickle=False)
    equality_count=int(np.count_nonzero(old['target_log_depths'][:,None]==old['boundaries'][None]))
    old_lower=np.searchsorted(old['boundaries'],old['target_log_depths'],side='left')
    old_upper=np.searchsorted(old['boundaries'],old['target_log_depths'],side='right')
    changed_labels=int(np.count_nonzero(old_lower!=old_upper))
    np.savez(directory/'matched_predictions.npz',log_losses=log_losses,squared_errors=squared_errors,
             predictive_means=np.stack(means),risk_probabilities=np.stack(risks),risk_outcomes=np.stack(labels),
             public_mean_log_advantages=mean_advantages,bootstrap_maximum_centered_errors=maximum_errors)
    summary={'status':'complete','stage':configuration['stage'],'development_screen_pass':bool((mean_advantages>0).all() and mean_advantages.min()-critical>0),
             'conditional_negative_log_likelihood_nats':float(log_losses.mean()),'clipped_depth_mse':float(squared_errors.mean()),
             'best_public_reader':names[best_index],'best_public_negative_log_likelihood_nats':float(source['log_losses'][:,best_index].mean()),
             'best_public_clipped_depth_mse':float(source['squared_errors'][:,best_index].mean()),
             'minimum_mean_log_advantage_nats':float(mean_advantages.min()),'simultaneous_bootstrap_critical_value':critical,
             'minimum_simultaneous_lower_bound_nats':float(mean_advantages.min()-critical),
             'mean_by_history_stratum_nats':[float(log_losses[:,start:start+16].mean()) for start in range(0,64,16)],
             'mse_by_history_stratum':[float(squared_errors[:,start:start+16].mean()) for start in range(0,64,16)],
             'maximum_path_error':maximum_error,'source_files_verified':checked,'strict_boundary_values_checked':63,
             'historical_integration_exact_boundary_ties':equality_count,'historical_integration_changed_labels':changed_labels,
             'neural_training':False,'optimizer_steps':0,'wall_seconds':time.perf_counter()-started}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
