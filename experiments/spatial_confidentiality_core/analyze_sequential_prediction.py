"""Compare conditional scene forecasts against fixed public inference procedures."""

from pathlib import Path
import hashlib
import json
import math
import sys
import time
import numpy as np
import torch
from analyze import features, thumbnail
from conditional_scene_distribution import disclosure_path
from data import read_frame
from joint_scene_mixture import JointSceneMixture
from sequential_scene_prediction import clipped_depth_mean, sequential_bin_log_probabilities, sequential_marginals


def main(directory,root,evaluation_root):
    started=time.perf_counter()
    configuration=json.loads((directory/'config.json').read_text())
    selection=json.loads((directory/'selection.json').read_text())
    parameters=np.load(directory/'public_parameters.npz',allow_pickle=False)
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    coordinates=np.array([[row,column] for row in configuration['rows'] for column in configuration['columns']])
    rows,columns=coordinates.T
    images,truths=[],[]
    for record in selection:
        image,truth,_=read_frame(root/'acquisition01',record)
        images.append(image);truths.append(truth.astype(np.float64))
    calibration=np.stack(truths[:16])[:,rows,columns]
    invalid=(~np.isfinite(calibration))|(calibration<=.1)|(calibration>=10)
    calibration[invalid]=float(parameters['global_mean'])
    templates=torch.tensor(np.log(calibration))
    spatial_coordinates=np.meshgrid(np.linspace(-1,1,640),np.linspace(-1,1,480))
    distance_square=torch.tensor(((coordinates[:,None]-coordinates[None])**2).sum(-1))
    public_factors=[]
    for length in configuration['public_length_scales_pixels']:
        correlation=torch.exp(-distance_square/(2*length**2))
        # Numerical diagonal stabilizer belongs to the declared covariance.
        factor=torch.linalg.cholesky(correlation+torch.eye(len(coordinates))*1e-12)
        for scale in configuration['public_log_depth_scales']:
            public_factors.append((f'length_{length}_scale_{scale:g}',factor*scale))
    names=['learned_parent']
    mean_names=['global_mean','spatial_mean','quadratic_position','position_rgb','nearest_image','three_nearest_images']
    names.extend(f'{name}_{label}' for name in mean_names for label,_ in public_factors)
    names.extend(f'calibration_mixture_scale_{scale:g}' for scale in configuration['template_standard_deviations'])
    names.append('uniform_bins')
    boundaries=torch.linspace(math.log(.1),math.log(10),configuration['finite_boundaries'])
    assert torch.equal(torch.bucketize(boundaries,boundaries,right=False),torch.arange(len(boundaries)))
    log_losses=np.zeros((16,len(names),64));squared_errors=np.zeros_like(log_losses)
    predictive_means=np.zeros_like(log_losses)
    parent_path_probabilities=[];parent_path_outcomes=[];public_means_saved=[]
    source_hashes=[];maximum_path_error=0.
    comparison_started=time.perf_counter()
    with torch.no_grad():
        for scene_number,index in enumerate(configuration['development_indices']):
            path=evaluation_root/f'inference{scene_number//4+1:02d}'/f'scene_{index:03d}.npz'
            content=path.read_bytes();source_hashes.append({'path':str(path),'sha256':hashlib.sha256(content).hexdigest()})
            saved=np.load(path,allow_pickle=False)
            assert np.array_equal(saved['coordinates'],coordinates)
            depths=torch.tensor(saved['depths']);observations=depths.log()
            assert np.array_equal(saved['depths'],truths[index][rows,columns])
            outcomes=torch.bucketize(observations,boundaries,right=False)
            parent=JointSceneMixture(*[torch.tensor(saved[name],dtype=torch.float64) for name in ['logits','locations','factors','diagonal']])
            distances=np.mean((parameters['thumbnails']-thumbnail(images[index]))**2,axis=1)
            neighbors=np.argsort(distances,kind='stable')[:3]
            weights=1/(distances[neighbors]+1e-8);weights/=weights.sum()
            design=features(images[index],spatial_coordinates)[rows,columns]
            means=np.stack([np.full(64,float(parameters['global_mean'])),parameters['spatial_mean'][rows,columns],
                            design[:,:6]@parameters['position_coefficients'],design@parameters['rgb_coefficients'],
                            calibration[neighbors[0]],weights@calibration[neighbors]])
            means=np.clip(means,.1,10);public_means_saved.append(means)
            readers=[parent]
            for mean in means:
                for _,factor in public_factors:
                    readers.append(JointSceneMixture(torch.zeros(1),torch.tensor(np.log(mean))[None],factor[None],
                                                     torch.full((1,64),configuration['public_diagonal_standard_deviation']**2)))
            for scale in configuration['template_standard_deviations']:
                readers.append(JointSceneMixture(torch.zeros(16),templates,torch.zeros(16,64,1),torch.full((16,64),scale**2)))
            assert len(readers)==len(names)-1
            for reader_index,reader in enumerate(readers):
                marginal=sequential_marginals(reader,observations)
                probabilities=sequential_bin_log_probabilities(marginal,boundaries)
                assert torch.isfinite(probabilities).all()
                assert torch.logsumexp(probabilities,-1).abs().max()<1e-10
                mean=clipped_depth_mean(marginal)
                log_losses[scene_number,reader_index]=-probabilities[torch.arange(64),outcomes].numpy()
                squared_errors[scene_number,reader_index]=(mean-depths.clamp(.1,10)).square().numpy()
                predictive_means[scene_number,reader_index]=mean.numpy()
                if reader_index==0:
                    paths=disclosure_path(probabilities,outcomes)
                    maximum_path_error=max(maximum_path_error,float((paths['selected_log_probabilities'].sum(-1)-probabilities[torch.arange(64),outcomes]).abs().max()))
                    parent_path_probabilities.append(paths['upper_probabilities'].numpy())
                    parent_path_outcomes.append(paths['upper_outcomes'].numpy())
            # Fixed midpoint representatives for the bounded uniform-bin mean.
            finite_metres=boundaries.exp()
            representatives=torch.cat((torch.tensor([.1]),(finite_metres[:-1]+finite_metres[1:])/2,torch.tensor([10.])))
            uniform_mean=float(representatives.mean())
            log_losses[scene_number,-1]=math.log(64)
            squared_errors[scene_number,-1]=(uniform_mean-depths.clamp(.1,10)).square().numpy()
            predictive_means[scene_number,-1]=uniform_mean
    comparison_seconds=time.perf_counter()-comparison_started
    differences=log_losses[:,1:].mean(-1)-log_losses[:,0:1].mean(-1)
    means=differences.mean(0)
    generator=np.random.default_rng(configuration['seed'])
    indices=generator.integers(0,16,size=(configuration['bootstrap_repetitions'],16))
    resampled=differences[indices].mean(1)
    maximum_errors=(means[None]-resampled).max(1)
    critical=float(np.quantile(maximum_errors,.95))
    lower=means-critical
    best_index=int(np.argmin(log_losses.mean((0,2))[1:]))+1
    records=[]
    for index,name in enumerate(names):
        records.append({'reader':name,'mean_negative_log_likelihood_nats':float(log_losses[:,index].mean()),
                        'clipped_depth_mse':float(squared_errors[:,index].mean()),
                        'mean_by_history_stratum_nats':[float(log_losses[:,index,start:start+16].mean()) for start in range(0,64,16)],
                        'mse_by_history_stratum':[float(squared_errors[:,index,start:start+16].mean()) for start in range(0,64,16)]})
    np.savez(directory/'per_scene_predictions.npz',reader_names=np.array(names),coordinates=coordinates,log_losses=log_losses,
             squared_errors=squared_errors,predictive_means=predictive_means,public_mean_fields=np.stack(public_means_saved),
             calibration_depths=calibration,parent_risk_probabilities=np.stack(parent_path_probabilities),
             parent_risk_outcomes=np.stack(parent_path_outcomes),bootstrap_indices=indices.astype(np.int16),
             bootstrap_maximum_centered_errors=maximum_errors)
    summary={'status':'complete','scenes':16,'rays_per_scene':64,'public_readers':len(names)-1,
             'qualification_pass':bool((means>0).all() and (lower>0).all()),
             'best_public_reader':names[best_index], 'minimum_mean_advantage_nats':float(means.min()),
             'simultaneous_bootstrap_critical_value':critical,'minimum_simultaneous_lower_bound_nats':float(lower.min()),
             'maximum_selected_path_identity_error':maximum_path_error,'imputed_calibration_values':int(invalid.sum()),
             'covariance_diagonal_stabilizer':1e-12,'learned_reader':records[0], 'best_public_record':records[best_index],
             'all_readers':records,'source_hashes':source_hashes,'comparison_seconds':comparison_seconds,
             'neural_training':False,'optimizer_steps':0,'wall_seconds':time.perf_counter()-started}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({key:value for key,value in summary.items() if key not in ['all_readers','source_hashes']},indent=2))


if __name__=='__main__':
    main(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
