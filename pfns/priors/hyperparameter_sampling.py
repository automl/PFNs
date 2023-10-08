# this is a wrapper prior that samples hyperparameters which are set to be ConfigSpace parameters
import torch
from .prior import Batch

from ConfigSpace import hyperparameters as CSH
import ConfigSpace as CS
from copy import deepcopy


def list_all_hps_in_nested(config):
    if isinstance(config, CSH.Hyperparameter):
        return [config]
    elif isinstance(config, dict):
        result = []
        for k, v in config.items():
            result += list_all_hps_in_nested(v)
        return result
    else:
        return []


def create_configspace_from_hierarchical(config):
    cs = CS.ConfigurationSpace()
    for hp in list_all_hps_in_nested(config):
        cs.add_hyperparameter(hp)
    return cs


def fill_in_configsample(config, configsample):
    # config is our dict that defines config distribution
    # configsample is a CS.Configuration
    hierarchical_configsample = deepcopy(config)
    for k, v in config.items():
        if isinstance(v, CSH.Hyperparameter):
            hierarchical_configsample[k] = configsample[v.name]
        elif isinstance(v, dict):
            hierarchical_configsample[k] = fill_in_configsample(v, configsample)
    return hierarchical_configsample


def sample_configspace_hyperparameters(hyperparameters):
    cs = create_configspace_from_hierarchical(hyperparameters)
    cs_sample = cs.sample_configuration()
    return fill_in_configsample(hyperparameters, cs_sample)


def get_batch(batch_size, *args, hyperparameters, get_batch, **kwargs):
    num_models = min(hyperparameters.get('num_hyperparameter_samples_per_batch', 1), batch_size)
    if num_models == -1:
        num_models = batch_size
    assert batch_size % num_models == 0, 'batch_size must be a multiple of num_models'
    cs = create_configspace_from_hierarchical(hyperparameters)
    sub_batches = []
    for i in range(num_models):
        cs_sample = cs.sample_configuration()
        hyperparameters_sample = fill_in_configsample(hyperparameters, cs_sample)
        sub_batch = get_batch(batch_size//num_models, *args, hyperparameters=hyperparameters_sample, **kwargs)
        sub_batches.append(sub_batch)

    # concat x, y, target (and maybe style)
    #assert 3 <= len(sub_batch) <= 4
    #return tuple(torch.cat([sb[i] for sb in sub_batches], dim=(0 if i == 3 else 1)) for i in range(len(sub_batch)))
    assert all(not b.other_filled_attributes(set_of_attributes=('x', 'y', 'target_y')) for b in sub_batches)
    return Batch(x=torch.cat([b.x for b in sub_batches], dim=1),
                 y=torch.cat([b.y for b in sub_batches], dim=1),
                 target_y=torch.cat([b.target_y for b in sub_batches], dim=1))
