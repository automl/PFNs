from typing import List
import torch

import utils
from .prior import Batch
from .utils import get_batch_to_dataloader
from ..utils import default_device

def get_batch(batch_size, seq_len, num_features, device=default_device
              , hyperparameters=None, batch_size_per_gp_sample=None, **kwargs):
    batch_size_per_gp_sample = batch_size_per_gp_sample or (min(64, batch_size))
    num_models = batch_size // batch_size_per_gp_sample
    assert num_models * batch_size_per_gp_sample == batch_size, f'Batch size ({batch_size}) not divisible by batch_size_per_gp_sample ({batch_size_per_gp_sample})'

    args = {'device': device,
            'seq_len': seq_len,
            'num_features': num_features,
            'batch_size': batch_size_per_gp_sample}

    prior_bag_priors_get_batch = hyperparameters['prior_bag_get_batch']
    prior_bag_priors_p = [1.0] + [hyperparameters[f'prior_bag_exp_weights_{i}'] for i in range(1, len(prior_bag_priors_get_batch))]

    weights = torch.tensor(prior_bag_priors_p, dtype=torch.float)  # create a tensor of weights
    batch_assignments = torch.multinomial(torch.softmax(weights, 0), num_models, replacement=True).numpy()

    if 'verbose' in hyperparameters and hyperparameters['verbose']:
        print('PRIOR_BAG:', weights, batch_assignments, num_models, batch_size_per_gp_sample, batch_size)

    sample: List[Batch] = \
        [prior_bag_priors_get_batch[int(prior_idx)](hyperparameters=hyperparameters, **args, **kwargs) for prior_idx in batch_assignments]

    def merge(sample, k):
        x = [getattr(x_,k) for x_ in sample]
        if torch.is_tensor(x[0]):
            return torch.cat(x, 1).detach()
        else:
            return [*x]
    utils.print_once('prior bag, merging attributes', [s.other_filled_attributes([]) for s in sample])
    sample = {k: merge(sample, k) for k in sample[0].other_filled_attributes([])}
    if hyperparameters.get('verbose'):
        print({k: v.shape for k,v in sample.items()})

    return Batch(**sample)

DataLoader = get_batch_to_dataloader(get_batch)