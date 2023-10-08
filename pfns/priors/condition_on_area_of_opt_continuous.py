import torch

from .prior import Batch
from ..utils import default_device


@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, get_batch, epoch, device=default_device, hyperparameters={}, **kwargs):
    """
    This is not part of the paper, but feel welcome to use this to write a better version of our user prior.


    This function assumes that every x is in the range [0, 1].
    Style shape is (batch_size, 3*num_features) under the assumption that get_batch returns a batch
    with shape (seq_len, batch_size, num_features).
    The style is build the following way: [prob_of_feature_1_in_range, range_min_of_feature_1, range_max_of_feature_1, ...]



    :param batch_size:
    :param seq_len:
    :param num_features:
    :param get_batch:
    :param epoch:
    :param device:
    :param hyperparameters:
    :param kwargs:
    :return:
    """

    maximize = hyperparameters.get('condition_on_area_maximize', True)
    size_range = hyperparameters.get('condition_on_area_size_range', (0.1, 0.5))
    distribution = hyperparameters.get('condition_on_area_distribution', 'uniform')
    assert distribution in ['uniform']


    batch: Batch = get_batch(batch_size=batch_size, seq_len=seq_len,
                             num_features=num_features, device=device,
                             hyperparameters=hyperparameters,
                             epoch=epoch, **kwargs)
    assert batch.style is None

    d = batch.x.shape[2]

    prob_correct = torch.rand(batch_size, d, device=device)
    correct_opt = torch.rand(batch_size, d, device=device) < prob_correct
    division_size = torch.rand(batch_size, d, device=device) * (size_range[1] - size_range[0]) + size_range[0]

    optima = batch.target_y.argmax(0).squeeze() if maximize else batch.target_y.argmin(0).squeeze()  # batch_size, d

    optima_hints = batch.x[optima, torch.arange(batch_size, device=device)] - division_size/2 + torch.rand(batch_size, d, device=device)*division_size # shape: (batch_size, d)
    optima_hints = optima_hints.clamp(0, 1)

    optima_division_lower_bound = (optima_hints - division_size/2).clamp(0, 1)
    optima_division_upper_bound = (optima_hints + division_size/2).clamp(0, 1)

    random_hints = torch.rand(batch_size, d, device=device) - division_size/2 + torch.rand(batch_size, d, device=device)*division_size # shape: (batch_size, d)
    random_hints = random_hints.clamp(0, 1)

    random_division_lower_bound = (random_hints - division_size/2).clamp(0, 1)
    random_division_upper_bound = (random_hints + division_size/2).clamp(0, 1)


    lower_bounds = torch.where(correct_opt, optima_division_lower_bound, random_division_lower_bound)
    upper_bounds = torch.where(correct_opt, optima_division_upper_bound, random_division_upper_bound)

    batch.style = torch.stack([prob_correct, lower_bounds, upper_bounds], 2).view(batch_size, -1)  # shape: (batch_size, 3*d)

    return batch







