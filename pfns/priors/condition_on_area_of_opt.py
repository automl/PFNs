import torch

from .prior import Batch
from ..utils import default_device


@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, get_batch, epoch, device=default_device, hyperparameters={}, **kwargs):
    """
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

    max_num_divisions = hyperparameters.get('condition_on_area_max_num_divisions', 5)
    maximize = hyperparameters.get('condition_on_area_maximize', True)
    remove_correct_from_rand = hyperparameters.get('condition_on_area_remove_correct_from_rand', False)
    assert remove_correct_from_rand is False, 'implement it'

    batch: Batch = get_batch(batch_size=batch_size, seq_len=seq_len,
                             num_features=num_features, device=device,
                             hyperparameters=hyperparameters,
                             epoch=epoch, **kwargs)
    assert batch.style is None

    d = batch.x.shape[2]

    prob_correct = torch.rand(batch_size, d, device=device)
    correct_opt = torch.rand(batch_size, d, device=device) < prob_correct
    division_size = torch.randint(1, max_num_divisions + 1, (batch_size, d), device=device, dtype=torch.long)

    optima_inds = batch.target_y.argmax(0).squeeze() if maximize else batch.target_y.argmin(0).squeeze()  # batch_size
    optima = batch.x[optima_inds, torch.arange(batch_size, device=device)]  # shape: (batch_size, d)

    optima_sections = torch.min(torch.floor(optima * division_size).long(), division_size - 1)
    random_sections = torch.min(torch.floor(torch.rand(batch_size, batch.x.shape[2], device=device) * division_size).long(), division_size - 1)

    sections = torch.where(correct_opt, optima_sections, random_sections).float()  # shape: (batch_size, d)
    sections /= division_size.float()
    assert tuple(sections.shape) == (batch_size, d)
    batch.style = torch.stack([prob_correct, sections, sections + 1 / division_size], 2).view(batch_size, -1)  # shape: (batch_size, 3*d)

    return batch







