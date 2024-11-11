# this is a wrapper prior that samples hyperparameters which are set to be ConfigSpace parameters
import math
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
    num_models = min(
        hyperparameters.get("num_hyperparameter_samples_per_batch", 1), batch_size
    )
    if num_models == -1:
        num_models = batch_size
    assert batch_size % num_models == 0, "batch_size must be a multiple of num_models"
    cs = create_configspace_from_hierarchical(hyperparameters)
    sub_batches = []
    sub_hps = []
    for i in range(num_models):
        cs_sample = cs.sample_configuration()
        hyperparameters_sample = fill_in_configsample(hyperparameters, cs_sample)
        sub_batch = get_batch(
            batch_size // num_models,
            *args,
            hyperparameters=hyperparameters_sample,
            **kwargs,
        )
        sub_batches.append(sub_batch)
        sub_hps.append(hyperparameters_sample)

    # concat x, y, target (and maybe style)
    # assert 3 <= len(sub_batch) <= 4
    # return tuple(torch.cat([sb[i] for sb in sub_batches], dim=(0 if i == 3 else 1)) for i in range(len(sub_batch)))
    assert all(
        not b.other_filled_attributes(set_of_attributes=("x", "y", "target_y"))
        for b in sub_batches
    ), f"Batch {[b.other_filled_attributes(set_of_attributes=('x', 'y', 'target_y')) for b in sub_batches if b.other_filled_attributes(set_of_attributes=('x', 'y', 'target_y'))]} has other attributes filled in."

    batch = Batch(
        x=torch.cat([b.x for b in sub_batches], dim=1),
        y=torch.cat([b.y for b in sub_batches], dim=1),
        target_y=torch.cat([b.target_y for b in sub_batches], dim=1),
        # log_likelihoods=torch.cat([b.log_likelihoods for b in sub_batches], dim=0) if sub_batches[0].log_likelihoods is not None else None,
    )

    if hps_as_style := hyperparameters.get(
        "hyperparameter_sampling_add_hps_to_style", []
    ):
        for i, b in enumerate(sub_batches):
            assert b.style is None
            b.style = torch.zeros((b.x.shape[1], len(hps_as_style)))
            for j, hp in enumerate(hps_as_style):
                hp_value = sub_hps[i][hp]
                assert isinstance(hp_value, (int, float, bool))
                b.style[:, j] = float(hp_value)
        batch.style = torch.cat([b.style for b in sub_batches], dim=0)
    return batch


class HyperparameterNormalizer(torch.nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.hyperparameters = hyperparameters

    def forward(self, x):
        to_be_encoded_hyperparameters = self.hyperparameters[
            "hyperparameter_sampling_add_hps_to_style"
        ]
        encoded_x = torch.zeros(
            (x.shape[0], len(to_be_encoded_hyperparameters)),
            device=x.device,
            dtype=x.dtype,
        )
        for i, hp in enumerate(to_be_encoded_hyperparameters):
            hp_value = self.hyperparameters[hp]
            if isinstance(
                hp_value,
                (CSH.UniformFloatHyperparameter, CSH.UniformIntegerHyperparameter),
            ):
                min_value = hp_value.lower
                max_value = hp_value.upper
                if hp_value.log:
                    encoded_x[:, i] = (torch.log(x[:, i]) - math.log(min_value)) / (
                        math.log(max_value) - math.log(min_value)
                    )
                else:
                    encoded_x[:, i] = (x[:, i] - min_value) / (max_value - min_value)
            else:
                raise NotImplementedError(
                    f"Hyperparameter type {type(hp_value)} not implemented"
                )
        # now encoded x is in [0, 1]
        # we need to normalize it to [-1, 1]
        encoded_x = 2 * encoded_x - 1
        return encoded_x
