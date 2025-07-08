from copy import deepcopy

import torch


@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, hyperparameters, get_batch, **kwargs):
    hyperparameters = deepcopy(hyperparameters)
    sample_num_features = hyperparameters.pop("sample_num_features", True)

    if sample_num_features and kwargs.get("epoch", 1) > 0:  # don't sample on test batch
        num_features = torch.randint(1, num_features + 1, size=[1]).item()
    return get_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        num_features=num_features,
        hyperparameters=hyperparameters,
        **kwargs,
    )
