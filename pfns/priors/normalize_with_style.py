import torch

from ..utils import default_device


@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, get_batch, model, single_eval_pos, epoch, device=default_device, hyperparameters={}, **kwargs):
    if hyperparameters.get('normalize_x', False):
        uniform_float = torch.rand(tuple()).clamp(.1,1.).item()
        new_hyperparameters = {**hyperparameters, 'sampling': uniform_float * hyperparameters['sampling']}
    else:
        new_hyperparameters = hyperparameters
    returns = get_batch(batch_size=batch_size, seq_len=seq_len,
                              num_features=num_features, device=device,
                              hyperparameters=new_hyperparameters, model=model,
                              single_eval_pos=single_eval_pos, epoch=epoch, **kwargs)

    style = []

    if normalize_x_mode := hyperparameters.get('normalize_x', False):
        returns.x, mean_style, std_style = normalize_data_by_first_k(returns.x, single_eval_pos if normalize_x_mode == 'train' else len(returns.x))
        if hyperparameters.get('style_includes_mean_from_normalization', True) or normalize_x_mode == 'train':
             style.append(mean_style)
        style.append(std_style)

    if hyperparameters.get('normalize_y', False):
        returns.y, mean_style, std_style = normalize_data_by_first_k(returns.y, single_eval_pos)
        style += [mean_style, std_style]

    returns.style = torch.cat(style,1) if style else None
    return returns


def normalize_data_by_first_k(x, k):
    # x has shape seq_len, batch_size, num_features or seq_len, num_features
    # k is the number of elements to normalize by
    unsqueezed_x = False
    if len(x.shape) == 2:
        x.unsqueeze_(2)
        unsqueezed_x = True

    if k > 1:
        relevant_x = x[:k]
        mean_style = relevant_x.mean(0)
        std_style = relevant_x.std(0)
        x = (x - relevant_x.mean(0, keepdim=True)) / relevant_x.std(0, keepdim=True)
    elif k == 1:
        mean_style = x[0]
        std_style = torch.ones_like(x[0])
        x = (x - x[0])
    else: # it is 0
        mean_style = torch.zeros_like(x[0])
        std_style = torch.ones_like(x[0])

    if unsqueezed_x:
        x.squeeze_(2)

    return x, mean_style, std_style

