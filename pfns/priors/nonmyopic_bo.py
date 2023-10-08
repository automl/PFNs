import torch

from .prior import Batch
from ..utils import default_device


loaded_models = {}

def get_model(model_name, device):
    if model_name not in loaded_models:
        import submitit
        group, index = model_name.split(':')
        ex = submitit.get_executor()
        model = ex.get_group(group)[int(index)].results()[0][2]
        model.to(device)
        loaded_models[model_name] = model
    return loaded_models[model_name]

@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, get_batch, model, single_eval_pos, epoch, device=default_device, hyperparameters=None, **kwargs):
    """
    Important Assumptions:
        'inf_batch_size', 'max_level', 'sample_only_one_level', 'eval_seq_len' and 'epochs_per_level' in hyperparameters

    You can train a new model, based on an old one to only sample from a single level.
    You specify `level_0_model` as a group:index string and the model will be loaded from the checkpoint.



    :param batch_size:
    :param seq_len:
    :param num_features:
    :param get_batch:
    :param model:
    :param single_eval_pos:
    :param epoch:
    :param device:
    :param hyperparameters:
    :param kwargs:
    :return:
    """
    if level_0_model := hyperparameters.get('level_0_model', None):
        assert hyperparameters['sample_only_one_level'], "level_0_model only makes sense if you sample only one level"
        assert hyperparameters['max_level'] == 1, "level_0_model only makes sense if you sample only one level"
        level_0_model = get_model(level_0_model, device)
        model = level_0_model

    # the level describes how many fantasized steps are possible. This starts at 0 for the first epochs.
    epochs_per_level = hyperparameters['epochs_per_level']
    share_predict_mean_distribution = hyperparameters.get('share_predict_mean_distribution', 0.)
    use_mean_prediction = share_predict_mean_distribution or\
                          (model.decoder_dict_once is not None and 'mean_prediction' in model.decoder_dict_once)
    num_evals = seq_len - single_eval_pos
    level = min(min(epoch // epochs_per_level, hyperparameters['max_level']), num_evals - 1)
    if level_0_model:
        level = 1
    eval_seq_len = hyperparameters['eval_seq_len']
    add_seq_len = 0 if use_mean_prediction else eval_seq_len
    long_seq_len = seq_len + add_seq_len

    if level_0_model:
        styles = torch.ones(batch_size, 1, device=device, dtype=torch.long)
    elif hyperparameters['sample_only_one_level']:
        styles = torch.randint(level + 1, (1, 1), device=device).repeat(batch_size, 1)  # styles are sorted :)
    else:
        styles = torch.randint(level + 1, (batch_size,1), device=device).sort(0).values # styles are sorted :)

    predict_mean_distribution = None
    if share_predict_mean_distribution:
        max_used_level = max(styles)
        # below code assumes epochs are base 0!
        share_of_training = epoch / epochs_per_level
        #print(share_of_training, (max_used_level + 1. - share_predict_mean_distribution), max_used_level, level, epoch)
        predict_mean_distribution = (share_of_training >= (max_used_level + 1. - share_predict_mean_distribution)) and (max_used_level < hyperparameters['max_level'])

    x, y, targets = [], [], []

    for considered_level in range(level+1):
        num_elements = (styles == considered_level).sum()
        if not num_elements:
            continue
        returns: Batch = get_batch(batch_size=num_elements, seq_len=long_seq_len,
                                   num_features=num_features, device=device,
                                   hyperparameters=hyperparameters, model=model,
                                   single_eval_pos=single_eval_pos, epoch=epoch,
                                   **kwargs)
        levels_x, levels_y, levels_targets = returns.x, returns.y, returns.target_y
        assert not returns.other_filled_attributes(), f"Unexpected filled attributes: {returns.other_filled_attributes()}"

        assert levels_y is levels_targets
        levels_targets = levels_targets.clone()
        if len(levels_y.shape) == 2:
            levels_y = levels_y.unsqueeze(2)
            levels_targets = levels_targets.unsqueeze(2)
        if considered_level > 0:

            feed_x = levels_x[:single_eval_pos + 1 + add_seq_len].repeat(1, num_evals, 1)
            feed_x[single_eval_pos, :] = levels_x[single_eval_pos:seq_len].reshape(-1, *levels_x.shape[2:])
            if not use_mean_prediction:
                feed_x[single_eval_pos + 1:] = levels_x[seq_len:].repeat(1, num_evals, 1)

            feed_y = levels_y[:single_eval_pos + 1 + add_seq_len].repeat(1, num_evals, 1)
            feed_y[single_eval_pos, :] = levels_y[single_eval_pos:seq_len].reshape(-1, *levels_y.shape[2:])
            if not use_mean_prediction:
                feed_y[single_eval_pos + 1:] = levels_y[seq_len:].repeat(1, num_evals, 1)

            model.eval()
            means = []
            for feed_x_b, feed_y_b in zip(torch.split(feed_x, hyperparameters['inf_batch_size'], dim=1),
                                          torch.split(feed_y, hyperparameters['inf_batch_size'], dim=1)):
                with torch.cuda.amp.autocast():
                    style = torch.zeros(feed_x_b.shape[1], 1, dtype=torch.int64, device=device) + considered_level - 1
                    if level_0_model is not None and level_0_model.style_encoder is None:
                        style = None
                    out = model(
                        (style, feed_x_b, feed_y_b),
                        single_eval_pos=single_eval_pos+1, only_return_standard_out=False
                    )
                    if isinstance(out, tuple):
                        output, once_output = out
                    else:
                        output = out
                        once_output = {}

                if once_output and 'mean_prediction' in once_output:
                    mean_pred_logits = once_output['mean_prediction'].float()
                    assert tuple(mean_pred_logits.shape) == (feed_x_b.shape[1], model.criterion.num_bars),\
                        f"{tuple(mean_pred_logits.shape)} vs {(feed_x_b.shape[1], model.criterion.num_bars)}"
                    means.append(model.criterion.icdf(mean_pred_logits, 1.-1./eval_seq_len))
                else:
                    logits = output['standard'].float()
                    means.append(model.criterion.mean(logits).max(0).values)
            means = torch.cat(means, 0)
            levels_targets_new = means.view(seq_len-single_eval_pos, *levels_y.shape[1:])
            levels_targets[single_eval_pos:seq_len] = levels_targets_new #- levels_targets_new.mean(0)
            model.train()

        levels_x = levels_x[:seq_len]
        levels_y = levels_y[:seq_len]
        levels_targets = levels_targets[:seq_len]

        x.append(levels_x)
        y.append(levels_y)
        targets.append(levels_targets)

    x = torch.cat(x, 1)
    # if predict_mean_distribution: print(f'predict mean dist in b, {epoch=}, {max_used_level=}')
    return Batch(x=x, y=torch.cat(y, 1), target_y=torch.cat(targets, 1), style=styles,
                 mean_prediction=predict_mean_distribution.item() if predict_mean_distribution is not None else None)










