import contextlib
import math

import scipy
import torch

from pfns.model import bar_distribution
from sklearn.preprocessing import PowerTransformer

from ..utils import to_tensor


def log01(x, eps=0.0000001, input_between_zero_and_one=False):
    logx = torch.log(x + eps)
    if input_between_zero_and_one:
        return (logx - math.log(eps)) / (math.log(1 + eps) - math.log(eps))
    return (logx - logx.min(0)[0]) / (logx.max(0)[0] - logx.min(0)[0])


def log01_batch(x, eps=0.0000001, input_between_zero_and_one=False):
    x = x.repeat(1, x.shape[-1] + 1, 1)
    for b in range(x.shape[-1]):
        x[:, b, b] = log01(
            x[:, b, b],
            eps=eps,
            input_between_zero_and_one=input_between_zero_and_one,
        )
    return x


def lognormed_batch(x, eval_pos, eps=0.0000001):
    x = x.repeat(1, x.shape[-1] + 1, 1)
    for b in range(x.shape[-1]):
        logx = torch.log(x[:, b, b] + eps)
        x[:, b, b] = (logx - logx[:eval_pos].mean(0)) / logx[:eval_pos].std(0)
    return x


def _rank_transform(x_train, x):
    assert len(x_train.shape) == len(x.shape) == 1
    relative_to = torch.cat(
        (
            torch.zeros_like(x_train[:1]),
            x_train.unique(
                sorted=True,
            ),
            torch.ones_like(x_train[-1:]),
        ),
        -1,
    )
    higher_comparison = (relative_to < x[..., None]).sum(-1).clamp(min=1)
    pos_inside_interval = (x - relative_to[higher_comparison - 1]) / (
        relative_to[higher_comparison] - relative_to[higher_comparison - 1]
    )
    x_transformed = higher_comparison - 1 + pos_inside_interval
    return x_transformed / (len(relative_to) - 1.0)


def rank_transform(x_train, x):
    assert x.shape[1] == x_train.shape[1], f"{x.shape=} and {x_train.shape=}"
    # make sure everything is between 0 and 1
    assert (x_train >= 0.0).all() and (x_train <= 1.0).all(), f"{x_train=}"
    assert (x >= 0.0).all() and (x <= 1.0).all(), f"{x=}"
    return_x = x.clone()
    for feature_dim in range(x.shape[1]):
        return_x[:, feature_dim] = _rank_transform(
            x_train[:, feature_dim], x[:, feature_dim]
        )
    return return_x


def general_power_transform(x_train, x_apply, eps, less_safe=False):
    if eps > 0:
        try:
            pt = PowerTransformer(method="box-cox")
            pt.fit(x_train.cpu() + eps)
            x_out = torch.tensor(
                pt.transform(x_apply.cpu() + eps),
                dtype=x_apply.dtype,
                device=x_apply.device,
            )
        except ValueError as e:
            print(e)
            x_out = x_apply - x_train.mean(0)
    else:
        pt = PowerTransformer(method="yeo-johnson")
        if not less_safe and (x_train.std() > 1_000 or x_train.mean().abs() > 1_000):
            x_apply = (x_apply - x_train.mean(0)) / x_train.std(0)
            x_train = (x_train - x_train.mean(0)) / x_train.std(0)
            print("inputs are LAARGEe, normalizing them")
        try:
            pt.fit(x_train.cpu().double())
        except ValueError as e:
            print("caught this errrr", e)
            if less_safe:
                x_train = (x_train - x_train.mean(0)) / x_train.std(0)
                x_apply = (x_apply - x_train.mean(0)) / x_train.std(0)
            else:
                x_train = x_train - x_train.mean(0)
                x_apply = x_apply - x_train.mean(0)
            pt.fit(x_train.cpu().double())
        x_out = torch.tensor(
            pt.transform(x_apply.cpu()),
            dtype=x_apply.dtype,
            device=x_apply.device,
        )
    if torch.isnan(x_out).any() or torch.isinf(x_out).any():
        print("WARNING: power transform failed")
        print(f"{x_train=} and {x_apply=}")
        x_out = x_apply - x_train.mean(0)
    return x_out


# @torch.inference_mode()
def general_acq_function(
    model,
    x_given,
    y_given,
    x_eval,
    apply_power_transform=True,
    rand_sample=False,
    znormalize=False,
    pre_normalize=False,
    pre_znormalize=False,
    predicted_mean_fbest=False,
    input_znormalize=False,
    max_dataset_size=10_000,
    remove_features_with_one_value_only=False,
    return_actual_ei=False,
    acq_function="ei",
    ucb_rest_prob=0.05,
    ensemble_log_dims=False,
    ensemble_type="mean_probs",  # in ('mean_probs', 'max_acq')
    input_power_transform=False,
    power_transform_eps=0.0,
    input_power_transform_eps=0.0,
    input_rank_transform=False,
    ensemble_input_rank_transform=False,
    ensemble_power_transform=False,
    ensemble_feature_rotation=False,
    style=None,
    outlier_stretching_interval=0.0,
    verbose=False,
    unsafe_power_transform=False,
):
    """
    Differences to HEBO:
        - The noise can't be set in the same way, as it depends on the tuning of HPs via VI.
        - Log EI and PI are always used directly instead of using the approximation.

    This is a stochastic function, relying on torch.randn

    :param model:
    :param x_given: torch.Tensor of shape (N, D)
    :param y_given: torch.Tensor of shape (N, 1) or (N,)
    :param x_eval: torch.Tensor of shape (M, D)
    :param kappa:
    :param eps:
    :return:
    """
    assert ensemble_type in ("mean_probs", "max_acq")
    if rand_sample is not False and (
        len(x_given) == 0
        or (
            (1 + x_given.shape[1] if rand_sample is None else max(2, rand_sample))
            > x_given.shape[0]
        )
    ):
        print("rando")
        return torch.zeros_like(x_eval[:, 0])  # torch.randperm(x_eval.shape[0])[0]
    y_given = y_given.reshape(-1)
    assert len(y_given) == len(x_given)
    if apply_power_transform:
        if pre_normalize:
            y_normed = y_given / y_given.std()
            if not torch.isinf(y_normed).any() and not torch.isnan(y_normed).any():
                y_given = y_normed
        elif pre_znormalize:
            y_znormed = (y_given - y_given.mean()) / y_given.std()
            if not torch.isinf(y_znormed).any() and not torch.isnan(y_znormed).any():
                y_given = y_znormed
        y_given = general_power_transform(
            y_given.unsqueeze(1),
            y_given.unsqueeze(1),
            power_transform_eps,
            less_safe=unsafe_power_transform,
        ).squeeze(1)
        if verbose:
            print(f"{y_given=}")
        # y_given = torch.tensor(power_transform(y_given.cpu().unsqueeze(1), method='yeo-johnson', standardize=znormalize), device=y_given.device, dtype=y_given.dtype,).squeeze(1)
    y_given_std = torch.tensor(1.0, device=y_given.device, dtype=y_given.dtype)
    if znormalize and not apply_power_transform:
        if len(y_given) > 1:
            y_given_std = y_given.std()
        y_given_mean = y_given.mean()
        y_given = (y_given - y_given_mean) / y_given_std

    if remove_features_with_one_value_only:
        x_all = torch.cat([x_given, x_eval], dim=0)
        only_one_value_feature = (
            torch.tensor(
                [len(torch.unique(x_all[:, i])) for i in range(x_all.shape[1])]
            )
            == 1
        )
        x_given = x_given[:, ~only_one_value_feature]
        x_eval = x_eval[:, ~only_one_value_feature]

    if outlier_stretching_interval > 0.0:
        tx = torch.cat([x_given, x_eval], dim=0)
        m = outlier_stretching_interval
        eps = 1e-10
        small_values = (tx < m) & (tx > 0.0)
        tx[small_values] = (
            m
            * (torch.log(tx[small_values] + eps) - math.log(eps))
            / (math.log(m + eps) - math.log(eps))
        )

        large_values = (tx > 1.0 - m) & (tx < 1.0)
        tx[large_values] = 1.0 - m * (
            torch.log(1 - tx[large_values] + eps) - math.log(eps)
        ) / (math.log(m + eps) - math.log(eps))
        x_given = tx[: len(x_given)]
        x_eval = tx[len(x_given) :]

    if input_znormalize:  # implementation that relies on the test set, too...
        std = x_given.std(dim=0)
        std[std == 0.0] = 1.0
        mean = x_given.mean(dim=0)
        x_given = (x_given - mean) / std
        x_eval = (x_eval - mean) / std

    if input_power_transform:
        x_given = general_power_transform(x_given, x_given, input_power_transform_eps)
        x_eval = general_power_transform(x_given, x_eval, input_power_transform_eps)

    if (
        input_rank_transform is True or input_rank_transform == "full"
    ):  # uses test set x statistics...
        x_all = torch.cat((x_given, x_eval), dim=0)
        for feature_dim in range(x_all.shape[-1]):
            uniques = torch.sort(torch.unique(x_all[..., feature_dim])).values
            x_eval[..., feature_dim] = torch.searchsorted(
                uniques, x_eval[..., feature_dim]
            ).float() / (len(uniques) - 1)
            x_given[..., feature_dim] = torch.searchsorted(
                uniques, x_given[..., feature_dim]
            ).float() / (len(uniques) - 1)
    elif input_rank_transform is False:
        pass
    elif input_rank_transform == "train":
        x_given = rank_transform(x_given, x_given)
        x_eval = rank_transform(x_given, x_eval)
    elif input_rank_transform.startswith("train"):
        likelihood = float(input_rank_transform.split("_")[-1])
        if torch.rand(1).item() < likelihood:
            print("rank transform")
            x_given = rank_transform(x_given, x_given)
            x_eval = rank_transform(x_given, x_eval)
    else:
        raise NotImplementedError

    # compute logits
    criterion: bar_distribution.BarDistribution = model.criterion
    x_predict = torch.cat([x_given, x_eval], dim=0)

    logits_list = []
    for x_feed in torch.split(x_predict, max_dataset_size, dim=0):
        x_full_feed = torch.cat([x_given, x_feed], dim=0).unsqueeze(1)
        y_full_feed = y_given.unsqueeze(1)
        if ensemble_log_dims == "01":
            x_full_feed = log01_batch(x_full_feed)
        elif ensemble_log_dims == "global01" or ensemble_log_dims is True:
            x_full_feed = log01_batch(x_full_feed, input_between_zero_and_one=True)
        elif ensemble_log_dims == "01-10":
            x_full_feed = torch.cat(
                (
                    log01_batch(x_full_feed)[:, :-1],
                    log01_batch(1.0 - x_full_feed),
                ),
                1,
            )
        elif ensemble_log_dims == "norm":
            x_full_feed = lognormed_batch(x_full_feed, len(x_given))
        elif ensemble_log_dims is not False:
            raise NotImplementedError

        if ensemble_feature_rotation:
            x_full_feed = torch.cat(
                [
                    x_full_feed[
                        :,
                        :,
                        (i + torch.arange(x_full_feed.shape[2])) % x_full_feed.shape[2],
                    ]
                    for i in range(x_full_feed.shape[2])
                ],
                dim=1,
            )

        if (
            ensemble_input_rank_transform == "train"
            or ensemble_input_rank_transform is True
        ):
            x_full_feed = torch.cat(
                [
                    rank_transform(x_given, x_full_feed[:, i, :])[:, None]
                    for i in range(x_full_feed.shape[1])
                ]
                + [x_full_feed],
                dim=1,
            )

        if ensemble_power_transform:
            assert apply_power_transform is False
            y_full_feed = torch.cat(
                (
                    general_power_transform(
                        y_full_feed, y_full_feed, power_transform_eps
                    ),
                    y_full_feed,
                ),
                dim=1,
            )

        if style is not None:
            if callable(style):
                style = style()

            if isinstance(style, torch.Tensor):
                style = style.to(x_full_feed.device)
            else:
                style = (
                    torch.tensor(style, device=x_full_feed.device)
                    .view(1, 1)
                    .repeat(x_full_feed.shape[1], 1)
                )

        logits = model(
            (
                style,
                x_full_feed.repeat_interleave(dim=1, repeats=y_full_feed.shape[1]),
                y_full_feed.repeat(1, x_full_feed.shape[1]),
            ),
            single_eval_pos=len(x_given),
        )
        if ensemble_type == "mean_probs":
            logits = (
                logits.softmax(-1).mean(1, keepdim=True).log_()
            )  # (num given + num eval, 1, num buckets)

        logits_list.append(logits)  # (< max_dataset_size, 1 , num_buckets)
    logits = torch.cat(
        logits_list, dim=0
    )  # (num given + num eval, 1 or (num_features+1), num buckets)
    del logits_list, x_full_feed
    if torch.isnan(logits).any():
        print("nan logits")
        print(f"y_given: {y_given}, x_given: {x_given}, x_eval: {x_eval}")
        print(f"logits: {logits}")
        return torch.zeros_like(x_eval[:, 0])

    # logits = model((torch.cat([x_given, x_given, x_eval], dim=0).unsqueeze(1),
    #               torch.cat([y_given, torch.zeros(len(x_eval)+len(x_given), device=y_given.device)], dim=0).unsqueeze(1)),
    #               single_eval_pos=len(x_given))[:,0] # (N + M, num_buckets)
    logits_given = logits[: len(x_given)]
    logits_eval = logits[len(x_given) :]

    # tau = criterion.mean(logits_given)[torch.argmax(y_given)] # predicted mean at the best y
    if predicted_mean_fbest:
        tau = criterion.mean(logits_given)[torch.argmax(y_given)].squeeze(0)
    else:
        tau = torch.max(y_given)
    # log_ei = torch.stack([criterion.ei(logits_eval[:,i], noisy_best_f[i]).log() for i in range(len(logits_eval))],0)

    def acq_ensembling(acq_values):  # (points, ensemble dim)
        return acq_values.max(1).values

    if isinstance(acq_function, (dict, list)):
        acq_function = acq_function[style]

    if acq_function == "ei":
        acq_value = acq_ensembling(criterion.ei(logits_eval, tau))
    elif acq_function == "ei_or_rand":
        if torch.rand(1).item() < 0.5:
            acq_value = torch.rand(len(x_eval))
        else:
            acq_value = acq_ensembling(criterion.ei(logits_eval, tau))
    elif acq_function == "pi":
        acq_value = acq_ensembling(criterion.pi(logits_eval, tau))
    elif acq_function == "ucb":
        acq_function = criterion.ucb
        if ucb_rest_prob is not None:
            acq_function = lambda *args: criterion.ucb(*args, rest_prob=ucb_rest_prob)  # noqa: E731
        acq_value = acq_ensembling(acq_function(logits_eval, tau))
    elif acq_function == "mean":
        acq_value = acq_ensembling(criterion.mean(logits_eval))
    elif acq_function.startswith("hebo"):
        noise, upsi, delta, eps = (float(v) for v in acq_function.split("_")[1:])
        noise = y_given_std * math.sqrt(2 * noise)
        kappa = math.sqrt(
            upsi
            * 2
            * (
                (2.0 + x_given.shape[1] / 2.0) * math.log(max(1, len(x_given)))
                + math.log(3 * math.pi**2 / (3 * delta))
            )
        )
        rest_prob = 1.0 - 0.5 * (
            1 + torch.erf(torch.tensor(kappa / math.sqrt(2), device=logits.device))
        )
        ucb = (
            acq_ensembling(criterion.ucb(logits_eval, None, rest_prob=rest_prob))
            + torch.randn(len(logits_eval), device=logits_eval.device) * noise
        )
        noisy_best_f = (
            tau
            + eps
            + noise
            * torch.randn(len(logits_eval), device=logits_eval.device)[:, None].repeat(
                1, logits_eval.shape[1]
            )
        )

        log_pi = acq_ensembling(criterion.pi(logits_eval, noisy_best_f).log())
        # log_ei = torch.stack([criterion.ei(logits_eval[:,i], noisy_best_f[i]).log() for i in range(len(logits_eval))],0)
        log_ei = acq_ensembling(criterion.ei(logits_eval, noisy_best_f).log())

        acq_values = torch.stack([ucb, log_ei, log_pi], dim=1)

        def is_pareto_efficient(costs):
            """
            Find the pareto-efficient points
            :param costs: An (n_points, n_costs) array
            :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
            """
            is_efficient = torch.ones(costs.shape[0], dtype=bool, device=costs.device)
            for i, c in enumerate(costs):
                if is_efficient[i]:
                    is_efficient[is_efficient.clone()] = (costs[is_efficient] < c).any(
                        1
                    )  # Keep any point with a lower cost
                    is_efficient[i] = True  # And keep self
            return is_efficient

        acq_value = is_pareto_efficient(-acq_values)
    else:
        raise ValueError(f"Unknown acquisition function: {acq_function}")

    max_acq = acq_value.max()

    return acq_value if return_actual_ei else (acq_value == max_acq)


def simple_ei_acquisition(
    model,
    x_given,
    y_given,
    x_eval,
    return_actual_ei=False,
    y_eval_for_finetuning=None,
    do_rand_search=False,
):
    """
    Simple Expected Improvement (EI) acquisition function using the provided model's criterion.

    :param model: Model object with a criterion that has an ei method
    :param x_given: torch.Tensor of shape (..., N, D) where ... is optional batch dimensions
    :param y_given: torch.Tensor of shape (..., N,) where ... is optional batch dimensions
    :param x_eval: torch.Tensor of shape (..., M, D) where ... is optional batch dimensions
    :param return_actual_ei: If True, return the actual EI values; otherwise, return a boolean mask of the max EI
    :param y_eval_for_finetuning: Optional tensor for finetuning the model
    :param do_rand_search: If True, randomly select from points with high acquisition values
    :param kwargs: Additional keyword arguments
    :return: torch.Tensor of shape (..., M) representing the EI values or a boolean mask
    """
    if len(x_eval.shape) == len(y_given.shape):
        y_given = y_given.squeeze(-1)

    extras = {}

    criterion = model.criterion
    tau = y_given.max(
        dim=-1, keepdim=True
    ).values  # Assuming tau is the maximum of y_given for EI calculation
    # Reshape x_given, y_given, and x_eval to combine leading batch dimensions into the second last dimension
    original_shape = x_eval.shape
    x_given = x_given.view(-1, x_given.shape[-2], x_given.shape[-1])
    y_given = y_given.view(-1, y_given.shape[-1])
    x_eval = x_eval.view(-1, x_eval.shape[-2], x_eval.shape[-1])

    # Forward pass on logits
    x_combined = torch.concat((x_given.transpose(0, 1), x_eval.transpose(0, 1)), 0)
    y_transposed = y_given.transpose(0, 1)
    # print(f"x_combined shape: {x_combined.shape}, y_transposed shape: {y_transposed.shape}")
    with torch.set_grad_enabled(y_eval_for_finetuning is not None):
        logits = model(
            (x_combined, y_transposed), single_eval_pos=y_given.shape[1]
        ).transpose(0, 1)
        if y_eval_for_finetuning is not None:
            losses = criterion(logits, y_eval_for_finetuning).transpose(0, 1)
            mean_loss = losses.mean()
            mean_loss.backward()
            extras["loss"] = mean_loss.item()

    # Undo the reshaping to restore original shape
    logits = logits.view(*original_shape[:-2], logits.shape[-2], logits.shape[-1])
    tau = tau.repeat(1, logits.shape[1])
    ei_values = criterion.ei(logits, tau)

    if do_rand_search:
        ei_values[:] = 0.0

    if return_actual_ei:
        return ei_values, extras

    max_ei = ei_values.max(dim=-1, keepdim=True).values
    return ei_values == max_ei, extras


def optimize_acq(
    model,
    known_x,
    known_y,
    num_grad_steps=10,
    num_random_samples=100,
    lr=0.01,
    **kwargs,
):
    """
    intervals are assumed to be between 0 and 1
    only works with ei
    recommended extra kwarg: ensemble_input_rank_transform=='train'

    :param model: model to optimize, should already handle different num_features with its encoder
    You can add this simply with `model.encoder = encoders.VariableNumFeaturesEncoder(model.encoder, model.encoder.num_features)`
    :param known_x: (N, num_features)
    :param known_y: (N,)
    :param num_grad_steps: int
    :param num_random_samples: int
    :param lr: float
    :param kwargs: will be given to `general_acq_function`
    :return:
    """
    x_eval = torch.rand(num_random_samples, known_x.shape[1]).requires_grad_(True)
    opt = torch.optim.Adam(params=[x_eval], lr=lr)
    best_acq, best_x = -float("inf"), x_eval[0].detach()
    for _grad_step in range(num_grad_steps):
        acq = general_acq_function(
            model, known_x, known_y, x_eval, return_actual_ei=True, **kwargs
        )
        max_acq = acq.detach().max().item()
        if max_acq > best_acq:
            best_x = x_eval[acq.argmax()].detach()
            best_acq = max_acq

        (-acq.mean()).backward()
        assert (x_eval.grad != 0.0).any()
        if torch.isfinite(x_eval.grad).all():
            opt.step()
        opt.zero_grad()
        with torch.no_grad():
            x_eval.clamp_(min=0.0, max=1.0)

    return best_x


def optimize_acq_w_lbfgs(
    model,
    known_x,
    known_y,
    num_grad_steps=15_000,
    num_candidates=100,
    pre_sample_size=100_000,
    device="cpu",
    verbose=False,
    dims_wo_gradient_opt=tuple(),
    rand_sample_func=None,
    **kwargs,
):
    """
    intervals are assumed to be between 0 and 1
    only works with deterministic acq
    recommended extra kwarg: ensemble_input_rank_transform=='train'

    :param model: model to optimize, should already handle different num_features with its encoder
    You can add this simply with `model.encoder = encoders.VariableNumFeaturesEncoder(model.encoder, model.encoder.num_features)`
    :param known_x: (N, num_features)
    :param known_y: (N,)
    :param num_grad_steps: int: how many steps to take inside of scipy, can be left high, as it stops most of the time automatically early
    :param num_candidates: int: how many candidates to optimize with LBFGS, increases costs when higher
    :param pre_sample_size: int: how many settings to try first with a random search, before optimizing the best with grads
    :param dims_wo_gradient_opt: int: which dimensions to not optimize with gradients, but with random search only
    :param rand_sample_func: function: how to sample random points, should be a function that takes a number of samples and returns a tensor
    For example `lambda n: torch.rand(n, known_x.shape[1])`.
    :param kwargs: will be given to `general_acq_function`
    :return:
    """
    num_features = known_x.shape[1]
    dims_w_gradient_opt = sorted(set(range(num_features)) - set(dims_wo_gradient_opt))
    known_x = known_x.to(device)
    known_y = known_y.to(device)
    pre_sample_size = max(pre_sample_size, num_candidates)
    rand_sample_func = rand_sample_func or (
        lambda n: torch.rand(n, num_features, device=device)
    )
    if len(known_x) < pre_sample_size:
        x_initial = torch.cat(
            (
                rand_sample_func(pre_sample_size - len(known_x)).to(device),
                known_x,
            ),
            0,
        )
    else:
        x_initial = rand_sample_func(pre_sample_size)
    x_initial = x_initial.clamp(min=0.0, max=1.0)
    x_initial_all = x_initial
    model.to(device)

    with torch.no_grad():
        acq = general_acq_function(
            model,
            known_x,
            known_y,
            x_initial.to(device),
            return_actual_ei=True,
            **kwargs,
        )
        if verbose:
            import matplotlib.pyplot as plt

            if x_initial.shape[1] == 2:
                plt.title("initial acq values, red -> blue")
                plt.scatter(
                    x_initial[:, 0][:100],
                    x_initial[:, 1][:100],
                    c=acq.cpu().numpy()[:100],
                    cmap="RdBu",
                )
        x_initial = x_initial[
            acq.argsort(descending=True)[:num_candidates].cpu()
        ].detach()  # num_candidates x num_features

    x_initial_all_ei = acq.cpu().detach()

    def opt_f(x):
        x_eval = (
            torch.tensor(x)
            .view(-1, len(dims_w_gradient_opt))
            .float()
            .to(device)
            .requires_grad_(True)
        )
        x_eval_new = x_initial.clone().detach().to(device)
        x_eval_new[:, dims_w_gradient_opt] = x_eval

        assert x_eval_new.requires_grad
        assert not torch.isnan(x_eval_new).any()
        model.requires_grad_(False)
        acq = general_acq_function(
            model,
            known_x,
            known_y,
            x_eval_new,
            return_actual_ei=True,
            **kwargs,
        )
        neg_mean_acq = -acq.mean()
        neg_mean_acq.backward()
        # print(neg_mean_acq.detach().numpy(), x_eval.grad.detach().view(*x.shape).numpy())
        with torch.no_grad():
            x_eval.grad[x_eval.grad != x_eval.grad] = 0.0
        return (
            neg_mean_acq.detach().cpu().to(torch.float64).numpy(),
            x_eval.grad.detach().view(*x.shape).cpu().to(torch.float64).numpy(),
        )

    # Optimize best candidates with LBFGS
    if num_grad_steps > 0 and len(dims_w_gradient_opt) > 0:
        # the columns not in dims_wo_gradient_opt will be optimized with gradients
        x_initial_for_gradient_opt = (
            x_initial[:, dims_w_gradient_opt].detach().cpu().flatten().numpy()
        )  # x_initial.cpu().flatten().numpy()
        res = scipy.optimize.minimize(
            opt_f,
            x_initial_for_gradient_opt,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0, 1)] * x_initial_for_gradient_opt.size,
            options={"maxiter": num_grad_steps},
        )
        results = x_initial.cpu()
        results[:, dims_w_gradient_opt] = (
            torch.tensor(res.x).float().view(-1, len(dims_w_gradient_opt))
        )

    else:
        results = x_initial.cpu()

    results = results.clamp(min=0.0, max=1.0)

    # Recalculate the acq values for the best candidates
    with torch.no_grad():
        acq = general_acq_function(
            model,
            known_x,
            known_y,
            results.to(device),
            return_actual_ei=True,
            verbose=verbose,
            **kwargs,
        )
        # print(acq)
        if verbose:
            import matplotlib.pyplot as plt
            from scipy.stats import rankdata

            if results.shape[1] == 2:
                plt.scatter(
                    results[:, 0],
                    results[:, 1],
                    c=rankdata(acq.cpu().numpy()),
                    marker="x",
                    cmap="RdBu",
                )
                plt.show()
        best_x = results[acq.argmax().item()].detach()

    acq_order = acq.argsort(descending=True).cpu()
    all_order = x_initial_all_ei.argsort(descending=True).cpu()

    return (
        best_x.detach(),
        results[acq_order].detach(),
        acq.cpu()[acq_order].detach(),
        x_initial_all.cpu()[all_order].detach(),
        x_initial_all_ei.cpu()[all_order].detach(),
    )


class TransformerBOMethod:
    def __init__(
        self,
        model,
        acq_f=general_acq_function,
        device="cpu",
        fit_encoder=None,
        **kwargs,
    ):
        print(kwargs)
        self.model = model
        self.device = device
        self.kwargs = kwargs
        self.acq_function = acq_f
        self.fit_encoder = fit_encoder

    @torch.no_grad()
    def observe_and_suggest(
        self,
        X_obs,
        y_obs,
        X_pen,
        return_actual_ei=False,
        y_pen_for_finetuning=None,
    ):
        # assert X_pen is not None
        # assumptions about X_obs and X_pen:
        # X_obs is a numpy array of shape (batch_size, n_samples, n_features) or (n_samples, n_features)
        # y_obs is a numpy array of shape (batch_size, n_samples) or (n_samples,), between 0 and 1
        # X_pen is a numpy array of shape (batch_size, n_samples_left, n_features) or (n_samples_left, n_features)

        # Ensure acq_function is the batched simple_ei_acquisition

        # Convert inputs to tensors and handle optional batch dimension
        X_obs = to_tensor(X_obs, device=self.device).to(torch.float32)
        y_obs = to_tensor(y_obs, device=self.device).to(torch.float32)
        X_pen = to_tensor(X_pen, device=self.device).to(torch.float32)
        # print(X_obs.shape, y_obs.shape, X_pen.shape)

        no_batch_use = X_obs.dim() == 2
        if no_batch_use:
            X_obs = X_obs.unsqueeze(0)
            y_obs = y_obs.unsqueeze(0)
            X_pen = X_pen.unsqueeze(0)
        else:
            assert (
                self.acq_function.__name__ == "simple_ei_acquisition"
            ), "acq_function must be simple_ei_acquisition"

        assert X_obs.size(1) == y_obs.size(
            1
        ), "make sure both X_obs and y_obs have the same length."

        self.model.to(self.device)

        if self.fit_encoder is not None:
            w = self.fit_encoder(self.model, X_obs, y_obs)
            X_obs = w(X_obs)
            X_pen = w(X_pen)

        kwargs = {**self.kwargs}
        if y_pen_for_finetuning is not None:
            kwargs["y_eval_for_finetuning"] = y_pen_for_finetuning

        with (
            torch.cuda.amp.autocast()
            if self.device[:3] != "cpu"
            else contextlib.nullcontext()
        ):
            acq_values, extras = self.acq_function(
                self.model,
                X_obs,
                y_obs,
                X_pen,
                return_actual_ei=return_actual_ei,
                **kwargs,
            )
            acq_values = acq_values.cpu().clone()
            acq_mask = acq_values == acq_values.max(dim=1, keepdim=True)[0]

        possible_next = [torch.arange(X_pen.size(1))[mask] for mask in acq_mask]
        possible_next = [
            pn if len(pn) > 0 else torch.arange(X_pen.size(1)) for pn in possible_next
        ]

        r = [pn[torch.randperm(len(pn))[0]].cpu().item() for pn in possible_next]

        if no_batch_use:
            r = r[0]
            acq_values = acq_values[0]

        if return_actual_ei:
            return r, acq_values, extras
        else:
            return r, extras
