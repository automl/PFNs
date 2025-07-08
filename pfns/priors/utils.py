import inspect
import random
import time
import types
from functools import partial

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import torch
import torch.utils.data  # Added for get_worker_info
from torch import nn

from ..utils import normalize_data

# only for pickle backwards compatibility
from .data_loading import (  # noqa: F401
    _BatchedIterableDataset,
    StandardDataLoader,
    worker_init_fn,
)
from .prior import Batch


@torch.no_grad()
def zero_time_get_batch(batch_size, seq_len, num_features, device="cpu", **kwargs):
    y = torch.rand(batch_size, seq_len, 1, device=device)
    return Batch(
        x=torch.rand(batch_size, seq_len, num_features, device=device),
        y=y,
        target_y=y.clone(),
    )


def plot_features(data, targets, fig=None, categorical=True, plot_diagonal=True):
    import seaborn as sns

    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

    fig2 = fig if fig else plt.figure(figsize=(8, 8))
    spec2 = gridspec.GridSpec(ncols=data.shape[1], nrows=data.shape[1], figure=fig2)
    for d in range(0, data.shape[1]):
        for d2 in range(0, data.shape[1]):
            if d > d2:
                continue
            sub_ax = fig2.add_subplot(spec2[d, d2])
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])
            if d == d2:
                if plot_diagonal:
                    if categorical:
                        sns.histplot(
                            data[:, d],
                            hue=targets[:],
                            ax=sub_ax,
                            legend=False,
                            palette="deep",
                        )
                    else:
                        sns.histplot(data[:, d], ax=sub_ax, legend=False)
                sub_ax.set(ylabel=None)
            else:
                if categorical:
                    sns.scatterplot(
                        x=data[:, d],
                        y=data[:, d2],
                        hue=targets[:],
                        legend=False,
                        palette="deep",
                    )
                else:
                    sns.scatterplot(
                        x=data[:, d],
                        y=data[:, d2],
                        hue=targets[:],
                        legend=False,
                    )
                # plt.scatter(data[:, d], data[:, d2],
                #               c=targets[:])
            # sub_ax.get_xaxis().set_ticks([])
            # sub_ax.get_yaxis().set_ticks([])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig2.show()


def plot_prior(prior, samples=1000, buckets=50):
    s = np.array([prior() for _ in range(0, samples)])
    count, bins, ignored = plt.hist(s, buckets, density=True)
    print(s.min())
    plt.show()


def trunc_norm_sampler_f(mu, sigma):
    def sampler():
        return stats.truncnorm(
            (0 - mu) / sigma, (1000000 - mu) / sigma, loc=mu, scale=sigma
        ).rvs(1)[0]

    return sampler


def beta_sampler_f(a, b):
    def sampler():
        return np.random.beta(a, b)

    return sampler


def gamma_sampler_f(a, b):
    def sampler():
        return np.random.gamma(a, b)

    return sampler


def uniform_sampler_f(a, b):
    def sampler():
        return np.random.uniform(a, b)

    return sampler


def uniform_int_sampler_f(a, b):
    def sampler():
        return round(np.random.uniform(a, b))

    return sampler


def zipf_sampler_f(a, b, c):
    x = np.arange(b, c)
    weights = x ** (-a)
    weights /= weights.sum()
    return lambda: stats.rv_discrete(name="bounded_zipf", values=(x, weights)).rvs(1)


def scaled_beta_sampler_f(a, b, scale, minimum):
    def sampler():
        return minimum + round(beta_sampler_f(a, b)() * (scale - minimum))

    return sampler


def normalize_by_used_features_f(
    x, num_features_used, num_features, normalize_with_sqrt=False
):
    if normalize_with_sqrt:
        return x / (num_features_used / num_features) ** (1 / 2)
    return x / (num_features_used / num_features)


def order_by_y(x, y):
    order = torch.argsort(y if random.randint(0, 1) else -y, dim=0)[:, 0, 0]
    order = order.reshape(2, -1).transpose(0, 1).reshape(-1)  # .reshape(seq_len)
    x = x[
        order
    ]  # .reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).reshape(seq_len, 1, -1)
    y = y[order]  # .reshape(2, -1).transpose(0, 1).reshape(-1).reshape(seq_len, 1, -1)

    return x, y


def randomize_classes(x, num_classes):
    classes = torch.arange(0, num_classes, device=x.device)
    random_classes = torch.randperm(num_classes, device=x.device).type(x.type())
    x = ((x.unsqueeze(-1) == classes) * random_classes).sum(-1)
    return x


default_zipf_sampler = zipf_sampler_f(0.8, 1, 10)


class CategoricalActivation(nn.Module):
    def __init__(
        self,
        categorical_p=0.1,
        ordered_p=0.7,
        keep_activation_size=False,
        num_classes_sampler=default_zipf_sampler,
    ):
        self.categorical_p = categorical_p
        self.ordered_p = ordered_p
        self.keep_activation_size = keep_activation_size
        self.num_classes_sampler = num_classes_sampler

        super().__init__()

    def forward(self, x):
        # x shape: T, B, H

        x = nn.Softsign()(x)

        num_classes = self.num_classes_sampler()
        hid_strength = (
            torch.abs(x).mean(0).unsqueeze(0) if self.keep_activation_size else None
        )

        categorical_classes = torch.rand((x.shape[1], x.shape[2])) < self.categorical_p
        class_boundaries = torch.zeros(
            (num_classes - 1, x.shape[1], x.shape[2]),
            device=x.device,
            dtype=x.dtype,
        )
        # Sample a different index for each hidden dimension, but shared for all batches
        for b in range(x.shape[1]):
            for h in range(x.shape[2]):
                ind = torch.randint(0, x.shape[0], (num_classes - 1,))
                class_boundaries[:, b, h] = x[ind, b, h]

        for b in range(x.shape[1]):
            x_rel = x[:, b, categorical_classes[b]]
            boundaries_rel = class_boundaries[:, b, categorical_classes[b]].unsqueeze(1)
            x[:, b, categorical_classes[b]] = (x_rel > boundaries_rel).sum(
                dim=0
            ).float() - num_classes / 2

        ordered_classes = torch.rand((x.shape[1], x.shape[2])) < self.ordered_p
        ordered_classes = torch.logical_and(ordered_classes, categorical_classes)
        x[:, ordered_classes] = randomize_classes(x[:, ordered_classes], num_classes)

        x = x * hid_strength if self.keep_activation_size else x

        return x


class QuantizationActivation(torch.nn.Module):
    def __init__(self, n_thresholds, reorder_p=0.5) -> None:
        super().__init__()
        self.n_thresholds = n_thresholds
        self.reorder_p = reorder_p
        self.thresholds = torch.nn.Parameter(torch.randn(self.n_thresholds))

    def forward(self, x):
        x = normalize_data(x).unsqueeze(-1)
        x = (x > self.thresholds).sum(-1)

        if random.random() < self.reorder_p:
            x = randomize_classes(x.unsqueeze(-1), self.n_thresholds).squeeze(-1)
        # x = ((x.float() - self.n_thresholds/2) / self.n_thresholds)# * data_std + data_mean
        x = normalize_data(x)
        return x


class NormalizationActivation(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        x = normalize_data(x)
        return x


class PowerActivation(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.exp = torch.nn.Parameter(0.5 * torch.ones(1))
        self.shared_exp_strength = 0.5
        # TODO: Somehow this is only initialized once, so it's the same for all runs

    def forward(self, x):
        # print(torch.nn.functional.softplus(x), self.exp)
        shared_exp = torch.randn(1)
        exp = torch.nn.Parameter(
            (
                shared_exp * self.shared_exp_strength
                + shared_exp * torch.randn(x.shape[-1]) * (1 - self.shared_exp_strength)
            )
            * 2
            + 0.5
        ).to(x.device)
        x_ = torch.pow(torch.nn.functional.softplus(x) + 0.001, exp)
        if False:
            print(
                x[0:3, 0, 0].cpu().numpy(),
                torch.nn.functional.softplus(x[0:3, 0, 0]).cpu().numpy(),
                x_[0:3, 0, 0].cpu().numpy(),
                normalize_data(x_)[0:3, 0, 0].cpu().numpy(),
                self.exp.cpu().numpy(),
            )
        return x_


def lambda_time(f, name="", enabled=True):
    if not enabled:
        return f()
    start = time.time()
    r = f()
    print("Timing", name, time.time() - start)
    return r


def pretty_get_batch(get_batch):
    """
    Genereate string representation of get_batch function
    :param get_batch:
    :return:
    """
    if isinstance(get_batch, types.FunctionType):
        return f"<{get_batch.__module__}.{get_batch.__name__} {inspect.signature(get_batch)}"
    else:
        return repr(get_batch)


class get_batch_sequence(list):
    """
    This will call the get_batch_methods in order from the back and pass the previous as `get_batch` kwarg.
    For example for `get_batch_methods=[get_batch_1, get_batch_2, get_batch_3]` this will produce a call
    equivalent to `get_batch_3(*args,get_batch=partial(partial(get_batch_2),get_batch=get_batch_1,**kwargs))`.
    get_batch_methods: all priors, but the first, muste have a `get_batch` argument
    """

    def __init__(self, *get_batch_methods):
        if len(get_batch_methods) == 0:
            raise ValueError("Must have at least one get_batch method")
        super().__init__(get_batch_methods)

    def __repr__(self):
        s = ",\n\t".join([f"{pretty_get_batch(get_batch)}" for get_batch in self])
        return f"get_batch_sequence(\n\t{s}\n)"

    def __call__(self, *args, **kwargs):
        """

        Standard kwargs are: batch_size, seq_len, num_features
        This returns a priors.Batch object.
        """
        final_get_batch = self[0]
        for get_batch in self[1:]:
            final_get_batch = partial(get_batch, get_batch=final_get_batch)
        return final_get_batch(*args, **kwargs)
