from __future__ import annotations

import inspect
import math
import multiprocessing
import random
import time
import types
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Iterator, Optional, TypeVar, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from ..utils import normalize_data, set_locals_in_self
from .prior import Batch, PriorDataLoader

# Set multiprocessing start method to 'spawn'
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)


class _BatchedIterableDataset(IterableDataset):
    def __init__(
        self,
        get_batch_fn: Callable[..., Batch],
        num_steps: int,
        **get_batch_kwargs: Any,
    ) -> None:
        super().__init__()
        self.get_batch_fn = get_batch_fn
        self.num_steps = num_steps
        self.get_batch_kwargs = get_batch_kwargs

    def __iter__(self) -> Iterator[Batch]:
        for _ in range(self.num_steps):
            yield self.get_batch_fn(**self.get_batch_kwargs)


@dataclass
class SequentialEvalPosSampler:
    """A sampler that generates sequential positions for evaluation"""
    seq_len: int
    num_positions: int
    offset: int = 0
    current_pos: int = 0
    
    def __call__(self) -> int:
        pos = ((self.current_pos + self.offset) * self.seq_len) // (self.num_positions + 1)
        self.current_pos = (self.current_pos + 1) % self.num_positions
        return pos

@dataclass
class UniformSingleEvalPosSampler:
    """A picklable sampler that generates uniform positions"""
    seq_len: int
    min_pos: int
    max_pos: int
    
    def __call__(self) -> int:
        return random.randint(self.min_pos, min(self.max_pos, self.seq_len - 1))


@dataclass
class EvalPosSeqLenSampler:
    """A picklable class to handle eval position and sequence length sampling"""
    seq_len: int
    single_eval_pos_sampler: UniformSingleEvalPosSampler | None = None

    def __call__(self) -> Tuple[int, int]:
        if self.single_eval_pos_sampler is None:
            # Default behavior: return middle position
            return self.seq_len // 2, self.seq_len
        return self.single_eval_pos_sampler(), self.seq_len


def get_uniform_single_eval_pos_sampler(seq_len: int, min_pos: int = 1, max_pos: int | None = None) -> UniformSingleEvalPosSampler:
    """Creates a sampler that uniformly samples positions between min_pos and max_pos"""
    if max_pos is None:
        max_pos = seq_len - 1
    return UniformSingleEvalPosSampler(seq_len=seq_len, min_pos=min_pos, max_pos=max_pos)


class ParallelPriorDataLoader(PriorDataLoader):
    """A parallel data loader implementation that uses multiple workers."""

    def __init__(
        self,
        num_steps: int,
        batch_size: int,
        eval_pos_seq_len_sampler: EvalPosSeqLenSampler,
        seq_len_maximum: int,
        device: str,
        get_batch_method: Callable[..., Batch],
        num_workers: int = 4,
        **get_batch_kwargs: Any,
    ) -> None:
        self.get_batch_kwargs = get_batch_kwargs
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.epoch_count = 0
        self.eval_pos_seq_len_sampler = eval_pos_seq_len_sampler
        self.seq_len_maximum = seq_len_maximum
        self.device = device
        self.get_batch_method = get_batch_method

        # Extract num_features from kwargs or class attribute
        self.num_features = get_batch_kwargs.get("num_features", None)

        super().__init__(
            num_steps=num_steps,
            batch_size=batch_size,
            eval_pos_seq_len_sampler=eval_pos_seq_len_sampler,
            seq_len_maximum=seq_len_maximum,
            device=device,
            **get_batch_kwargs,
        )

        self.dataset = _BatchedIterableDataset(
            get_batch_fn=self.gbm,
            num_steps=num_steps,
            eval_pos_seq_len_sampler=eval_pos_seq_len_sampler,
            **get_batch_kwargs,
        )

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=None,  # We handle batching in get_batch
            num_workers=8,
            persistent_workers=True,
            prefetch_factor=2,
            worker_init_fn=self._worker_init_fn,
            multiprocessing_context="spawn",  # Explicitly set spawn context
        )

    @staticmethod
    def _worker_init_fn(worker_id: int) -> None:
        # Set different random seeds for each worker
        seed = torch.initial_seed() + worker_id
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def gbm(
        self,
        *args: Any,
        eval_pos_seq_len_sampler: EvalPosSeqLenSampler,
        **kwargs: Any,
    ) -> Batch:
        single_eval_pos, seq_len = eval_pos_seq_len_sampler()

        # Prepare all required arguments
        batch_kwargs = {
            "batch_size": self.batch_size,
            "seq_len": seq_len,
            "num_features": self.num_features,
            "device": self.device,
            "single_eval_pos": single_eval_pos,
            **kwargs,
        }

        # Handle dynamic batch sizing
        if "dynamic_batch_size" in kwargs and kwargs["dynamic_batch_size"] > 0:
            batch_kwargs["batch_size"] = batch_kwargs["batch_size"] * math.floor(
                math.pow(kwargs["seq_len_maximum"], kwargs["dynamic_batch_size"])
                / math.pow(seq_len, kwargs["dynamic_batch_size"])
            )

        batch = self.get_batch_method(**batch_kwargs)
        if batch.single_eval_pos is None:
            batch.single_eval_pos = single_eval_pos
        return batch

    def get_test_batch(self, **kwargs: Any) -> Batch:
        return self.gbm(
            eval_pos_seq_len_sampler=self.eval_pos_seq_len_sampler,
            epoch=self.epoch_count,
            model=self.model if hasattr(self, "model") else None,
            **{**self.get_batch_kwargs, **kwargs},
        )

    def __len__(self) -> int:
        return self.num_steps

    def __iter__(self) -> Iterator[Batch]:
        assert hasattr(
            self, "model"
        ), "Please assign model with `dl.model = ...` before training."
        self.epoch_count += 1
        return iter(self.dataloader)


def get_batch_to_dataloader(
    get_batch_method_: Callable[..., Batch],
) -> type[ParallelPriorDataLoader]:
    """Factory function to create a ParallelPriorDataLoader with specific get_batch method"""

    def create_loader(*args, **kwargs):
        return ParallelPriorDataLoader(
            *args, get_batch_method=get_batch_method_, **kwargs
        )

    return create_loader


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
                        x=data[:, d], y=data[:, d2], hue=targets[:], legend=False
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


trunc_norm_sampler_f = lambda mu, sigma: lambda: stats.truncnorm(
    (0 - mu) / sigma, (1000000 - mu) / sigma, loc=mu, scale=sigma
).rvs(1)[0]
beta_sampler_f = lambda a, b: lambda: np.random.beta(a, b)
gamma_sampler_f = lambda a, b: lambda: np.random.gamma(a, b)
uniform_sampler_f = lambda a, b: lambda: np.random.uniform(a, b)
uniform_int_sampler_f = lambda a, b: lambda: round(np.random.uniform(a, b))


def zipf_sampler_f(a, b, c):
    x = np.arange(b, c)
    weights = x ** (-a)
    weights /= weights.sum()
    return lambda: stats.rv_discrete(name="bounded_zipf", values=(x, weights)).rvs(1)


scaled_beta_sampler_f = lambda a, b, scale, minimum: lambda: minimum + round(
    beta_sampler_f(a, b)() * (scale - minimum)
)


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


@torch.no_grad()
def sample_num_feaetures_get_batch(
    batch_size, seq_len, num_features, hyperparameters, get_batch, **kwargs
):
    if (
        hyperparameters.get("sample_num_features", True) and kwargs["epoch"] > 0
    ):  # don't sample on test batch
        num_features = random.randint(1, num_features)
    return get_batch(
        batch_size, seq_len, num_features, hyperparameters=hyperparameters, **kwargs
    )


class CategoricalActivation(nn.Module):
    def __init__(
        self,
        categorical_p=0.1,
        ordered_p=0.7,
        keep_activation_size=False,
        num_classes_sampler=zipf_sampler_f(0.8, 1, 10),
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
            (num_classes - 1, x.shape[1], x.shape[2]), device=x.device, dtype=x.dtype
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
