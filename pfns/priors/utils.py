import inspect
import math
import os
import random
import time
import types
from copy import deepcopy
from functools import partial
from typing import Callable, Iterator, TypeAlias

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import torch
import torch.distributed as dist
import torch.utils.data  # Added for get_worker_info
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from typing_extensions import override

from ..utils import (
    get_uniform_single_eval_pos_sampler,
    normalize_data,
    set_locals_in_self,
)
from .prior import Batch, PriorDataLoader


# Moved this function outside the class
def compute_eval_pos_sequence(
    num_steps,
    eval_pos_seq_len_sampler,
    batch_size=None,
    seq_len_maximum=None,
    dynamic_batch_size=None,
):
    """Pre-computes single_eval_pos, seq_len and adjusted batch_size for all steps in the dataloader.

    Args:
        num_steps: Total number of steps (batches) in the epoch.
        eval_pos_seq_len_sampler: Callable that returns (single_eval_pos, seq_len) tuples
        batch_size: Base batch size before dynamic adjustment
        seq_len_maximum: Maximum sequence length for dynamic batch size scaling
        dynamic_batch_size: Power factor for dynamic batch size scaling

    Returns:
        List of (single_eval_pos, seq_len, adjusted_batch_size) tuples for each step
    """
    eval_pos_sequence = []
    for _ in range(num_steps):
        single_eval_pos, seq_len = eval_pos_seq_len_sampler()

        # Calculate dynamic batch size if parameters are provided
        adjusted_batch_size = batch_size
        # Check specifically if dynamic_batch_size is provided and non-zero/non-None
        if dynamic_batch_size and all(
            x is not None for x in [batch_size, seq_len_maximum]
        ):
            adjusted_batch_size = batch_size * math.floor(
                math.pow(seq_len_maximum, dynamic_batch_size)
                / math.pow(seq_len, dynamic_batch_size)
            )

        eval_pos_sequence.append(
            (single_eval_pos, seq_len, adjusted_batch_size)
        )

    return eval_pos_sequence


class _BatchedIterableDataset(IterableDataset[Batch]):
    def __init__(
        self,
        get_batch_method: Callable[[], Batch],
        num_steps: int,
        # Add sampler and dynamic batch size params needed for sequence computation
        eval_pos_seq_len_sampler: Callable,
        **get_batch_kwargs,
    ) -> None:
        super().__init__()
        self.get_batch_method = get_batch_method
        self.num_steps = num_steps
        self.eval_pos_seq_len_sampler = eval_pos_seq_len_sampler
        self.kwargs = get_batch_kwargs

    @override
    def __iter__(self) -> Iterator[Batch]:
        # Compute the sequence once per iterator creation (i.e., per worker)
        # Extract necessary parameters from stored kwargs for clarity
        batch_size = self.kwargs.get("batch_size")
        seq_len_maximum = self.kwargs.get("seq_len_maximum")
        dynamic_batch_size = self.kwargs.get("dynamic_batch_size")

        eval_pos_sequence = compute_eval_pos_sequence(
            self.num_steps,
            self.eval_pos_seq_len_sampler,
            batch_size,
            seq_len_maximum,
            dynamic_batch_size,
        )

        worker_info = torch.utils.data.get_worker_info()
        if (
            worker_info is None
        ):  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = self.num_steps
        else:  # multiple workers, split the work
            per_worker = int(
                math.ceil(self.num_steps / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.num_steps)

        for i in range(iter_start, iter_end):
            single_eval_pos, seq_len, adjusted_batch_size = eval_pos_sequence[
                i
            ]

            # Prepare kwargs for the actual batch fetching method
            kwargs = dict(self.kwargs)
            kwargs["single_eval_pos"] = single_eval_pos
            kwargs["seq_len"] = seq_len
            # Use the pre-calculated adjusted_batch_size
            kwargs["batch_size"] = adjusted_batch_size
            # Remove sampler from kwargs passed to get_batch_method if it's not expected
            kwargs.pop("eval_pos_seq_len_sampler", None)
            kwargs.pop(
                "dynamic_batch_size", None
            )  # Also remove dynamic_batch_size if not expected
            kwargs.pop(
                "seq_len_maximum", None
            )  # Remove seq_len_maximum if not expected

            b = self.get_batch_method(**kwargs)
            # Ensure single_eval_pos is set on the batch object if get_batch_method doesn't handle it
            if b.single_eval_pos is None:
                b.single_eval_pos = single_eval_pos
            yield b


def worker_init_fn(epoch, worker_id):
    # 1. Set PyTorch threads
    torch.set_num_threads(1)

    # 2. Set environment variables for underlying libraries (NumPy, MKL, OpenMP)
    #    Often, torch.set_num_threads(1) might handle this, but setting explicitly is safer.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    # You might need others depending on your NumPy/SciPy build:
    # os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    # os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # 3. Original seeding logic
    worker_seed = worker_id

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    seed = worker_seed + rank * 100 + epoch * 10000

    torch.manual_seed(seed)
    # No need to seed cuda in worker_init_fn if data loading is CPU-only
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)  # Also seed Python's random module if used


class StandardDataLoader(DataLoader):
    # Caution, you might need to set self.num_features manually if it is not part of the args.
    def __init__(
        self,
        get_batch_method,
        num_steps,
        num_workers=4,
        eval_pos_seq_len_sampler=None,
        **get_batch_kwargs,
    ):
        set_locals_in_self(locals())

        # The stuff outside the or is set as class attribute before instantiation.
        self.num_features = (
            get_batch_kwargs.get("num_features") or self.num_features
        )
        self.epoch_count = 0
        self.importance_sampling_infos = None

        # Extract sampler for _BatchedIterableDataset init
        if eval_pos_seq_len_sampler is None:
            # Default sampler if not provided in kwargs (matches old _BatchedIterableDataset logic)
            # Requires seq_len to be in get_batch_kwargs if used.
            seq_len = get_batch_kwargs.get("seq_len")
            if seq_len is None:
                raise ValueError(
                    "seq_len must be provided in get_batch_kwargs if eval_pos_seq_len_sampler is not."
                )
            eval_pos_seq_len_sampler = lambda: (
                get_uniform_single_eval_pos_sampler(seq_len - 1)(),
                seq_len,
            )

        self.eval_pos_seq_len_sampler = eval_pos_seq_len_sampler

        print("DataLoader.__dict__", self.__dict__)

        if "device" in self.get_batch_kwargs:
            self.get_batch_kwargs["device"] = "cpu"

        # Pass sampler and all kwargs to the dataset
        super().__init__(
            dataset=_BatchedIterableDataset(
                get_batch_method,
                num_steps,
                eval_pos_seq_len_sampler=eval_pos_seq_len_sampler,
                **get_batch_kwargs,
            ),
            batch_size=None,  # Batching is handled by the dataset iterator
            batch_sampler=None,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=lambda worker_id: worker_init_fn(
                self.epoch_count, worker_id
            ),  # Use the updated function
            collate_fn=lambda x: x,  # Dataset yields complete batches
        )

    def __len__(self):
        return self.num_steps

    def __iter__(self):
        assert hasattr(
            self, "model"
        ), "Please assign model with `dl.model = ...` before training."
        self.epoch_count += 1
        # The iteration logic is now handled by _BatchedIterableDataset passed to super().__init__
        return super().__iter__()


class DiscreteImportanceSamplingDataLoader(StandardDataLoader):
    def __init__(
        self,
        get_batch_method,
        num_steps,
        importance_hyperparameter: str,
        importance_hyperparameter_options: list,
        do_not_adapt: bool = False,
        importance_sampling_based_on_square: bool = False,
        importance_probs_init: list | None = None,
        grad_magnitude_adam_normalized: bool = False,
        importance_sampling_based_on_loss_improvement: bool = False,
        multiplicative_loss_improvement: bool = False,
        normalize_loss_improvement_by: str | None = None,
        **get_batch_kwargs,
    ):
        # default assumption is that we want to sample them all equally
        super().__init__(get_batch_method, num_steps, **get_batch_kwargs)
        self.importance_hyperparameter = importance_hyperparameter
        self.importance_hyperparameter_options = (
            importance_hyperparameter_options
        )
        self.importance_sampling_infos = None
        self.do_not_adapt = do_not_adapt
        self.importance_sampling_based_on_square = (
            importance_sampling_based_on_square
        )
        self.importance_probs_init = importance_probs_init
        self.importance_sampling_based_on_loss_improvement = (
            importance_sampling_based_on_loss_improvement
        )
        self.multiplicative_loss_improvement = multiplicative_loss_improvement
        self.normalize_loss_improvement_by = normalize_loss_improvement_by
        self.grad_magnitude_adam_normalized = grad_magnitude_adam_normalized

    def gbm(self, *args, eval_pos_seq_len_sampler, **kwargs):
        kwargs["single_eval_pos"], kwargs["seq_len"] = (
            eval_pos_seq_len_sampler()
        )
        # Scales the batch size dynamically with the power of 'dynamic_batch_size'.
        # A transformer with quadratic memory usage in the seq len would need a power of 2 to keep memory constant.
        if kwargs.get("dynamic_batch_size"):
            kwargs["batch_size"] = kwargs["batch_size"] * math.floor(
                math.pow(
                    kwargs["seq_len_maximum"], kwargs["dynamic_batch_size"]
                )
                / math.pow(kwargs["seq_len"], kwargs["dynamic_batch_size"])
            )
        batch: Batch = self.get_batch_method(*args, **kwargs)
        if batch.single_eval_pos is None:
            batch.single_eval_pos = kwargs["single_eval_pos"]
        return batch

    def __iter__(self):
        assert hasattr(
            self, "model"
        ), "Please assign model with `dl.model = ...` before training."
        if self.epoch_count > 0:
            # did an iter before
            assert self.importance_sampling_infos is not None
        self.epoch_count += 1

        if (
            self.importance_sampling_infos is not None
        ) and not self.do_not_adapt:
            if self.importance_sampling_based_on_loss_improvement:
                # Group losses by hyperparameter option and by epoch

                # Calculate current epoch's average loss per option
                current_epoch_losses = {}
                for _, option_idx, loss, *_ in self.importance_sampling_infos:
                    assert (
                        0
                        <= option_idx
                        < len(self.importance_hyperparameter_options)
                    ), f"Option index {option_idx} is out of bounds for hyperparameter options {self.importance_hyperparameter_options}"
                    if option_idx not in current_epoch_losses:
                        current_epoch_losses[option_idx] = []
                    current_epoch_losses[option_idx].append(loss)

                current_epoch_option_counts = torch.zeros(
                    len(self.importance_hyperparameter_options)
                )

                # Average the losses for each option in the current epoch
                for option_idx in current_epoch_losses:
                    current_epoch_option_counts[option_idx] = len(
                        current_epoch_losses[option_idx]
                    )
                    current_epoch_losses[option_idx] = (
                        torch.tensor(current_epoch_losses[option_idx])
                        .mean()
                        .item()
                    )

                if not hasattr(self, "previous_epoch_losses"):
                    probs = torch.ones(
                        len(self.importance_hyperparameter_options)
                    ) / len(self.importance_hyperparameter_options)
                else:
                    print("current_epoch_losses", current_epoch_losses)

                    # Calculate improvements compared to previous epoch
                    improvements = torch.zeros(
                        len(self.importance_hyperparameter_options)
                    )
                    for (
                        option_idx,
                        current_loss,
                    ) in current_epoch_losses.items():
                        if option_idx in self.previous_epoch_losses:
                            if self.multiplicative_loss_improvement:
                                # Calculate geometric mean of improvement per step
                                improvements[option_idx] = (
                                    current_loss
                                    / self.previous_epoch_losses[option_idx]
                                )
                            else:
                                # Original additive improvement
                                improvements[option_idx] = (
                                    self.previous_epoch_losses[option_idx]
                                    - current_loss
                                )

                    print("improvements", improvements)

                    if self.normalize_loss_improvement_by == "count":
                        total_counts = (
                            self.previous_epoch_counts
                            + current_epoch_option_counts
                        ) / 2
                        if self.multiplicative_loss_improvement:
                            improvement_ratios = improvements ** (
                                1.0 / total_counts
                            )

                            expected_losses = torch.tensor(
                                [
                                    [
                                        improvement_ratios[i] ** steps
                                        * current_epoch_losses[i]
                                        for steps in range(self.num_steps + 1)
                                    ]
                                    for i in range(
                                        len(
                                            self.importance_hyperparameter_options
                                        )
                                    )
                                ]
                            )

                            config = torch.zeros(
                                len(self.importance_hyperparameter_options),
                                dtype=torch.int64,
                            )
                            print("expected_losses", expected_losses)
                            for step in range(self.num_steps):
                                current_losses = expected_losses[
                                    torch.arange(len(config)), config
                                ]
                                possible_improvements = (
                                    current_losses
                                    - expected_losses[
                                        torch.arange(len(config)), config + 1
                                    ]
                                )
                                best_option_idx = torch.argmax(
                                    possible_improvements
                                )
                                config[best_option_idx] += 1

                            print("config", config)

                            probs = config / config.sum()
                            probs = (
                                probs * 0.8
                                + torch.ones(len(probs)) / len(probs) * 0.2
                            )

                        else:
                            # Use combined counts from both epochs for normalization
                            # Avoid division by zero by setting improvements to 0 where count is 0
                            mask = total_counts > 0
                            improvements[mask] = improvements[
                                mask
                            ] / torch.sqrt(total_counts[mask])
                            improvements[~mask] = 0.0
                            print("normalized improvements", improvements)

                            # Ensure all improvements are positive by shifting
                            improvements = (
                                improvements - improvements.min() + 1e-8
                            )

                            probs = improvements / improvements.sum()

                            # make sure that the smallest prob is at least 1/len(options)/10
                            probs = (
                                probs * 0.9
                                + torch.ones(len(probs)) / len(probs) * 0.1
                            )
                    else:
                        assert (
                            self.normalize_loss_improvement_by is None
                        ), f"Invalid normalization method: {self.normalize_loss_improvement_by}"

                # Store current losses and update counts for next epoch comparison
                self.previous_epoch_losses = current_epoch_losses
                self.previous_epoch_counts = current_epoch_option_counts
            else:
                # Original gradient-based importance sampling
                if self.grad_magnitude_adam_normalized:
                    grad_mags = torch.tensor(
                        [nm for m, i, _, nm in self.importance_sampling_infos]
                    )
                else:
                    grad_mags = torch.tensor(
                        [m for m, i, *_ in self.importance_sampling_infos]
                    )
                if not self.importance_sampling_based_on_square:
                    grad_mags = grad_mags.sqrt()
                hyperparameter_option_index = torch.tensor(
                    [i for m, i, *_ in self.importance_sampling_infos]
                )
                grad_mag_per_option = torch.zeros(
                    len(self.importance_hyperparameter_options)
                )
                recorded_magnitudes = torch.scatter_reduce(
                    grad_mag_per_option,
                    0,
                    hyperparameter_option_index[torch.isfinite(grad_mags)],
                    grad_mags[torch.isfinite(grad_mags)],
                    reduce="mean",
                )
                if self.importance_sampling_based_on_square:
                    recorded_magnitudes = recorded_magnitudes.sqrt()
                probs = recorded_magnitudes / recorded_magnitudes.sum()
        else:
            print("using uniform dist")
            if self.importance_probs_init is None:
                probs = torch.ones(
                    len(self.importance_hyperparameter_options)
                ) / len(self.importance_hyperparameter_options)
            else:
                probs = torch.tensor(self.importance_probs_init)
                probs = probs / probs.sum()

        if (probs == 0.0).any():
            probs[probs == 0.0] = probs[probs > 0.0].min()
            probs = probs / probs.sum()

        print("importance sampling probs", probs)

        scales = 1 / (probs * len(probs))

        self.last_probs = probs
        get_batch_kwargs = deepcopy(self.get_batch_kwargs)

        # Need the sampler for the gbm call below
        sampler = get_batch_kwargs.get("eval_pos_seq_len_sampler")
        if sampler is None:
            seq_len = get_batch_kwargs.get("seq_len")
            if seq_len is None:
                raise ValueError(
                    "seq_len must be provided in get_batch_kwargs for DiscreteImportanceSamplingDataLoader if eval_pos_seq_len_sampler is not."
                )
            sampler = lambda: (
                get_uniform_single_eval_pos_sampler(seq_len - 1)(),
                seq_len,
            )

        hp_indices = []
        hyperparameters_for_all_batches = []
        multipliers = []
        for step in range(self.num_steps):
            hp_index = torch.multinomial(
                probs, 1
            ).item()  # todo check this correct
            hp_indices.append(hp_index)
            hp_value = self.importance_hyperparameter_options[hp_index]
            # Ensure 'hyperparameters' exists in get_batch_kwargs before trying to merge
            base_hps = get_batch_kwargs.get("hyperparameters", {})
            hyperparameters_for_all_batches.append(
                {**base_hps, self.importance_hyperparameter: hp_value}
            )
            multipliers.append(scales[hp_index])

        for hp_index, hps, m in zip(
            hp_indices, hyperparameters_for_all_batches, multipliers
        ):
            # This call uses self.gbm, which still does its own sampling via the sampler
            b = self.gbm(
                eval_pos_seq_len_sampler=sampler,  # Pass the sampler to gbm
                **{**self.get_batch_kwargs, "hyperparameters": hps},
                epoch=self.epoch_count - 1,
                model=self.model,
            )
            b.gradient_multipliers = m
            b.info_used_with_gradient_magnitudes = hp_index
            yield b

    def __len__(self):
        return self.num_steps

    def __iter__(self):
        assert hasattr(
            self, "model"
        ), "Please assign model with `dl.model = ...` before training."
        self.epoch_count += 1
        return iter(
            self.gbm(
                **self.get_batch_kwargs,
                epoch=self.epoch_count - 1,
                model=self.model,
            )
            for _ in range(self.num_steps)
        )


@torch.no_grad()
def zero_time_get_batch(
    batch_size, seq_len, num_features, device="cpu", **kwargs
):
    y = torch.rand(seq_len, batch_size, 1, device=device)
    return Batch(
        x=torch.rand(seq_len, batch_size, num_features, device=device),
        y=y,
        target_y=y.clone(),
    )


def plot_features(
    data, targets, fig=None, categorical=True, plot_diagonal=True
):
    import seaborn as sns

    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

    fig2 = fig if fig else plt.figure(figsize=(8, 8))
    spec2 = gridspec.GridSpec(
        ncols=data.shape[1], nrows=data.shape[1], figure=fig2
    )
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
    return lambda: stats.rv_discrete(
        name="bounded_zipf", values=(x, weights)
    ).rvs(1)


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
    order = (
        order.reshape(2, -1).transpose(0, 1).reshape(-1)
    )  # .reshape(seq_len)
    x = x[
        order
    ]  # .reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).reshape(seq_len, 1, -1)
    y = y[
        order
    ]  # .reshape(2, -1).transpose(0, 1).reshape(-1).reshape(seq_len, 1, -1)

    return x, y


def randomize_classes(x, num_classes):
    classes = torch.arange(0, num_classes, device=x.device)
    random_classes = torch.randperm(num_classes, device=x.device).type(
        x.type()
    )
    x = ((x.unsqueeze(-1) == classes) * random_classes).sum(-1)
    return x


@torch.no_grad()
def sample_num_feaetures_get_batch(
    batch_size, seq_len, num_features, hyperparameters, get_batch, **kwargs
):
    if (
        hyperparameters.get("sample_num_features", True)
        and kwargs.get("epoch", 1) > 0
    ):  # don't sample on test batch
        num_features = torch.randint(1, num_features + 1, size=[1]).item()
    return get_batch(
        batch_size,
        seq_len,
        num_features,
        hyperparameters=hyperparameters,
        **kwargs,
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
            torch.abs(x).mean(0).unsqueeze(0)
            if self.keep_activation_size
            else None
        )

        categorical_classes = (
            torch.rand((x.shape[1], x.shape[2])) < self.categorical_p
        )
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
            boundaries_rel = class_boundaries[
                :, b, categorical_classes[b]
            ].unsqueeze(1)
            x[:, b, categorical_classes[b]] = (x_rel > boundaries_rel).sum(
                dim=0
            ).float() - num_classes / 2

        ordered_classes = torch.rand((x.shape[1], x.shape[2])) < self.ordered_p
        ordered_classes = torch.logical_and(
            ordered_classes, categorical_classes
        )
        x[:, ordered_classes] = randomize_classes(
            x[:, ordered_classes], num_classes
        )

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
            x = randomize_classes(x.unsqueeze(-1), self.n_thresholds).squeeze(
                -1
            )
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
                + shared_exp
                * torch.randn(x.shape[-1])
                * (1 - self.shared_exp_strength)
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
        s = ",\n\t".join(
            [f"{pretty_get_batch(get_batch)}" for get_batch in self]
        )
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
