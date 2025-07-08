import math
import os
import random

from copy import deepcopy

from functools import partial
from typing import Callable, Iterator

import numpy as np
import torch
import torch.distributed as dist
from pfns.batch_shape_sampler import BatchShape
from pfns.priors.prior import Batch
from pfns.utils import set_locals_in_self
from torch.utils.data import DataLoader, IterableDataset
from typing_extensions import override


def worker_init_fn(worker_id: int, epoch: int):
    print(f"worker_init_fn {worker_id=} {epoch=}")

    # make sure each worker does not multi-thread
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    worker_seed = worker_id
    seed = worker_seed + rank * 100 + epoch * 10000

    torch.manual_seed(seed)
    # No need to seed cuda in worker_init_fn if data loading is CPU-only
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)  # Also seed Python's random module if used

    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.epoch_count = epoch


class _BatchedIterableDataset(IterableDataset[Batch]):
    def __init__(
        self,
        get_batch_method: Callable[[], Batch],
        num_steps: int,
        batch_shape_sampler_function: Callable[[int, int], BatchShape],
        **get_batch_kwargs,
    ) -> None:
        super().__init__()
        self.get_batch_method = get_batch_method
        self.num_steps = num_steps
        self.batch_shape_sampler_function = batch_shape_sampler_function
        self.kwargs = get_batch_kwargs
        self.epoch_count = 1

    @override
    def __iter__(self) -> Iterator[Batch]:
        # Compute the sequence once per iterator creation (i.e., per worker)
        # Extract necessary parameters from stored kwargs for clarity
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            per_worker = self.num_steps
            worker_id = 0
            num_workers = 1
        else:  # multiple workers, split the work
            per_worker = int(math.ceil(self.num_steps / float(worker_info.num_workers)))
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        for local_step in range(per_worker):
            batch_shape = self.batch_shape_sampler_function(
                epoch=self.epoch_count,
                step=local_step * num_workers + worker_id,
            )

            # Prepare kwargs for the actual batch fetching method
            kwargs = batch_shape.as_get_batch_kwargs()
            kwargs.update(self.kwargs)

            b = self.get_batch_method(**kwargs)

            assert (
                len(b.x) == len(b.y) == len(b.target_y) == batch_shape.batch_size
            ), "Our code was updated to use the more intuitive batch first format, please make sure your get_batch function returns data with shapes (batch_size, seq_len, ...)"

            # Ensure single_eval_pos is set on the batch object if get_batch_method doesn't handle it
            if b.single_eval_pos is None:
                b.single_eval_pos = batch_shape.single_eval_pos
            b.step = local_step * num_workers + worker_id + 10000 * self.epoch_count
            yield b
        self.epoch_count += 1


class StandardDataLoader(DataLoader):
    # Caution, you might need to set self.num_features manually if it is not part of the args.
    def __init__(
        self,
        get_batch_method,
        num_steps,
        batch_shape_sampler_function: Callable[[int, int], BatchShape],
        num_workers=4,
        persistent_workers=False,  # can't update epoch_count from the outside if True and num_workers > 0
        **get_batch_kwargs,
    ):
        set_locals_in_self(locals())

        # The stuff outside the or is set as class attribute before instantiation.
        self.epoch_count = 0
        self.importance_sampling_infos = None

        self.batch_shape_sampler_function = batch_shape_sampler_function

        print("DataLoader.__dict__", self.__dict__)

        if "device" in self.get_batch_kwargs:
            self.get_batch_kwargs["device"] = "cpu"

        # Pass sampler and all kwargs to the dataset
        super().__init__(
            dataset=_BatchedIterableDataset(
                get_batch_method,
                num_steps,
                batch_shape_sampler_function=batch_shape_sampler_function,
                **get_batch_kwargs,
            ),
            batch_size=None,  # Batching is handled by the dataset iterator
            batch_sampler=None,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=None,  # Dataset yields complete batches
            persistent_workers=persistent_workers
            if num_workers > 0
            else False,  # don't keep workers as we need to update the epoch count
        )

    def __len__(self):
        return self.num_steps

    def __iter__(self):
        assert hasattr(
            self, "model"
        ), "Please assign model with `dl.model = ...` before training."
        self.epoch_count += 1
        self.worker_init_fn = partial(worker_init_fn, epoch=self.epoch_count)
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
        self.importance_hyperparameter_options = importance_hyperparameter_options
        self.importance_sampling_infos = None
        self.do_not_adapt = do_not_adapt
        self.importance_sampling_based_on_square = importance_sampling_based_on_square
        self.importance_probs_init = importance_probs_init
        self.importance_sampling_based_on_loss_improvement = (
            importance_sampling_based_on_loss_improvement
        )
        self.multiplicative_loss_improvement = multiplicative_loss_improvement
        self.normalize_loss_improvement_by = normalize_loss_improvement_by
        self.grad_magnitude_adam_normalized = grad_magnitude_adam_normalized

    def gbm(self, *args, eval_pos_seq_len_sampler, **kwargs):
        kwargs["single_eval_pos"], kwargs["seq_len"] = eval_pos_seq_len_sampler()
        # A transformer with quadratic memory usage in the seq len would need a power of 2 to keep memory constant.
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

        if (self.importance_sampling_infos is not None) and not self.do_not_adapt:
            if self.importance_sampling_based_on_loss_improvement:
                # Group losses by hyperparameter option and by epoch

                # Calculate current epoch's average loss per option
                current_epoch_losses = {}
                for _, option_idx, loss, *_ in self.importance_sampling_infos:
                    assert (
                        0 <= option_idx < len(self.importance_hyperparameter_options)
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
                        torch.tensor(current_epoch_losses[option_idx]).mean().item()
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
                            self.previous_epoch_counts + current_epoch_option_counts
                        ) / 2
                        if self.multiplicative_loss_improvement:
                            improvement_ratios = improvements ** (1.0 / total_counts)

                            expected_losses = torch.tensor(
                                [
                                    [
                                        improvement_ratios[i] ** steps
                                        * current_epoch_losses[i]
                                        for steps in range(self.num_steps + 1)
                                    ]
                                    for i in range(
                                        len(self.importance_hyperparameter_options)
                                    )
                                ]
                            )

                            config = torch.zeros(
                                len(self.importance_hyperparameter_options),
                                dtype=torch.int64,
                            )
                            print("expected_losses", expected_losses)
                            for _step in range(self.num_steps):
                                current_losses = expected_losses[
                                    torch.arange(len(config)), config
                                ]
                                possible_improvements = (
                                    current_losses
                                    - expected_losses[
                                        torch.arange(len(config)), config + 1
                                    ]
                                )
                                best_option_idx = torch.argmax(possible_improvements)
                                config[best_option_idx] += 1

                            print("config", config)

                            probs = config / config.sum()
                            probs = (
                                probs * 0.8 + torch.ones(len(probs)) / len(probs) * 0.2
                            )

                        else:
                            # Use combined counts from both epochs for normalization
                            # Avoid division by zero by setting improvements to 0 where count is 0
                            mask = total_counts > 0
                            improvements[mask] = improvements[mask] / torch.sqrt(
                                total_counts[mask]
                            )
                            improvements[~mask] = 0.0
                            print("normalized improvements", improvements)

                            # Ensure all improvements are positive by shifting
                            improvements = improvements - improvements.min() + 1e-8

                            probs = improvements / improvements.sum()

                            # make sure that the smallest prob is at least 1/len(options)/10
                            probs = (
                                probs * 0.9 + torch.ones(len(probs)) / len(probs) * 0.1
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
                probs = torch.ones(len(self.importance_hyperparameter_options)) / len(
                    self.importance_hyperparameter_options
                )
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

        hp_indices = []
        hyperparameters_for_all_batches = []
        multipliers = []
        for _step in range(self.num_steps):
            hp_index = torch.multinomial(probs, 1).item()  # todo check this correct
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
