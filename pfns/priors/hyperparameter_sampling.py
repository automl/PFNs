# this is a wrapper prior that samples hyperparameters which are set to be custom distribution parameters
import math
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from pfns.base_config import BaseConfig
from pfns.priors import Batch


@dataclass(frozen=True)
class DistributionConfig(BaseConfig):
    """Base class for hyperparameter distributions"""

    def sample(self):
        raise NotImplementedError

    def normalize(self, value):
        raise NotImplementedError

    def encode_to_torch(self, value):
        return value


@dataclass(frozen=True)
class UniformFloatDistConfig(DistributionConfig):
    lower: float
    upper: float
    log: bool = False

    def __post_init__(self):
        if self.log:
            assert self.lower > 0, "lower must be positive for log sampling"
        return super().__post_init__()

    def sample(self):
        if self.log:
            log_lower = math.log(self.lower)
            log_upper = math.log(self.upper)
            return math.exp(random.uniform(log_lower, log_upper))
        return random.uniform(self.lower, self.upper)

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        if self.log:
            return (torch.log(value) - math.log(self.lower)) / (
                math.log(self.upper) - math.log(self.lower)
            )
        return (value - self.lower) / (self.upper - self.lower)

    def encode_to_torch(self, value):
        assert (value >= self.lower) and (
            value <= self.upper
        ), f"Value {value} not in range [{self.lower}, {self.upper}]"
        return value


@dataclass(frozen=True)
class PowerUniformFloatDistConfig(DistributionConfig):
    """Distribution that samples using a power transformation, providing a middle ground
    between uniform and log-uniform sampling.

    The transformation uses x^(1/power) where:
    - power = 1 gives uniform sampling
    - power approaching infinity gives behavior similar to log sampling
    - power = 2 gives a square root transformation (good middle ground)
    """

    lower: float
    upper: float
    power: float

    def __post_init__(self):
        assert self.lower >= 0, "lower bound must be non-negative for power sampling"
        assert self.upper > self.lower, "upper bound must be greater than lower bound"
        assert self.power > 0, "power must be positive"
        return super().__post_init__()

    def sample(self):
        # Sample uniformly in transformed space then transform back
        u = random.uniform(0, 1)
        transformed_lower = self.lower ** (1 / self.power)
        transformed_upper = self.upper ** (1 / self.power)
        transformed_value = (
            u * (transformed_upper - transformed_lower) + transformed_lower
        )
        return transformed_value**self.power

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        transformed_value = torch.pow(value, 1 / self.power)
        transformed_lower = self.lower ** (1 / self.power)
        transformed_upper = self.upper ** (1 / self.power)
        return (transformed_value - transformed_lower) / (
            transformed_upper - transformed_lower
        )

    def encode_to_torch(self, value):
        assert (value >= self.lower) and (
            value <= self.upper
        ), f"Value {value} not in range [{self.lower}, {self.upper}]"
        return value


@dataclass(frozen=True)
class UniformIntegerDistConfig(DistributionConfig):
    lower: int
    upper: int
    log: bool = False

    def __post_init__(self):
        if self.log:
            assert self.lower > 0, "lower must be positive for log sampling"
        return super().__post_init__()

    def sample(self):
        if self.log:
            log_lower = math.log(self.lower)
            log_upper = math.log(self.upper)
            return round(math.exp(random.uniform(log_lower, log_upper)))
        return random.randint(self.lower, self.upper)

    def normalize(self, value):
        if self.log:
            return (torch.log(value) - math.log(self.lower)) / (
                math.log(self.upper) - math.log(self.lower)
            )
        return (value - self.lower) / (self.upper - self.lower)

    def encode_to_torch(self, value):
        assert (value >= self.lower) and (
            value <= self.upper
        ), f"Value {value} not in range [{self.lower}, {self.upper}]"
        return value


@dataclass(frozen=True)
class ChoiceDistConfig(DistributionConfig):
    choices: list[str]

    def sample(self):
        return random.choice(self.choices)

    def encode_to_torch(self, value: str):
        assert value in self.choices, f"Value {value} not in choices {self.choices}"
        return torch.tensor(self.choices.index(value))

    def normalize(self, value: torch.Tensor):
        # Return one-hot encoding of the choice
        return value / (len(self.choices) - 1)


def sample_hyperparameters(config):
    """Sample values for all hyperparameters in the config"""
    sampled_config = deepcopy(config)
    for k, v in config.items():
        if isinstance(v, DistributionConfig):
            sampled_config[k] = v.sample()
        elif isinstance(v, dict):
            sampled_config[k] = sample_hyperparameters(v)
    return sampled_config


def find_all_distribution_hps(hyperparameters, path=None):
    if path is None:
        path = []
    for k, v in hyperparameters.items():
        current_path = path + [k]
        if isinstance(v, DistributionConfig):
            yield ".".join(current_path)
        elif isinstance(v, dict):
            yield from find_all_distribution_hps(v, current_path)


def access_dict_with_path(d: dict[str, Any], path: str | list[str]):
    if isinstance(path, str):
        path = path.split(".")
    for k in path:
        d = d[k]
    return d


def get_all_styled_hps(hyperparameters):
    """Get a list of hyperparameter names that should be added as style.

    Args:
        hyperparameters: The hyperparameter config dict

    Returns:
        List of hyperparameter names that should be added as style, or empty list if none
    """
    hps_as_style: str | list[str] = hyperparameters[
        "hyperparameter_sampling_add_hps_to_style"
    ]
    if hps_as_style == "all_sampled":
        return list(find_all_distribution_hps(hyperparameters))
    else:
        assert isinstance(
            hps_as_style, list
        ), "hyperparameter_sampling_add_hps_to_style must be a list of strings or 'all_sampled'"
        return hps_as_style


def get_batch(
    batch_size, *args, hyperparameters, get_batch, batch_size_per_gp_sample=1, **kwargs
):
    hyperparameters = deepcopy(hyperparameters)
    assert (
        batch_size % batch_size_per_gp_sample == 0
    ), f"batch_size {batch_size} must be a multiple of batch_size_per_gp_sample {batch_size_per_gp_sample}"
    num_models = batch_size // batch_size_per_gp_sample
    skip_prob = hyperparameters.pop("hyperparameter_sampling_skip_style_prob")
    hps_as_style = get_all_styled_hps(hyperparameters)
    del hyperparameters["hyperparameter_sampling_add_hps_to_style"]

    if num_models == -1:
        num_models = batch_size
    assert batch_size % num_models == 0, "batch_size must be a multiple of num_models"

    sub_batches = []
    sub_hps = []
    for _i in range(num_models):
        hyperparameters_sample = sample_hyperparameters(hyperparameters)
        sub_batch = get_batch(
            batch_size // num_models,
            *args,
            hyperparameters=hyperparameters_sample,
            batch_size_per_gp_sample=batch_size_per_gp_sample,
            **kwargs,
        )
        sub_batches.append(sub_batch)
        sub_hps.append(hyperparameters_sample)

    assert all(
        not b.other_filled_attributes(set_of_attributes=("x", "y", "target_y"))
        for b in sub_batches
    ), f"Batch {[b.other_filled_attributes(set_of_attributes=('x', 'y', 'target_y')) for b in sub_batches if b.other_filled_attributes(set_of_attributes=('x', 'y', 'target_y'))]} has other attributes filled in."

    batch = Batch(
        x=torch.cat([b.x for b in sub_batches], dim=0),
        y=torch.cat([b.y for b in sub_batches], dim=0),
        target_y=torch.cat([b.target_y for b in sub_batches], dim=0),
    )

    if hps_as_style:
        # Probability to not add hyperparameters as style
        for i, b in enumerate(sub_batches):
            assert b.style is None
            b.style = torch.zeros((b.x.shape[0], len(hps_as_style)))
            for j, hp in enumerate(hps_as_style):
                hp_value = access_dict_with_path(sub_hps[i], hp)
                # Randomly set some hyperparameters to NaN based on skip_prob
                if random.random() < skip_prob:
                    b.style[:, j] = float("nan")
                else:
                    b.style[:, j] = float(
                        access_dict_with_path(hyperparameters, hp).encode_to_torch(
                            hp_value
                        )
                    )

        batch.style = torch.cat([b.style for b in sub_batches], dim=0)

        batch.y_style = batch.style
    return batch


class HyperparameterNormalizer(torch.nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.to_be_encoded_hyperparameters = get_all_styled_hps(self.hyperparameters)
        self.num_hps = len(self.to_be_encoded_hyperparameters)

    def hyperparameters_dict_to_tensor(self, raw_hyperparameters: dict) -> torch.Tensor:
        """Convert a dictionary of hyperparameters to a tensor format.
        All non-set hyperparameters are set to nan, which means "unknown", if the "hyperparameter_sampling_skip_style_prob" is set > 0.

        Args:
            raw_hyperparameters: Dictionary mapping hyperparameter names to their values

        Returns:
            torch.Tensor: A tensor containing the hyperparameter values in the correct order

        Raises:
            ValueError: If there are missing or unexpected hyperparameters
        """
        # Convert sets for comparison
        expected_keys = set(self.to_be_encoded_hyperparameters)
        provided_keys = set(raw_hyperparameters.keys())

        extra_keys = provided_keys - expected_keys

        if extra_keys:
            raise ValueError(f"Unexpected hyperparameters provided: {extra_keys}")

        # Convert dict to tensor format
        values = [
            access_dict_with_path(self.hyperparameters, hp).encode_to_torch(
                raw_hyperparameters[hp]
            )
            if hp in raw_hyperparameters
            else float("nan")
            for hp in self.to_be_encoded_hyperparameters
        ]
        return torch.tensor([values], dtype=torch.float32)

    def forward(self, raw_hyperparameters: torch.Tensor):
        # Create output tensor with twice the number of features
        # First half for normalized values, second half for nan indicators
        encoded_x = torch.zeros(
            (raw_hyperparameters.shape[0], self.num_hps * 2),
            device=raw_hyperparameters.device,
            dtype=raw_hyperparameters.dtype,
        )

        for i, hp in enumerate(self.to_be_encoded_hyperparameters):
            hp_value = access_dict_with_path(self.hyperparameters, hp)

            # Create nan mask for this hyperparameter
            is_nan = torch.isnan(raw_hyperparameters[:, i])

            # Set nan indicator in the second half of features
            encoded_x[:, i + self.num_hps] = torch.where(is_nan, 1.0, -1.0)

            # Process non-nan values
            non_nan_indices = ~is_nan
            if non_nan_indices.any():
                if isinstance(hp_value, DistributionConfig):
                    normalized = hp_value.normalize(
                        raw_hyperparameters[non_nan_indices, i]
                    )
                    encoded_x[non_nan_indices, i] = normalized
                else:
                    raise NotImplementedError(
                        f"Hyperparameter type {type(hp_value)} not implemented"
                    )

                # Scale from [0,1] to [-1,1]
                encoded_x[non_nan_indices, i] = 2 * encoded_x[non_nan_indices, i] - 1

        assert encoded_x.isnan().sum() == 0, "Encoded styles have nans"

        return encoded_x
