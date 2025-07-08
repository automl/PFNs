#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

from dataclasses import dataclass

from typing import Any

import numpy as np
import torch

from pfns import base_config
from pfns.model import encoders
from pfns.priors.hyperparameter_sampling import (
    DistributionConfig,
    HyperparameterNormalizer,
)
from torch import nn

### Simple Encoders


def get_linear_y_encoder(emsize):
    return SequentialEncoder(
        NanHandlingEncoderStep(),
        LinearInputEncoderStep(
            num_features=2,  # 2 for the value and the nan indicator
            emsize=emsize,
            out_keys=("output",),
            in_keys=("main", "nan_indicators"),
        ),
    )


def get_linear_x_encoder(emsize, features_per_group=1):
    return SequentialEncoder(
        VariableNumFeaturesEncoderStep(
            num_features=features_per_group,
        ),
        LinearInputEncoderStep(
            num_features=features_per_group,
            emsize=emsize,
        ),
    )


### Encoder Config


@dataclass(frozen=True)
class EncoderConfig(base_config.BaseConfig):
    variable_num_features_normalization: bool = False
    nan_handling: bool = False
    constant_normalization_mean: float = 0.0
    constant_normalization_std: float = 1.0
    train_normalization: bool = False

    def __post_init__(self):
        assert not (
            self.train_normalization
            and (
                self.constant_normalization_mean != 0.0
                or self.constant_normalization_std != 1.0
            )
        )
        return super().__post_init__()

    def create_encoder(self, features, emsize):
        encoder_sequence = []
        in_keys_for_linear = ("main",)
        if (
            self.constant_normalization_mean is not None
            or self.constant_normalization_std is not None
        ):
            encoder_sequence.append(
                ConstantNormalizationInputEncoderStep(
                    mean=self.constant_normalization_mean,
                    std=self.constant_normalization_std,
                )
            )
        if self.train_normalization:
            encoder_sequence.append(
                InputNormalizationEncoderStep(
                    normalize_on_train_only=True,
                    normalize_x=True,
                    remove_outliers=True,
                )
            )
        if self.variable_num_features_normalization:
            encoder_sequence.append(
                VariableNumFeaturesEncoderStep(
                    num_features=features,
                )
            )
        if self.nan_handling:
            encoder_sequence.append(NanHandlingEncoderStep())
            in_keys_for_linear = ("main", "nan_indicators")

        encoder_sequence.append(
            LinearInputEncoderStep(
                num_features=features * len(in_keys_for_linear),
                emsize=emsize,
                in_keys=in_keys_for_linear,
                out_keys=("output",),
            )
        )
        return SequentialEncoder(*encoder_sequence)


### Style Encoders


def linear_style_encoder(num_styles, emsize):
    return nn.Linear(num_styles, emsize)


@dataclass(frozen=True)
class StyleEncoderConfig(base_config.BaseConfig):
    num_styles: int | None = None
    normalize_to_hyperparameters: (
        dict[str, base_config.BaseTypes | DistributionConfig] | None
    ) = None
    encoder_type: str = "linear"

    def create_encoder(self, emsize):
        num_features = self.num_styles

        modules = []

        if self.normalize_to_hyperparameters is not None:
            assert (
                self.num_styles is None
            ), "num_styles must be None if normalize_to_hyperparameters is given"
            hpn = HyperparameterNormalizer(self.normalize_to_hyperparameters)
            num_features = hpn.num_hps * 2
            modules.append(hpn)

        if self.encoder_type == "linear":
            modules.append(encoders.linear_style_encoder(num_features, emsize))
            return nn.Sequential(*modules)
        else:
            raise ValueError(
                f"Style encoder generator {self.encoder_type} not supported"
            )


# Custom Encoders
# These encoders are meant to be used for both x and y


class SequentialEncoder(nn.Sequential):
    """Our Encoder class, which applies a sequence of encoder steps.

    Each step accepts a set of inputs, specified by `in_keys`, and outputs a set of outputs, specified by `out_keys`.
    The steps are worked off one after another, writing back the results into a dict.
    """

    def __init__(self, *args: SeqEncStep, output_key: str = "output", **kwargs: Any):
        """Initialize the SequentialEncoder.

        Args:
            *args: A list of SeqEncStep instances to apply in order.
            output_key:
                The key to use for the output of the encoder in the state dict.
                Defaults to "output", i.e. `state["output"]` will be returned.
            **kwargs: Additional keyword arguments passed to `nn.Sequential`.
        """
        super().__init__(*args, **kwargs)
        self.output_key = output_key

    def forward(
        self, input: dict[str, torch.Tensor] | torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Apply the sequence of encoder steps to the input.

        Args:
            input:
                The input state dictionary.
                If the input is not a dict and the first layer expects one input key,
                the input tensor is mapped to the key expected by the first layer.
            **kwargs: Additional keyword arguments passed to each encoder step.

        Returns:
            The output of the final encoder step.
        """
        # If the input is not a dict, we assume it is the main input and wrap it in a dict
        if not isinstance(input, dict):
            assert (
                len(self[0].in_keys) == 1 and self[0].in_keys[0] == "main"
            ), "The first encoder step must expect a single input key 'main', if the input is not a dict"
            input = {"main": input}

        for module in self:
            input = module(input, **kwargs)

        return input[self.output_key] if self.output_key is not None else input


class SeqEncStep(nn.Module):
    """Abstract base class for sequential encoder steps.

    SeqEncStep is a wrapper around a module that defines the expected input keys
    and the produced output keys. The outputs are assigned to the output keys
    in the order specified by `out_keys`.

    Subclasses should either implement `_forward` or `_fit` and `_transform`.
    Subclasses that transform `x` should always use `_fit` and `_transform`,
    creating any state that depends on the train set in `_fit` and using it in `_transform`.
    This allows fitting on data first and doing inference later without refitting.
    Subclasses that work with `y` can alternatively use `_forward` instead.
    """

    def __init__(
        self,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("main",),
    ):
        """Initialize the SeqEncStep.

        Args:
            in_keys: The keys of the input tensors.
            out_keys: The keys to assign the output tensors to.
        """
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys

    # Either implement _forward:

    def _forward(self, *x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        """Forward pass of the encoder step.

        Implement this if not implementing _fit and _transform.

        Args:
            *x: The input tensors. A single tensor or a tuple of tensors.
            **kwargs: Additional keyword arguments passed to the encoder step.

        Returns:
            The output tensor or a tuple of output tensors.
        """
        raise NotImplementedError()

    # Or implement _fit and _transform:

    def _fit(
        self,
        *x: torch.Tensor,
        single_eval_pos: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Fit the encoder step on the training set.

        Args:
            *x: The input tensors. A single tensor or a tuple of tensors.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments passed to the encoder step.
        """
        raise NotImplementedError

    def _transform(
        self,
        *x: torch.Tensor,
        single_eval_pos: int | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor]:
        """Transform the data using the fitted encoder step.

        Args:
            *x: The input tensors. A single tensor or a tuple of tensors.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments passed to the encoder step.

        Returns:
            The transformed output tensor or a tuple of output tensors.
        """
        raise NotImplementedError

    def forward(
        self,
        state: dict,
        cache_trainset_representation: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Perform the forward pass of the encoder step.

        Args:
            state: The input state dictionary containing the input tensors.
            cache_trainset_representation:
                Whether to cache the training set representation. Only supported for
                _fit and _transform (not _forward).
            **kwargs: Additional keyword arguments passed to the encoder step.

        Returns:
            The updated state dictionary with the output tensors assigned to the output keys.
        """
        args = [state[in_key] for in_key in self.in_keys]
        if hasattr(self, "_fit"):
            if kwargs["single_eval_pos"] or not cache_trainset_representation:
                self._fit(*args, **kwargs)
            out = self._transform(*args, **kwargs)
        else:
            assert not cache_trainset_representation
            out = self._forward(*args, **kwargs)

        assert isinstance(out, tuple)
        assert len(out) == len(self.out_keys)
        state.update({out_key: out[i] for i, out_key in enumerate(self.out_keys)})
        return state


class LinearInputEncoderStep(SeqEncStep):
    """A simple linear input encoder step."""

    def __init__(
        self,
        *,
        num_features: int,
        emsize: int,
        replace_nan_by_zero: bool = False,
        bias: bool = True,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("output",),
    ):
        """Initialize the LinearInputEncoderStep.

        Args:
            num_features: The number of input features.
            emsize: The embedding size, i.e. the number of output features.
            replace_nan_by_zero: Whether to replace NaN values in the input by zero. Defaults to False.
            bias: Whether to use a bias term in the linear layer. Defaults to True.
            in_keys: The keys of the input tensors. Defaults to ("main",).
            out_keys: The keys to assign the output tensors to. Defaults to ("output",).
        """
        super().__init__(in_keys, out_keys)
        self.layer = nn.Linear(num_features, emsize, bias=bias)
        self.replace_nan_by_zero = replace_nan_by_zero

    def _fit(self, *x: torch.Tensor, **kwargs: Any):
        """Fit the encoder step. Does nothing for LinearInputEncoderStep."""
        pass

    def _transform(self, *x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        """Apply the linear transformation to the input.

        Args:
            *x: The input tensors to concatenate and transform.
            **kwargs: Unused keyword arguments.

        Returns:
            A tuple containing the transformed tensor.
        """
        x = torch.cat(x, dim=-1)
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)
        return (self.layer(x),)


class ConstantNormalizationInputEncoderStep(SeqEncStep):
    """
    An encoder step that subtracts all inputs by a constant mean and divides by a constant standard deviation.
    """

    def __init__(
        self,
        *,
        mean: float,
        std: float,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("main",),
    ):
        assert len(in_keys) == 1 and len(out_keys) == 1
        super().__init__(in_keys, out_keys)
        self.mean = mean
        self.std = std

    def _fit(self, *x: torch.Tensor, **kwargs: Any):
        pass

    def _transform(self, *x: torch.Tensor, **kwargs: Any):
        assert len(x) == 1
        x = x[0]
        return ((x - self.mean) / self.std,)


class SqueezeBetween0and1(nn.Module):  # take care of test set here
    def forward(self, x):
        width = x.max(0).values - x.min(0).values
        result = (x - x.min(0).values) / width
        result[(width == 0)[None].repeat(len(x), *[1] * (len(x.shape) - 1))] = 0.5
        return result


class NanHandlingEncoderStep(SeqEncStep):
    """Encoder step to handle NaN and infinite values in the input."""

    nan_indicator = -2.0
    inf_indicator = 2.0
    neg_inf_indicator = 4.0

    def __init__(
        self,
        keep_nans: bool = True,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("main", "nan_indicators"),
    ):
        """Initialize the NanHandlingEncoderStep.

        Args:
            keep_nans: Whether to keep NaN values as separate indicators. Defaults to True.
            in_keys: The keys of the input tensors. Must be a single key.
            out_keys: The keys to assign the output tensors to.
        """
        assert len(in_keys) == 1, "NanHandlingEncoderStep expects a single input key"
        super().__init__(in_keys, out_keys)
        self.keep_nans = keep_nans
        self.register_buffer("feature_means_", torch.tensor([]), persistent=False)

    def _fit(self, x: torch.Tensor, single_eval_pos: int, **kwargs: Any) -> None:
        """Compute the feature means on the training set for replacing NaNs.

        Args:
            x: The input tensor.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments (unused).
        """
        self.feature_means_ = torch_nanmean(x[:single_eval_pos], axis=0)

    def _transform(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace NaN and infinite values in the input tensor.

        Args:
            x: The input tensor.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the transformed tensor and optionally the NaN indicators.
        """
        nans_indicator = None
        if self.keep_nans:
            # TODO: There is a bug here: The values arriving here are already mapped to nan if they were inf before
            nans_indicator = (
                torch.isnan(x) * NanHandlingEncoderStep.nan_indicator
                + torch.logical_and(torch.isinf(x), torch.sign(x) == 1)
                * NanHandlingEncoderStep.inf_indicator
                + torch.logical_and(torch.isinf(x), torch.sign(x) == -1)
                * NanHandlingEncoderStep.neg_inf_indicator
            ).to(x.dtype)

        nan_mask = torch.logical_or(torch.isnan(x), torch.isinf(x))
        # replace nans with the mean of the corresponding feature
        x = x.clone()  # clone to avoid inplace operations
        x[nan_mask] = self.feature_means_.unsqueeze(0).expand_as(x)[nan_mask]

        return x, nans_indicator


class VariableNumFeaturesEncoderStep(SeqEncStep):
    """Encoder step to handle variable number of features.

    Transforms the input to a fixed number of features by appending zeros.
    Also normalizes the input by the number of used features to keep the variance
    of the input constant, even when zeros are appended.
    """

    def __init__(
        self,
        num_features: int,
        normalize_by_used_features: bool = True,
        normalize_by_sqrt: bool = True,
        **kwargs: Any,
    ):
        """Initialize the VariableNumFeaturesEncoderStep.

        Args:
            num_features: The number of features to transform the input to.
            normalize_by_used_features: Whether to normalize by the number of used features.
            normalize_by_sqrt: Legacy option to normalize by sqrt instead of the number of used features.
            **kwargs: Keyword arguments passed to the parent SeqEncStep.
        """
        super().__init__(**kwargs)
        self.normalize_by_used_features = normalize_by_used_features
        self.num_features = num_features
        self.normalize_by_sqrt = normalize_by_sqrt
        self.number_of_used_features_ = None

    def _fit(self, x: torch.Tensor, **kwargs: Any) -> None:
        """Compute the number of used features on the training set.

        Args:
            x: The input tensor.
            **kwargs: Additional keyword arguments (unused).
        """
        sel = (x[1:] == x[0]).sum(0) != (x.shape[0] - 1)
        self.number_of_used_features_ = torch.clip(
            sel.sum(-1).unsqueeze(-1),
            min=1,
        ).cpu()

    def _transform(self, x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        """Transform the input tensor to have a fixed number of features.

        Args:
            x: The input tensor of shape (seq_len, batch_size, num_features_old).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the transformed tensor of shape (seq_len, batch_size, num_features).
        """
        if x.shape[2] == 0:
            return torch.zeros(
                x.shape[0],
                x.shape[1],
                self.num_features,
                device=x.device,
                dtype=x.dtype,
            )
        if self.normalize_by_used_features:
            if self.normalize_by_sqrt:
                # Verified that this gives indeed unit variance with appended zeros
                x = x * torch.sqrt(
                    self.num_features / self.number_of_used_features_.to(x.device),
                )
            else:
                x = x * (self.num_features / self.number_of_used_features_.to(x.device))

        zeros_appended = torch.zeros(
            *x.shape[:-1],
            self.num_features - x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat([x, zeros_appended], -1)
        return (x,)


class InputNormalizationEncoderStep(SeqEncStep):
    """Encoder step to normalize the input in different ways.

    Can be used to normalize the input to a ranking, remove outliers,
    or normalize the input to have unit variance.
    """

    def __init__(
        self,
        normalize_on_train_only: bool,
        normalize_x: bool,
        remove_outliers: bool,
        remove_outliers_sigma: float = 4.0,
        seed: int = 0,
        **kwargs: Any,
    ):
        """Initialize the InputNormalizationEncoderStep.

        Args:
            normalize_on_train_only: Whether to compute normalization only on the training set.
            normalize_x: Whether to normalize the input to have unit variance.
            remove_outliers: Whether to remove outliers from the input.
            remove_outliers_sigma: The number of standard deviations to use for outlier removal.
            seed: Random seed for reproducibility.
            **kwargs: Keyword arguments passed to the parent SeqEncStep.
        """
        super().__init__(**kwargs)
        self.normalize_on_train_only = normalize_on_train_only
        self.normalize_x = normalize_x
        self.remove_outliers = remove_outliers
        self.remove_outliers_sigma = remove_outliers_sigma
        self.seed = seed
        self.reset_seed()
        self.lower_for_outlier_removal = None
        self.upper_for_outlier_removal = None
        self.mean_for_normalization = None
        self.std_for_normalization = None

    def reset_seed(self) -> None:
        """Reset the random seed."""

    def _fit(self, x: torch.Tensor, single_eval_pos: int, **kwargs: Any) -> None:
        """Compute the normalization statistics on the training set.

        Args:
            x: The input tensor.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments (unused).
        """
        normalize_position = single_eval_pos if self.normalize_on_train_only else -1
        if self.remove_outliers:
            (
                x,
                (
                    self.lower_for_outlier_removal,
                    self.upper_for_outlier_removal,
                ),
            ) = remove_outliers(
                x,
                normalize_positions=normalize_position,
                n_sigma=self.remove_outliers_sigma,
            )

        if self.normalize_x:
            (
                x,
                (
                    self.mean_for_normalization,
                    self.std_for_normalization,
                ),
            ) = normalize_data(
                x,
                normalize_positions=normalize_position,
                return_scaling=True,
            )

    def _transform(
        self,
        x: torch.Tensor,
        single_eval_pos: int,
        **kwargs: Any,
    ) -> tuple[torch.Tensor]:
        """Normalize the input tensor.

        Args:
            x: The input tensor.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the normalized tensor.
        """
        normalize_position = single_eval_pos if self.normalize_on_train_only else -1

        if self.remove_outliers:
            assert (
                self.remove_outliers_sigma > 1.0
            ), "remove_outliers_sigma must be > 1.0"

            x, _ = remove_outliers(
                x,
                normalize_positions=normalize_position,
                lower=self.lower_for_outlier_removal,
                upper=self.upper_for_outlier_removal,
                n_sigma=self.remove_outliers_sigma,
            )

        if self.normalize_x:
            x = normalize_data(
                x,
                normalize_positions=normalize_position,
                mean=self.mean_for_normalization,
                std=self.std_for_normalization,
            )

        return (x,)


### Helper functions


# usage of custom implementations is required to support ONNX export
def torch_nansum(x: torch.Tensor, axis=None, keepdim=False, dtype=None):
    nan_mask = torch.isnan(x)
    masked_input = torch.where(
        nan_mask,
        torch.tensor(0.0, device=x.device, dtype=x.dtype),
        x,
    )
    return torch.sum(masked_input, axis=axis, keepdim=keepdim, dtype=dtype)


def torch_nanmean(
    x: torch.Tensor,
    axis: int = 0,
    *,
    return_nanshare: bool = False,
    include_inf: bool = False,
):
    nan_mask = torch.isnan(x)
    if include_inf:
        nan_mask = torch.logical_or(nan_mask, torch.isinf(x))

    num = torch.where(nan_mask, torch.full_like(x, 0), torch.full_like(x, 1)).sum(  # type: ignore
        axis=axis,
    )
    value = torch.where(nan_mask, torch.full_like(x, 0), x).sum(axis=axis)  # type: ignore
    if return_nanshare:
        return value / num, 1.0 - num / x.shape[axis]
    return value / num.clip(min=1.0)


def torch_nanstd(x: torch.Tensor, axis: int = 0):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum(  # type: ignore
        axis=axis,
    )
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum(axis=axis)  # type: ignore
    mean = value / num
    mean_broadcast = torch.repeat_interleave(
        mean.unsqueeze(axis),
        x.shape[axis],
        dim=axis,
    )
    return torch.sqrt(
        torch_nansum(torch.square(mean_broadcast - x), axis=axis) / (num - 1),  # type: ignore
    )


def normalize_data(
    data: torch.Tensor,
    *,
    normalize_positions: int = -1,
    return_scaling: bool = False,
    clip: bool = True,
    std_only: bool = False,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Normalize data to mean 0 and std 1.

    Args:
        data: The data to normalize. (T, B, H)
        normalize_positions: If > 0, only use the first `normalize_positions` positions for normalization.
        return_scaling: If True, return the scaling parameters as well (mean, std).
        std_only: If True, only divide by std.
        clip: If True, clip the data to [-100, 100].
        mean: If given, use this value instead of computing it.
        std: If given, use this value instead of computing it.
    """
    # TODO(eddiebergman): I feel like this function is easier to just do what you need
    # where you need it, rather than supporting all these variations
    assert (mean is None) == (
        std is None
    ), "Either both or none of mean and std must be given"
    if mean is None:
        if normalize_positions is not None and normalize_positions > 0:
            mean = torch_nanmean(data[:normalize_positions], axis=0)  # type: ignore
            std = torch_nanstd(data[:normalize_positions], axis=0) + 1e-20
        else:
            mean = torch_nanmean(data, axis=0)  # type: ignore
            std = torch_nanstd(data, axis=0) + 1e-20

        if len(data) == 1 or normalize_positions == 1:
            std[:] = 1.0

        if std_only:
            mean[:] = 0  # type: ignore
    data = (data - mean) / std

    if clip:
        data = torch.clip(data, min=-100, max=100)

    if return_scaling:
        return data, (mean, std)  # type: ignore
    return data


def select_features(x: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    """Select features from the input tensor based on the selection mask,
    and arrange them contiguously in the last dimension.
    If batch size is bigger than 1, we pad the features with zeros to make the number of features fixed.

    Args:
        x: The input tensor of shape (sequence_length, batch_size, total_features)
        sel: The boolean selection mask indicating which features to keep of shape (batch_size, total_features)

    Returns:
        The tensor with selected features.
        The shape is (sequence_length, batch_size, number_of_selected_features) if batch_size is 1.
        The shape is (sequence_length, batch_size, total_features) if batch_size is greater than 1.
    """
    B, total_features = sel.shape
    sequence_length = x.shape[0]

    # If B == 1, we don't need to append zeros, as the number of features don't need to be fixed.
    if B == 1:
        return x[:, :, sel[0]]

    new_x = torch.zeros(
        (sequence_length, B, total_features),
        device=x.device,
        dtype=x.dtype,
    )

    # For each batch, compute the number of selected features.
    sel_counts = sel.sum(dim=-1)  # shape: (B,)

    for b in range(B):
        s = int(sel_counts[b])
        if s > 0:
            new_x[:, b, :s] = x[:, b, sel[b]]

    return new_x


def remove_outliers(
    X: torch.Tensor,
    n_sigma: float = 4,
    normalize_positions: int = -1,
    lower: None | torch.Tensor = None,
    upper: None | torch.Tensor = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    # Expects T, B, H
    assert (lower is None) == (upper is None), "Either both or none of lower and upper"
    assert len(X.shape) == 3, "X must be T,B,H"
    # for b in range(X.shape[1]):
    # for col in range(X.shape[2]):
    if lower is None:
        data = X if normalize_positions == -1 else X[:normalize_positions]
        data_clean = data[:].clone()
        data_mean, data_std = (
            torch_nanmean(data, axis=0),
            torch_nanstd(data, axis=0),
        )
        cut_off = data_std * n_sigma
        lower, upper = data_mean - cut_off, data_mean + cut_off

        data_clean[torch.logical_or(data_clean > upper, data_clean < lower)] = np.nan
        data_mean, data_std = (
            torch_nanmean(data_clean, axis=0),
            torch_nanstd(data_clean, axis=0),
        )
        cut_off = data_std * n_sigma
        lower, upper = data_mean - cut_off, data_mean + cut_off

    X = torch.maximum(-torch.log(1 + torch.abs(X)) + lower, X)
    X = torch.minimum(torch.log(1 + torch.abs(X)) + upper, X)
    return X, (lower, upper)
