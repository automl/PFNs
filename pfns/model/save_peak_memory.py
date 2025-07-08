#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

from collections.abc import Callable
from types import MethodType
from typing import Any

import torch


def support_save_peak_mem_factor(method: MethodType) -> Callable:
    """Can be applied to a method acting on a tensor 'x' whose first dimension is a
    flat batch dimension
    (i.e. the operation is trivially parallel over the first dimension).

    For additional tensor arguments, it is assumed that the first dimension is again
    the batch dimension, and that non-tensor arguments can be passed as-is
    to splits when parallelizing over the batch dimension.

    The decorator adds options 'add_input' to add the principal input 'x' to the
    result of the method and 'allow_inplace'.
    By setting 'allow_inplace', the caller indicates that 'x'
    is not used after the call and its buffer can be reused for the output.

    Setting 'allow_inplace' does not ensure that the operation will be inplace,
    and the return value should be used for clarity and simplicity.

    Moreover, it adds an optional int parameter 'save_peak_mem_factor' that is
    only supported in combination with 'allow_inplace' during inference and subdivides
    the operation into the specified number of chunks to reduce peak memory consumption.
    """

    def method_(
        self: torch.nn.Module,
        x: torch.Tensor,
        *args: torch.Tensor,
        add_input: bool = False,
        allow_inplace: bool = False,
        save_peak_mem_factor: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            x: The input tensor, over which we want to go sequentially in 0th dimension.
            *args: Additional arguments of the wrapped function.
            add_input: Whether to add the input to the result, to get a residual connection.
            allow_inplace: Whether to allow inplace operations.
            save_peak_mem_factor: The number of chunks to split the input into.
            **kwargs: Additional keyword arguments of the wrapped function.
        """
        assert isinstance(self, torch.nn.Module)
        assert (
            save_peak_mem_factor is None or allow_inplace
        ), "The parameter save_peak_mem_factor only supported with 'allow_inplace' set."
        assert isinstance(x, torch.Tensor)

        tensor_inputs = list(tuple(self.parameters()) + tuple(args))

        assert (
            save_peak_mem_factor is None
            or not any(t.requires_grad for t in tensor_inputs)
            or not torch.is_grad_enabled()
        ), "The parameter save_peak_mem_factor is only supported during inference."

        if save_peak_mem_factor is not None:
            assert isinstance(save_peak_mem_factor, int)
            assert save_peak_mem_factor > 1
            split_size = (x.size(0) + save_peak_mem_factor - 1) // save_peak_mem_factor

            split_args = zip(
                *[
                    (
                        torch.split(arg, split_size)
                        if isinstance(arg, torch.Tensor)
                        else [arg] * save_peak_mem_factor
                    )
                    for arg in (x, *args)
                ],
            )

            for x_, *args_ in split_args:
                if add_input:
                    x_[:] += method(self, x_, *args_, **kwargs)
                else:
                    x_[:] = method(self, x_, *args_, **kwargs)
            return x

        if add_input:
            return x + method(self, x, *args, **kwargs)

        return method(self, x, *args, **kwargs)

    return method_
