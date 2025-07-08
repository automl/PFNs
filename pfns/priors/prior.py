import importlib
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, fields
from functools import partial
from typing import Callable, ClassVar, Optional, Set

import torch

from pfns.base_config import BaseConfig
from torch.utils.data import DataLoader


class PriorConfig(BaseConfig, metaclass=ABCMeta):
    @abstractmethod
    def create_get_batch_method(self) -> Callable:
        pass


@dataclass(frozen=True)
class AdhocPriorConfig(PriorConfig):
    # Set as a class variable instead of being set at init
    prior_names: str | Sequence[str] | None = None
    get_batch_methods: Callable | Sequence[Callable] | None = None
    prior_kwargs: dict | None = None

    strict_field_types: ClassVar[bool] = False

    def create_get_batch_method(self) -> Callable:
        # Local import to avoid circular import
        from pfns.priors import get_batch_sequence

        assert (
            (self.prior_names is None) != (self.get_batch_methods is None)
        ), f"Either prior_name or get_batch_method must be provided, got prior_names={self.prior_names} and get_batch_methods={self.get_batch_methods}"

        if self.prior_names is not None:
            get_batch_methods = []
            for prior_name in (
                self.prior_names
                if isinstance(self.prior_names, Sequence)
                else [self.prior_names]
            ):
                prior_module = importlib.import_module(f"pfns.priors.{prior_name}")
                get_batch_methods.append(prior_module.get_batch)
        else:
            get_batch_methods = (
                self.get_batch_methods
                if isinstance(self.get_batch_methods, Sequence)
                else [self.get_batch_methods]
            )

        return partial(get_batch_sequence(*get_batch_methods), **self.prior_kwargs)


@dataclass
class Batch:
    """
    A batch of data, with non-optional x, y, and target_y attributes.
    All other attributes are optional.

    If you want to add an attribute for testing only, you can just assign it after creation like:
    ```
        batch = Batch(x=x, y=y, target_y=target_y)
        batch.test_attribute = test_attribute
    ```
    """

    # Required entries
    x: torch.Tensor
    y: torch.Tensor
    target_y: torch.Tensor

    # Optional Batch Entries
    style: Optional[torch.Tensor] = None
    y_style: Optional[torch.Tensor] = None
    style_hyperparameter_values: Optional[torch.Tensor] = None
    single_eval_pos: Optional[torch.Tensor] = None
    causal_model_dag: Optional[object] = None
    mean_prediction: Optional[bool] = (
        None  # this controls whether to do mean prediction in bar_distribution for nonmyopic BO
    )
    info_used_with_gradient_magnitudes: Optional[dict] = None
    gradient_multipliers: Optional[torch.Tensor] = None

    def other_filled_attributes(
        self, set_of_attributes: Set[str] = frozenset(("x", "y", "target_y"))
    ):
        return [
            f.name
            for f in fields(self)
            if f.name not in set_of_attributes and getattr(self, f.name) is not None
        ]


class PriorDataLoader(DataLoader, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        num_steps,
        batch_size,
        eval_pos_seq_len_sampler,
        seq_len_maximum,
        device,
        **kwargs,
    ):
        """

        :param num_steps: int, first argument, the number of steps to take per epoch, i.e. iteration of the DataLoader
        :param batch_size: int, number of datasets per batch
        :param eval_pos_seq_len_sampler: callable, it takes no arguments and returns a tuple (single eval pos, bptt)
        :param kwargs: for future compatibility it is good to have a final all catch, as new kwargs might be introduced
        """
        pass

    # A class or object variable `num_features`: int
    # Optional: `validate` function that accepts a transformer model

    # The DataLoader iter should return batches of the form ([style], x, y), target_y, single_eval_pos
    # We follow sequence len (s) first, batch size (b) second. So x: (s,b,num_features), y,target_y: (s,b)
    # and style: Optional[(b,num_style_params)], style can be omitted or set to None, if it is not intended to be used.

    # For more references, see `priors/utils.py` for a pretty general implementation of a DataLoader
    # and `train.py` for the only call of it.
