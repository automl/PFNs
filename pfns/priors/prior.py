import importlib
from abc import ABCMeta, abstractmethod
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
    prior_name: str | None = None
    get_batch_method: Callable | None = None
    prior_kwargs: dict | None = None

    strict_field_types: ClassVar[bool] = False

    def create_get_batch_method(self) -> Callable:
        assert (
            (self.prior_name is None) != (self.get_batch_method is None)
        ), f"Either prior_name or get_batch_method must be provided, got prior_name={self.prior_name} and get_batch_method={self.get_batch_method}"

        if self.prior_name is not None:
            prior_module = importlib.import_module(
                f"pfns.priors.{self.prior_name}"
            )
            get_batch = getattr(prior_module, "get_batch")
        else:
            get_batch = self.get_batch_method

        return partial(get_batch, **self.prior_kwargs)


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
            if f.name not in set_of_attributes
            and getattr(self, f.name) is not None
        ]


def safe_merge_batches_in_batch_dim(*batches, ignore_attributes=[]):
    """
    Merge all supported non-None fields in a pre-specified (general) way,
    e.g. mutliple batch.x are concatenated in the batch dimension.
    :param ignore_attributes: attributes to remove from the merged batch, treated as if they were None.
    :return:
    """
    not_none_fields = [
        f.name
        for f in fields(batches[0])
        if f.name not in ignore_attributes
        and getattr(batches[0], f.name) is not None
    ]
    assert all(
        [
            set(not_none_fields)
            == set(
                [
                    f.name
                    for f in fields(b)
                    if f.name not in ignore_attributes
                    and getattr(b, f.name) is not None
                ]
            )
            for b in batches
        ]
    ), "All batches must have the same fields!"
    merge_funcs = {
        "x": lambda xs: torch.cat(xs, 1),
        "y": lambda ys: torch.cat(ys, 1),
        "target_y": lambda target_ys: torch.cat(target_ys, 1),
        "style": lambda styles: torch.cat(styles, 0),
    }
    assert all(
        f in merge_funcs for f in not_none_fields
    ), f"Unknown fields encountered in `safe_merge_batches_in_batch_dim`."
    return Batch(
        **{
            f: merge_funcs[f]([getattr(batch, f) for batch in batches])
            for f in not_none_fields
        }
    )


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
