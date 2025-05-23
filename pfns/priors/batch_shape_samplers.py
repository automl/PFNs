import math
import random
from dataclasses import dataclass
from typing import Literal

from pfns import base_config


def get_weighted_single_eval_pos_sampler(max_len, min_len=0, p=1.0):
    """
    This gives a sampler that can be used for `single_eval_pos` which yields good performance for all positions p,
    where p <= `max_len`. At most `max_len` - 1 examples are shown to the Transformer.
    :return: Sampler that can be fed to `train()` as `single_eval_pos_gen`.
    """


@dataclass(frozen=True)
class BatchShapeSamplerConfig(base_config.BaseConfig):
    seq_len: int
    min: int = 0
    max: int | None = None
    type: Literal["uniform", "weighted"] = "uniform"

    def __post_init__(self):
        assert self.max is None or (self.min < self.max)
        assert self.max is None or (self.max < self.seq_len)

    def get_sampler(self):
        if self.max is None:
            max = self.seq_len - 1
        else:
            max = self.max
        if self.type == "uniform":
            return lambda: (
                random.choices(range(self.min, max))[0],
                self.seq_len,
            )
        else:
            p = 1.0  # the power of the weighted sampler
            return lambda: random.choices(
                range(self.min, max),
                [
                    1 / math.pow(((max - self.min) - i), p)
                    for i in range(max - self.min)
                ],
            )[0]
