import math
import random
from dataclasses import dataclass
from typing import Literal

from pfns import base_config


@dataclass(frozen=True)
class BatchShapeSamplerConfig(base_config.BaseConfig):
    seq_len: int
    min: int = 0
    max: int | None = None
    type: Literal["uniform", "weighted"] = "uniform"

    def __post_init__(self):
        assert self.max is None or (self.min < self.max)
        assert self.max is None or (self.max < self.seq_len)

    def sample(self, rng: random.Random = random) -> tuple[int, int]:
        if self.max is None:
            max = self.seq_len - 1
        else:
            max = self.max
        if self.type == "uniform":
            return rng.choices(range(self.min, max))[0], self.seq_len
        else:
            p = 1.0  # the power of the weighted sampler
            return rng.choices(
                range(self.min, max),
                [
                    1 / math.pow(((max - self.min) - i), p)
                    for i in range(max - self.min)
                ],
            )[0], self.seq_len
