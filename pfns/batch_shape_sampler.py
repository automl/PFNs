# We create a batch shape sampler, that determines the parts of the batch that determine how compute intensive it is.
# Putting this into one module is important to allow multi-gpu training, as we want to seed the sampling of this shape the same across workers
# s.t. they all take the same amount of time.

# It includes batch_size, seq_len, num_features, and single_eval_pos.

import random
from dataclasses import dataclass
from typing import Optional

from pfns.base_config import BaseConfig


@dataclass
class BatchShape:
    batch_size: int
    seq_len: int
    num_features: int
    single_eval_pos: Optional[int] = None

    def as_get_batch_kwargs(self):
        return {
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "num_features": self.num_features,
            "single_eval_pos": self.single_eval_pos,
        }


@dataclass(frozen=True)
class BatchShapeSamplerConfig(BaseConfig):
    batch_size: int = 32
    max_seq_len: int = 1000
    min_num_features: int = 1
    max_num_features: int = 16
    fixed_num_test_instances: Optional[int] = None

    seed: int = 42

    def sample_batch_shape(self, epoch: int, step: int) -> BatchShape:
        # Create deterministic seed based on epoch and step
        seed = self.seed + epoch * 10000 + step
        rng = random.Random(seed)

        # it seems to be beneficial to oversample small numbers of features
        num_features = rng.randint(self.min_num_features, self.max_num_features)

        single_eval_pos = rng.randint(
            0,
            self.max_seq_len
            - 1
            - (
                self.fixed_num_test_instances
                if self.fixed_num_test_instances is not None
                else 0
            ),
        )

        seq_len = self.max_seq_len
        if self.fixed_num_test_instances is not None:
            seq_len = self.fixed_num_test_instances + single_eval_pos

        # future todo: adapt batch_size and num_features based on seq_len -> shrinking them for large seq_lens
        return BatchShape(
            batch_size=self.batch_size,
            seq_len=seq_len,
            num_features=num_features,
            single_eval_pos=single_eval_pos,
        )
