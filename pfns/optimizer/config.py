from dataclasses import dataclass

import torch

from pfns import base_config


@dataclass(frozen=True)
class OptimizerConfig(base_config.BaseConfig):
    optimizer: str
    lr: float | None
    weight_decay: float = 0.0
    frequency_of_heavy_lifting: int | None = None
    optim_warmup_steps: int = 0

    def create_optimizer(self, model_parameters):
        if self.optim_warmup_steps > 0:
            assert (
                self.optimizer == "sf_adamw"
            ), "warmup_steps is only supported for sf_adamw"

        if self.frequency_of_heavy_lifting is not None:
            assert (
                self.optimizer == "shampoo"
            ), "frequency_of_heavy_lifting is only supported for shampoo"

        if self.optimizer == "adam":
            return torch.optim.Adam(
                model_parameters, lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == "adamw":
            return torch.optim.AdamW(
                model_parameters, lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == "sf_adamw":
            from .adamw_schedulefree import AdamWScheduleFree

            return AdamWScheduleFree(
                model_parameters,
                lr=self.lr,
                weight_decay=self.weight_decay,
                warmup_steps=self.optim_warmup_steps,
            )
        elif self.optimizer == "shampoo":
            try:
                from optimizers.distributed_shampoo import (
                    AdamGraftingConfig,
                    DistributedShampoo,
                )
            except ImportError:
                raise ImportError(
                    "shampoo optimizer not found, please install optimizers package"
                )

            return DistributedShampoo(
                model_parameters,
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                epsilon=1e-12,
                max_preconditioner_dim=8192,
                precondition_frequency=self.frequency_of_heavy_lifting,  # todo, if this is good ablate larger values and see perf impact
                use_decoupled_weight_decay=True,
                grafting_config=AdamGraftingConfig(
                    beta2=0.999,
                    epsilon=1e-08,
                ),
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer} not supported")
