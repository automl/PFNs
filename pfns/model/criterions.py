import torch
import torch.nn as nn
from pfns import base_config

from .bar_distribution import BarDistributionConfig


class CrossEntropyConfig(base_config.BaseConfig):
    reduction: str = "none"
    num_classes: int

    def get_criterion(self):
        return nn.CrossEntropyLoss(
            reduction=self.reduction, weight=torch.ones(self.num_classes)
        )


__all__ = ["BarDistributionConfig", "CrossEntropyConfig"]
