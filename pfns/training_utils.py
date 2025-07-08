import typing as tp
from dataclasses import dataclass

import torch


class EpochResult(tp.NamedTuple):
    loss: float  # total loss for the epoch
    data_time: float  # time spent getting batch data
    forward_time: float  # time spent in forward pass
    step_time: float  # total time per step
    nan_share: float  # share of NaN values
    ignore_share: float  # share of ignored values (-100)
    importance_sampling_infos: list  # gradient magnitude info


@dataclass
class Metrics:
    steps_per_epoch: int
    total_loss: float = 0.0
    nan_steps: float = 0.0
    ignore_steps: float = 0.0
    forward_time: float = 0.0
    step_time: float = 0.0
    time_to_get_batch: float = 0.0

    @torch.no_grad()
    def update(
        self,
        loss: torch.Tensor,
        nan_share: float,
        targets: torch.Tensor,
        forward_time: float,
        step_time: float,
        time_to_get_batch: float,
    ):
        self.total_loss += loss.cpu().detach().item()

        self.nan_steps += nan_share
        self.ignore_steps += (targets == -100).float().mean()
        self.forward_time += forward_time
        self.step_time += step_time
        self.time_to_get_batch += time_to_get_batch

    def get_epoch_result(self, importance_sampling_infos: list[tuple]):
        return EpochResult(
            loss=self.total_loss / self.steps_per_epoch,
            data_time=self.time_to_get_batch / self.steps_per_epoch,
            forward_time=self.forward_time / self.steps_per_epoch,
            step_time=self.step_time / self.steps_per_epoch,
            nan_share=self.nan_steps.cpu().item() / self.steps_per_epoch,
            ignore_share=self.ignore_steps.cpu().item() / self.steps_per_epoch,
            importance_sampling_infos=importance_sampling_infos,
        )


@torch.no_grad()
def update_importance_sampling_infos(
    importance_sampling_infos: list,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: float,
    info_used_with_gradient_magnitudes: list,
):
    squared_grad_magnitudes = {
        name: (w.grad**2).sum().cpu().item()
        for name, w in model.named_parameters()
        if w.grad is not None
    }
    total_grad_magnitude = sum(squared_grad_magnitudes.values())

    normalized_squared_grad_magnitudes = {}
    total_normalized_grad_magnitude = None
    # Compute grad magnitude normalized by Adam's beta2 parameter if Adam optimizer is used
    if squared_grad_magnitudes and isinstance(
        optimizer, (torch.optim.Adam, torch.optim.AdamW)
    ):
        beta2 = optimizer.param_groups[0]["betas"][1]
        # Get the current state of Adam's running average of squared gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                state = optimizer.state.get(param, {})
                if "exp_avg_sq" in state:
                    # Normalize the squared gradient by the running average
                    normalized_grad_magnitude = (
                        (
                            (param.grad**2)
                            / (
                                state["exp_avg_sq"]
                                * (1 - beta2 ** state.get("step", 1))
                                + 1e-8
                            )
                        )
                        .sum()
                        .cpu()
                        .item()
                    )
                    normalized_squared_grad_magnitudes[name] = normalized_grad_magnitude
        total_normalized_grad_magnitude = sum(
            v for k, v in normalized_squared_grad_magnitudes.items()
        )

    importance_sampling_infos.append(
        (
            total_grad_magnitude,
            info_used_with_gradient_magnitudes,
            loss,
            total_normalized_grad_magnitude,
        )
    )


def set_model_to(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    mode: tp.Literal["train", "eval"],
):
    assert mode in [
        "train",
        "eval",
    ], f"mode must be 'train' or 'eval', got {mode}"
    if mode == "train":
        model.train()
        if hasattr(optimizer, "train"):
            optimizer.train()
    else:
        model.eval()
        if hasattr(optimizer, "eval"):
            optimizer.eval()


def move_y_style_and_check_shape(
    y_style: torch.Tensor | None, y: torch.Tensor, device: torch.device
) -> torch.Tensor | None:
    y_style = y_style.to(device) if y_style is not None else None
    if y_style is not None:
        if y_style.dim() == 2:
            broken = y_style.shape[0] != y.shape[0]
        else:
            raise ValueError(f"y_style must have 2 dimensions, got {y_style.shape}")
        if broken:
            raise ValueError(
                f"y_style must have the same batch size as y, got {y_style.shape=} "
                f"and {y.shape=}"
            )
    return y_style


def move_style_and_check_shape(
    style: torch.Tensor | None, x: torch.Tensor, device: torch.device
) -> torch.Tensor | None:
    style = style.to(device) if style is not None else None
    if style is not None:
        if style.dim() == 2:
            broken = style.shape[0] != x.shape[0]
        elif style.dim() == 3:
            broken = style.shape[0] != x.shape[0] or style.shape[1] != x.shape[2]
        else:
            raise ValueError(f"style must have 2 or 3 dimensions, got {style.shape}")
        if broken:
            raise ValueError(
                f"style must have the same batch size as x and if it has 3 dimensions, "
                f"the middle dimension must match the number of features, got {style.shape=} "
                f"and {x.shape=}"
            )
    return style
