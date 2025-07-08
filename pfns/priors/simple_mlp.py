import math

import torch
from torch import nn

from ..utils import default_device, print_once
from .prior import Batch


class MLP(torch.nn.Module):
    def __init__(
        self,
        num_inputs,
        num_layers,
        num_hidden,
        num_outputs,
        init_std=None,
        sparseness=0.0,
        preactivation_noise_std=0.0,
        activation="tanh",
    ):
        super(MLP, self).__init__()
        if num_layers == 1:
            self.linears = nn.ModuleList([nn.Linear(num_inputs, num_outputs)])
        else:
            self.linears = nn.ModuleList(
                [nn.Linear(num_inputs, num_hidden)]
                + [nn.Linear(num_hidden, num_hidden) for _ in range(num_layers - 2)]
                + [nn.Linear(num_hidden, num_outputs)]
            )

        self.init_std = init_std
        self.sparseness = sparseness
        self.reset_parameters()

        self.preactivation_noise_std = preactivation_noise_std
        self.activation = {
            "tanh": torch.nn.Tanh(),
            "relu": torch.nn.ReLU(),
            "elu": torch.nn.ELU(),
            "identity": torch.nn.Identity(),
        }[activation]

    def reset_parameters(self, init_std=None, sparseness=None):
        init_std = init_std if init_std is not None else self.init_std
        sparseness = sparseness if sparseness is not None else self.sparseness
        for linear in self.linears:
            linear.reset_parameters()

        with torch.no_grad():
            if init_std is not None:
                for linear in self.linears:
                    linear.weight.normal_(0, init_std)
                    linear.bias.normal_(0, init_std)

            if sparseness > 0.0:
                for linear in self.linears[1:-1]:
                    linear.weight /= (1.0 - sparseness) ** (1 / 2)
                    linear.weight *= torch.bernoulli(
                        torch.ones_like(linear.weight) * (1.0 - sparseness)
                    )

    def forward(self, x):
        for linear in self.linears[:-1]:
            x = linear(x)
            x = x + torch.randn_like(x) * self.preactivation_noise_std
            x = self.activation(x)
        x = self.linears[-1](x)
        return x


def sample_input(
    input_sampling_setting,
    batch_size,
    seq_len,
    num_features,
    device=default_device,
):
    if input_sampling_setting == "normal":
        x = torch.randn(batch_size, seq_len, num_features, device=device)
        x_for_mlp = x
    elif input_sampling_setting == "uniform":
        x = torch.rand(batch_size, seq_len, num_features, device=device)
        x_for_mlp = (x - 0.5) / math.sqrt(1 / 12)
    else:
        raise ValueError(f"Unknown input_sampling: {input_sampling_setting}")
    return x, x_for_mlp


@torch.no_grad()
def get_batch(
    batch_size,
    seq_len,
    num_features,
    hyperparameters,
    device=default_device,
    num_outputs=1,
    n_targets_per_input=1,
    **kwargs,
):
    print_once("Surplus kwargs to get_batch:", kwargs.keys())
    if hyperparameters is None:
        hyperparameters = {
            "mlp_num_layers": 2,
            "mlp_num_hidden": 64,
            "mlp_init_std": 0.1,
            "mlp_sparseness": 0.2,
            "mlp_input_sampling": "normal",
            "mlp_output_noise": 0.0,
            "mlp_noisy_targets": True,
            "mlp_preactivation_noise_std": 0.0,
        }

    x, x_for_mlp = sample_input(
        hyperparameters.get("mlp_input_sampling", "normal"),
        batch_size,
        seq_len,
        num_features,
        device=device,
    )

    model = MLP(
        num_features,
        hyperparameters["mlp_num_layers"],
        hyperparameters["mlp_num_hidden"],
        num_outputs,
        hyperparameters["mlp_init_std"],
        hyperparameters["mlp_sparseness"],
        hyperparameters["mlp_preactivation_noise_std"],
        hyperparameters.get("activation", "tanh"),
    ).to(device)

    ys = []
    for x_ in x_for_mlp:  # iterating across batch size
        model.reset_parameters()
        y = model(
            x_[None].repeat(n_targets_per_input, 1, 1) / math.sqrt(num_features)
        ).squeeze(-1)
        ys.append(y.T)

    y = torch.stack(ys, dim=1)
    targets = y

    noisy_y = y + torch.randn_like(y) * hyperparameters["mlp_output_noise"]

    # return x.transpose(0, 1), noisy_y, (noisy_y if hyperparameters['mlp_noisy_targets'] else targets)
    return Batch(
        x.transpose(0, 1),
        noisy_y[:, :, :1],
        (noisy_y if hyperparameters["mlp_noisy_targets"] else targets),
    )
