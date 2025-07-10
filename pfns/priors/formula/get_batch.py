import random
from typing import Literal

import numpy as np

import torch

from .. import Batch

from .ops import binary_ops, unary_ops
from .trees import evaluate_tree, sample_tree


def sample_x(num_samples, num_features, num_tree_leaves, num_constants):
    """
    Sample input data for tree evaluation.
    Only part of the input of the tree is used.

    Args:
        num_samples: Number of data points to generate
        num_features: Total number of dimensions in the input space
        num_tree_inputs: Number of inputs the tree expects
        num_constants: Number of constant inputs to use. The remaining (num_tree_inputs - num_constants) will be random dimensions.

    Returns:
        x: Actual input data of shape [num_samples, num_features]
        tree_inputs: Inputs for the tree of shape [num_samples, num_tree_inputs]
    """
    assert (
        num_tree_leaves > num_constants
    ), f"num_tree_leaves ({num_tree_leaves}) must be greater than num_constants ({num_constants})"

    # Generate random input data
    x = torch.randn(num_samples, num_features)

    # Initialize tree inputs, the first num_constants are constant inputs and thus left 0.
    tree_inputs = torch.zeros(num_samples, num_tree_leaves)

    num_dims_left_to_fill = num_tree_leaves - num_constants
    # Sample with replacement to allow redraws and unused features
    selected_dims = random.choices(range(num_features), k=num_dims_left_to_fill)

    # Map selected dimensions to remaining tree inputs
    if (
        selected_dims
    ):  # Ensure selected_dims is not empty to prevent errors with slicing
        tree_inputs[:, num_constants : num_constants + len(selected_dims)] = x[
            :, selected_dims
        ]

    return x, tree_inputs


def boring_y(y: torch.Tensor):
    """Check if a vector y is 'boring', meaning it is close to 0 (due to normalization) almost everywhere.

    A vector is considered boring if less than 2% of its values deviate from the median by more than 0.1.

    Args:
        y: A vector of values

    Returns:
        bool: True if the vector is boring, False otherwise
    """
    median = y.median()
    return (((y > median + 0.1) | (y < median - 0.1)).sum().item() / len(y)) < 0.02


# todo get first trainings running
# todo fix num workers > 0
# todo get first hebo-based trainings running, too


def get_batch(
    batch_size,
    seq_len,
    num_features,
    hyperparameters=None,
    surplus_samples_share_for_hiding_normalization=0.1,
    n_targets_per_input=1,
    single_eval_pos=None,  # not using this
    device="cpu",  # ignoring this
):
    assert (
        n_targets_per_input == 1
    ), "n_targets_per_input must be 1 for now (can be changed later)"
    hyperparameters = hyperparameters or {}

    batch_as_list = []

    while len(batch_as_list) < batch_size:
        x, y, tree = sample_dataset(
            num_samples=seq_len
            + int(surplus_samples_share_for_hiding_normalization * seq_len),
            num_features=num_features,
            **hyperparameters,
        )
        if random.random() < 0.1 or not boring_y(y):
            batch_as_list.append((x, y, tree))

    batch = Batch(
        x=torch.stack([x for x, _, _ in batch_as_list])[:, :seq_len],
        y=torch.stack([y for _, y, _ in batch_as_list])[:, :seq_len],
        target_y=torch.stack([y for _, y, _ in batch_as_list])[:, :seq_len],
    )

    # longterm todo: add styles based on the tree

    return batch


def sample_dataset(
    num_samples,
    num_features,
    max_share_oversampled_tree_leaves=0.0,
    min_constant_share=0.0,
    max_constant_share=1.0,
    binary_op_likelihoods: dict[str, float] | None = None,
    unary_op_likelihoods: dict[str, float] | None = None,
    unary_op_likelihood: float = 0.5,
    factor_dist: Literal["uniform", "normal"] = "uniform",
    bias_dist: Literal["uniform", "normal"] = "uniform",
    factor_std: float = 1.0,
    bias_std: float = 1.0,
    max_unary_op_noise_std: float = 0.0,
    max_binary_op_noise_std: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Sample a dataset based on a formula tree.

    Args:
        num_samples: Number of samples to generate
        num_features: Number of features in the input space
        share_oversampled_tree_leaves: Share of tree leaves that we want more than num_features, we will use (1. + this) * num_features as the number of tree leaves.
        num_constants: Number of constant inputs to use. The remaining (num_tree_inputs - num_constants) will be random dimensions.
        binary_op_likelihoods: Likelihoods of each binary operation, normalized.
        unary_op_likelihoods: Likelihoods of each unary operation, normalized.
    """

    max_tree_leave_share = 1.0 + max_share_oversampled_tree_leaves
    max_num_tree_leaves = int(num_features * max_tree_leave_share)

    num_tree_leaves = random.randint(max(num_features, 2), max(max_num_tree_leaves, 2))

    num_constants = random.randint(
        round(min_constant_share * num_tree_leaves),
        min(round(max_constant_share * num_tree_leaves), num_tree_leaves - 1),
    )

    if binary_op_likelihoods is None:
        binary_op_likelihoods = {op: 1.0 for op in binary_ops.keys()}
    if unary_op_likelihoods is None:
        unary_op_likelihoods = {op: 1.0 for op in unary_ops.keys()}

    def binary_op_sampler():
        return random.choices(
            list(binary_op_likelihoods.keys()),
            weights=list(binary_op_likelihoods.values()),
        )[0]

    def unary_op_sampler():
        if random.random() < unary_op_likelihood:
            return random.choices(
                list(unary_op_likelihoods.keys()),
                weights=list(unary_op_likelihoods.values()),
            )[0]
        else:
            return None

    def factor_bias_sampler(op, num_inputs):
        def get_sample(dist, std):
            if dist == "uniform":
                half_width = (3**0.5) * std
                return (torch.rand(num_inputs) * 2 - 1) * half_width
            elif dist == "normal":
                return torch.randn(num_inputs) * std
            else:
                raise ValueError(f"Unknown factor distribution: {factor_dist}")

        return get_sample(factor_dist, factor_std), get_sample(bias_dist, bias_std)

    unary_noise_std = random.random() * max_unary_op_noise_std
    binary_noise_std = random.random() * max_binary_op_noise_std

    def node_noise_sampler(node_type, op_or_input):
        if node_type == "leaf":
            return unary_noise_std
        elif node_type == "binary":
            return binary_noise_std
        else:
            return 0.0

    tree, leaf_indices = sample_tree(
        num_leaves=num_tree_leaves,
        binary_op_sampler=binary_op_sampler,
        unary_op_sampler=unary_op_sampler,
        factor_bias_sampler=factor_bias_sampler,
        node_noise_sampler=node_noise_sampler,
    )

    # print("Sampled Tree (NumPy Array):")
    # print_tree(sampled_tree_np)
    # print("Leaf Indices:")
    # print(leaf_indices)
    # print("Num Constants:", num_constants, "Num Tree Leaves:", num_tree_leaves)

    x, tree_inputs = sample_x(num_samples, num_features, num_tree_leaves, num_constants)
    y = evaluate_tree(tree, tree_inputs)

    return x, y, tree
