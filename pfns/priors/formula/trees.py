import random
from typing import Callable, List, Tuple

import numpy as np
import torch

from .ops import binary_ops, unary_ops

node_dtype = np.dtype(
    [
        (
            "type",
            "U6",
        ),  # The type of the node can be 'binary', 'unary', or 'leaf'
        (
            "op_or_input",
            "U12",
        ),  # The operation (e.g. 'add') or input index, if it is a leaf.
        (
            "left",
            "i4",
        ),  # The index of the left child in the tree, -1 if no left child.
        (
            "right",
            "i4",
        ),  # The index of the right child in the tree, -1 if no right child.
        (
            "print",
            "U3",
        ),  # Whether to print ('yes') the node during evaluation, empty ('') otherwise. This flag is there to be used rather freely.
        (
            "factor_left",
            "f4",
        ),  # The factor applied to the left child before applying the operation.
        (
            "factor_right",
            "f4",
        ),  # The factor applied to the right child before applying the operation.
        (
            "bias_left",
            "f4",
        ),  # The bias applied to the left child before applying the operation.
        (
            "bias_right",
            "f4",
        ),  # The bias applied to the right child before applying the operation.
        (
            "noise_std",
            "f4",
        ),  # Standard deviation of Gaussian noise to add to the node's output.
    ]
)


def sample_tree(
    num_leaves: int,
    binary_op_sampler: Callable[[], str],
    unary_op_sampler: Callable[[], str | None] = None,
    factor_bias_sampler: Callable[[str, int], Tuple[List[float], List[float]]]
    | None = None,
    node_noise_sampler: Callable[[str, str], float] | None = None,
) -> Tuple[np.ndarray, List[int]]:  # the array is a list of type node_dtype
    """
    Samples a binary operation tree using NumPy structured array representation.

    Starts with a single leaf node and iteratively replaces random leaf nodes with
    binary operations (which create two new leaf nodes) until the target number of
    leaves is reached. Each binary operation node has two children, and each leaf
    represents an input variable.

    Each node (binary and leaf) is (potentially) wrapped with a unary operation, the
    unary operation is chosen randomly by the unary_op_sampler.

    Args:
        num_leaves: The desired number of leaves in the final tree.
        binary_op_sampler: A function that returns a binary operation name (str) and does not take any arguments.
        unary_op_sampler: A function that returns a unary operation name (str) or None, if it returns None, no unary operations will be added.

        factor_bias_sampler: A function that takes (op, num_inputs) and returns
                             a tuple of lists (factors, biases), where each list
                             has length num_inputs. These factors and biases
                             are applied to the inputs of the operation.
                             The outputs of each operation are z-normalized.

        node_noise_sampler: A function that takes (node_type, op_or_input) and returns the standard deviation of Gaussian noise to add to the node's output.
                            If None, defaults to 0.0 (no noise) for all nodes.

    Returns:
        A tuple of a NumPy structured array representing the tree using `node_dtype`
        and a list of indices of the leaf nodes.
        The root of the tree is at index 0.
    """
    if num_leaves < 1:
        raise ValueError("Target leaves must be at least 1.")

    # Assume binary_ops and unary_ops dictionaries are defined in the global scope
    # Assume node_dtype is defined in the global scope

    # Default factor_bias_sampler returns 1.0 for factors and 0.0 for biases
    if factor_bias_sampler is None:
        factor_bias_sampler = lambda op, num_inputs: (  # noqa: E731
            [1.0] * num_inputs,  # factors
            [0.0] * num_inputs,  # biases
        )

    if node_noise_sampler is None:
        node_noise_sampler = (
            lambda node_type, op_or_input: 0.0
        )  # Default: no noise  # noqa: E731

    # Handle base case: single leaf
    if num_leaves == 1:
        # For the single leaf case, its state is final, so set noise_std directly.
        noise_std = node_noise_sampler("leaf", "0")
        return np.array(
            [("leaf", "0", -1, -1, "", 1.0, 1.0, 0.0, 0.0, noise_std)],
            dtype=node_dtype,
        ), [0]

    tree_nodes = []  # List to store node tuples before converting to NumPy array
    leaf_indices = []  # Stores indices of *actual leaf nodes* in tree_nodes
    next_input_idx = 0

    # Helper function to add a node to the list and return its index
    def add_node(node_data):
        idx = len(tree_nodes)
        # Ensure node_data matches node_dtype structure
        tree_nodes.append(node_data)
        return idx

    # Helper function to potentially wrap a node with a unary operation
    def maybe_wrap_with_unary(node_idx_to_wrap):
        # This function assumes unary_op_sampler and add_node are available in its scope
        if unary_op_sampler is None:
            return node_idx_to_wrap
        unary_op = unary_op_sampler()
        if unary_op is None:
            return node_idx_to_wrap

        # Get factors and biases for the unary operation
        factors, biases = factor_bias_sampler(unary_op, 1)
        # Initialize noise_std to 0.0; will be set in the final loop.

        # Insert the unary node pointing to the original node
        unary_node_idx = add_node(
            (
                "unary",
                unary_op,
                node_idx_to_wrap,
                -1,
                "",
                factors[0],
                1.0,
                biases[0],
                0.0,
                0.0,
            )
        )  # noise_std is 0.0 for now
        return unary_node_idx

    # Initialize the tree with the first leaf node
    initial_leaf_op_or_input = str(next_input_idx)
    # Initialize noise_std to 0.0; will be set in the final loop.
    root_leaf_idx = add_node(
        ("leaf", initial_leaf_op_or_input, -1, -1, "", 1.0, 1.0, 0.0, 0.0, 0.0)
    )  # noise_std is 0.0 for now
    leaf_indices.append(root_leaf_idx)
    next_input_idx += 1

    # Iteratively expand leaves until the target number is reached
    while len(leaf_indices) < num_leaves:
        # Choose a random leaf index from the list of current leaf indices
        idx_in_leaves_list = random.randrange(len(leaf_indices))
        # Get the actual index of the node in the tree_nodes list
        node_to_expand_idx = leaf_indices[idx_in_leaves_list]

        # Retrieve the input variable name of the leaf being expanded
        # Important: Access tree_nodes *before* potentially overwriting the node
        current_input_var = tree_nodes[node_to_expand_idx][1]

        # Sample a binary operation
        binary_op = binary_op_sampler()

        # Get factors and biases for the binary operation
        factors, biases = factor_bias_sampler(binary_op, 2)

        # Create the new left leaf node (reusing the current input variable)
        # Initialize noise_std to 0.0; will be set in the final loop.
        left_leaf_idx = add_node(
            ("leaf", current_input_var, -1, -1, "", 1.0, 1.0, 0.0, 0.0, 0.0)
        )  # noise_std is 0.0 for now
        # Optionally wrap the new left leaf node with unary op(s)
        # The maybe_wrap function returns the index of the node that should be the child (either the leaf or the unary op)
        left_child_node_idx = maybe_wrap_with_unary(left_leaf_idx)

        # Create the new right leaf node (using the next available input variable)
        right_leaf_op_or_input = str(next_input_idx)
        # Initialize noise_std to 0.0; will be set in the final loop.
        right_leaf_idx = add_node(
            (
                "leaf",
                right_leaf_op_or_input,
                -1,
                -1,
                "",
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
            )
        )  # noise_std is 0.0 for now
        next_input_idx += 1
        # Optionally wrap the new right leaf node with unary op(s)
        right_child_node_idx = maybe_wrap_with_unary(right_leaf_idx)

        # Update the node that was expanded: change it from a leaf to a binary node
        # Its children are the (potentially wrapped) new leaf nodes
        # Initialize noise_std to 0.0; will be set in the final loop.
        tree_nodes[node_to_expand_idx] = (
            "binary",
            binary_op,
            left_child_node_idx,
            right_child_node_idx,
            "",
            factors[0],
            factors[1],
            biases[0],
            biases[1],
            0.0,
        )  # noise_std is 0.0 for now

        # Update the list of leaf indices:
        # Replace the index of the expanded leaf with the index of the new left leaf
        leaf_indices[idx_in_leaves_list] = left_leaf_idx
        # Add the index of the new right leaf to the list
        leaf_indices.append(right_leaf_idx)

    tree_nodes = np.array(tree_nodes, dtype=node_dtype)

    # After the tree structure is built, iterate through nodes to set final noise_std
    # This ensures noise_std is based on the final type and op_or_input of each node.
    # The num_leaves == 1 case is handled separately above.
    if num_leaves > 1:
        for i in range(len(tree_nodes)):
            current_node_tuple = tree_nodes[i]
            node_type = current_node_tuple["type"]  # type is at index 0
            op_or_input = current_node_tuple["op_or_input"]  # op_or_input is at index 1

            actual_noise_std = node_noise_sampler(node_type, op_or_input)
            tree_nodes[i]["noise_std"] = actual_noise_std

    # Convert the list of node tuples to a NumPy structured array
    return tree_nodes, leaf_indices


def evaluate_tree(
    tree: np.ndarray,
    inputs: torch.Tensor,
    node_idx: int = 0,
    normalize_output_after_noise: bool = False,
) -> torch.Tensor:
    """
    Evaluates a binary syntax tree on the given inputs.
    This function recursively calls itself to evaluate.
    This might be a bit slow, but at least it is batched

    Args:
        tree: NumPy structured array representing the tree (from sample_tree)
        inputs: Tensor of shape [batch_size, num_inputs] containing input values
        node_idx: Index of the node to evaluate (default: 0 for root)
        normalize_output_after_noise: Whether to normalize again after noise, to make sure the output is mean 0 and std 1
    Returns:
        Tensor of shape [batch_size] with the result of evaluating the tree
    """
    node = tree[node_idx]
    node_type = node["type"]

    if node_type == "leaf":
        # Leaf node: return the corresponding input
        input_idx: int = int(node["op_or_input"])
        result = inputs[:, input_idx]
        if node["print"] == "yes":
            print(f"Leaf node {node_idx} (input {input_idx}) unnormalized: {result}")

    elif node_type == "unary":
        # Unary node: apply unary operation to child
        op_name: str = node["op_or_input"]
        child_idx = node["left"]  # Unary nodes only have left child

        if op_name not in unary_ops:
            raise ValueError(f"Unknown unary operation: {op_name}")

        # Apply factor and bias to the child value
        child_value = evaluate_tree(tree, inputs, child_idx)
        transformed_value = node["factor_left"] * child_value + node["bias_left"]

        # Apply the unary operation
        result = unary_ops[op_name](transformed_value)

        if node["print"] == "yes":
            print(f"Unary node {node_idx} (op {op_name}) unnormalized: {result}")

    elif node_type == "binary":
        # Binary node: apply binary operation to children
        op_name = node["op_or_input"]
        left_idx = node["left"]
        right_idx = node["right"]

        if op_name not in binary_ops:
            raise ValueError(f"Unknown binary operation: {op_name}")

        # Get child values
        left_value = evaluate_tree(tree, inputs, left_idx)
        right_value = evaluate_tree(tree, inputs, right_idx)

        # Apply factors and biases
        transformed_left = node["factor_left"] * left_value + node["bias_left"]
        transformed_right = node["factor_right"] * right_value + node["bias_right"]

        # Apply the binary operation
        result = binary_ops[op_name](transformed_left, transformed_right)

        if node["print"] == "yes":
            print(f"Binary node {node_idx} (op {op_name}) unnormalized: {result}")
            print(f"left: {transformed_left}, right: {transformed_right}")
    else:
        raise ValueError(f"Unknown node type: {node_type}")

    # Apply z-normalization across the batch
    result = (result - result.mean()) / (result.std() + 1e-8)

    noise = torch.randn_like(result) * node["noise_std"]
    result = result + noise

    if normalize_output_after_noise:
        # Add noise to the result
        result = (result - result.mean()) / (result.std() + 1e-8)

    return result
