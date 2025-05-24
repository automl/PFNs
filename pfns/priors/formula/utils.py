def bias_str(bias):
    bias_sign = "+" if bias >= 0 else "-"
    return f"{bias_sign}{abs(bias):.2f}"


def print_tree(tree, node_idx=0, indent=0, prefix=""):
    """
    Print a tree in a human-readable format with indentation to show hierarchy.

    Args:
        tree: The tree structure (list of nodes)
        node_idx: Current node index to print
        indent: Current indentation level
        edge_label: Label for the edge leading to this node
    """
    node = tree[node_idx]
    node_type = node["type"]

    # Print the current node with indentation
    indent_str = "  " * indent
    prefix = f"{indent_str}{prefix}"

    if node_type == "input" or node_type == "leaf":
        print(
            f"{prefix}{node_type.upper()} {node['op_or_input']} + ~ {node['noise_std']:.2f}"
        )
    elif node_type == "unary":
        op_name = node["op_or_input"]

        # Print child with increased indentation
        print(f"{prefix}{op_name} + ~ {node['noise_std']:.2f}")
        factor = node["factor_left"]
        bias = node["bias_left"]
        print_tree(
            tree,
            node["left"],
            indent + 1,
            prefix=f"(x{factor:.2f}){bias_str(bias)} ",
        )
    elif node_type == "binary":
        op_name = node["op_or_input"]
        print(f"{prefix}{op_name} + ~ {node['noise_std']:.2f}")

        # Print left child with increased indentation
        factor = node["factor_left"]
        bias = node["bias_left"]
        print_tree(
            tree,
            node["left"],
            indent + 1,
            prefix=f"(x{factor:.2f}){bias_str(bias)} ",
        )

        # Print right child with increased indentation
        factor = node["factor_right"]
        bias = node["bias_right"]
        print_tree(
            tree,
            node["right"],
            indent + 1,
            prefix=f"(x{factor:.2f}){bias_str(bias)} ",
        )
