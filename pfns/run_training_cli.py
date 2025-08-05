#!/usr/bin/env python3
"""
Command-line interface for training PFNs models.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pfns.train


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a PFNs model using configuration from a Python file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the Python configuration file that defines a 'config' variable",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training (e.g., 'cuda:0', 'cpu', 'mps'). If not specified, will auto-detect cuda, but not mps.",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to save/load checkpoint (overrides config setting)",
    )

    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to load checkpoint from (overrides config setting)",
    )

    return parser.parse_args()


def load_config_from_python(config_file: str) -> pfns.train.MainConfig:
    """Load MainConfig from a Python file by accessing the 'config' variable."""
    config_path = Path(config_file)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    if not config_path.suffix.lower() == ".py":
        print(f"Warning: Config file {config_file} doesn't have .py extension")

    try:
        # Load the Python file as a module
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load spec for {config_file}")

        config_module = importlib.util.module_from_spec(spec)

        # Add the config file's directory to sys.path temporarily
        config_dir = str(config_path.parent.absolute())
        if config_dir not in sys.path:
            sys.path.insert(0, config_dir)
            path_added = True
        else:
            path_added = False

        try:
            spec.loader.exec_module(config_module)

            # Check if the module has a config variable
            if not hasattr(config_module, "config"):
                raise AttributeError(
                    f"Config file {config_file} must define a 'config' variable"
                )

            # Get the config variable
            config = config_module.config

            # Validate that it is a MainConfig instance
            if not isinstance(config, pfns.train.MainConfig):
                raise TypeError(
                    f"'config' variable must be a MainConfig instance, got {config.__class__.__name__}"
                )

            print(f"Successfully loaded config from {config_file}")
            return config

        finally:
            # Remove the added path
            if path_added:
                sys.path.remove(config_dir)

    except Exception as e:
        raise ValueError(f"Failed to load config from {config_file}: {e}")


def main():
    """Main CLI entry point."""
    args = parse_args()

    # Load configuration from Python file
    config = load_config_from_python(args.config_file)

    # Override checkpoint paths if specified via CLI
    if args.checkpoint_path is not None:
        config = config.__class__(
            **{
                **config.__dict__,
                "train_state_dict_save_path": args.checkpoint_path,
            }
        )

    if args.load_checkpoint is not None:
        config = config.__class__(
            **{
                **config.__dict__,
                "train_state_dict_load_path": args.load_checkpoint,
            }
        )

    # We overwrite the config with the one from the checkpoint if it exists
    # as there is some randomness in the config and we want to use the exact
    # same config again.
    if pfns.train.should_load_checkpoint(config):
        config = pfns.train.load_config(config.train_state_dict_load_path)

    print("Starting training with configuration:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Steps per epoch: {config.steps_per_epoch}")
    print(f"  Device: {args.device or 'auto-detect'}")
    print(f"  Mixed precision: {config.train_mixed_precision}")

    try:
        result = pfns.train.train(
            c=config,
            device=args.device,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)

    print("\nTraining completed successfully!")
    print(f"Total training time: {result['total_time']:.2f} seconds")
    print(f"Final loss: {result['total_loss']:.6f}")

    if args.checkpoint_path:
        print(f"Model saved to: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
