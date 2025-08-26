# PFNs Training CLI

This document explains how to use the command-line interface for training PFNs models.

## Overview

The training CLI allows you to train PFNs models using configuration from Python files. This provides a flexible and programmable way to configure training parameters, allowing for dynamic configuration generation, conditional logic, and easy reuse of configuration components. Configuration files define a `config` variable containing the training configuration.

## Usage

### Basic Usage

```bash
python run_training_cli.py config.py
```

### Advanced Usage

```bash
python run_training_cli.py config.py \
    --device cuda:0 \
    --compile \
    --checkpoint-path ./my_checkpoint.pt \
    --load-checkpoint ./previous_checkpoint.pt
```

## Command Line Arguments

- `config_file` (required): Path to the Python configuration file that defines a `config` variable
- `--device`: Device to use for training (e.g., 'cuda:0', 'cpu'). If not specified, will auto-detect.
- `--reusable-config` / `--no-reusable-config`: Enable/disable reusable config validation (default: enabled)
- `--compile`: Use torch.compile for the model (requires PyTorch 2.0+)
- `--checkpoint-path`: Path to save checkpoint (overrides config setting)
- `--load-checkpoint`: Path to load checkpoint from (overrides config setting)

## Configuration File Format

The Python configuration file must define a `config` variable that is a `MainConfig` instance. Here's an example structure:

```python
#!/usr/bin/env python3
"""
Example configuration file for PFNs training.
"""

from pfns.train import MainConfig
from pfns.optimizer import OptimizerConfig
from pfns.model.transformer_config import TransformerConfig
from pfns.batch_shape_sampler import BatchShapeSamplerConfig
from pfns.priors.formula.get_batch import FormulaPriorConfig


# Configure your priors
priors = [
    FormulaPriorConfig(
        # Add your prior-specific configuration here
    )
]

# Configure optimizer
optimizer = OptimizerConfig(
    # Add your optimizer configuration here
)

# Configure model
model = TransformerConfig(
    # Add your model configuration here
)

# Configure batch shape sampler
batch_shape_sampler = BatchShapeSamplerConfig(
    # Add your batch shape sampler configuration here
)

# Create the main configuration
config = MainConfig(
    epochs=100,
    steps_per_epoch=1000,
    aggregate_k_gradients=1,
    n_targets_per_input=1,
    train_mixed_precision=True,
    scheduler="cosine_decay",
    warmup_epochs=10,
    train_state_dict_save_path="./checkpoints/model_checkpoint.pt",
    validation_period=10,
    verbose=True,
    progress_bar=True,
    num_workers=4,
    priors=priors,
    optimizer=optimizer,
    model=model,
    batch_shape_sampler=batch_shape_sampler,
)
```

## Creating Configuration Files

Configuration files are standard Python files, so you have the full power of Python for creating dynamic configurations:

```python
#!/usr/bin/env python3
"""
Advanced configuration example showing dynamic configuration.
"""

import os
from pfns.train import MainConfig
from pfns.priors.formula.get_batch import FormulaPriorConfig
from pfns.optimizer import OptimizerConfig
from pfns.model.transformer_config import TransformerConfig
from pfns.batch_shape_sampler import BatchShapeSamplerConfig


# You can use environment variables
epochs = int(os.getenv('EPOCHS', '100'))
use_gpu = os.getenv('USE_GPU', 'true').lower() == 'true'

# You can have conditional logic
if use_gpu:
    train_mixed_precision = True
    batch_size = 32
else:
    train_mixed_precision = False
    batch_size = 16

# You can dynamically create configurations
priors = []
if os.getenv('USE_FORMULA_PRIOR', 'true').lower() == 'true':
    priors.append(FormulaPriorConfig(
        # formula-specific config
    ))

# You can reuse configuration components
base_model_config = {
    'd_model': 512,
    'n_heads': 8,
    'n_layers': 6,
}

model = TransformerConfig(**base_model_config)
optimizer = OptimizerConfig()
batch_shape_sampler = BatchShapeSamplerConfig()

config = MainConfig(
    epochs=epochs,
    train_mixed_precision=train_mixed_precision,
    priors=priors,
    model=model,
    optimizer=optimizer,
    batch_shape_sampler=batch_shape_sampler,
    # ... other configuration
)


# You can also define helper functions
def create_development_config():
    """Helper function for development configuration."""
    return MainConfig(
        epochs=10,  # Fewer epochs for development
        steps_per_epoch=50,  # Fewer steps for faster iteration
        verbose=True,
        progress_bar=True,
        # ... other development settings
    )
```

## Examples

### Example 1: Basic Training
```bash
python run_training.py example_config.py
```

### Example 2: Resume Training from Checkpoint
```bash
python run_training.py example_config.py --load-checkpoint ./checkpoints/model_checkpoint.pt
```

### Example 3: Training with Custom Checkpoint Path
```bash
python run_training.py example_config.py --checkpoint-path ./my_models/model.pt
```

### Example 4: Dynamic Configuration with Environment Variables
```bash
EPOCHS=200 USE_GPU=true python run_training.py dynamic_config.py
```

## Adding tensorboard

You can add tensorboards by passing a `tensorboard_path` in the `MainConfig`.
You can then view your training logs by starting the tensorboard with `tensorboard --logdir YOUR_TENSOR_BOARD_PATH`.

## Error Handling

The CLI provides informative error messages for common issues:

- **Config file not found**: Check the file path
- **Missing config variable**: Ensure your Python file defines a `config` variable
- **Invalid variable type**: Ensure `config` is a `MainConfig` instance
- **Python syntax errors**: Check your Python file for syntax errors
- **Import errors**: Ensure all required modules are importable

## Integration with Existing Code

The CLI is designed to work seamlessly with the existing `train` function. You can:

1. Use the CLI for production training
2. Use the `train` function directly in scripts
3. Mix both approaches as needed

The CLI simply provides a convenient interface to the same underlying training functionality. 



## Inference

Our trainings save checkpoints and to run inference on these checkpoints one needs to rebuild the model using the config.
This can be done like so:

```python
from pfns.train import MainConfig

checkpoint = torch.load(CHECKPOINT_PATH)
c = MainConfig.from_dict(checkpoint['config'])
model = c.model.create_model()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```
