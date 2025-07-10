import numpy as np
import torch

from pfns.priors.hyperparameter_sampling import (
    PowerUniformFloatDistConfig,
    UniformFloatDistConfig,
)
from scipy import stats


def test_distribution_normalizers():
    """Test that all distribution configs produce roughly uniform distributions after normalization.
    This is done by sampling many times and checking the histogram of normalized values.
    """

    def check_distribution_uniformity(
        samples: np.ndarray, name: str, p_threshold: float = 0.02
    ):
        """Use Kolmogorov-Smirnov test to check if normalized samples are uniform"""
        # Test against uniform distribution on [0,1]
        ks_stat, p_value = stats.kstest(samples, "uniform")
        assert (
            p_value > p_threshold
        ), f"{name} normalized samples failed uniformity test with p={p_value:.4f}"

    # Test cases for each distribution type
    test_cases = [
        (
            UniformFloatDistConfig(lower=0.1, upper=10.0, log=False),
            "UniformFloat",
        ),
        (
            UniformFloatDistConfig(lower=0.1, upper=10.0, log=True),
            "UniformFloat-Log",
        ),
        (
            PowerUniformFloatDistConfig(lower=0.1, upper=10.0, power=2.0),
            "PowerUniform-2",
        ),
        (
            PowerUniformFloatDistConfig(lower=0.1, upper=10.0, power=4.0),
            "PowerUniform-4",
        ),
        # we don't test the integer distributions because they are not **really** uniform
    ]

    n_samples = 100000
    for dist, name in test_cases:
        # Generate samples
        samples = [dist.sample() for _ in range(n_samples)]

        samples_tensor = torch.tensor(samples, dtype=torch.float32)

        normalized = dist.normalize(samples_tensor)

        # Check uniformity
        check_distribution_uniformity(normalized.numpy(), name)

        # Basic range check
        assert torch.all(normalized >= 0) and torch.all(
            normalized <= 1
        ), f"{name} normalized values outside [0,1] range"
