#!/usr/bin/env python3
"""
Example configuration file for PFN training.
This is a Hebo+ prior configuration, as found in the PFNs4BO paper.
This file demonstrates how to configure the MainConfig for training using Python.
"""

import math

import torch
from pfns.model import bar_distribution
from pfns.model.encoders import EncoderConfig, StyleEncoderConfig
from pfns.priors.hyperparameter_sampling import ChoiceDistConfig, UniformFloatDistConfig
from pfns.priors.prior import AdhocPriorConfig
from pfns.train import (
    BatchShapeSamplerConfig,
    MainConfig,
    OptimizerConfig,
    TransformerConfig,
)
from pfns.utils import product_dict

from tqdm import tqdm


def get_config(config_index: int):
    config_dicts = product_dict(
        {
            "sampled_hp_prior": [True, False],
            "emsize": [128, 256, 512],
            "epochs": [100, 400],
            "lr": [1e-4, 3e-4],
        }
    )

    config_dict = list(config_dicts)[config_index]

    sampled_hp_prior = config_dict["sampled_hp_prior"]
    emsize = config_dict["emsize"]
    epochs = config_dict["epochs"]
    lr = config_dict["lr"]

    num_workers = 6
    steps_per_epoch = 1000

    def get_prior_config(sampled_hp_prior=sampled_hp_prior, plotting=False):
        hyperparameters = {
            "lengthscale_mean": 0.7958,  # uniform dist
            "lengthscale_std": 0.7233,  # uniform dist
            "outputscale_mean": 2.1165,  # uniform dist
            "outputscale_std": 2.3021,  # uniform dist
            "add_linear_kernel": False,  # dist as likelihood
            "unused_feature_likelihood": 0.3,  # dist, uniform 0. to .6
            "observation_noise": True,  # dist
            "x_sampler": "normal",  # fixed
            "batch_size_per_gp_sample": 1,
            "hebo_noise_logmean": -4.63,
            "hebo_noise_std": 0.5,
        }
        if sampled_hp_prior:
            hyperparameters.update(
                {
                    "num_hyperparameter_samples_per_batch": -1,
                    "hyperparameter_sampling_add_hps_to_style": "all_sampled",
                    "hyperparameter_sampling_skip_style_prob": 0.1,
                }
            )
            hyperparameters["lengthscale_mean"] = UniformFloatDistConfig(0.5, 1.5)
            hyperparameters["outputscale_mean"] = UniformFloatDistConfig(0.5, 3.0)
            hyperparameters["lengthscale_std"] = UniformFloatDistConfig(0.1, 1.5)
            hyperparameters["outputscale_std"] = UniformFloatDistConfig(0.1, 3.0)
            hyperparameters["unused_feature_likelihood"] = UniformFloatDistConfig(
                0.0, 0.6
            )
            hyperparameters["add_linear_kernel"] = UniformFloatDistConfig(0.0, 1.0)
            hyperparameters["observation_noise"] = ChoiceDistConfig([True, False])
            hyperparameters["hebo_noise_logmean"] = UniformFloatDistConfig(-8.0, -2.0)
            hyperparameters["hebo_noise_std"] = UniformFloatDistConfig(0.1, 5.0)

        prior_config = AdhocPriorConfig(
            prior_names=["hebo_prior"]
            + (["hyperparameter_sampling"] if sampled_hp_prior else []),
            prior_kwargs={
                "num_features": 1 if plotting else 18,
                "hyperparameters": {**hyperparameters},
            },
        )
        return prior_config, hyperparameters

    prior_config, hps = get_prior_config(sampled_hp_prior=sampled_hp_prior)

    gb = prior_config.create_get_batch_method()

    ys = []
    for num_features in tqdm(list(range(1, 11)) * 3):
        ys.append(
            gb(batch_size=16, seq_len=100, num_features=num_features).target_y.flatten()
        )

    ys = torch.cat(ys)
    print(f"{len(ys)=}")

    borders = bar_distribution.get_bucket_borders(1000, ys=ys)

    config = MainConfig(
        priors=[prior_config],
        optimizer=OptimizerConfig("adamw", lr=lr, weight_decay=0.0),
        scheduler="constant",
        model=TransformerConfig(
            criterion=bar_distribution.BarDistributionConfig(
                borders.tolist(), full_support=True
            ),
            emsize=emsize,
            nhead=emsize // 32,
            nhid=emsize * 4,
            nlayers=8,
            encoder=EncoderConfig(
                variable_num_features_normalization=True,
                constant_normalization_mean=0.5,
                constant_normalization_std=1 / math.sqrt(12),
            ),
            y_encoder=EncoderConfig(nan_handling=True),
            attention_between_features=True,
            style_encoder=StyleEncoderConfig(normalize_to_hyperparameters=hps)
            if sampled_hp_prior
            else None,
            y_style_encoder=StyleEncoderConfig(normalize_to_hyperparameters=hps)
            if sampled_hp_prior
            else None,
        ),
        batch_shape_sampler=BatchShapeSamplerConfig(
            batch_size=32,
            max_seq_len=60,
            fixed_num_test_instances=10,
            max_num_features=18,
        ),
        epochs=epochs,
        warmup_epochs=epochs // 10,
        steps_per_epoch=steps_per_epoch,
        num_workers=num_workers,
        train_mixed_precision=False,
    )
    return config


# View with: tensorboard --logdir=runs
