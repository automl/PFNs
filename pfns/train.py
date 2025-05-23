from __future__ import annotations

import importlib

import itertools
import os
import time
import typing as tp
from contextlib import nullcontext
from dataclasses import dataclass

import einops
import torch
from pfns.model.transformer_config import TransformerConfig
from torch import nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from . import base_config, utils
from .priors import prior, utils as priors_utils
from .priors.batch_shape_samplers import BatchShapeSamplerConfig
from .utils import get_cosine_schedule_with_warmup, init_dist


@dataclass(frozen=True)
class OptimizerConfig(base_config.BaseConfig):
    optimizer: str
    lr: float | None
    weight_decay: float = 0.0

    def create_optimizer(self, model_parameters):
        if self.optimizer == "adam":
            return torch.optim.Adam(
                model_parameters, lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == "adamw":
            return torch.optim.AdamW(
                model_parameters, lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer} not supported")


@dataclass(frozen=True)
class MainConfig(base_config.BaseConfig):
    # Training configuration
    priors: tp.List[prior.PriorConfig]
    optimizer: OptimizerConfig

    # Model (includes criterion)
    model: TransformerConfig

    # Training
    batch_shape_sampler: BatchShapeSamplerConfig
    epochs: int = 10
    steps_per_epoch: int = 100
    batch_size: int = 200
    aggregate_k_gradients: int = 1
    n_targets_per_input: int = 1
    train_mixed_precision: bool = True

    # LR Scheduler
    scheduler: str = "cosine_decay"
    warmup_epochs: int = 10

    # Checkpointing
    train_state_dict_save_path: tp.Optional[str] = None
    train_state_dict_load_path: tp.Optional[str] = None

    # Logging
    validation_period: int = 10
    verbose: bool = True
    progress_bar: bool = False

    # Data loading
    dataloader_class: str | None = None
    num_workers: tp.Optional[int] = None


class EpochResult(tp.NamedTuple):
    loss: float  # total loss for the epoch
    positional_losses: list  # list of losses per position
    data_time: float  # time spent getting batch data
    forward_time: float  # time spent in forward pass
    step_time: float  # total time per step
    nan_share: float  # share of NaN values
    ignore_share: float  # share of ignored values (-100)
    importance_sampling_infos: list  # gradient magnitude info


def train(c: MainConfig, device: str | None = None):
    # Arguments from original signature not in MainConfig are set to defaults here
    load_weights_from_this_state_dict = None
    epoch_callback = None
    step_callback = None
    continue_model = None

    total_start_time = time.time()

    default_device: str = "cuda:0" if torch.cuda.is_available() else "cpu:0"
    if device is None:
        device = default_device
    print(f"Found default device {device}.")
    using_dist, rank, device = init_dist(device)

    # Resolve single_eval_pos_gen, todo make ready for multi gpu by including step info
    eval_pos_seq_len_sampler = c.batch_shape_sampler.get_sampler()
    print(eval_pos_seq_len_sampler)

    # Resolve dataloader_class string to actual class
    if c.dataloader_class is None:
        actual_dataloader_class = priors_utils.StandardDataLoader
    else:
        parts = c.dataloader_class.split(".")
        module_path = ".".join(parts[:-1])
        class_name = parts[-1]
        try:
            module = importlib.import_module(module_path)
            actual_dataloader_class = getattr(module, class_name)
        except Exception as e:
            raise ImportError(
                f"Could not import dataloader_class '{c.dataloader_class}': {e}"
            ) from e

    if not c.priors:
        raise ValueError("main_config.priors cannot be empty.")

    if len(c.priors) != 1:
        raise ValueError(
            "Currently only supporting a single prior. Later this should be a seqeunce that is called in order by wrapping."
        )

    if len(c.priors) == 1 and callable(c.priors[0]):  # Simplistic assumption
        get_batch_method_instance = c.priors[0]
    elif isinstance(c.priors[0], prior.PriorConfig):
        get_batch_method_instance = c.priors[0].create_get_batch_method()
    else:
        raise ValueError(
            "main_config.priors must be a list of PriorConfig objects or a single callable."
        )

    current_extra_prior_kwargs_dict = {}
    if c.num_workers is not None:
        current_extra_prior_kwargs_dict["num_workers"] = c.num_workers

    dl = actual_dataloader_class(
        get_batch_method=get_batch_method_instance,  # Use the constructed/resolved instance
        num_steps=c.steps_per_epoch,
        batch_size=c.batch_size,
        eval_pos_seq_len_sampler=eval_pos_seq_len_sampler,
        device=device,  # Pass the torch device object
        n_targets_per_input=c.n_targets_per_input,
        **current_extra_prior_kwargs_dict,
    )

    if continue_model:  # This was a non-config arg, defaulted to None
        model = continue_model
    else:
        assert (
            c.model.features_per_group > 0 or c.model.features_per_group == -1
        ), "features_per_group must be > 0 or -1"

        model = c.model.create_model()
    criterion = model.criterion

    if load_weights_from_this_state_dict is not None:
        model.load_state_dict(load_weights_from_this_state_dict)

    print(
        f"Using a Transformer with {sum(p.numel() for p in model.parameters())/1000/1000:.{2}f} M parameters"
    )

    model.to(device)

    if hasattr(c.optimizer, "create_optimizer"):
        optimizer = c.optimizer.create_optimizer(model.parameters())
    else:
        raise ValueError(
            "main_config.optimizer must have a 'create_optimizer' method"
        )

    # Resolve scheduler string to function
    if c.scheduler == "cosine_decay":
        scheduler_fn = get_cosine_schedule_with_warmup
    else:
        raise ValueError(f"Scheduler {c.scheduler} not supported")

    scheduler = scheduler_fn(  # todo move warmup epochs into scheduler args, ideally as steps instead!?
        optimizer, c.warmup_epochs, c.epochs if c.epochs is not None else 100
    )

    start_epoch = 1  # Default start epoch

    if c.train_state_dict_load_path:
        if (c.train_state_dict_save_path != c.train_state_dict_load_path) or (
            (c.train_state_dict_save_path == c.train_state_dict_load_path)
            and os.path.exists(c.train_state_dict_load_path)
        ):
            # load_checkpoint needs the scheduler instance, not the factory
            start_epoch = (
                load_checkpoint(  # load_checkpoint might return start_epoch
                    model,
                    optimizer,
                    scheduler,
                    c.train_state_dict_load_path,
                    device,
                )
            )
        else:
            print(
                f"Checkpoint file {c.train_state_dict_load_path} not found or load/save paths are identical and file doesn't exist. Starting from scratch."
            )

    if using_dist:
        print("Distributed training")
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            broadcast_buffers=False,
        )
        dl.model = (
            model.module
        )  # use local model, should not use multi-gpu functionality..
    else:
        dl.model = model

    scaler = GradScaler() if c.train_mixed_precision else None

    # check that everything uses up-to-date APIs
    utils.check_compatibility(dl)

    def train_epoch():
        model.train()
        total_loss = 0.0
        total_positional_losses = 0.0
        total_positional_losses_recorded = 0
        nan_steps = 0
        ignore_steps = 0
        before_get_batch = time.time()
        assert (
            len(dl) % c.aggregate_k_gradients == 0
        ), "Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it."
        tqdm_iter = (
            tqdm(range(len(dl)), desc="Training Epoch")
            if rank == 0 and c.progress_bar
            else None
        )
        importance_sampling_infos = []  # Renamed from squared_grad_magnitudes_and_infos

        for batch_index, batch in enumerate(dl):
            batch: prior.Batch = (
                batch  # Type annotation for better IDE support
            )
            assert (
                c.batch_size
                == len(batch.x)
                == len(batch.y)
                == len(batch.target_y)
            ), "Our code was updated to use the more intuitive batch first format, please make sure your get_batch function returns data with shapes (batch_size, seq_len, ...)"
            if not model.attention_between_features:
                assert (
                    model.features_per_group == batch.x.shape[2]
                ), "features_per_group must match the number of features in the input, if attention_between_features is False"
            targets = batch.target_y.to(device)
            single_eval_pos = batch.single_eval_pos
            seq_len = batch.x.shape[1]

            tqdm_iter.update() if tqdm_iter is not None else None
            if using_dist and not (
                batch_index % c.aggregate_k_gradients
                == c.aggregate_k_gradients - 1
            ):
                cm = model.no_sync()
            else:
                cm = nullcontext()
            with cm:
                time_to_get_batch = time.time() - before_get_batch
                before_forward = time.time()
                try:
                    metrics_to_log = {}
                    with autocast(
                        device.split(":")[0], enabled=scaler is not None
                    ):
                        style = (
                            batch.style.to(device)
                            if batch.style is not None
                            else None
                        )
                        if style is not None:
                            if style.dim() == 2:
                                broken = style.shape[0] != batch.x.shape[0]
                            elif style.dim() == 3:
                                broken = (
                                    style.shape[0] != batch.x.shape[0]
                                    or style.shape[1] != batch.x.shape[2]
                                )
                            else:
                                raise ValueError(
                                    f"style must have 2 or 3 dimensions, got {style.shape}"
                                )
                            if broken:
                                raise ValueError(
                                    f"style must have the same batch size as x and if it has 3 dimensions, "
                                    f"the middle dimension must match the number of features, got {style.shape=} "
                                    f"and {batch.x.shape=}"
                                )
                        y_style = (
                            batch.y_style.to(device)
                            if batch.y_style is not None
                            else None
                        )
                        if y_style is not None:
                            if y_style.dim() == 2:
                                broken = y_style.shape[0] != batch.y.shape[0]
                            else:
                                raise ValueError(
                                    f"y_style must have 2 dimensions, got {y_style.shape}"
                                )
                            if broken:
                                raise ValueError(
                                    f"y_style must have the same batch size as y, got {y_style.shape=} "
                                    f"and {batch.y.shape=}"
                                )
                        x = batch.x.to(device)
                        y = batch.y.to(device)
                        out = model(
                            x,
                            y[:, :single_eval_pos],
                            style=style,
                            y_style=y_style,
                            only_return_standard_out=False,
                        )

                        out, output_once = (
                            out if isinstance(out, tuple) else (out, None)
                        )
                        output = (
                            out["standard"] if isinstance(out, dict) else out
                        )

                        forward_time = time.time() - before_forward

                        if single_eval_pos is not None:
                            targets = targets[:, single_eval_pos:]

                        # Repeat output in the semi-last dimension n_targets_per_input times
                        output = output.unsqueeze(2).expand(
                            *output.shape[:2],
                            c.n_targets_per_input,
                            output.shape[-1],
                        )

                        if len(targets.shape) == 2:
                            # This implies we only have a single target per input
                            targets = targets.unsqueeze(2)

                        assert targets.shape == output.shape[:-1], (
                            f"Target shape {targets.shape} "
                            f"does not match output shape {output.shape}."
                            f"This might be because you are missing trailing "
                            "1 dimension in the target."
                        )

                        output = einops.rearrange(
                            output, "b s t l -> (b t) s l"
                        )
                        targets = einops.rearrange(targets, "b s t -> (b t) s")

                        if isinstance(criterion, nn.GaussianNLLLoss):
                            assert (
                                output.shape[-1] == 2
                            ), "need to write a little bit of code to handle multiple regression targets at once"

                            mean_pred = output[..., 0]
                            var_pred = output[..., 1].abs()
                            losses = criterion(
                                mean_pred.flatten(),
                                targets.flatten(),
                                var=var_pred.flatten(),
                            )
                        elif isinstance(
                            criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)
                        ):
                            targets[torch.isnan(targets)] = -100
                            losses = criterion(
                                output.flatten(), targets.flatten()
                            )
                            losses = losses.view(*targets.shape)
                        elif isinstance(criterion, nn.CrossEntropyLoss):
                            targets[torch.isnan(targets)] = -100
                            losses = criterion(
                                output.reshape(-1, len(criterion.weight)),
                                targets.long().flatten(),
                            )
                        else:
                            losses = criterion(output, targets.unsqueeze(-1))
                        losses = einops.rearrange(
                            losses, "(b t) s -> b s t", t=c.n_targets_per_input
                        )
                        losses = losses.mean(-1)
                        loss, nan_share = utils.torch_nanmean(
                            losses.mean(
                                1
                            ),  # loss per sequence without nanmean
                            return_nanshare=True,
                        )
                        loss_scaled = loss / c.aggregate_k_gradients

                    if scaler:
                        loss_scaled = scaler.scale(loss_scaled)
                    loss_scaled.backward()

                    if (
                        batch_index % c.aggregate_k_gradients
                        == c.aggregate_k_gradients - 1
                    ):
                        if scaler:
                            scaler.unscale_(optimizer)
                        squared_grad_magnitudes = {
                            name: (w.grad**2).sum().cpu().item()
                            for name, w in model.named_parameters()
                            if w.grad is not None
                        }
                        total_grad_magnitude = sum(
                            squared_grad_magnitudes.values()
                        )

                        normalized_squared_grad_magnitudes = {}
                        # Compute grad magnitude normalized by Adam's beta2 parameter if Adam optimizer is used
                        if isinstance(
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
                                                    * (
                                                        1
                                                        - beta2
                                                        ** state.get("step", 1)
                                                    )
                                                    + 1e-8
                                                )
                                            )
                                            .sum()
                                            .cpu()
                                            .item()
                                        )
                                        normalized_squared_grad_magnitudes[
                                            name
                                        ] = normalized_grad_magnitude
                            total_normalized_grad_magnitude = sum(
                                v
                                for k, v in normalized_squared_grad_magnitudes.items()
                            )
                        # print('normalized_squared_grad_magnitudes and squared_grad_magnitudes', {k: (normalized_squared_grad_magnitudes[k]/total_normalized_grad_magnitude, squared_grad_magnitudes[k]/total_grad_magnitude) for k in normalized_squared_grad_magnitudes.keys()})
                        importance_sampling_infos.append(
                            (
                                total_grad_magnitude,
                                batch.info_used_with_gradient_magnitudes,
                                loss.cpu().item(),
                                total_normalized_grad_magnitude,
                            )
                        )
                        # todo i should run metrics on this
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                        if batch.gradient_multipliers is not None:
                            assert (
                                c.aggregate_k_gradients == 1
                            ), "Scaling grads is only supported if you don't do grad acc."
                            assert all(
                                batch.gradient_multipliers.view(-1)[0]
                                == batch.gradient_multipliers.view(-1)[i]
                                for i in range(
                                    batch.gradient_multipliers.numel()
                                )
                            ), "we don't scale losses for now to be able to try the interaction with gradient clipping, and thus we can only support the same scaler"
                            # todo make print to see that this is actually running
                            with torch.no_grad():
                                for w in model.parameters():
                                    w.grad = (
                                        w.grad
                                        * batch.gradient_multipliers.view(-1)[
                                            0
                                        ]
                                    )

                        if scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()

                    step_time = time.time() - before_forward

                    if not torch.isnan(loss):
                        total_loss += loss.cpu().detach().item()
                        total_positional_losses += (
                            losses.mean(1).cpu().detach()
                            if single_eval_pos is None
                            else nn.functional.one_hot(
                                torch.tensor(single_eval_pos), seq_len
                            )
                            * utils.torch_nanmean(
                                losses[: seq_len - single_eval_pos].mean(0)
                            )
                            .cpu()
                            .detach()
                        )

                        total_positional_losses_recorded += (
                            torch.ones(seq_len)
                            if single_eval_pos is None
                            else nn.functional.one_hot(
                                torch.tensor(single_eval_pos), seq_len
                            )
                        )

                        metrics_to_log = {
                            **metrics_to_log,
                            **{
                                f"loss": loss,
                                "single_eval_pos": single_eval_pos,
                            },
                        }
                        if step_callback is not None and rank == 0:
                            step_callback(metrics_to_log)
                        nan_steps += nan_share
                        ignore_steps += (targets == -100).float().mean()
                    else:
                        print("Invalid step encountered, skipping...")
                        nan_position = torch.isnan(
                            losses
                        )  # (seq_len, batch_size)

                except Exception as e:
                    print("Invalid step encountered, skipping...")
                    print(e)
                    raise (e)

            if tqdm_iter:
                tqdm_iter.set_postfix(
                    {
                        "data_time": time_to_get_batch,
                        "step_time": step_time,
                        "mean_loss": total_loss / (batch_index + 1),
                    }
                )

            before_get_batch = time.time()
        return EpochResult(
            loss=total_loss / c.steps_per_epoch,
            positional_losses=(
                total_positional_losses / total_positional_losses_recorded
            ).tolist(),
            data_time=time_to_get_batch,
            forward_time=forward_time,
            step_time=step_time,
            nan_share=nan_steps.cpu().item() / (batch_index + 1),
            ignore_share=ignore_steps.cpu().item() / (batch_index + 1),
            importance_sampling_infos=importance_sampling_infos,
        )

    total_loss = float("inf")
    total_positional_losses = float("inf")
    importance_sampling_infos = (
        None  # Renamed from squared_grad_magnitudes_and_infos
    )
    try:
        # Initially test the epoch callback function only if starting from epoch 1
        if epoch_callback is not None and rank == 0 and start_epoch == 1:
            epoch_callback(
                model, 0, data_loader=dl, scheduler=scheduler
            )  # Call with epoch 0 for initial state

        # Adjust epoch range based on start_epoch
        epoch_iterator = (
            range(start_epoch, c.epochs + 1)
            if c.epochs is not None
            else itertools.count(start_epoch)
        )

        for epoch in epoch_iterator:
            epoch_start_time = time.time()
            try:
                epoch_result = train_epoch()
                total_loss = epoch_result.loss
                total_positional_losses = epoch_result.positional_losses
                dl.importance_sampling_infos = (
                    epoch_result.importance_sampling_infos
                )

            except Exception as e:
                print("Invalid epoch encountered, skipping...")
                print(e)
                raise (e)
            if hasattr(dl, "validate") and epoch % c.validation_period == 0:
                with torch.no_grad():
                    val_score = dl.validate(model)

            else:
                val_score = None

            if c.verbose:
                print("-" * 89)
                print(
                    f"| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | mean loss {epoch_result.loss:5.2f} | "
                    f"pos losses {','.join([f'{l:5.2f}' for l in epoch_result.positional_losses])}, lr {scheduler.get_last_lr()[0]}"
                    f" data time {epoch_result.data_time:5.2f} step time {epoch_result.step_time:5.2f}"
                    f" forward time {epoch_result.forward_time:5.2f}"
                    f" nan share {epoch_result.nan_share:5.2f} ignore share (for classification tasks) {epoch_result.ignore_share:5.4f}"
                    + (
                        f"val score {val_score}"
                        if val_score is not None
                        else ""
                    )
                )
                print("-" * 89)

            # stepping with wallclock time based scheduler
            if epoch_callback is not None and rank == 0:
                epoch_callback(
                    model, epoch, data_loader=dl, scheduler=scheduler
                )
            scheduler.step()

            # Save model state dict after each epoch if path is provided (on rank 0)
            if c.train_state_dict_save_path is not None and rank == 0:
                save_checkpoint(
                    model, optimizer, c.train_state_dict_save_path, epoch
                )

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        pass

    if rank == 0:  # trivially true for non-parallel training
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
            dl = None
        return {
            "total_loss": total_loss,
            "total_positional_losses": total_positional_losses,
            "model": model.to("cpu"),
            "data_loader": dl,
            "total_time": time.time() - total_start_time,
        }


def load_checkpoint(
    model, optimizer, scheduler, train_state_dict_load_path, device
):
    print(f"Loading checkpoint from {train_state_dict_load_path}")
    try:
        checkpoint = torch.load(
            train_state_dict_load_path, map_location=device
        )
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # New format with model, optimizer state and epoch
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resuming from epoch {start_epoch}")
            # Fast-forward the scheduler to the correct epoch
            for _ in range(start_epoch - 1):
                scheduler.step()
        else:
            raise ValueError(
                f"Checkpoint does not contain 'model_state_dict' or 'optimizer_state_dict'. Checkpoint: {checkpoint}"
            )
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise e


def save_checkpoint(model, optimizer, train_state_dict_save_path, epoch):
    save_model = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )
    print(f"Saving checkpoint to {train_state_dict_save_path} (epoch {epoch})")
    os.makedirs(os.path.dirname(train_state_dict_save_path), exist_ok=True)
    try:
        # Save model state dict, optimizer state dict, and current epoch
        checkpoint = {
            "model_state_dict": save_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, train_state_dict_save_path)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
