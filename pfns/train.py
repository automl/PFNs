from __future__ import annotations

import itertools
import os
import time
import typing as tp
from contextlib import nullcontext

import einops
import torch
import yaml
from pfns.model import transformer
from pfns.model.bar_distribution import BarDistribution
from torch import nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from . import priors, utils
from .priors import prior
from .utils import get_cosine_schedule_with_warmup, init_dist


class Losses:
    gaussian = nn.GaussianNLLLoss(full=True, reduction="none")
    mse = nn.MSELoss(reduction="none")
    ce = lambda num_classes: nn.CrossEntropyLoss(
        reduction="none", weight=torch.ones(num_classes)
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")
    get_BarDistribution = BarDistribution


class EpochResult(tp.NamedTuple):
    loss: float  # total loss for the epoch
    positional_losses: list  # list of losses per position
    data_time: float  # time spent getting batch data
    forward_time: float  # time spent in forward pass
    step_time: float  # total time per step
    nan_share: float  # share of NaN values
    ignore_share: float  # share of ignored values (-100)
    importance_sampling_infos: list  # gradient magnitude info


def train(
    get_batch_method: prior.PriorDataLoader | callable,
    criterion,
    encoder_generator=None,
    emsize=200,
    nhid=200,
    nlayers=6,
    nhead=2,
    epochs=10,
    steps_per_epoch=100,
    batch_size=200,
    seq_len=10,
    lr=None,
    weight_decay=0.0,
    warmup_epochs=10,
    train_state_dict_save_path=None,
    train_state_dict_load_path=None,
    y_encoder_generator=None,
    decoder_dict={},
    extra_prior_kwargs_dict={},
    scheduler=get_cosine_schedule_with_warmup,
    load_weights_from_this_state_dict=None,
    validation_period=10,
    single_eval_pos_gen=None,
    gpu_device="cuda:0",
    aggregate_k_gradients=1,
    verbose=True,
    style_encoder_generator=None,
    y_style_encoder_generator=None,
    epoch_callback=None,
    step_callback=None,
    continue_model=None,
    features_per_group=-1,
    train_mixed_precision=True,
    progress_bar=False,
    n_targets_per_input=1,
    dataloader_class=priors.utils.StandardDataLoader,
    num_workers=None,
    **model_extra_args,
):
    total_start_time = time.time()
    device: str = gpu_device if torch.cuda.is_available() else "cpu:0"
    print(f"Using {device} device")
    using_dist, rank, device = init_dist(device)
    single_eval_pos_gen = (
        single_eval_pos_gen
        if callable(single_eval_pos_gen)
        else lambda: single_eval_pos_gen
    )

    def eval_pos_seq_len_sampler():
        single_eval_pos = single_eval_pos_gen()
        return single_eval_pos, seq_len

    if num_workers is not None:
        extra_prior_kwargs_dict["num_workers"] = num_workers

    dl = dataloader_class(
        get_batch_method=get_batch_method,
        num_steps=steps_per_epoch,
        batch_size=batch_size,
        eval_pos_seq_len_sampler=eval_pos_seq_len_sampler,
        seq_len_maximum=seq_len,
        device=device,
        n_targets_per_input=n_targets_per_input,
        **extra_prior_kwargs_dict,
    )

    if isinstance(criterion, nn.GaussianNLLLoss):
        n_out = 2
    elif (
        isinstance(criterion, BarDistribution)
        or "BarDistribution" in criterion.__class__.__name__
    ):  # TODO remove this fix (only for dev)
        n_out = criterion.num_bars
    elif isinstance(criterion, nn.CrossEntropyLoss):
        n_out = criterion.weight.shape[0]
    else:
        n_out = 1

    if continue_model:
        model = continue_model
    else:
        decoder_dict = (
            decoder_dict if decoder_dict else {"standard": (None, n_out)}
        )

        assert (
            features_per_group > 0 or features_per_group == -1
        ), "features_per_group must be > 0 or -1"
        architectural_features_per_group = (
            dl.num_features if features_per_group == -1 else features_per_group
        )

        encoder = (
            encoder_generator(architectural_features_per_group, emsize)
            if encoder_generator
            else None
        )
        y_encoder = (
            y_encoder_generator(1, emsize) if y_encoder_generator else None
        )
        model = transformer.TableTransformer(
            encoder=encoder,
            y_encoder=y_encoder,
            features_per_group=architectural_features_per_group,
            decoder_dict=decoder_dict,
            ninp=emsize,
            nhid=nhid,
            nlayers=nlayers,
            nhead=nhead,
            attention_between_features=features_per_group != -1,
            style_encoder=(
                style_encoder_generator(
                    architectural_features_per_group, emsize
                )
                if style_encoder_generator is not None
                else None
            ),  # the style encoder maps a tensor (batch_size [* num feature groups], [features in this group,] num styles) to (batch_size [* num feature groups], emsize)
            y_style_encoder=(
                y_style_encoder_generator(emsize)
                if y_style_encoder_generator is not None
                else None
            ),  # the y_style encoder maps a tensor (batch_size, num y styles) to (batch_size, emsize)
            **model_extra_args,
        )
    model.criterion = criterion
    if load_weights_from_this_state_dict is not None:
        model.load_state_dict(load_weights_from_this_state_dict)

    print(
        f"Using a Transformer with {sum(p.numel() for p in model.parameters())/1000/1000:.{2}f} M parameters"
    )

    try:
        for (k, v), (k2, v2) in zip(
            model.state_dict().items(),
            initialize_with_model.state_dict().items(),
        ):
            print(k, ((v - v2) / v).abs().mean(), v.shape)
    except Exception:
        pass

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = scheduler(
        optimizer, warmup_epochs, epochs if epochs is not None else 100
    )  # when training for fixed time lr schedule takes 100 steps

    start_epoch = 1  # Default start epoch

    # Load checkpoint if provided
    if train_state_dict_load_path:
        if (train_state_dict_save_path != train_state_dict_load_path) or (
            (train_state_dict_save_path == train_state_dict_load_path)
            and os.path.exists(train_state_dict_load_path)
        ):
            load_checkpoint(
                model, optimizer, scheduler, train_state_dict_load_path, device
            )
        else:
            print(
                f"Not loading checkpoint as {train_state_dict_load_path} does not exist yet. Will create it during training."
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

    scaler = GradScaler() if train_mixed_precision else None

    # check that everything uses up-to-date APIs
    utils.check_compatibility(dl)

    def train_epoch():
        model.train()  # Turn on the train mode
        total_loss = 0.0
        total_positional_losses = 0.0
        total_positional_losses_recorded = 0
        nan_steps = 0
        ignore_steps = 0
        before_get_batch = time.time()
        assert (
            len(dl) % aggregate_k_gradients == 0
        ), "Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it."
        tqdm_iter = (
            tqdm(range(len(dl)), desc="Training Epoch")
            if rank == 0 and progress_bar
            else None
        )
        importance_sampling_infos = []  # Renamed from squared_grad_magnitudes_and_infos

        for batch_index, batch in enumerate(dl):
            assert batch.batch_first, "batch_first must be true, make sure to adapt your prior to provide shapes like (batch_size, seq_len, ...)"
            targets = batch.target_y.to(device)
            single_eval_pos = batch.single_eval_pos

            tqdm_iter.update() if tqdm_iter is not None else None
            if using_dist and not (
                batch_index % aggregate_k_gradients
                == aggregate_k_gradients - 1
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
                            n_targets_per_input,
                            output.shape[-1],
                        )

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
                                output.reshape(-1, n_out),
                                targets.long().flatten(),
                            )
                        else:
                            losses = criterion(output, targets.unsqueeze(-1))
                        losses = einops.rearrange(
                            losses, "(b t) s -> b s t", t=n_targets_per_input
                        )
                        losses = losses.mean(-1)
                        loss, nan_share = utils.torch_nanmean(
                            losses.mean(
                                1
                            ),  # loss per sequence without nanmean
                            return_nanshare=True,
                        )
                        loss_scaled = loss / aggregate_k_gradients

                    if scaler:
                        loss_scaled = scaler.scale(loss_scaled)
                    loss_scaled.backward()

                    if (
                        batch_index % aggregate_k_gradients
                        == aggregate_k_gradients - 1
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
                                aggregate_k_gradients == 1
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
            loss=total_loss / steps_per_epoch,
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
            range(start_epoch, epochs + 1)
            if epochs is not None
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
            if hasattr(dl, "validate") and epoch % validation_period == 0:
                with torch.no_grad():
                    val_score = dl.validate(model)

            else:
                val_score = None

            if verbose:
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
            if train_state_dict_save_path is not None and rank == 0:
                save_checkpoint(
                    model, optimizer, train_state_dict_save_path, epoch
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
