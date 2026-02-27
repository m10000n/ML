from __future__ import annotations

import math
from contextlib import suppress
from datetime import datetime, timedelta
from typing import Literal, cast

import torch
import torch.distributed as dist
from torch import amp
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from config.file_names import TRAINED_MODEL_FILE_NAME
from helper import model_ as helper_model
from helper import system, time
from helper.exception import NanError
from helper.helper_ import round_to_str
from helper.print import print_color, print_end, print_info, print_start
from model import model_helper
from model.experiment import Experiment
from model.scheduler import (
    MetricLRScheduler_,
    StepLRScheduler_,
)


def main(
    exp: Experiment,
) -> None:
    pu = system.get_pu()
    pu_str = ",".join([str(gpu_idx) for gpu_idx in pu]) if pu != "cpu" else "CPU"
    num_workers = system.get_num_workers()

    print_start(
        text=(
            f"Start Training | {time.now_str()} | Experiment: {exp.get_name()}"
            f" | Batch Size: {exp.get_batch_size()} | Batch Accum.: {exp.get_batch_accumulation()}"
            f" | PU: {pu_str} | Number of Workers: {num_workers} | Prefetch Factor: {system.get_prefetch_factor()}"
            f" | Autocast: {'enabled' if system.with_autocast() else 'disabled'}"
        ),
        mode="primary",
    )

    world_size = system.get_world_size()
    system.set_num_workers(max(1, num_workers // world_size) if world_size > 1 else num_workers)

    exp.start("train")

    if world_size <= 1:
        _train(rank=0, exp=exp)
    else:
        torch.multiprocessing.spawn(
            fn=_train,
            args=(exp,),
            nprocs=world_size,
        )

    system.set_num_workers(num_workers)

    exp.end("train")
    exp.write()

    print_end(text="Finished training.", mode="primary")


def _train(rank: int, exp: Experiment) -> None:
    world_size = system.get_world_size()
    multi_gpu = system.is_multi_gpu()
    num_workers = system.get_num_workers()
    prefetch_factor = system.get_prefetch_factor()
    with_autocast = system.with_autocast()

    pu = torch.device(system.get_device_str(rank))

    if system.is_gpu():
        torch.cuda.set_device(rank)

    if multi_gpu:
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank, device_id=pu)

    try:
        model: Module | DDP = exp.create_model(pu)

        if multi_gpu:
            model = DDP(model, device_ids=[pu.index])

        train_loader = exp.create_loader(
            type="train", world_size=world_size, rank=rank, num_workers=num_workers, prefetch_factor=prefetch_factor
        )
        val_loader = exp.create_loader(
            type="val", world_size=world_size, rank=rank, num_workers=num_workers, prefetch_factor=prefetch_factor
        )

        criterion = exp.create_criterion(pu)

        optimizer = exp.create_optimizer(model)
        scheduler = exp.create_scheduler(optimizer)
        stop_criterion = exp.create_stop_criterion()
        clip_grad = exp.create_clip_grad(model)
        no_learn_limit = exp.get_no_learn_limit()

        track_time = exp.track_time_spend()
        warmup_ = exp.create_warmup(optimizer)
        accum_steps = exp.get_batch_accumulation()

        if warmup_ is not None:
            n_steps = warmup_.get_n_total_steps()

            model.train()

            n_accum = torch.tensor(0, device=pu)
            accum_loss = torch.tensor(0.0, device=pu)

            if rank == 0:
                pbar = tqdm(desc=f"Warmup(step)", initial=0, total=n_steps, ncols=120, leave=False)
                warmup_start = datetime.now()
                spend_loading = timedelta(0)

                if track_time:
                    spend_moving = timedelta(0)
                    spend_forward = timedelta(0)
                    spend_backprop = timedelta(0)

            loader = iter(train_loader)
            i = 0

            while not warmup_.done():
                i += 1
                is_update_step = i % accum_steps == 0

                start_loading = datetime.now()
                try:
                    data, labels, _ = next(loader)
                except StopIteration:
                    loader = iter(train_loader)
                    data, labels, _ = next(loader)
                if rank == 0:
                    spend_loading += datetime.now() - start_loading

                # move to device
                if track_time and rank == 0:
                    start_moving = datetime.now()
                if with_autocast:
                    data = data.to(pu, dtype=torch.bfloat16, non_blocking=True)
                else:
                    data = data.to(pu, non_blocking=True)
                labels = labels.to(pu, non_blocking=True)
                if track_time and rank == 0:
                    if world_size > 1:
                        torch.cuda.synchronize()
                    spend_moving += datetime.now() - start_moving

                # forward
                if track_time and rank == 0:
                    start_forward = datetime.now()
                with amp.autocast(device_type=pu.type, dtype=torch.bfloat16, enabled=with_autocast):
                    output = model(data)
                    loss = criterion(input=output, target=labels)
                try:
                    model_helper.assert_no_nan(tensor=loss, tensor_name="warmup loss")
                except AssertionError as e:
                    raise NanError(message=e.args[0])
                if track_time and rank == 0:
                    if world_size > 1:
                        torch.cuda.synchronize()
                    spend_forward += datetime.now() - start_forward

                # backward
                if track_time and rank == 0:
                    start_backprop = datetime.now()
                (loss / accum_steps).backward()
                if is_update_step:
                    if clip_grad:
                        clip_grad.clip()
                    optimizer.step()
                    optimizer.zero_grad()
                if track_time and rank == 0:
                    if world_size > 1:
                        torch.cuda.synchronize()
                    spend_backprop += datetime.now() - start_backprop

                # accumulation loss
                batch_size = int(data.size(0))
                n_accum += batch_size
                accum_loss += loss.detach() * batch_size

                if is_update_step:
                    warmup_.step()

                    if multi_gpu:
                        dist.all_reduce(n_accum, op=dist.ReduceOp.SUM)
                        dist.all_reduce(accum_loss, op=dist.ReduceOp.SUM)
                    accum_loss_ = float((accum_loss / n_accum).item())

                    if rank == 0:
                        post_fix = {
                            "lr": f"{optimizer.get_learning_rate():.2e}",
                            "loss": round_to_str(x=accum_loss_, digits=4),
                        }
                        pbar.update(1)
                        pbar.set_postfix(post_fix)

                    n_accum = torch.tensor(0, device=pu)
                    accum_loss = torch.tensor(0.0, device=pu)

                if rank == 0:
                    exp.append_loss(loss=loss.item(), select="warmup")
                    start_loading = datetime.now()

            if rank == 0:
                warmup_duration = datetime.now() - warmup_start
                pbar.close()
                print_color(
                    text=f"{n_steps} warmup step{'s' if n_steps > 1 else ''} completed in {time.h_min_sec_str(delta=warmup_duration, truncate=True)}.",
                    color_="green",
                )
                exp.add_time_spend(task="loading", time=spend_loading)
                if track_time:
                    exp.add_time_spend(task="to_pu", time=spend_moving)
                    exp.add_time_spend(task="forward", time=spend_forward)
                    exp.add_time_spend(task="backward", time=spend_backprop)
                exp.write()

        epoch = 0
        lowest_val_loss = float("inf")
        lr = -1.0

        early_stop: Literal["early_stop", "no_learn", ""] = ""
        val_acc_ = []

        cross_validation = exp.for_cross_validation()

        while not stop_criterion.stop() and not early_stop:
            if rank == 0:
                start_epoch = datetime.now()
                spend_loading = timedelta(0)
                if track_time:
                    spend_moving = timedelta(0)
                    spend_forward = timedelta(0)
                    spend_backprop = timedelta(0)

            # training
            model.train()

            if multi_gpu:
                cast(DistributedSampler, train_loader.sampler).set_epoch(epoch)

            n_train = torch.tensor(0, device=pu)
            train_loss = torch.tensor(0.0, device=pu)
            n_accum = torch.tensor(0, device=pu)
            accum_loss = torch.tensor(0.0, device=pu)

            if rank == 0:
                pbar = tqdm(
                    desc=f"Training({epoch})",
                    initial=0,
                    total=math.ceil(len(train_loader) / accum_steps),
                    ncols=120,
                    leave=False,
                )
                start_loading = datetime.now()

            for i, (data, labels, _) in enumerate(train_loader):
                if rank == 0:
                    spend_loading += datetime.now() - start_loading
                is_update_step = (i + 1) % accum_steps == 0 or i == len(train_loader) - 1

                lr = optimizer.get_learning_rate()

                # move to device
                if track_time and rank == 0:
                    start_moving = datetime.now()
                if with_autocast:
                    data = data.to(pu, dtype=torch.bfloat16, non_blocking=True)
                else:
                    data = data.to(pu, non_blocking=True)
                labels = labels.to(pu, non_blocking=True)
                if track_time and rank == 0:
                    if world_size > 1:
                        torch.cuda.synchronize()
                    spend_moving += datetime.now() - start_moving

                # forward
                if track_time and rank == 0:
                    start_forward = datetime.now()
                with amp.autocast(device_type=pu.type, dtype=torch.bfloat16, enabled=with_autocast):
                    output = model(data)
                    loss = criterion(input=output, target=labels)
                try:
                    model_helper.assert_no_nan(tensor=loss, tensor_name="training loss")
                except AssertionError as e:
                    raise NanError(message=e.args[0])
                if track_time and rank == 0:
                    if world_size > 1:
                        torch.cuda.synchronize()
                    spend_forward += datetime.now() - start_forward

                # backward
                if track_time and rank == 0:
                    start_backprop = datetime.now()
                (loss / accum_steps).backward()
                if is_update_step:
                    if clip_grad:
                        clip_grad.clip()
                    optimizer.step()
                    optimizer.zero_grad()
                if track_time and rank == 0:
                    if world_size > 1:
                        torch.cuda.synchronize()
                    spend_backprop += datetime.now() - start_backprop

                # batch size, training loss, accumulation loss
                batch_size = int(data.size(0))

                n_train += batch_size
                n_accum += batch_size

                if criterion.config.reduction == "mean":
                    train_loss += loss.detach() * batch_size
                    accum_loss += loss.detach() * batch_size
                else:
                    train_loss += loss.detach()
                    accum_loss += loss.detach()

                if is_update_step:
                    if isinstance(scheduler, StepLRScheduler_):
                        scheduler.step(type="step")

                    if multi_gpu:
                        dist.all_reduce(n_accum, op=dist.ReduceOp.SUM)
                        dist.all_reduce(accum_loss, op=dist.ReduceOp.SUM)
                    accum_loss_ = float((accum_loss / n_accum).item())

                    if rank == 0:
                        post_fix = {"loss": round_to_str(x=accum_loss_, digits=4)}
                        pbar.update(1)
                        pbar.set_postfix(post_fix)

                    n_accum = torch.tensor(0, device=pu)
                    accum_loss = torch.tensor(0.0, device=pu)

                if rank == 0:
                    start_loading = datetime.now()

            if rank == 0:
                pbar.close()

            if multi_gpu:
                dist.all_reduce(n_train, op=dist.ReduceOp.SUM)
                dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)

            train_loss_ = float((train_loss / n_train).item())

            # validation
            model.eval()

            if multi_gpu:
                cast(DistributedSampler, val_loader.sampler).set_epoch(epoch)

            n_val = torch.tensor(0, device=pu)
            val_loss = torch.tensor(0.0, device=pu)
            correct = torch.tensor(0, device=pu)

            if rank == 0:
                pbar = tqdm(desc=f"Validation({epoch})", initial=0, total=len(val_loader), ncols=120, leave=False)
                start_loading = datetime.now()

            with torch.no_grad():
                for data, labels, _ in val_loader:
                    if rank == 0:
                        spend_loading += datetime.now() - start_loading

                    # move to device
                    if track_time and rank == 0:
                        start_moving = datetime.now()
                    if with_autocast:
                        data = data.to(pu, dtype=torch.bfloat16, non_blocking=True)
                    else:
                        data = data.to(pu, non_blocking=True)
                    labels = labels.to(pu, non_blocking=True)
                    if track_time and rank == 0:
                        if world_size > 1:
                            torch.cuda.synchronize()
                        spend_moving += datetime.now() - start_moving

                    # forward
                    if track_time and rank == 0:
                        start_forward = datetime.now()
                    with amp.autocast(device_type=pu.type, dtype=torch.bfloat16, enabled=with_autocast):
                        output = model(data)
                        loss = criterion(input=output, target=labels)
                    model_helper.assert_no_nan(tensor=loss, tensor_name="validation loss")
                    if track_time and rank == 0:
                        if world_size > 1:
                            torch.cuda.synchronize()
                        spend_forward += datetime.now() - start_forward

                    # batch size, validation loss, correct
                    batch_size = int(data.size(0))
                    n_val += batch_size

                    if criterion.config.reduction == "mean":
                        val_loss += loss * batch_size
                    else:
                        val_loss += loss

                    _, predicted = torch.max(output, 1)
                    correct_ = (predicted == labels).sum()
                    correct += correct_

                    if rank == 0:
                        pbar.set_postfix(
                            {
                                "loss": round_to_str(x=loss.item(), digits=4),
                                "acc": round_to_str(x=correct_.item() / batch_size, digits=2),
                            }
                        )
                        pbar.update(1)
                        start_loading = datetime.now()

            if rank == 0:
                pbar.close()

            if multi_gpu:
                dist.all_reduce(n_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(correct, op=dist.ReduceOp.SUM)

            val_loss_ = float((val_loss / n_val).item())
            val_acc_.append(float((correct / n_val).item()))

            if rank == 0:
                epoch_duration = datetime.now() - start_epoch
                if epoch % 10 == 0:
                    header = " Epoch | Learning Rate | Train Loss | Validation Loss | Validation Accuracy | Duration "
                    print(header)
                data = (
                    f"{epoch:^7}|{lr:^15.2e}|{train_loss_:^12.6f}|{val_loss_:^17.6f}|"
                    f"{val_acc_[-1]:^21.4f}| {time.h_min_sec_str(delta=epoch_duration, truncate=True)}"
                )

                if val_loss_ < lowest_val_loss:
                    print_color(text=data, color_="green")
                else:
                    print(data)

                exp.set_total_epochs(epoch + 1)
                exp.append_epoch_duration(epoch_duration)
                exp.append_learning_rate(lr)
                exp.append_loss(loss=train_loss_, select="train")
                exp.append_loss(loss=val_loss_, select="val")
                exp.append_val_accuracy(accuracy=val_acc_[-1])
                exp.add_time_spend(task="loading", time=spend_loading)
                if track_time:
                    exp.add_time_spend(task="to_pu", time=spend_moving)
                    exp.add_time_spend(task="forward", time=spend_forward)
                    exp.add_time_spend(task="backward", time=spend_backprop)

            if val_loss_ < lowest_val_loss:
                lowest_val_loss = val_loss_
                if rank == 0:
                    torch.save(
                        cast(Module, (model.module if hasattr(model, "module") else model)).state_dict(),
                        exp.get_dir(with_status=True) / TRAINED_MODEL_FILE_NAME.format(exp_name=exp.get_name()),
                    )
                    exp.set_model_epoch(epoch)

            if rank == 0:
                exp.write()

            epoch += 1

            if isinstance(scheduler, MetricLRScheduler_):
                scheduler.step(metric=val_loss_)
            elif isinstance(scheduler, StepLRScheduler_):
                scheduler.step(type="epoch")
            else:
                raise ValueError(f"Scheduler type `{type(scheduler)}` not supported.")

            stop_criterion.step(metric=val_loss_)

            if (
                no_learn_limit is not None
                and len(val_acc_) - 1 >= no_learn_limit
                and all(math.isclose(v_a, val_acc_[-1]) for v_a in val_acc_[-(no_learn_limit + 1) :])
            ):
                early_stop = "no_learn"
            elif not cross_validation and helper_model.early_stop_is_active(exp.get_name()) and epoch >= 2:
                early_stop = "early_stop"

            if multi_gpu:
                dist.barrier()

        if rank == 0:
            print_color(text=f"{epoch} epoch{'s' if epoch > 1 else ''} of training completed.", color_="green")

            if early_stop and not stop_criterion.stop():
                if early_stop == "early_stop":
                    msg = "User triggered early stop."
                    helper_model.deactivate_early_stop(exp.get_name())
                    exp.set_early_stop_reason(reason="user")
                elif early_stop == "no_learn":
                    msg = "No learning triggered early stop."
                    exp.set_early_stop_reason(reason="no_learn")
                print_info(msg + f" Training stopped at epoch `{epoch}`.")
                exp.set_status(status="early_stop")

    except BaseException:
        if multi_gpu and dist.is_initialized():
            with suppress(Exception):
                dist.abort()  # type: ignore[attr-defined]
        raise
    finally:
        if multi_gpu and dist.is_initialized():
            with suppress(Exception):
                dist.destroy_process_group()
