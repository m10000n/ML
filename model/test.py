from contextlib import suppress
from datetime import datetime, timedelta

import torch
import torch.distributed as dist
from torch import amp
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from helper import system
from helper import time as time_
from helper.print import print_end, print_start
from model import model_helper
from model.experiment import Experiment


def main(
    exp: Experiment,
) -> None:
    pu = exp.get_pu()
    pu_str = ",".join([str(gpu_idx) for gpu_idx in pu]) if pu != "cpu" else "CPU"
    num_workers = system.get_num_workers()

    print_start(
        text=(
            f"Start Testing | {time_.now_str()} | Experiment: {exp.get_name()}"
            f" | Batch Size: {exp.get_batch_size()} | PU: {pu_str} | Number of Workers: {num_workers}"
            f" | Autocast: {'enabled' if system.with_autocast() else 'disabled'}"
        ),
        mode="primary",
    )

    world_size = system.get_world_size()
    system.set_num_workers(max(1, num_workers // world_size) if world_size > 1 else num_workers)

    exp.start("test")

    if world_size <= 1:
        _test(
            rank=0,
            exp=exp,
        )
    else:
        torch.multiprocessing.spawn(
            fn=_test,
            args=(exp,),
            nprocs=world_size,
        )

    system.set_num_workers(num_workers)

    exp.end("test")
    exp.write()

    print_end(text="Finished testing.", mode="primary")


def _test(rank: int, exp: Experiment) -> None:
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

        model: Module | DDP = exp.load_model(pu)

        if multi_gpu:
            model = DDP(model, device_ids=[pu.index])

        test_loader = exp.create_loader(
            type="test", world_size=world_size, rank=rank, num_workers=num_workers, prefetch_factor=prefetch_factor
        )

        criterion = exp.create_criterion(pu)

        model.eval()
        ids = []
        n_test = torch.tensor(0, device=pu)
        test_loss = torch.tensor(0.0, device=pu)
        actual = torch.tensor([], dtype=torch.int, device=pu)
        predicted = torch.tensor([], dtype=torch.int, device=pu)

        track_time = exp.track_time_spend()

        if rank == 0:
            if track_time:
                spend_moving = timedelta(0)
                spend_forward = timedelta(0)

            pbar = tqdm(desc="Test", initial=0, total=len(test_loader), ncols=120, leave=False)
            spend_loading = timedelta(0)
            start_loading = datetime.now()

        with torch.no_grad():
            for data, labels, id in test_loader:
                if rank == 0:
                    spend_loading += datetime.now() - start_loading

                ids.extend(id)

                # move to device
                if track_time and rank == 0:
                    start_moving = datetime.now()
                if with_autocast:
                    data = data.to(pu, dtype=torch.bfloat16, non_blocking=True)
                else:
                    data = data.to(pu, non_blocking=True)
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
                model_helper.assert_no_nan(tensor=loss, tensor_name="test loss")
                if track_time and rank == 0:
                    if world_size > 1:
                        torch.cuda.synchronize()
                    spend_forward += datetime.now() - start_forward

                # batch size, test loss, actual, predicted
                batch_size = int(data.size(0))
                n_test += batch_size

                if criterion.config.reduction == "mean":
                    test_loss += loss * batch_size
                else:
                    test_loss += loss

                actual = torch.cat((actual, labels))
                _, p = torch.max(output, 1)
                predicted = torch.cat((predicted, p))

                if rank == 0:
                    pbar.update(1)
                    start_loading = datetime.now()

            if multi_gpu:
                dist.all_reduce(n_test, op=dist.ReduceOp.SUM)
                dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)

                actual_gathered = [torch.empty_like(actual) for _ in range(dist.get_world_size())]
                predicted_gathered = [torch.empty_like(predicted) for _ in range(dist.get_world_size())]

                dist.all_gather(actual_gathered, actual)
                dist.all_gather(predicted_gathered, predicted)

                actual = torch.cat(actual_gathered)
                predicted = torch.cat(predicted_gathered)

            test_loss_ = float((test_loss / n_test).item())
            actual_ = actual.cpu().tolist()
            predicted_ = predicted.cpu().tolist()

            if rank == 0:
                exp.set_loss(loss=test_loss_, select="test")
                exp.set_confusion_actual(actual=actual_)
                exp.set_confusion_predicted(predicted=predicted_)
                exp.set_confusion_ids(ids=ids)
                if track_time:
                    exp.add_time_spend(task="loading", time=spend_loading)
                    exp.add_time_spend(task="to_pu", time=spend_moving)
                    exp.add_time_spend(task="forward", time=spend_forward)
                exp.write()

            if multi_gpu:
                dist.barrier()

        if rank == 0:
            pbar.close()

    except BaseException:
        if multi_gpu and dist.is_initialized():
            with suppress(Exception):
                dist.abort()  # type: ignore[attr-defined]
        raise
    finally:
        if multi_gpu and dist.is_initialized():
            with suppress(Exception):
                dist.destroy_process_group()
