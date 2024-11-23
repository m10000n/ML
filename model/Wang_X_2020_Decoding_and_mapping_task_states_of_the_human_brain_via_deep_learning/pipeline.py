import inspect
import shutil
import sys
from pathlib import Path

import torch.cuda
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

import data.hcp_openacces.dataloader as dataloader
from data.hcp_openacces.dataset import TaskDataset
from helper.env import save_dependencies
from helper.folder import create_results
from helper.log import Log
from helper.system import get_max_workers, get_num_physical_cores
from model.Wang_X_2020_Decoding_and_mapping_task_states_of_the_human_brain_via_deep_learning.model import (
    Model,
)

EXP_NAME = "original"


def get_pu(world_size, rank):
    if world_size == 0:
        return "cpu"
    elif world_size == 1:
        return "cuda"
    else:
        return rank


def train(rank, world_size, datasets, model_path, criterion, log):
    if rank == 0:
        print(f"----->Start training. Available GPUs: {world_size}<<-----")

    multi_gpu = world_size > 1

    if multi_gpu:
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    pu = get_pu(world_size, rank)

    model = Model().to(pu)
    if multi_gpu:
        with model.join():
            model = DDP(model, device_ids=[rank])

    train_loader = dataloader.get_loader(
        dataset=datasets["train"],
        batch_size=32,
        shuffle=True,
        world_size=world_size,
        rank=rank,
        num_workers=max(1, int(get_num_physical_cores() / 2)),
    )
    val_loader = dataloader.get_loader(
        dataset=datasets["val"],
        batch_size=32,
        shuffle=False,
        world_size=world_size,
        rank=rank,
        num_workers=max(1, int(get_num_physical_cores() / 2)),
    )

    n_train = len(train_loader.dataset)
    n_train_batches = len(train_loader)
    n_val = len(val_loader.dataset)

    train_loss = []
    val_loss = []
    val_accuracy = []

    criterion = criterion.to(pu)
    optimizer = optim.Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.1, patience=15
    )

    # patience is not mentioned in paper
    patience = 30
    best_val_loss = float("inf")
    no_improve_epochs = 0
    early_stop = False
    epoch = 0

    while not early_stop:
        epoch += 1
        if multi_gpu:
            train_loader.sampler.set_epoch(epoch)

        model.train()
        train_loss_epoch = 0.0

        for i_batch, (data, labels) in enumerate(train_loader):
            if rank == 0:
                sys.stdout.write(
                    f"\repoch: {epoch} | batch: {i_batch + 1}/{n_train_batches}"
                )
                sys.stdout.flush()

            data = data.to(pu)
            labels = labels.to(pu)

            output = model(data)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item() * data.size(0)
        print()

        train_loss_epoch /= n_train

        train_loss.append(train_loss_epoch)
        if rank == 0:
            print(f"\ttraining loss: {train_loss[-1]:.4f}")

        model.eval()
        val_loss_epoch = 0.0
        correct = 0

        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(pu)
                labels = labels.to(pu)

                output = model(data)
                loss = criterion(output, labels)

                val_loss_epoch += loss.item() * data.size(0)
                _, predicted = torch.max(output, 1)
                correct += (predicted == labels).sum().item()

        val_loss_epoch /= n_val

        scheduler.step(val_loss_epoch)

        val_loss.append(val_loss_epoch)
        val_accuracy.append(correct / n_val)

        if rank == 0:
            log.n_epochs = epoch
            log.set_loss(train=train_loss, val=val_loss)
            log.set_accuracy(val=val_accuracy)
            log.update()
            print(f"\tvalidation loss: {val_loss[-1]:.4f}")
            print(f"\tvalidation accuracy: {val_accuracy[-1]:.4f}")

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            no_improve_epochs = 0
            if rank == 0:
                torch.save(model.state_dict(), model_path)
                print(f"\t---saved model---")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            early_stop = True

    if rank == 0:
        print("----->>Finished training<<-----")

    if multi_gpu:
        dist.destroy_process_group()


def test(rank, world_size, dataset, model_path, criterion, log):
    if rank == 0:
        print(f"----->Start testing. Available GPUs: {world_size}<<-----")

    multi_gpu = world_size > 1

    if multi_gpu:
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    pu = get_pu(world_size, rank)

    state_dict = torch.load(f=model_path, map_location=pu, weights_only=True)
    model = Model()
    model.load_state_dict(state_dict=state_dict)
    if multi_gpu:
        with model.join():
            model = DDP(model, device_ids=[rank])

    test_loader = dataloader.get_loader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        world_size=world_size,
        rank=rank,
        num_workers=max(1, int(get_num_physical_cores() / 2)),
    )

    n_test = len(test_loader.dataset)

    criterion = criterion.to(pu)

    test_loss = 0.0
    predicted = torch.tensor([], dtype=torch.int32)
    actual = torch.tensor([], dtype=torch.int32)

    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(pu)
            labels = labels.to(pu)

            output = model(data)
            loss = criterion(output, labels)

            test_loss += loss.item() * data.size(0)
            _, predicted_ = torch.max(output, 1)
            predicted = torch.cat((predicted, predicted_))
            actual = torch.cat((actual, labels))

    test_loss /= n_test
    correct = (predicted == actual).sum().item()
    test_accuracy = correct / n_test

    if rank == 0:
        log.set_loss(test=test_loss)
        log.set_accuracy(test=test_accuracy)
        log.set_confusion(predicted=predicted.tolist(), actual=actual.tolist())
        log.update()
        print(f"\ttest loss: {test_loss:.4f}")
        print(f"\ttest accuracy: {test_accuracy:.4f}")
        print("----->>Finished training<<-----")

    if multi_gpu:
        dist.destroy_process_group()


def main():
    folder_path = Path(__file__).parent
    results_path = folder_path / "results" / EXP_NAME
    trained_model_path = results_path / f"trained_model_{EXP_NAME}.pth"

    create_results(model_path=folder_path, model_name=EXP_NAME)

    log = Log(file_path=results_path / f"log_{EXP_NAME}.json")
    save_dependencies(path=results_path / f"dependencies_{EXP_NAME}.txt")

    shutil.copy(inspect.getfile(Model), results_path / f"model_{EXP_NAME}.py")
    shutil.copy(Path(__file__), results_path / f"pipeline_{EXP_NAME}.py")

    if not TaskDataset.fully_downloaded():
        TaskDataset.download_dataset(num_workers=get_max_workers())

    ids = dataloader.get_ids(ratios=(0.7, 0.1, 0.2))
    log.set_dataset_ids(ids)

    datasets = {"train": TaskDataset(ids["train"]), "val": TaskDataset(ids["val"])}
    test_dataset = TaskDataset(ids["test"])

    criterion = nn.CrossEntropyLoss()

    world_size = torch.cuda.device_count()
    if world_size <= 1:
        train(
            world_size=0,
            rank=0,
            datasets=datasets,
            model_path=trained_model_path,
            criterion=criterion,
            log=log,
        )
        test(
            world_size=0,
            rank=0,
            dataset=test_dataset,
            model_path=trained_model_path,
            criterion=criterion,
            log=log,
        )
    else:
        torch.multiprocessing.spawn(
            fn=train,
            args=(world_size, datasets, trained_model_path, criterion, log),
            nprocs=world_size,
        )
        torch.multiprocessing.spawn(
            fn=test,
            args=(world_size, test_dataset, trained_model_path, criterion, log),
            nprocs=world_size,
        )


if __name__ == "__main__":
    main()
