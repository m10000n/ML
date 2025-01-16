import sys

import torch.cuda
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

import data.hcp_openacces.dataloader as dataloader
from helper.model import get_pu


def train(rank, world_size, model_class, datasets, num_workers, model_path, log):
    if rank == 0:
        print(
            f"----->>> Start training. Model: {model_class.__name__} | Available GPUs: {world_size} | Workers: {num_workers} <<<-----"
        )

    multi_gpu = world_size > 1

    if multi_gpu:
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    pu = get_pu(world_size, rank)

    model = model_class()
    model = model.to(pu)
    if multi_gpu:
        with model.join():
            model = DDP(model, device_ids=[rank])

    train_loader = dataloader.get_loader(
        dataset=datasets["train"],
        batch_size=32,
        shuffle=True,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
    )
    val_loader = dataloader.get_loader(
        dataset=datasets["val"],
        batch_size=32,
        shuffle=False,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
    )

    n_train = len(train_loader.dataset)
    n_train_batches = len(train_loader)
    n_val = len(val_loader.dataset)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.1, patience=15
    )

    learning_rate = []
    train_loss = []
    val_loss = []
    val_accuracy = []

    # patience is not mentioned in paper
    patience = 30
    best_val_loss = float("inf")
    no_improve_epochs = 0
    early_stop = False
    epoch = -1

    while not early_stop:
        epoch += 1
        if multi_gpu:
            train_loader.sampler.set_epoch(epoch)

        model.train()
        lr = optimizer.param_groups[0]["lr"]
        learning_rate.append(lr)

        train_loss_epoch = 0.0

        for i_batch, (data, labels) in enumerate(train_loader):
            if rank == 0:
                sys.stdout.write(
                    f"\repoch: {epoch + 1} | batch: {i_batch + 1}/{n_train_batches} | learning rate: {lr:.0e}"
                )
                sys.stdout.flush()

            data = data.to(pu, non_blocking=True)
            labels = labels.to(pu, non_blocking=True)

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
            print(f"\ttraining loss: {train_loss_epoch:.4f}")

        model.eval()
        val_loss_epoch = 0.0
        correct = 0

        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(pu, non_blocking=True)
                labels = labels.to(pu, non_blocking=True)

                output = model(data)
                loss = criterion(output, labels)

                val_loss_epoch += loss.item() * data.size(0)
                _, predicted = torch.max(output, 1)
                correct += (predicted == labels).sum().item()

        val_loss_epoch /= n_val
        val_loss.append(val_loss_epoch)
        val_accuracy.append(correct / n_val)

        scheduler.step(val_loss_epoch)

        if rank == 0:
            log.n_epochs = epoch + 1
            log.learning_rate = learning_rate
            log.set_loss(train=train_loss, val=val_loss)
            log.set_accuracy(val=val_accuracy)
            print(f"\tvalidation loss: {val_loss_epoch:.4f}")
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
            log.update()

    if rank == 0:
        print("----->>> Finished training <<<-----")

    if multi_gpu:
        dist.destroy_process_group()
