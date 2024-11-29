import torch.cuda
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import data.hcp_openacces.dataloader as dataloader
from helper.model import get_pu


def test(rank, world_size, model_class, dataset, num_workers, model_path, log):
    if rank == 0:
        print(
            f"----->>> Start testing. Model: {model_class.__name__} | Available GPUs: {world_size} | Workers: {num_workers} <<<-----"
        )

    multi_gpu = world_size > 1

    if multi_gpu:
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    pu = get_pu(world_size, rank)

    state_dict = torch.load(f=model_path, map_location=pu, weights_only=True)
    model = model_class()
    model.load_state_dict(state_dict=state_dict)
    model.to(pu)
    if multi_gpu:
        with model.join():
            model = DDP(model, device_ids=[rank])

    test_loader = dataloader.get_loader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
    )

    n_test = len(test_loader.dataset)

    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    predicted = torch.tensor([], dtype=torch.int32)
    actual = torch.tensor([], dtype=torch.int32)

    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(pu, non_blocking=True)
            labels = labels.to(pu, non_blocking=True)

            output = model(data)
            loss = criterion(output, labels)

            test_loss += loss.item() * data.size(0)
            _, predicted_ = torch.max(output, 1)
            predicted_ = predicted_.to("cpu")
            predicted = torch.cat((predicted, predicted_))
            labels = labels.to("cpu")
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
        print("----->>> Finished training <<<-----")

    if multi_gpu:
        dist.destroy_process_group()
