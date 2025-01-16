import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def get_dataset(size, dimensions, n_classes):
    data = torch.randn(size, *dimensions)
    labels = torch.randint(0, n_classes, (size,))
    return TensorDataset(data, labels)


def get_loader(batch_size, size, dimensions, n_classes):
    dataset = get_dataset(size=size, dimensions=dimensions, n_classes=n_classes)
    return DataLoader(dataset=dataset, batch_size=batch_size)


def get_distributed_loader(batch_size, size, dimensions, n_classes, world_size, rank):
    dataset = get_dataset(size=size, dimensions=dimensions, n_classes=n_classes)
    sampler = DistributedSampler(dataset=dataset, num_replicas=world_size, rank=rank)
    return DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)


def get_ids(size_train, size_val, size_test):
    return {
        "train": [i for i in range(size_train)],
        "val": [i for i in range(size_val)],
        "test": [i for i in range(size_test)],
    }
