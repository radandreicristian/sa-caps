import torch.utils.data

from src.datasets.nlub.nlub_dataset import NluBenchmarkDataset


def get_nlub_dataloaders(dataset: NluBenchmarkDataset,
                         batch_size: int,
                         shuffle: bool,
                         p_train: float):
    train_data_len = int(p_train * len(dataset))
    test_data_len = len(dataset) - train_data_len

    # Todo - Balanced sampling pls
    train_indices, test_indices = torch.utils.data.random_split(dataset=dataset,
                                                                lengths=(train_data_len, test_data_len))
    train_dataloader = torch.utils.data.DataLoader(dataset=train_indices,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_indices,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle)

    return train_dataloader, test_dataloader
