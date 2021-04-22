import torch.utils.data

from src.datasets.nlub.nlub_dataset import NluBenchmarkDataset


def get_nlub_dataloaders(dataset: NluBenchmarkDataset,
                         batch_size: int,
                         shuffle: bool
                         ):
    l_splits = {
        'train': 0.6,
        'val': 0.2,
        'test': 0.2
    }

    assert sum(l_splits.values()) == 1., "Percentages of splits do not add up to 1."

    train_data_len = int(l_splits['train'] * len(dataset))
    val_data_len = int(l_splits['val'] * len(dataset))
    test_data_len = len(dataset) - train_data_len - val_data_len

    # Todo - Balanced sampling pls
    train_indices, val_indices, test_indices = torch.utils.data.random_split(dataset=dataset,
                                                                             lengths=(train_data_len,
                                                                                      val_data_len,
                                                                                      test_data_len),
                                                                             generator=torch.Generator().manual_seed(
                                                                                 42))
    train_dataloader = torch.utils.data.DataLoader(dataset=train_indices,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle)

    val_dataloader = torch.utils.data.DataLoader(dataset=val_indices,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_indices,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle)

    return train_dataloader, val_dataloader, test_dataloader
