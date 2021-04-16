import logging
import os
import pathlib

import torch
from einops import rearrange

import src.utils.constants as c
from src.losses.dot import dot_loss
from src.models.capsnet import SRCapsNet
from src.utils.date import get_hour_min


def train(optim_name='adam',
          epochs: int = 200,
          lang: str = 'en',
          batch_size: int = 8,
          shuffle: bool = True,
          p_train: float = 0.8,
          p_dropout_featurizer: float = 0.5,
          p_dropout_intent: float = 0.5,
          max_ngram_size: int = 3,
          max_seq_len: int = 35,
          n_chars: int = 26,
          n_heads: int = 8,
          n_layers: int = 2,
          d_dense: int = 300,
          d_model: int = 256,
          d_semantic_space: int = 20,
          dataset_name: str = 'nlub',
          use_sparse_features: bool = True,
          use_dense_features: bool = True,
          verbose: bool = True,
          debug: bool = False
          ):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug(f'Started training procedure at {get_hour_min()}.')

    if debug:
        torch.autograd.set_detect_anomaly(True)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    root_path = pathlib.Path(__file__).parent.parent

    sparse_embeddings_path = c.paths[f'{lang}_{max_ngram_size}.json'] if use_sparse_features else None
    dense_embeddings_path = c.paths['en_ft_300.bin'] if use_dense_features else None

    sparse_embeddings_abs_path = os.path.join(root_path, sparse_embeddings_path)
    dense_embeddings_abs_path = os.path.join(root_path, dense_embeddings_path)

    logging.debug(f'Creating the dataset object at {get_hour_min()}.')
    if dataset_name == 'nlub':
        from src.datasets.nlub.nlub_dataset import NluBenchmarkDataset
        from src.datasets.nlub.nlub_dataloader import get_nlub_dataloaders
        dataset_abs_path = os.path.join(root_path, c.paths['nlub.csv'])
        dataset = NluBenchmarkDataset(dataset_path=dataset_abs_path,
                                      max_ngram_size=max_ngram_size,
                                      n_chars=n_chars,
                                      max_seq_len=max_seq_len,
                                      dense_embeddings_path=dense_embeddings_abs_path,
                                      sparse_embeddings_path=sparse_embeddings_abs_path)
        n_slots, n_intents = dataset.get_counts()
        d_sparse = dataset.sparse_dim
        train_loader, test_loader = get_nlub_dataloaders(dataset, batch_size=batch_size, shuffle=shuffle,
                                                         p_train=p_train)
    else:
        raise ValueError('Unsupported dataset name :c')

    logging.debug(f'Creating the model object at {get_hour_min()}.')
    capsnet = SRCapsNet(n_slots=n_slots,
                        n_intents=n_intents,
                        d_dense=300,
                        d_sparse=d_sparse,
                        max_seq_len=max_seq_len,
                        n_heads=n_heads,
                        n_layers=n_layers,
                        d_model=d_model,
                        d_semantic_space=d_semantic_space,
                        p_dropout_featurizer=p_dropout_featurizer,
                        p_dropout_intent=p_dropout_intent,
                        device=device).to(device)
    # Todo: K fold Cross-val - For each i in K

    if optim_name == 'adam':
        optimizer = torch.optim.Adam(params=capsnet.parameters(), lr=0.001)
    else:
        raise ValueError('Unsupported dataset name :c')

    slot_criterion = torch.nn.CrossEntropyLoss()

    logging.debug(f'Started training at {get_hour_min()}.')
    for epoch in range(epochs):
        print(f'Starting epoch {epoch + 1}')
        epoch_loss = 0.0
        for i, data in enumerate(train_loader):
            x_dense, x_sparse, y_slots, y_intents = data

            x_dense = torch.reshape(x_dense, (batch_size, max_seq_len, d_dense)).to(device)
            x_sparse = torch.reshape(x_sparse, (batch_size, max_seq_len, d_sparse)).to(device)

            y_slots = y_slots.to(device)
            y_intents = torch.reshape(y_intents, (batch_size, n_intents)).to(device)

            optimizer.zero_grad()

            y_hat_slots, y_hat_intents = capsnet(x_dense, x_sparse)
            y_hat_slots = y_hat_slots.to(device)
            y_hat_slots = rearrange(y_hat_slots, 'b l c -> b c l')

            intent_embeddings = capsnet.get_intent_semantic_embeddings()

            slot_loss = slot_criterion(y_hat_slots, y_slots)
            intent_loss = dot_loss(y_hat_intents, y_intents, intent_embeddings, device)

            batch_loss = slot_loss + intent_loss

            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()
            print(f'[Epoch: {epoch + 1}, mini-batch: {i + 1}] Loss = {epoch_loss}')

    print('Finished training')


if __name__ == '__main__':
    train()
