import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from gensim.models.fasttext import load_facebook_vectors
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset

from src.utils.padding import pad_sequence


class NluBenchmarkDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 sparse_dim: int,
                 sparse_embeddings_path: Optional[str] = None,
                 pretrained_embeddings_path: Optional[str] = None,
                 max_seq_len: int = 35):
        """

        :param dataset_path:
        :param sparse_embeddings_path:
        :param pretrained_embeddings_path:
        :param max_seq_len:
        """
        super(NluBenchmarkDataset, self).__init__()

        root_path = Path(__file__).parent.parent.parent.parent
        csv_path = os.path.join(root_path, dataset_path)

        self.sparse_dim = sparse_dim
        self.max_seq_len = max_seq_len

        # Load the dataset
        df = pd.read_csv(csv_path, sep=';')

        # Make sure we have at least one type of embeddings
        if sparse_embeddings_path is None and pretrained_embeddings_path is None:
            raise Exception("There is no embedding path provided. Please specify at least one in the args")

        # Load pretrained embeddings
        # Todo: Treat the case when it's glove and not fasttext embeddings
        if pretrained_embeddings_path is not None:
            self.pretrained_embeddings = load_facebook_vectors(pretrained_embeddings_path).wv

        # Load sparse embeddings
        if sparse_embeddings_path is not None:
            sparse_embeddings_file = open(sparse_embeddings_path, 'r')
            self.sparse_embeddings = json.load(sparse_embeddings_file)

        # Pad tokens sequence with <pad> - Embedding of this token should be all zeros
        df['tokens'] = df.apply(lambda row: pad_sequence(row['tokens'], max_len=35, pad_value='<pad>'), axis=1)

        # Pad slots sequence with O
        df['slot_tags'] = df.apply(lambda row: pad_sequence(row['slot_tags'], max_len=35, pad_value='O'), axis=1)

        # Independent variable - Sentences containing utterances - (n_samples, max_seq_len)
        self.x = df.iloc[1:, 1]

        # Todo - Slot tags needs to be a tensor
        # Get the slots column and count the different slots
        slot_tags = df.iloc[1:, 2].values
        n_slots = df.iloc[1:, 2].count() + 1

        # Get the number of dims
        scatter_dim = len(slot_tags.size())
        y_tensor = slot_tags.view(*slot_tags.size(), -1)
        zeros = torch.zeros(*slot_tags.size(), n_slots, dtype=slot_tags.dtype)

        # self.y_slots (n_samples, max_seq_len, n_slots)
        self.y_slots = zeros.scatter(scatter_dim, y_tensor, 1)

        # Target variable - (n_samples, n_slots
        self.y_intents = OneHotEncoder().fit_transform(df.iloc[1:, 3]).toarray()

    def __getitem__(self, index):
        sentence = self.x[index]

        # pretrained (max_seq_len, d_pretrained)
        pretrained = np.array([self.pretrained_embeddings[i] for i in sentence])

        # sparse (max_seq_len, d_sparse)
        sparse_embeddings = np.zeros((self.max_seq_len, self.sparse_dim))
        for w_index, word in enumerate(sentence):
            sparse_indices = self.sparse_embeddings[word]
            for s_index in sparse_indices:
                sparse_embeddings[w_index][s_index] = 1

    def __len__(self):
        return len(self.x)
