import json
import os
from pathlib import Path
from typing import Optional, Tuple

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
                 max_ngram_size: int,
                 n_chars: int,
                 max_seq_len: int,
                 dense_features_path: Optional[str] = None,
                 sparse_features_path: Optional[str] = None
                 ):
        """

        :param dataset_path:
        :param sparse_features_path:
        :param dense_features_path:
        :param max_seq_len:
        """
        super(NluBenchmarkDataset, self).__init__()

        root_path = Path(__file__).parent.parent.parent.parent
        csv_path = os.path.join(root_path, dataset_path)

        self.d_sparse = sum([pow(n_chars, ngram_size) for ngram_size in range(1, max_ngram_size + 1)])
        self.max_seq_len = max_seq_len

        self.use_sparse_embeddings = True if sparse_features_path is not None else False
        self.use_dense_features = True if dense_features_path is not None else False

        # Load the dataset
        df = pd.read_csv(csv_path, sep=';')

        # Make sure we have at least one type of embeddings
        if not self.use_dense_features and not self.use_sparse_embeddings:
            raise Exception("There is no embedding path provided. Please specify at least one in the args")

        # Load pretrained embeddings
        # Todo: Treat the case when it's glove and not fasttext embeddings. Meh
        if self.use_dense_features:
            pretrained_embeddings = load_facebook_vectors(dense_features_path)
            self.pretrained_embeddings = pretrained_embeddings.wv

        # Load sparse embeddings
        if self.use_sparse_embeddings:
            sparse_embeddings_file = open(sparse_features_path, 'r')
            self.sparse_embeddings = json.load(sparse_embeddings_file)

        # Pad tokens sequence with <pad> - Embedding of this token should be all zeros
        df['tokens'] = df.apply(lambda row: pad_sequence(row['tokens'], max_len=35, pad_value='<pad>'), axis=1)

        # Each element is a str. No point OHE-ing, as the vocab size is huge.
        # Just leave them as strings and look them up in the pretrained/sparse dictionaries when loading in __getitem__
        # self.x (n_samples, max_seq_len)
        self.x = np.stack(np.array([np.array(el.split(), dtype=object) for el in df['tokens']]))

        # Pad slots sequence with O
        df['slot_tags'] = df.apply(lambda row: pad_sequence(row['slot_tags'], max_len=35, pad_value='O'), axis=1)

        # slot_tags (n_samples, max_seq_len). Each element needs to be one hot encoded
        slot_tags = np.array([np.array(el.split(), dtype=object) for el in df['slot_tags']])
        slot_tags = np.stack(slot_tags)

        # shape (n_samples, max_seq_len)
        initial_shape = slot_tags.shape

        # Flatten and one-hot encode the flattened array
        slot_tags_flattened = slot_tags.flatten()
        unique_slot_tags, inverse = np.unique(slot_tags_flattened, return_inverse=True)

        self.n_slots = len(unique_slot_tags)
        onehot_encodings = np.eye(unique_slot_tags.shape[0])[inverse]

        # self.slot_tags (n_samples, max_seq_len, n_slots) - Each element is a onehot encoding over the number of slots.
        # The reshape's last dim is -1, since we don't know how many slots there are in total.
        # Todo - Change from one-hot to label encoding
        self.slot_tags = torch.argmax(torch.Tensor(onehot_encodings.reshape(initial_shape + (-1,))), dim=-1,
                                      keepdim=False)

        # self.intents (n_samples, n_intents) - Each element is a onehot encoding over the number of intents.
        intents = np.array(df['intent']).reshape(-1, 1)
        unique_intents = np.unique(intents)
        self.n_intents = len(unique_intents)

        self.intents = torch.Tensor(OneHotEncoder().fit_transform(intents).toarray())

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the variables corresponding to the index.
        :param index: An integer value, in (0, len(self))
        :return: A tuple containing the following 4 torch tensors:
        - dense (max_seq_len, d_pretrained) - Pretrained embeddings for each token in the sentence.
        - sparse (max_seq_len, d_sparse) - Sparse embeddings for each token in the senetnece.
        - slot_tags (max_seq_len, n_slots) - One hot encoding of slot tags
        - intent (max_seq_len, n_intents) - One hot encoding of intents
        """
        sentence = self.x[index]

        # dense (max_seq_len, d_pretrained)
        dense = torch.Tensor(np.array([self.pretrained_embeddings[i] for i in sentence]))

        # sparse (max_seq_len, d_sparse)
        indices = [[], []]
        n_values = 0
        shape = (self.max_seq_len, self.d_sparse)

        if self.use_sparse_embeddings:
            for w_index, word in enumerate(sentence):
                sparse_indices = self.sparse_embeddings.get(word, [])
                for s_index in sparse_indices:
                    indices[0].append(w_index)
                    indices[1].append(s_index)
                    n_values += 1
            values = [1 for _ in range(n_values)]
        else:
            values = []
        sparse = torch.sparse_coo_tensor(torch.Tensor(indices), values, shape)
        # slot_tags (max_seq_len, n_slots) - Per-word one-hot
        slot_tags = self.slot_tags[index]

        # intent (n_intents) - One-hot
        intent = self.intents[index]
        return dense, sparse, slot_tags, intent

    def __len__(self):
        return len(self.x)

    def get_counts(self):
        return self.n_slots, self.n_intents
