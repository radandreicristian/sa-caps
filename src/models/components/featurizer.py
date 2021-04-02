import torch
import torch.nn as nn
import torch.nn.functional as f


class TokenFeaturizer(nn.Module):
    def __init__(self,
                 pretrained_embeddings: dict,
                 sparse_features: dict,
                 d_model: int = 256,
                 masked_training: bool = False):
        """

        :param pretrained_embeddings: A dictionary containing pre-trained embeddings. Key - word, Value - embedding vector
        :param sparse_features: A dictionary containing sparse features. Key - word, Value - Sparse features vector
        :param d_model:
        """
        super(TokenFeaturizer, self).__init__()

        self.pretrained = pretrained_embeddings
        self.sparse = sparse_features

        self.d_pretrained = len(list(self.pretrained.values())[0])
        self.d_sparse = len(list(self.sparse.values())[0])

        self.d_model = d_model

        self.spare_fc = nn.Linear(self.d_sparse, self.d_model)

        self.concat_fc = nn.Linear(self.d_pretrained + self.d_model, self.d_model)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        # x = (b, s_len)

        sparse_features = torch.Tensor(list(map(lambda e: self.sparse[e], x))) # (b, s_len, d_sparse)
        pretrained_embeddings = torch.Tensor(list(map(lambda e: self.pretrained[e], x)))  # (b, s_len, d_pretrained)
        features = torch.cat((sparse_features, pretrained_embeddings), dim=-1)  # (b, s_len, 2*d_model)
        return self.concat_fc(features)  # (b, s_len, d_model)
