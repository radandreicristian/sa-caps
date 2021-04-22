import torch
import torch.nn as nn


class SparseDenseTokenFeaturizer(nn.Module):
    """
    A sub-network with 2 hidden layers that transforms the sparse and dense features of a word into a representation
    into the d_model space.
    """

    def __init__(self,
                 d_dense: int,
                 d_sparse: int,
                 p_dropout: float,
                 d_model: int,
                 ):
        super(SparseDenseTokenFeaturizer, self).__init__()

        # A linear mapping from d_sparse to d_model
        self.spare_fc = nn.Linear(in_features=d_sparse, out_features=d_dense)

        # A linear mapping from d_pretrained + d_model (from concatenation) to d_model
        self.concat_fc = nn.Linear(2 * d_dense, d_model)

        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, dense, sparse):
        # pretrained (b, max_seq_len, d_dense)
        # sparse (b, max_seq_len, d_sparse)

        # Take the sparse features through the FC layer
        # sparse (b, max_seq_len, d_dense)
        sparse = self.dropout(self.spare_fc(sparse))

        # features (b, max_seq_len, 2*d_dense)
        features = torch.cat((sparse, dense), dim=-1)

        # return (b, max_seq_len, d_model)
        return self.concat_fc(features)


class SparseTokenFeaturizer(nn.Module):
    """
    A sub-network with 2 hidden layers that transforms the sparse and dense features of a word into a representation
    into the d_model space.
    """

    def __init__(self,
                 d_dense: int,
                 d_sparse: int,
                 p_dropout: float,
                 d_model: int,
                 ):
        super(SparseTokenFeaturizer, self).__init__()

        # A linear mapping from d_sparse to d_model
        self.spare_fc = nn.Linear(in_features=d_sparse, out_features=d_dense)

        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, dense, sparse):
        # sparse (b, max_seq_len, d_sparse)

        # Take the sparse features through the FC layer
        return self.dropout(self.spare_fc(sparse))


class DenseTokenFeaturizer(nn.Module):
    """
    A sub-network with 2 hidden layers that transforms the sparse and dense features of a word into a representation
    into the d_model space.
    """

    def __init__(self,
                 d_dense: int,
                 d_sparse: int,
                 p_dropout: float,
                 d_model: int,
                 ):
        super(DenseTokenFeaturizer, self).__init__()

        # A linear mapping from d_sparse to d_model
        self.spare_fc = nn.Linear(in_features=d_dense, out_features=d_model)

    def forward(self, dense, sparse):
        # dense (b, max_seq_len, d_dense)
        return self.spare_fc(dense)
