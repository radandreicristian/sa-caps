import torch
import torch.nn as nn


class TokenFeaturizer(nn.Module):
    def __init__(self,
                 d_pretrained: int,
                 d_sparse: int,
                 d_model: int = 256):
        super(TokenFeaturizer, self).__init__()

        self.spare_fc = nn.Linear(d_sparse, d_model)

        self.concat_fc = nn.Linear(d_pretrained + d_model, d_model)

    def forward(self, pretrained, sparse):
        # pretrained (b, max_seq_len, d_pretrained)
        # sparse (b, max_seq_len, d_sparse)
        sparse = self.spare_fc(sparse)

        # Take the sparse features through the FC layer
        # sparse (b, max_seq_len, d_model)
        sparse = self.spare_fc(sparse)

        # features (b, max_seq_len, d_pretrained+d_model)
        features = torch.cat((sparse, pretrained), dim=-1)

        # return (b, max_seq_len, d_model)
        return self.concat_fc(features)
