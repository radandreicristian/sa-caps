from typing import Union

import torch


def dot_loss(y_predicted: torch.Tensor,
             y_true: torch.Tensor,
             ys_other: torch.Tensor) -> Union[float, torch.float, torch.Tensor]:
    """

    :param y_predicted: Per-batch predicted embeddings, shape (b, d_embedding)
    :param y_true: Per-batch true embeddings, shape (b, d_embedding)
    :param ys_other: (per-batch?) embeddings, shape (b, n_embeddings-1, d_embeddings)
    :return: the dot product loss, for the batch.
    """
    # si_pluses [b, 1]
    si_pluses = torch.einsum('bi, bj -> b', y_predicted, y_true)

    # si_minuses [b, n-1, 1]
    si_minuses = torch.einsum('bni, bj -> bn', ys_other, y_predicted)

    # inner_sum [b, 1]
    inner_sum = si_pluses - torch.log(torch.exp(si_pluses) + torch.sum(torch.exp(si_minuses), dim=1, keepdim=True))

    # loss [1]
    loss = -torch.mean(inner_sum, dim=0)
    return loss
