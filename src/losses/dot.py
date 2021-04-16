import torch


def dot_loss(y_predicted: torch.Tensor,
             y_true: torch.Tensor,
             embeddings: torch.Tensor,
             device: torch.device) -> float:
    """

    :param device:
    :param y_predicted: Per-batch predicted embeddings, shape (b, d_semantic_space)
    :param y_true: Per-batch one-hot encodings of the correct label (b, n_intents)
    :param embeddings: embeddings, shape (n_intents, d_semantic_space)
    :return: the dot product loss, for the batch.
    """

    # h_pluses (b, d_embedding)
    h_pluses = torch.einsum('bn, nd -> bd', y_true, embeddings).to(device)

    # truth_index (b,)
    truth_indices = torch.argmax(y_true, dim=-1).to(device)

    _, d = y_predicted.shape
    b, ni = y_true.shape

    # h_minuses (b, n_intents-1, d_semantic_space)
    h_minuses = torch.zeros((b, ni - 1, d)).to(device)

    for i in range(b):
        h_minuses[i, :, :] = torch.cat((embeddings[:truth_indices[i], :], embeddings[truth_indices[i] + 1:, :]), dim=0)

    # si_pluses (b,)
    si_pluses = torch.einsum('bi, bj -> b', y_predicted, h_pluses).to(device)

    # si_minuses (b, n_intents-1)
    si_minuses = torch.einsum('bni, bj -> bn', h_minuses, y_predicted).to(device)

    # inner_sum (b,)
    inner_sum = (si_pluses - torch.log(torch.exp(si_pluses) + torch.sum(torch.exp(si_minuses), dim=1))).to(device)

    # loss (1,)
    loss = -torch.mean(inner_sum, dim=0)
    return loss
