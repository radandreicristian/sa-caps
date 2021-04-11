def pad_sequence(sequence: str, max_len: int = 35, pad_value: str = '<pad>') -> str:
    """
    Pads a sequence of strings with a string pad value, equally to the left/right.
    :param sequence: The input string, consisting of a sentence.
    :param max_len: The maximum sequence length.
    :param pad_value: The value to be padded with.
    :return: Padded sequence of length max_len
    """

    seq_split = sequence.split(" ")
    diff = max_len - len(seq_split)
    assert diff > 0, f"fasfasd ({len(seq_split)}) of sequence '{sequence}' is greater than the max_length {max_len}"
    if max_len % 2 == 0:
        left_pad_len = right_pad_len = diff // 2
    else:
        left_pad_len = diff // 2
        right_pad_len = diff // 2 + 1

    left_padding = list([pad_value for _ in range(left_pad_len)])
    right_padding = list([pad_value for _ in range(right_pad_len)])

    assert type(seq_split) == list

    padded_sequence = left_padding + seq_split + right_padding
    return " ".join(padded_sequence)


if __name__ == '__main__':
    p = pad_sequence('set an alarm for two hours from now', 35, 'O')
    print(p)
