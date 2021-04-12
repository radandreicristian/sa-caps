def pad_sequence(sequence: str, max_len: int = 35, pad_value: str = '<pad>') -> str:
    """
    Pads a sequence of strings with a string pad value, equally to the left/right.
    :param sequence: The input string, consisting of a sentence.
    :param max_len: The maximum sequence length.
    :param pad_value: The value to be padded with.
    :return: Padded sequence of length max_len
    """

    seq_split = sequence.split(" ")
    seq_len = len(seq_split)
    diff = max_len - seq_len

    assert diff > 0, \
        f"Length of sequence '{sequence}' is greater than the max_length - {len(seq_split)} > {max_len}."

    if diff % 2 == 0:
        left_pad_len = diff // 2
        right_pad_len = diff // 2
    else:
        left_pad_len = diff // 2
        right_pad_len = diff // 2 + 1

    assert left_pad_len + right_pad_len + seq_len == max_len, \
        f'Sum of {left_pad_len}, {right_pad_len}, and {diff} != {max_len}'

    left_padding = list([pad_value for _ in range(left_pad_len)])
    right_padding = list([pad_value for _ in range(right_pad_len)])

    assert type(seq_split) == list

    padded_sequence = left_padding + seq_split + right_padding

    assert len(padded_sequence) == max_len, \
        f'Final length ({len(padded_sequence)}) of padded sequence {padded_sequence} is different from {max_len}.'

    print(f'Seq len {len(padded_sequence)}')

    return " ".join(padded_sequence)


if __name__ == '__main__':
    p = pad_sequence('wake me up at nine am on friday', 35, '<pad>')
    print(len(p.split()))
