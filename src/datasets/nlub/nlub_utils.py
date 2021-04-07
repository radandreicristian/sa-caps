"""
NLU Benchmark Dataset & Preprocessing
Benchmarking Natural Language Understanding Services for building Conversational Agents - X. Liu et. al., 2019
Under the Creative Commons 4.0 License - https://creativecommons.org/licenses/by/4.0/legalcode
"""
import os
from pathlib import Path

import pandas as pd


def get_slot_tags_from_annotated(annotated: str):
    """
    Returns a sequence of tokens corresponding to the slot tags for the given annotated sequence
    :param annotated: Sequence where slot tags are tagged -
        WORD_1, ..., WORD_N-1, [SLOT_TYPE_J : WORD_N ... WORD_N+K], WORD_N+K+1..., WORD_L.
    :return: The corresponding tagged sequence
        CTX, ..., CTX, SLOT_TYPE_J, ... SLOT_TYPE_J, CTX, ... CTX.
    """
    tokens = annotated.split(sep=' ')
    slot_tags = []
    context_tag = 'O'
    current_tag = context_tag
    for token in tokens:
        if '[' in token:
            current_tag = token.replace('[', '')
            continue
        if ':' in token:
            continue
        if ']' in token:
            slot_tags.append(current_tag)
            current_tag = context_tag
            continue
        slot_tags.append(current_tag)
    slots = ' '.join(slot_tags)
    return slots


def get_words_from_annotated(annotated: str):
    """
    Returns a sequence of tokens corresponding to the words for the given annotated sequence
    :param annotated: Sequence where slot tags are tagged -
          WORD_1, ..., WORD_N-1, [SLOT_TYPE_J : WORD_N ... WORD_N+K], WORD_N+K+1..., WORD_L.
    :return: The corresponding words
         WORD_1, ..., WORD_L.
       """
    tokens = annotated.split(sep=' ')
    words = []
    for token in tokens:
        if '[' in token:
            continue
        if ':' in token:
            continue
        if ']' in token:
            words.append(token.replace(']', ''))
            continue
        words.append(token)
    return ' '.join(words)


def preprocess_nlu_benchmark_dataset(file_name: str = 'nlub-raw.csv') -> pd.DataFrame:
    """
    Preprocesses the raw dataframe. The resulting dataframe contains only 3 columns - [tokens, slot_tags, intent]
    :param file_name: The name of the file (relative to root/data/nlu-data)
    :return: The dataframe containing only 3 columns.
    """
    # Build the path to the dataset - projectroot/data/nlu-data/nlu-evaluation-raw.csv
    root_path = Path(__file__).parent.parent.parent.parent
    csv_path = os.path.join(root_path, 'data', 'nlu-data', file_name)

    # Read the CSV into a pandas dataframe
    df = pd.read_csv(csv_path, sep=';')

    # Concat the scenario and intent into a new column
    df['intent'] = df['scenario'] + '_' + df['intent']

    # Create new columns for tokens and slots, based on the annotated answer
    df['tokens'] = df.apply(lambda row: get_words_from_annotated(row['answer_annotation']), axis=1)
    df['slot_tags'] = df.apply(lambda row: get_slot_tags_from_annotated(row['answer_annotation']), axis=1)

    # Drop the unnecessary columns
    df.drop(['userid', 'answerid', 'status', 'notes', 'answer', 'question',
             'answer_normalised', 'suggested_entities', 'answer_annotation', 'scenario'],
            axis=1, inplace=True)

    # Drop sentences with unknown columns
    df = df[~df['tokens'].str.contains("<unk>")]

    # Drop sentences with more than 35 words
    df = df[df['tokens'].map(lambda x: x.split()).map(len) < 35]

    cols = df.columns.tolist()
    cols = [cols[1], cols[2], cols[0]]
    df = df[cols]

    return df


def preprocess_and_save(
        input_file_name: str = 'nlub-raw.csv',
        output_file_name: str = 'nlub.csv') -> None:
    # Build the path to the dataset - projectroot/data/nlu-data/nlu-evaluation-raw.csv
    root_path = Path(__file__).parent.parent.parent.parent
    out_csv_path = os.path.join(root_path, 'data', 'nlu-data', output_file_name)
    df = preprocess_nlu_benchmark_dataset(input_file_name)
    df.to_csv(out_csv_path, sep=';')


if __name__ == '__main__':
    preprocess_and_save()
