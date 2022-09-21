import os

from tqdm import tqdm

import pandas as pd


def store_data(path, lines):
    with open(path, 'w') as f:
        for line in tqdm(lines):
            f.write(line.strip() + '\n')


def main():

    dst = '/home/fernandol/transformer-translator-pytorch/partition/'

    tsv_path = '/home/fernandol/transformer-translator-pytorch/data/hypothesis.tsv'
    df = pd.read_csv(tsv_path, header=0, sep='\t')

    # Shuffle data
    shuffled_df = df.sample(frac=1, random_state=1).reset_index()

    src_lines = shuffled_df['Hypothesis'].tolist()
    trg_lines = shuffled_df['Reference'].tolist()

    store_data(os.path.join(dst, 'src', 'raw_data.src'), src_lines)
    store_data(os.path.join(dst, 'trg', 'raw_data.trg'), trg_lines)

    print("Splitting data...")
    train_frac = 0.8
    first_split_idx = int(train_frac * len(src_lines))
    second_split_idx = first_split_idx + int(((1-train_frac)/2) * len(src_lines))
    print(first_split_idx)
    print(second_split_idx)

    # source data
    src_train_lines = src_lines[:first_split_idx]
    src_valid_lines = src_lines[first_split_idx:second_split_idx]
    src_test_lines = src_lines[second_split_idx:]

    # target
    trg_train_lines = trg_lines[:first_split_idx]
    trg_valid_lines = trg_lines[first_split_idx:second_split_idx]
    trg_test_lines = trg_lines[second_split_idx:]

    # Sore data
    store_data(os.path.join(dst, 'src', 'train.txt'), src_train_lines)
    store_data(os.path.join(dst, 'src', 'valid.txt'), src_valid_lines)
    store_data(os.path.join(dst, 'src', 'test.txt'), src_test_lines)

    store_data(os.path.join(dst, 'trg', 'train.txt'), trg_train_lines)
    store_data(os.path.join(dst, 'trg', 'valid.txt'), trg_valid_lines)
    store_data(os.path.join(dst, 'trg', 'test.txt'), trg_test_lines)





if __name__ == '__main__':
    main()