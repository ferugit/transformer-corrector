import os
import re
import json
import random
import pandas as pd
from tqdm import tqdm

from reporter import Reporter

# Set seed
random.seed(1234)

characters = ['E', 'A', 'O', 'S', 'N', 'R', 'I', 'L', 'D', 'T', 'C', 'U', 'M',
             'P', 'B', 'G', 'V', 'F', 'Ó', 'Y', 'H', 'Í', 'Á', 'J', 'Q', 'Z',
              'Ñ', 'X', 'Ú', 'K', 'W', 'É', 'Ü' ,' ']


def unicode_normalisation(text):

    try:
        text = unicode(text, "utf-8")
    except NameError:  # unicode is a default on python 3
        pass
    return str(text)


def normalize_text(words):
    words = unicode_normalisation(words)

    # !! Language specific cleaning !!
    # Important: feel free to specify the text normalization
    # corresponding to your alphabet.

    words = re.sub(
        "[^’'A-Za-z0-9À-ÖØ-öø-ÿЀ-ӿéæœâçèàûî]+", " ", words
    ).upper()
    
    # Spanish specific cleaning
    words = words.replace("\"", " ")
    words = words.replace("'", " ")
    words = words.replace("’", " ")

    # catalan = "ÏÀÒ"
    words = words.replace("Ï", "I")
    words = words.replace("À", "A")
    words = words.replace("Ò", "O")
    
    # Remove multiple spaces
    words = re.sub(" +", " ", words)

    # Remove spaces at the beginning and the end of the sentence
    words = words.lstrip().rstrip()
    return words


def store_data(path, lines):
    with open(path, 'w') as f:
        for line in tqdm(lines):
            f.write(line.strip() + '\n')


def repeated_char(word):
    pos = random.randint(0, len(word)-1)
    rep_times = 1 #random.randint(1,9) # use only one repetition in word
    out_str = word[:pos]

    for i in range(0,rep_times):
        out_str += word[pos]
        out_str += word[pos:]
    return out_str

# Inserting a char in a random position
def insertion_mistake(word):
    index = random.randint(0, len(word)-1)
    return word[:index] + get_random_char() + word[index:]


# Deleting a random char
def deletion_mistake(word):
    if (len(word) < 2):
        return word
    index = random.randint(0, len(word)-1)
    return word[:index] + word[index+1:]


# Substituting a random char with another one
def substitution_mistake(word):
    index = random.randint(0, len(word)-1)
    return word[:index] + get_random_char() + word[index+1:]


# Swaps two characters
def swap_mistake(word):
    if len(word) < 2:
        return word

    index = random.randint(0, len(word)-2)
    aux = word[:]
    word = word[:index] + word[index+1] + word[index]
    if index < len(aux)-2:
        word += aux[index+2:]
    return word


# Inserting a random typo mistake among the types previously defined
def typo_mistake(word, im=1/6, dm=1/6, sm=3/6, swm=1/6):
    p = random.random() * (im + dm + sm + swm)

    if p < sm:
        return insertion_mistake(word)
    if p < sm + dm:
        return deletion_mistake(word)
    if p < sm + dm + im:
        return substitution_mistake(word)
    return swap_mistake(word)


# Returns a sentence with one random misspelling among those that are possible
def spelling_mistake(word):

    # Defining all the misspellings. If there is more than one possible misspelling for a given correct spelling, it
    # must be introduced as a list
    misspelling_dict = {
        'U': 'W',
        'W': 'U',
        #'oo': 'u',
        'Y': 'I',
        'I': 'Y',
        'B': 'V',
        'V': 'B',
        'CA': 'KA',
        'KA': 'CA',
        'CE': 'SE',
        'SE': 'CE',
        'CI': 'SI',
        'SI': 'CI',
        'CO': 'KO',
        'KO': 'CU',
        'CU': 'KU',
        'KU': 'CU',
        #'ee': 'i',
        'LL': 'Y',
        ' A': ' HA',
        ' HA': ' A',
        ' O': ' HO',
        ' HO': ' O',
        ' I': ' HI', 
        ' HI': ' I',
        'GE': 'JE',
        'JE': 'GE',
        'GI': 'JI',
        'JI': 'GI',
        'CH': 'X',
        'X':'CH',
        ' I': ' HI',
        ' HI': ' I',
        'QUI': 'KI',
        'KI': 'QUI',
        'QUE': ['KE', 'K', 'Q'],
        'KE': 'QUE'
    }

    possible_missp = []
    
    for k in misspelling_dict.keys():
        
        if k in word:
            possible_missp.append(k)

    # No possible misspellings
    if not possible_missp:
        return word

    # Selecting a random misspelling; there might be different possible misspellings from a single source correct
    #  spelling
    missp = random.choice(possible_missp)
    missp_err = misspelling_dict[missp]
    
    if isinstance(missp_err, list):
        missp_err = random.choice(missp_err)

    # Choose one random appearance of the correct spelling to produce the misspelling
    missp_positions = []
    for i in range(len(word) - len(missp) + 1):
        if word[i:i+len(missp)] == missp:
            missp_positions.append(i)

    index = random.choice(missp_positions)
    return word[:index] + missp_err + word[index+len(missp):]

def get_random_char():
    return random.choice(characters)


def create_mistakes(clean_list):
    mistaken_list = []
    char_typo_prob = 1/5
    char_missp_prob = 1/9
    char_rep_prob = 1/125

    word_deletion_prob = 89348/519779 # from errors

    for idx, x in tqdm(enumerate(clean_list)):
        # Split in words
        original = x.strip()

        # deletion: remove word prob
        if (len(original.split(' ')) > 2) and (random.random() < word_deletion_prob):
            words = original.split(' ')
            remove_idx = random.randint(0, len(words) - 1)
            words = words[:remove_idx] + words[remove_idx+1:]
            mistaken = " ".join(words)

            if mistaken.strip() != '' and mistaken:
                mistaken_list.append(mistaken.upper())
        
        # char mistakes
        elif len(original) > 2:
            mistaken = ''

            # Split in words 
            for w in original.split(' '):
                
                if len(w) > 3:
                    
                    if random.random() < char_typo_prob:
                        w = w.strip()
                        w = typo_mistake(w)
                    
                    elif random.random() < char_missp_prob:
                        w = spelling_mistake(' ' + w + ' ')
                        w = w.strip()
                    
                    elif random.random() < char_rep_prob:
                        w = repeated_char(w)
                        w = w.strip()

                    mistaken += w + ' '
                
                else:
                    mistaken += w + ' '

            mistaken = mistaken[:-1]
            
            if mistaken.strip() != '':
                mistaken_list.append(mistaken)

        else:
            mistaken_list.append(original)

    if len(clean_list) != len(mistaken_list):
        print(len(clean_list))
        print(len(mistaken_list))
        raise Exception("Something strange happend while creating mistakes.")

    return mistaken_list 


def main():

    dst = 'data/'

    ckpt_0 = True
    ckpt_75 = True
    ckpt_95 = True
    n_augments = 1

    # Reporter
    reporter = Reporter(dst + "dataset.json")

    tsv_path = 'data/source/hypothesis.tsv'
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

    # source data
    src_train_lines = src_lines[:first_split_idx]
    src_valid_lines = src_lines[first_split_idx:second_split_idx]
    src_test_lines = src_lines[second_split_idx:]

    # target
    trg_train_lines = trg_lines[:first_split_idx]
    trg_valid_lines = trg_lines[first_split_idx:second_split_idx]
    trg_test_lines = trg_lines[second_split_idx:]

    ############################################
    # REAL DATA: Append data from checkpoints###
    ############################################

    if ckpt_0:
        print("Reading data generated with CKPT+2022-05-04+10-22-26+00...")
        tsv_path_ckpt0 = 'data/source/CKPT+2022-05-04+10-22-26+00_hypothesis.tsv'
        ckpt0_df = pd.read_csv(tsv_path_ckpt0, header=0, sep='\t')
        ckpt0_df = ckpt0_df.dropna() # drop nans
        ckpt0_df = ckpt0_df[~ckpt0_df["Reference"].isin(trg_lines)]
        ckpt0_df['Reference'] = ckpt0_df['Reference'].apply(lambda x: normalize_text(x))
        ckpt0_df['Hypothesis'] = ckpt0_df['Hypothesis'].apply(lambda x: normalize_text(x))
        src_train_lines += ckpt0_df['Hypothesis'].tolist()
        trg_train_lines += ckpt0_df['Reference'].tolist()

    if ckpt_75:
        print("Reading data generated with CKPT+2022-09-11+20-25-21+00...")
        tsv_path_ckpt4 = 'data/source/CKPT+2022-09-11+20-25-21+00_hypothesis.tsv'
        ckpt4_df = pd.read_csv(tsv_path_ckpt4, header=0, sep='\t')
        ckpt4_df = ckpt4_df.dropna() # drop nans
        ckpt4_df = ckpt4_df[~ckpt4_df["Reference"].isin(trg_lines)]
        ckpt4_df['Reference'] = ckpt4_df['Reference'].apply(lambda x: normalize_text(x))
        ckpt4_df['Hypothesis'] = ckpt4_df['Hypothesis'].apply(lambda x: normalize_text(x))
        src_train_lines += ckpt4_df['Hypothesis'].tolist()
        trg_train_lines += ckpt4_df['Reference'].tolist()

    if ckpt_95:
        print("Reading data generated with CKPT+2022-09-23+21-10-47+00...")
        tsv_path_ckpt5 = 'data/source/CKPT+2022-09-23+21-10-47+00_hypothesis.tsv'
        ckpt5_df = pd.read_csv(tsv_path_ckpt5, header=0, sep='\t')
        ckpt5_df = ckpt5_df.dropna() # drop nans
        ckpt5_df = ckpt5_df[~ckpt5_df["Reference"].isin(trg_lines)]
        ckpt5_df['Reference'] = ckpt5_df['Reference'].apply(lambda x: normalize_text(x))
        ckpt5_df['Hypothesis'] = ckpt5_df['Hypothesis'].apply(lambda x: normalize_text(x))
        src_train_lines += ckpt5_df['Hypothesis'].tolist()
        trg_train_lines += ckpt5_df['Reference'].tolist()

    ############################################
    ### Augmented data: Apped generated data ###
    ############################################

    # Read the rest of the aligned data
    aligned_path = 'data/source/albayzin_aligned.tsv'
    aligned_df = pd.read_csv(aligned_path, header=0, sep='\t')
    aligned_df = aligned_df.dropna() # drop nans

    # remove dev2 data: which contains real mistakes
    aligned_df = aligned_df[~aligned_df['Transcription'].isin(trg_lines)]
    aligned_df = aligned_df[~aligned_df['Transcription'].isin(["NAN"])] # remove nans
    aligned_df['Transcription'] = aligned_df['Transcription'].apply(lambda x: normalize_text(x))
    clean_list = list(filter(None, aligned_df['Transcription'].tolist()))

    # Augment with random mistakes
    for _ in range(n_augments):
        print("Generate augmented data...")
        # From ASR tokenizer
        mistaken_list = create_mistakes(clean_list)

        # Append augmented data
        src_train_lines += mistaken_list
        trg_train_lines += clean_list

    # used data
    used_data = {
        "ckpt_0": ckpt_0,
        "ckpt_75": ckpt_75,
        "ckpt_95": ckpt_95,
        "augmented": True if n_augments > 0 else False,
        }
    value_counts = {
        "n_sentences": {
            "train": len(src_train_lines),
            "dev": len(src_valid_lines),
            "test": len(src_test_lines),
        }
    }

    reporter.report("used_data", used_data)
    reporter.report("data_quantity", value_counts)

    ######################################
    ######## Post-process and save #######
    ######################################

    # lower case source
    src_train_lines = [each_string.lower() for each_string in src_train_lines]
    src_valid_lines = [each_string.lower() for each_string in src_valid_lines]
    src_test_lines = [each_string.lower() for each_string in src_test_lines]

    # Lowecase target
    trg_train_lines = [each_string.lower() for each_string in trg_train_lines]
    trg_valid_lines = [each_string.lower() for each_string in trg_valid_lines]
    trg_test_lines = [each_string.lower() for each_string in trg_test_lines]

    # Shuffle train lists
    temp = list(zip(src_train_lines, trg_train_lines))
    random.shuffle(temp)
    src_train_lines, trg_train_lines = zip(*temp)
    src_train_lines, trg_train_lines = list(src_train_lines), list(trg_train_lines)

    if '' in src_train_lines or '' in trg_train_lines:
        print("empty string found")
        return

    # Sore data as txt
    store_data(os.path.join(dst, 'src', 'train.txt'), src_train_lines)
    store_data(os.path.join(dst, 'src', 'valid.txt'), src_valid_lines)
    store_data(os.path.join(dst, 'src', 'test.txt'), src_test_lines)

    store_data(os.path.join(dst, 'trg', 'train.txt'), trg_train_lines)
    store_data(os.path.join(dst, 'trg', 'valid.txt'), trg_valid_lines)
    store_data(os.path.join(dst, 'trg', 'test.txt'), trg_test_lines)

    # Store as csvs
    train_df = pd.DataFrame({'src': src_train_lines , 'trg': trg_train_lines})
    train_df.to_csv(os.path.join(dst, 'csv', 'train.csv'), index=None)
    dev_df = pd.DataFrame({'src': src_valid_lines , 'trg': trg_valid_lines})
    dev_df.to_csv(os.path.join(dst, 'csv', 'valid.csv'), index=None)
    test_df = pd.DataFrame({'src': src_test_lines , 'trg': trg_test_lines})
    test_df.to_csv(os.path.join(dst, 'csv', 'test.csv'), index=None)

if __name__ == '__main__':
    main()