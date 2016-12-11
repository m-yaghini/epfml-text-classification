import re
import numpy as np
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def clean_files():
    positive_examples = list(open('../data/train_pos.txt', "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open('../data/train_neg.txt', "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    test_examples = list(open('../data/test_data.txt', "r").readlines())
    test_examples = [s.strip() for s in test_examples]
    # Split by words
    positive_text = [clean_str(sent) for sent in positive_examples]
    negative_text = [clean_str(sent) for sent in negative_examples]
    test_text = [clean_str(sent) for sent in test_examples]

    print("texts loaded and cleaned")

    with open('../data/train_pos_clean.txt', 'w') as f:
        for sent in positive_text:
            f.write(sent + '\n')

    with open('../data/train_neg_clean.txt', 'w') as f:
        for sent in negative_text:
            f.write(sent + '\n')

    with open('../data/test_data_clean.txt', 'w') as f:
        for sent in test_text:
            f.write(sent + '\n')