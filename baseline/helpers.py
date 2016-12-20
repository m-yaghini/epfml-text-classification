import re
import numpy as np
from nltk.corpus import stopwords

def clean_str_regex(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    Modified for the project, may be improved
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"^\d+", "", string)
    string = re.sub(r"\'m", "", string)
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", " not ", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"url", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\d*", "", string)
    return string.strip().lower()

def remove_stop_words(string, stop_words_list):
    words = [str(w.lower()) for w in string.split() if w not in stop_words_list]
    return " ".join(words)

def clean_str(string, stop_words_list):
    """ Perform both cleaning operations, evaluating, cleaning regex and removing stop words"""
    string = clean_str_regex(string)
    string = remove_stop_words(string, stop_words_list)
    return string


def load_data_and_labels(positive_data_file, negative_data_file, test_data_file):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels for the training sets and split sentences for the testing set
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    test_examples = list(open(test_data_file, "r").readlines())
    test_examples = [s.strip() for s in test_examples]
    # Split by words
    stop_words_list = set(stopwords.words('english'))
    stop_words_list.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',
                            '@', '<', '>', '-', '``', '--', '—', '&', '%', '*', '•', '#', "''",
                            'user', 'url', 'u'])
    train = positive_examples + negative_examples
    train = [clean_str(sent, stop_words_list) for sent in train]
    test = [clean_str(sent, stop_words_list) for sent in test_examples]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    labels = np.concatenate([positive_labels, negative_labels], 0)
    return [train, labels, test]
