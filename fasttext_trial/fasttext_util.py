from cleaners import clean_data
import fasttext
import os
import numpy as np


def extract_word_vectors(vector_filename):
    # Extracts word vectors from fasttext 'model.vec', outputs two dictionaries searchable by words, vector
    # dictionary and index dictionary, respectively.
    with open(vector_filename, 'r') as vec_file:
        skipped = False
        vec_dict = {}
        index_dict = {}
        index = 0
        for line in vec_file:
            if not skipped:  # skipping the first line
                skipped = True
                continue
            word, vec = line.split(sep=' ', maxsplit=1)
            # vec = np.array([vec.split])
            vec_dict.__setitem__(word, vec)
            index_dict.__setitem__(word, index)
            index += 1
    return vec_dict, index_dict


def text2vec_fast(data_file, method='cbow', modelname='model'):
    clean_data(data_file, 'cleaned.txt')
    if method == 'skipgram':
        model = fasttext.skipgram('cleaned.txt', modelname)
    else:
        model = fasttext.cbow('cleaned.txt', modelname)

    vector_dict, index_dict = extract_word_vectors(modelname + '.vec')
    os.remove('cleaned.txt')
    return vector_dict, index_dict
