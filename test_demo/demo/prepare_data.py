import numpy as np
import pickle
from helpers import load_data_and_labels, vectorize_set

def prepare_data():
    path = '../outputs/'
    path_data = '../data/'

    with open(path + 'train/vocab.pkl', 'rb') as f:
        vocab_train_dict = pickle.load(f)

    with open(path + '/test/vocab.pkl', 'rb') as f:
        vocab_test_dict = pickle.load(f)

    W = np.load(path + 'train/embeddings.npy')
    W_test = np.load(path + 'test/embeddings.npy')

    train, labels, test = load_data_and_labels(path_data + 'train_pos_clean.txt', path_data + 'train_neg_clean.txt', path_data + 'test_data_clean.txt')

    print("Vectorization of the tweet sets")
    # To be improved (does not work when passed as a function ???)
    ls = []
    for sent in train:
        ls_temp = []
        for word in list(sent.split()):
            try:
                ls_temp.append(vocab_train_dict[word])
            except:
                ls_temp.append(0)
        ls.append(ls_temp)

    ls_sum = []
    for ls_in in ls:
        sum_vect = 0
        for index in ls_in:
            sum_vect += W[index]
        ls_sum.append(sum_vect)
    X_train = ls_sum

    ls = []
    for sent in test:
        ls_temp = []
        for word in list(sent.split()):
            try:
                ls_temp.append(vocab_test_dict[word])
            except:
                ls_temp.append(0)
        ls.append(ls_temp)

    ls_sum = []
    for ls_in in ls:
        sum_vect = 0
        for index in ls_in:
            sum_vect += W_test[index]
        ls_sum.append(sum_vect)
    X_test = ls_sum
    print("Sets vectorized")

    return X_train, labels, X_test