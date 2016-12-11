import numpy as np
import pickle
from scipy.sparse import *
from cleaners import clean_str

def pickle_vocab(dir):
    vocab = dict()
    path = '../outputs/' + dir
    with open(path + '/vocab_cut.txt') as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open(path + '/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

def cooc_train():
    with open('../outputs/train/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    data, row, col = [], [], []
    for fn in ['../data/train_pos_clean.txt', '../data/train_neg_clean.txt']:
        with open(fn) as f:
            for line in f:
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                tokens = [t for t in tokens if t >= 0]
                for t in tokens:
                    for t2 in tokens:
                        data.append(1)
                        row.append(t)
                        col.append(t2)

    cooc = coo_matrix((data, (row, col)))
    cooc.sum_duplicates()
    with open('../outputs/train/cooc.pkl', 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)

def cooc_test():
    with open('../outputs/test/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    data, row, col = [], [], []
    for fn in ['../data/test_data_clean.txt']:
        with open(fn) as f:
            for line in f:
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                tokens = [t for t in tokens if t >= 0]
                for t in tokens:
                    for t2 in tokens:
                        data.append(1)
                        row.append(t)
                        col.append(t2)
    cooc = coo_matrix((data, (row, col)))
    cooc.sum_duplicates()
    with open('../outputs/test/cooc.pkl', 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)

def glove_embeddings(dir, nmax=100, embedding_dim=20, eta=0.001, alpha=0.75, epochs=10):
    print("loading cooccurrence matrix")
    path = '../outputs/' + dir
    with open(path + '/cooc.pkl', 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x
    np.save(path + '/embeddings', xs)


def load_data_and_labels(positive_data_file, negative_data_file, test_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    test_examples = list(open(test_data_file, "r").readlines())
    test_examples = [s.strip() for s in test_examples]
    # Split by words
    train = positive_examples + negative_examples
    train = [clean_str(sent) for sent in train]
    test = [clean_str(sent) for sent in test_examples]
    # Generate labels
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    labels = np.concatenate([positive_labels, negative_labels], 0)
    return [train, labels, test]


def vectorize_set(tweet_set, vocab_dict, W):
    list_indices_tweet = []
    for tweet in tweet_set:
        words_indices = []
        for word in list(tweet.split()):
            try:
                words_indices.append(vocab_dict[word])
            except:
                words_indices.append(0)
        list_indices_tweet.append(words_indices)
    print(list_indices_tweet[0:10])
    print(W[0])
    print(W[1])
    print(W[1879])
    list_indices_sum = []
    for ls in list_indices_tweet:
        sum_vect = np.zeros(20)
        for index in ls:
            sum_vect += W[index]
        list_indices_tweet.append(sum_vect)
    return list_indices_sum