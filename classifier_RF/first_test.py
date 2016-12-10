import numpy as np
import pickle
from helpers import load_data_and_labels

path = 'data/mini/'

with open(path + 'vocab.pkl', 'rb') as f:
    vocab_dict = pickle.load(f)

W = np.load(path + 'embeddings.npy')

x_text, y = load_data_and_labels(positive_data_file=path + 'pos_train.txt', negative_data_file=path + 'neg_train.txt')

ls = []
for sent in x_text:
    ls_temp = []
    for word in list(sent.split()):
        try:
            ls_temp.append(vocab_dict[word])
        except:
            ls_temp.append(0)
    ls.append(ls_temp)
print(ls)

ls_sum = []
for ls_in in ls:
    sum_vect = 0
    for index in ls_in:
        sum_vect += W[index]
    ls_sum.append(sum_vect)
print(ls_sum)

def test_sentence(sentence):
    words = list(sentence.split())
    ls = []
    for w in words:
        ls.append(vocab_dict[w])
    sum_vect = 0
    for index in ls:
        sum_vect += W[index]
    return sum_vect

test_sent = test_sentence("this is were i belong")
test_sent = np.reshape(test_sent, (1,-1))
print(test_sent)


X = ls_sum

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X, y)
res = rfc.predict(test_sent)[0]
if res == 1:
    res_text = 'positive'
else:
    res_text = 'negative'
print('Result : ' + res_text)
