from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import csv
from keras.preprocessing import sequence
from helpers import load_data_and_labels
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.models import load_model

# set parameters:
max_features = 3000
maxlen = 1000
batch_size = 200
embedding_dims = 300
nb_filter = 250
filter_length = 3
hidden_dims = 50
nb_epoch = 2

print('Loading data...')
x_train, labels, x_test = load_data_and_labels('data/train_pos.txt', 'data/train_neg.txt', 'data/test_data.txt')

# Tokenize the words and creating sequences and padding sequences
tokenizer = Tokenizer(nb_words=maxlen)
tokenizer.fit_on_texts(x_train)
sequences_test = tokenizer.texts_to_sequences(x_test)
X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)


print('Loading the model...')
model = load_model('model_base.h5')

# Predictions for the test data
probas = model.predict(X_test, batch_size=32)

# Replacing the predictions with -1 or +1
predictions = []
index = 1
for tweet in probas:
    if tweet[0] >= tweet[1]:
        predictions.append([index, -1])
    else:
        predictions.append([index, 1])
    index += 1
print(predictions)

# Doing a sum to check if the test data are predicted evenly
sum = 0
for pred in predictions:
    sum += pred[1]
print(sum)

# Creating the csv file for the submission on Kaggle
with open('submission.csv', 'w') as csvfile:
    fieldnames = ['Id', 'Prediction']
    sub_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    index = 0
    sub_writer.writeheader()
    for res in predictions:
        index += 1
        sub_writer.writerow({'Id': str(index), 'Prediction': str(res[1])})
print("Submission file created")
