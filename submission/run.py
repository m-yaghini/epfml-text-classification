from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from helpers import load_data_and_labels, create_csv_file
from keras.preprocessing.text import Tokenizer
from keras.models import load_model


# Fix a seed for reproducibility
np.random.seed(1337)

# set parameter for the maximum length of the tokenizer (this must be the same
# as the values used for the training set
maxlen = 400

print('Loading data...')
x_train, labels, x_test = load_data_and_labels('data/train_pos.txt', 'data/train_neg.txt', 'data/test_data.txt')

# Tokenize the words and creating sequences and padding sequences
tokenizer = Tokenizer(nb_words=maxlen)
tokenizer.fit_on_texts(x_train)
sequences_test = tokenizer.texts_to_sequences(x_test)
X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)


print('Loading the model...')
model = load_model('model_CNN_test.h5')

# Predictions for the test data
pred_matrix = model.predict(X_test, batch_size=32)
print("Predictions done")

# Replacing the predictions with -1 or +1
# The prediction matrix returned above is a list of 2x1 vectors where the first column is the probability that the
# tweet is a negative tweet and the second column represents the probabiltity that it is a positive tweet.
# Hence for each tweet, we check which probability is higher and we attribute the label 1 or -1 depending on the result.
predictions = []
index = 1
for tweet in pred_matrix:
    if tweet[0] >= tweet[1]:
       predictions.append([index, -1])
    else:
        predictions.append([index, 1])
    index += 1

# Creation of the csv file
create_csv_file(predictions)



