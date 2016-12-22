from __future__ import print_function
import numpy as np

# Import useful elements to preprocess the input
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
# Import models and layers from Keras to build the neural nets
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, MaxPooling1D, Flatten
from keras.layers import Convolution1D, GlobalMaxPooling1D
# Import of our other functions
from helpers import load_data_and_labels

# Fix a seed for reproducibility
np.random.seed(1337)
# set parameters:
# We can play with those parameters to get better results
max_features = 5000
maxlen = 400
batch_size = 1000
embedding_dims = 100
nb_filter = 250
filter_length = 3
hidden_dims = 50
nb_epoch = 2


print('Loading data...')
# Loading the files using our load_data functions, see the complete documentation in the helpers.py file

x_train, y_train, x_val, y_vals, x_test = load_data_and_labels('data/train_pos.txt', 'data/train_neg.txt', 'data/test_data.txt')

# Tokenize the words and creating sequences and padding sequences
tokenizer = Tokenizer(nb_words=maxlen)
tokenizer.fit_on_texts(x_train)
sequences_train = tokenizer.texts_to_sequences(x_train)
sequences_val = tokenizer.texts_to_sequences(x_val)
sequences_test = tokenizer.texts_to_sequences(x_test)

print(len(x_train), 'train sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(sequences_train, maxlen=maxlen)
X_val = sequence.pad_sequences(sequences_val, maxlen=maxlen)
X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_val shape:', X_val.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
# This basic model was taken from an online example, we can add
# much more layers and use the Standford embeddings to improve it

model = Sequential()
# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen,
                    dropout=0.2))

model.add(Convolution1D(nb_filter=100, filter_length=4, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=1))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='relu'))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, validation_data=[X_val, y_vals],
          batch_size=batch_size,
          nb_epoch=nb_epoch)

model.save('model_CNN_test.h5')
