from __future__ import print_function
import numpy as np
np.random.seed(1337)

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM
from keras.layers import Convolution1D, GlobalMaxPooling1D, MaxPooling1D
from helpers import load_data_and_labels
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt


# set parameters:
# We can play with those parameters to get better results
'''
max_features = 6000
maxlen = 2000
batch_size = 1000
embedding_dims = 300
nb_filter = 250
filter_length = 3
hidden_dims = 50
nb_epoch = 2
'''


max_features = 3000
maxlen = 1000
batch_size = 200
embedding_dims = 100
nb_filter = 250
filter_length = 3
hidden_dims = 50
nb_epoch = 2

'''
max_features = 20000
maxlen = 1000
embedding_size = 100

# Convolution
filter_length = 5
nb_filter = 64
pool_length = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 300
nb_epoch = 2
'''

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open('data/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

print('Loading data...')
x_train, labels, x_test = load_data_and_labels('data/train_pos.txt', 'data/train_neg.txt', 'data/test_data.txt')

# Tokenize the words and creating sequences and padding sequences
tokenizer = Tokenizer(nb_words=maxlen)
tokenizer.fit_on_texts(x_train)
sequences_train = tokenizer.texts_to_sequences(x_train)
sequences_test = tokenizer.texts_to_sequences(x_test)

word_index = tokenizer.word_index

print(len(x_train), 'train sequences')
print(len(labels), 'labels')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(sequences_train, maxlen=maxlen)
X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


# prepare embedding matrix
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words+1, embedding_dims))
print(nb_words)
for word, i in word_index.items():
    if i > max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


print('Build model...')
# This basic model was taken from an online example, we can add
# much more layers and use the Standford embeddings to improve it


model = Sequential()
# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(nb_words + 1,
                    embedding_dims,
                    weights=[embedding_matrix],
                    input_length=maxlen,
                    dropout=0.2))

'''
model.add(Dropout(0.25))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=2))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(LSTM(lstm_output_size))
model.add(Dense(2))
model.add(Activation('sigmoid'))
'''

'''
# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=2))


# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(2))
model.add(Activation('sigmoid'))
'''

model.add(GlobalMaxPooling1D())
model.add(Dense(100, init='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, init='normal', activation='relu'))
#x = Dense(1, init='normal', activation='sigmoid')(x)
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



history = model.fit(X_train, labels, batch_size=batch_size, nb_epoch=nb_epoch)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('accuracy.png')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('loss.png')

plt.savefig()

model.save('model_base.h5')
