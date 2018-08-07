#!/usr/bin/env python

'''
POS_LSTM_keras_w2v_original.py
Min Chen <mc43@iu.edu>
Project: Deep Learning and POS tagging

Corpus: Treebank from NLTK, Brown
Libary: keras
Model: BILSTM RNN
Word Embedding: Yes

Last Updated by Min Chen - Aug 2,2018

Some code modified from https://github.com/aneesh-joshi/LSTM_POS_Tagger

'''

import nltk, gensim, os
import numpy as np
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, Bidirectional
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping

# PARAMETERS ================
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300
CUSTOM_SEED = 42
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.25
CORPUS = 'Treebank'
#CORPUS = 'Brown'


if CORPUS == 'Treebank':
    tagged_sentences = nltk.corpus.treebank.tagged_sents()
    PATH = 'results/POS_LSTM_keras_w2v/'
else:
    tagged_sentences = nltk.corpus.brown.tagged_sents()
    PATH = 'results/POS_LSTM_keras_w2v_brown/'
print(tagged_sentences[0])
print("Tagged sentences: ", len(tagged_sentences))


def plot_model_performance(train_loss, train_acc, train_val_loss, train_val_acc, save_figure_path):
    """ Plot model loss and accuracy through epochs. """

    green = '#72C29B'
    orange = '#FFA577'

    with plt.xkcd():
        fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8))
        ax1.plot(range(1, len(train_loss) + 1), train_loss, green, linewidth=5,
                 label='training')
        ax1.plot(range(1, len(train_val_loss) + 1), train_val_loss, orange,
                 linewidth=5, label='validation')
        ax1.set_xlabel('# epoch')
        ax1.set_ylabel('loss')
        ax1.tick_params('y')
        ax1.legend(loc='upper right', shadow=False)
        ax1.set_title('Model loss through #epochs', fontweight='bold')

        ax2.plot(range(1, len(train_acc) + 1), train_acc, green, linewidth=5,
                 label='training')
        ax2.plot(range(1, len(train_val_acc) + 1), train_val_acc, orange,
                 linewidth=5, label='validation')
        ax2.set_xlabel('# epoch')
        ax2.set_ylabel('accuracy')
        ax2.tick_params('y')
        ax2.legend(loc='lower right', shadow=False)
        ax2.set_title('Model accuracy through #epochs', fontweight='bold')

    plt.tight_layout()
    plt.show()
    fig.savefig(save_figure_path)
    plt.close(fig)

def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]

def tag(tagged_sentence):
    return [t for w, t in tagged_sentence]

def untagged_whole(tagged_sentences):
    sent_list = []
    for s in tagged_sentences:
        sent_list.append(untag(s))
    return sent_list

def train_w2v(sent_list, size=100):
    model = gensim.models.Word2Vec(sent_list, min_count=1,size=size)
    model.delete_temporary_training_data(replace_word_vectors_with_normalized=True)
    return model

def token2vec(token,w2vmodel):
    return w2vmodel.wv[token]

def apply2int(list, dict):
    return [dict[l] for l in list]


w2vmodel = train_w2v(untagged_whole(tagged_sentences), EMBEDDING_DIM)

word2int = {}
tag2int = {}
counter1 = -1
counter2 = -1
for tagged in tagged_sentences:
    for (word,pos) in tagged:
        if word not in word2int:
            counter1 += 1
            word2int[word]= counter1
        if pos not in tag2int:
            counter2 += 1
            tag2int[pos] = counter2
print('size of volcabulary: ',len(word2int))
print('number of tags: ',len(tag2int))
#vol = set([t[0] for tagged in tagged_sentences for t in tagged])

# turn the words/tokens into input matrix, in terms of integers
X = np.array([apply2int(untag(sent),word2int) for sent in tagged_sentences])
# turn the tagss into input matrix (categorical, one-hot encoder vectors)
y = np.array([to_categorical(apply2int(tag(sent),tag2int),num_classes=len(tag2int)+1) for sent in tagged_sentences])
#print(y)

#padding 0s infront to make it same size
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
y = pad_sequences(y, maxlen=MAX_SEQUENCE_LENGTH)
#print(X)
#print(y)

# shuffle the data
X, y = shuffle(X, y)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT,random_state=CUSTOM_SEED)

# split training data into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VALIDATION_SPLIT, random_state=1)

n_train_samples = X_train.shape[0]
n_val_samples = X_val.shape[0]
n_test_samples = X_test.shape[0]

print('We have %d TRAINING samples' % n_train_samples)
print('We have %d VALIDATION samples' % n_val_samples)
print('We have %d TEST samples' % n_test_samples)


# + 1 to include the unkown word
embedding_matrix = np.random.random((len(word2int) + 1, EMBEDDING_DIM))

for word in word2int:
    embedding_vector = token2vec(word,w2vmodel)
    if embedding_vector is not None:
        # words not found in embeddings_index will remain unchanged and thus will be random.
        embedding_matrix[word2int[word]] = embedding_vector

print('Embedding matrix shape', embedding_matrix.shape)
print('X_train shape', X_train.shape)

embedding_layer = Embedding(len(word2int) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

l_lstm = Bidirectional(LSTM(64, return_sequences=True))(embedded_sequences)
preds = TimeDistributed(Dense(len(tag2int) + 1, activation='softmax'))(l_lstm)
model = Model(sequence_input, preds)


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Bidirectional LSTM")
model.summary()

#earlystopper = EarlyStopping(monitor='val_acc', patience=5, verbose=1)
#callbacks=[earlystopper]

x= model.fit(X_train, y_train,
          batch_size=256,
          epochs=8,
          validation_data=(X_val, y_val),
          shuffle = True,
          verbose = 1
          )

if not os.path.exists(PATH):
    print('MAKING DIRECTORY to save model file')
    os.makedirs(PATH)

plot_model_performance(
    train_loss=x.history.get('loss', []),
    train_acc=x.history.get('acc', []),
    train_val_loss=x.history.get('val_loss', []),
    train_val_acc=x.history.get('val_acc', []),
    save_figure_path = PATH +'model_performance.png'
)

# Visualize model architecture
plot_model(model, to_file=PATH +'model_structure.png', show_shapes=True)

test_results = model.evaluate(X_test, y_test, verbose=1)
print('TEST LOSS %f \nTEST ACCURACY: %f' % (test_results[0], test_results[1]))

print("see results in " + PATH)

