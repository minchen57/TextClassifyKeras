#!/usr/bin/env python

import nltk
import pandas as pd
import numpy as np
import os
from gensim.models import Word2Vec
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, precision_recall_fscore_support

# PARAMETERS ================
MAX_SEQUENCE_LENGTH = 100
CUSTOM_SEED = 43
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.25
START = 0
NoOfTopic = 2
end = 3 * NoOfTopic - 1

def cal_test_baseline(df, start, end):
    df1 = df[df.columns[start:end+1]]
    return (1 - df1.sum().sum()/(len(df1)*len(df1.columns)))

def token2vec(token,w2vmodel):
    return w2vmodel.wv[token]

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


relativePath = os.getcwd()
sentencePath = relativePath + "/data/sample5_sentences_08132018.csv"
sentences = pd.read_csv(sentencePath, index_col="Sentence#")
print(sentences.columns)
sentences = sentences[list(sentences.columns.values)[0:-2]+["Sentence"]]
numberOfClasses = len(sentences.columns)-1
#print(sentences.tail(4))
print("classes selected", sentences.columns[0:-2])
print("number of classes/labels: ", numberOfClasses)
print("total number of sentences: ", len(sentences))
w2vmodel = Word2Vec.load("word2vec.model")
print("vector size used in w2v: ",w2vmodel.vector_size)
path = "Results/08132018-"+ str(NoOfTopic)+"/"

# split data into train and test
train, test = train_test_split(sentences, test_size=TEST_SPLIT,random_state=CUSTOM_SEED + 10)

print(len(test))

word2int = {}
counter = -1

def prepare_inputs_X(df, word2int, counter, sent_token_list):
    dropped = []
    for index, row in df.iterrows():
        tokens = nltk.word_tokenize(row["Sentence"])
        tokenstoIDs = []
        for token in tokens:
            if token not in word2int:
                counter += 1
                word2int[token] = counter
            tokenstoIDs.append(word2int[token])
        if len(tokenstoIDs) <= MAX_SEQUENCE_LENGTH:
            sent_token_list.append(tokenstoIDs)
        else:
            dropped.append(index)
    X = np.array(sent_token_list)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    return X, word2int, counter, dropped

def prepare_inputs_y(df, start, end, multilabel,dropped):
    for index, row in df.iterrows():
        if index not in dropped:
            multilabel.append(list(row[start:end+1].values))
    y = np.array(multilabel)
    return y

(X,word2int,counter,dropped_train) = prepare_inputs_X(train, word2int,counter,[])
(X_test,word2int,counter,dropped_test) = prepare_inputs_X(test, word2int,counter,[])
print('size of volcabulary: ',len(word2int))
test.drop(dropped_test, axis=0, inplace=True)
print(len(test))


print("number of classes considered: ", NoOfTopic)
print("classes are: ", sentences.columns[START:end+1])
y = prepare_inputs_y(train, START, end, [], dropped_train)
y_test = prepare_inputs_y(test, START, end, [], dropped_test)
print(X.shape)
print(X_test.shape)
print(y.shape)
print(y_test.shape)

# split training data into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VALIDATION_SPLIT, random_state=CUSTOM_SEED)

n_train_samples = X_train.shape[0]
n_val_samples = X_val.shape[0]
n_test_samples = X_test.shape[0]

print('We have %d TRAINING samples' % n_train_samples)
print('We have %d VALIDATION samples' % n_val_samples)
print('We have %d TEST samples' % n_test_samples)

# + 1 to include the unkown word
embedding_matrix = np.random.random((len(word2int) + 1, w2vmodel.vector_size))

for word in word2int:
    embedding_vector = token2vec(word,w2vmodel)
    if embedding_vector is not None:
        # words not found in embeddings_index will remain unchanged and thus will be random.
        embedding_matrix[word2int[word]] = embedding_vector

print('Embedding matrix shape', embedding_matrix.shape)
print('X_train shape', X_train.shape)

model = Sequential()
embedding_layer = Embedding(len(word2int) + 1,
                            w2vmodel.vector_size,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

model.add(embedding_layer)
model.add(Bidirectional(LSTM(128, return_sequences=False)))
model.add(Dense(500, activation='relu'))
model.add(Dense(3 * NoOfTopic, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Bidirectional LSTM")
model.summary()


x= model.fit(X_train, y_train,
          batch_size=256,
          epochs=5,
          validation_data = (X_val, y_val),
          shuffle = True,
          verbose = 1
          )

if not os.path.exists(path):
    print('MAKING DIRECTORY to save model file')
    os.makedirs(path)

plot_model_performance(
    train_loss=x.history.get('loss', []),
    train_acc=x.history.get('acc', []),
    train_val_loss=x.history.get('val_loss', []),
    train_val_acc=x.history.get('val_acc', []),
    save_figure_path = path +'model_performance.png'
)

# Visualize model architecture
#plot_model(model, to_file=path +'model_structure.png', show_shapes=True)

preds = model.predict(X_test, verbose=1)
preds[preds>=0.5] = int(1)
preds[preds<0.5] = int(0)
columns = []
for col in sentences.columns[START:end+1]:
    columns.append(col+"_predicted")
predd = pd.DataFrame(preds, columns=columns, index=test.index)
re = pd.concat([test[test.columns[START:end+1]], predd], axis=1)
re.to_csv(path + 'predicted_result.csv')

y_test = test[test.columns[START:end+1]].values
print("baseline point-wise acc: ", cal_test_baseline(test, START, end))
print("poitwise accuracy", np.sum(preds == y_test)/(preds.shape[0]*preds.shape[1]))
print ("f1: ", f1_score(y_test, preds, average='weighted'))
print ("accuracy: ", accuracy_score(y_test, preds))
print ("precision: ", precision_score(y_test, preds, average='weighted'))
print ("recall: ", recall_score(y_test, preds, average='weighted'))
#print ("precision_recall_fscore_support: ", precision_recall_fscore_support(y_test, preds, average='weighted'))
print("see results in " + path)