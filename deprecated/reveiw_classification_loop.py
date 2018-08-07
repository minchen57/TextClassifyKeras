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
CUSTOM_SEED = 42
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.25


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
sentencePath = relativePath + "/data/sample1_sentences_08062018.csv"
sentences = pd.read_csv(sentencePath, index_col = "Sentence#")
print(sentences.columns)
sentences = sentences[list(sentences.columns.values)[0:2]+["Sentence"]]
numberOfClasses = len(sentences.columns)-1
#print(sentences.tail(4))
print("classes selected", sentences.columns[0:-1])
print("number of classes/labels: ", numberOfClasses)
print("total number of sentences: ", len(sentences))
w2vmodel = Word2Vec.load("word2vec.model")
print("vector size used in w2v: ",w2vmodel.vector_size)
path = "Results/08062018-"+ str(numberOfClasses)+"/"

multilabel = []
sent_token_list = []
word2int = {}
counter = -1
for index, row in sentences.iterrows():
    tokens = nltk.word_tokenize(row["Sentence"])
    tokenstoIDs = []
    for token in tokens:
        if token not in word2int:
            counter += 1
            word2int[token] = counter
        tokenstoIDs.append(word2int[token])
    if len(tokenstoIDs) <= MAX_SEQUENCE_LENGTH:
        sent_token_list.append(tokenstoIDs)
        multilabel.append(list(row[0:numberOfClasses].values))
print('size of volcabulary: ',len(word2int))
X = np.array(sent_token_list)
y = np.array(multilabel)
#print(type(X),X.shape, X[0:3])
#print(type(y),y.shape,y[0:3])
#print(len([len(l) for l in sent_token_list if len(l)>MAX_SEQUENCE_LENGTH]))
print("total number of sentences kept: ", len(X))

#padding 0s infront to make it same size
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)


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
embedding_matrix = np.random.random((len(word2int) + 1, w2vmodel.vector_size))

for word in word2int:
    embedding_vector = token2vec(word,w2vmodel)
    if embedding_vector is not None:
        # words not found in embeddings_index will remain unchanged and thus will be random.
        embedding_matrix[word2int[word]] = embedding_vector

print('Embedding matrix shape', embedding_matrix.shape)
print('X_train shape', X_train.shape)

def Run_Models(lstm_units, hidden_units, activation):
    model = Sequential()
    embedding_layer = Embedding(len(word2int) + 1,
                                w2vmodel.vector_size,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(lstm_units, return_sequences=False)))
    if hidden_units > 0:
        model.add(Dense(hidden_units, activation=activation))
    model.add(Dense(numberOfClasses, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    print("model fitting - Bidirectional LSTM")
    model.summary()

    x= model.fit(X_train, y_train,
              batch_size=256,
              epochs=20,
              validation_data=(X_val, y_val),
              shuffle = True,
              verbose = 1
              )

    preds = model.predict(X_test)
    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0

    return (accuracy_score(y_test, preds), precision_score(y_test, preds, average='weighted'),
            recall_score(y_test, preds, average='weighted'),f1_score(y_test, preds, average='weighted'))

Results = [("lstm-units", "hidden-units", "activation-function", "accuracy", "precision", "recall", "f1")]
with open('results/' + str(numberOfClasses) + '_result.txt', 'w') as f:
    f.write('%s %s %s %s %s %s %s\n' % Results[0])
for lstm_units in [32,64,128,256,512]:
    for hidden_units in [0, 50, 100, 200, 300, 500, 1000, 1500, 2000]:
        for activation in ['relu', 'tanh']:
            result = Run_Models(lstm_units,hidden_units,activation)
            Results.append((lstm_units,hidden_units,activation, result))
            with open('results/'+str(numberOfClasses)+'_result.txt', 'a') as f:
                f.write('%s %s %s ' % (lstm_units,hidden_units,activation))
                f.write('%s %s %s %s\n' % result)
            print(result)

print(Results)