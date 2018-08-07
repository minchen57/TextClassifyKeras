import nltk
import pandas as pd
import os
import gensim

# PARAMETERS ================
EMBEDDING_DIM = 300

def train_w2v(sent_list, size=100):
    model = gensim.models.Word2Vec(sent_list, min_count=1,size=size)
    model.delete_temporary_training_data(replace_word_vectors_with_normalized=True)
    return model

relativePath = os.getcwd()
allSentencePath = relativePath + "/data/all_sentences_08062018.csv"

allSentence = pd.read_csv(allSentencePath, index_col = "Sentence#")
print(allSentence.tail(4))

allSentenceList = []
for index, row in allSentence.iterrows():
    allSentenceList.append(nltk.word_tokenize(row["Sentence"]))
#print(allSentenceList[0:4])

w2vmodel = train_w2v(allSentenceList, EMBEDDING_DIM)
w2vmodel.save("word2vec.model")