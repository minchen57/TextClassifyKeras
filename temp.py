import os
import pandas as pd
from sklearn.model_selection import train_test_split
CUSTOM_SEED = 43
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.25

relativePath = os.getcwd()
sentencePath = relativePath + "/data/sample5_sentences_08132018.csv"
sentences = pd.read_csv(sentencePath, index_col="Sentence#")
sentences = sentences[list(sentences.columns.values)[0:18]+["Sentence"]]
train, test = train_test_split(sentences, test_size=TEST_SPLIT,random_state=CUSTOM_SEED + 10)
truetrain, val = train_test_split(sentences, test_size=VALIDATION_SPLIT,random_state=CUSTOM_SEED )
truetrain.to_csv(relativePath + "/data/sample5_sentences_08132018_18_train.csv")
test.to_csv(relativePath + "/data/sample5_sentences_08132018_18_test.csv")