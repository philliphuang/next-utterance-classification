import numpy as np
import pandas as pd
import gensim

train = pd.read_csv("data/train.csv", nrows=100)
train_data = np.append(train.Context.values,train.Utterance.values)
texts = [doc.split() for doc in train_data]
dictionary = gensim.corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lda = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics=100, workers=7)
return 'success'