from sklearn.cluster import KMeans
from gensim.models import doc2vec, word2vec
from collections import namedtuple
import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv")
train_data = np.append(train.Context.values,train.Utterance.values)

taggedMessage = namedtuple('TaggedMessage', 'words tags')
documents = []

# Preprocess messages
for i, message in enumerate(train_data):
    # Split into lists of words
    words = message.split()
    tags = [i]
    x = taggedMessage(words, tags)
    documents.append(taggedMessage(words, tags))

d2v2 = doc2vec.Doc2Vec(documents, size=200, workers=4, iter=20)

X = []
for v in d2v2.docvecs:
    X.append(v)

X = np.array(X)

# Cluster messages using k-means
kmeans = KMeans(n_clusters=100, n_jobs=-1).fit(X)
print 'successfully clustered'
