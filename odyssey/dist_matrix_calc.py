from sklearn_dist_matrix import sk_dist_matrix
from sklearn.cluster import AgglomerativeClustering
import pickle 
import pandas as pd

# get list of contexts
train = pd.read_csv("data/train.csv", nrows=10)
train_list = train['Context'].tolist()

# build distance matrix between all contexts
edit_matrix = sk_dist_matrix(train_list, 'edit')
bleu_matrix = sk_dist_matrix(train_list, 'bleu')

# cluster with edit distance
agc_model_edit = AgglomerativeClustering(n_clusters=1000, affinity='precomputed', linkage='average')
clusters_edit = agc_model.fit_predict(edit_matrix, train_list)

# cluster with bleu score
agc_model_bleu = AgglomerativeClustering(n_clusters=1000, affinity='precomputed', linkage='average')
clusters_bleu = agc_model.fit_predict(bleu_matrix, train_list)

# save cluster results
pickle.dump(clusters_edit, open('edit_clusters.p', 'wb'))
pickle.dump(clusters_bleu, open('bleu_clusters.p', 'wb'))