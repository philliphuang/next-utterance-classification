import imp
agglom = imp.load_source('agglom', '../modules/agglom.py')
from agglom import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

## Testing

# bleu distance is negative
assert(bleu_score("hello I am pizza", "I do enjoy pizza quite a bit") < 0)

# dist_matrix works
assert(np.array_str(dist_matrix(['b', 'a'], 'edit')) == np.array_str(np.array([[0.,1.],[1.,0.]])))
assert(np.array_str(dist_matrix(['b', 'greetings hello', 'hello'], 'bleu')) == np.array_str(np.array([[-1.,0.,0.],[0.,-1.,-0.36787944],[0.,-0.36787944,-1.]])))

# generate train data
train = '''a
b
c
aaaaaaaaaaa
d
aaaaaaaaaab
e
aaaaaaaaaac
aaaaaaaaaad'''
train_data1 = train.split('\n')

train = '''I enjoy pizza
I am also pizza
I somewhat enjoy pizza
pickles are tasty
I think pickles are also pretty good
pickles are quite good'''
train_data2 = train.split('\n')

# generate distance matrices
edit_matrix = dist_matrix(train_data1, 'edit')
bleu_matrix = dist_matrix(train_data2, 'bleu')

agc_model = AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage='average')
print agc_model.fit_predict(edit_matrix, train_data1)
print agc_model.fit_predict(bleu_matrix, train_data2)