import imp
agglom = imp.load_source('agglom', '../modules/agglom.py')
from agglom import *
from sklearn.cluster import AgglomerativeClustering

## Testing

# bleu distance is negative
assert(bleu_score("hello I am pizza", "I do enjoy pizza quite a bit") < 0)

# dist_matrix works
assert(np.array_str(dist_matrix(['b', 'a'], 'edit')) == np.array_str(np.array([[0.,1.],[1.,0.]])))
assert(np.array_str(dist_matrix(['b', 'greetings hello', 'hello'], 'bleu')) == np.array_str(np.array([[-1.,0.,0.],[0.,-1.,-0.36787944],[0.,-0.36787944,-1.]])))

# generate train data
train = '''Hello
Hi!
How are you?
I am doing well, how about you?
Great!
How are you today?
I am pretty good
What's up?
I need to poop'''
train_data = train.split('\n')

# generate distance matrices
edit_matrix = dist_matrix(train_data, 'edit')
bleu_matrix = dist_matrix(train_data, 'bleu')


