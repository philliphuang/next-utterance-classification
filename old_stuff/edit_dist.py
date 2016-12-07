import editdistance
from operator import itemgetter

print editdistance.eval('yo', 'bahama')

a = [1,2,4,0,5]

print min(enumerate(a), key=itemgetter(1))[0] 