import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import editdistance
import nltk

# code adapted from https://github.com/scipy/scipy/issues/4047

def bleu_score(x, y):
	'''
	return negative BLEU score, looking at unigrams only as some strings may only contain one word
	'''
	return (-1) * nltk.translate.bleu_score.sentence_bleu([x.split()], y.split(), weights=[1])

def homogenized(strings):
    lengths = [len(s) for s in strings]
    n = max(lengths)
    for s in strings:
        k = len(s)
        yield [k] + [ord(c) for c in s] + [0] * (n - k)

def dehomogenized(points):
    for p in points:
        k = int(p[0])
        yield ''.join(chr(int(x)) for x in p[1:k+1])

def edit_dist_dehomogenize(u, v):
    return editdistance.eval(*list(dehomogenized((u, v))))

def bleu_score_dehomogenize(u, v):
    return bleu_score(*list(dehomogenized((u, v))))

def sk_dist_matrix(data, affinity):
	'''
	Calculates distance matrix

	Input:
		data: list of strings
		affinity: how to compute distance between strings
			'edit': edit distance
			'bleu': BLEU score

	Output:
		np matrix of distances
	'''
	# choose distance function
	dist_funct = edit_dist_dehomogenize if affinity == 'edit' else bleu_score_dehomogenize

	points = np.array(list(homogenized(data)))

	return pairwise_distances(points, metric=dist_funct, n_jobs=-1)