# Agglomerative Clustering

import pandas as pd
import numpy as np
import editdistance
import nltk


def bleu_score(x, y):
	'''
	return negative BLEU score, looking at unigrams only as some strings may only contain one word
	'''
	return (-1) * nltk.translate.bleu_score.sentence_bleu([x.split()], y.split(), weights=[1])

def dist_matrix(data, affinity):
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

	# number of strings
	length = len(data) 

	# initialize distance matrix 
	distances = np.empty(length, length)

	# 
	dist_funct = editdistance.eval() if affinity == 'edit' else bleu_score() 

	# for each string
	for index, string in enumerate(data):

		# go to remaining strings
		i = index + 1
		for other in data[index:]:
			dist = dist_funct(string, other)
			distances[index, i] = dist
			distances[i, index] = dist

			# increment index of ne
			i = i + 1

def str_agglom_cluster(data, N, affinity, linkage):
	'''
	Agglomerative clustering algorithm for strings

	Input:
		data: list of strings to cluster
		N: number of clusters to return
		affinity: how to compute distance between strings
			'edit': edit distance
			'bleu': BLEU score
		linkage: how to compute distance between clusters
			'single': closest pair
			'complete': farthest pair
			'average': average between each set
			# note: ward cannot be implemented because euclidian distance is not applicable

	Output:
		pandas dataframe of 2 columns: cluster #, message
	'''

	# calculate distance matrix
	dists = dist_matrix(data, affinity)

	# number of clusters
	count = len(data)

	# 

	while 


