import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDF_Predictor:
	def __init__(self):
		self.vectorizer = TfidfVectorizer()

	def train(self, data):
		# data should be in the form of an array of strings
		self.vectorizer.fit(data)

	def predict(self, context, responses):
		# Vectorize the context and possible responses
		context_vectorized = self.vectorizer.transform([context])
		responses_vectorized = self.vectorizer.transform(responses)

		sims = []
		# Calculate cosine similarity between context and each of responses
		for response_vectorized in responses_vectorized:
			sim = cosine_similarity(context_vectorized.toarray(), response_vectorized.toarray())[0][0]
			sims.append(sim)

		# Sort by highest cosine similarity and return indices in descending order
		sorted_list = np.argsort(sims, axis=0)[::-1]
		# print sorted_list
		return sorted_list
