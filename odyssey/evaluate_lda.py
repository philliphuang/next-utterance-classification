import gensim
import numpy as np

def recall_at_k(y_pred, k=1):
	"""
	recall@k evaluation metric

	Args:
		y_pred: list of lists of predicted rankings for each of 10 possible responses (descending order)
				i.e. [0,3,1,2,5,6,4,7,8,9] means response 0 is rated most probable, 9 least probable
		k: number of tries model has to predict the correct response

	Returns:
		Accuracy percentage
	"""
	num_correct = 0
	num_total = len(y_pred)

	for ranking in y_pred:
		# Message at index 0 is always correct next message in our test data
		if 0 in ranking[:k]:
			num_correct += 1

	return float(num_correct) / num_total

# Convert gensim LDA topic distribution for a given document to feature vector of length number of topics
def get_lda_feature_vector(topic_dist, num_topics):
    vector = np.zeros(num_topics)
    
    # Fill in values for nonzero topics
    for index, value in topic_dist:
        vector[index] = value
            
    return vector

def lda_predict(model, dictionary, context, responses):
    '''
    Calculates the cosine similarity between the context and each of the possible responses,
    returning a ranked list of responses sorted in decreasing order by cosine similarity
    
    Args:
        model: a lda model
        dictionary: dictionary associated with the lda model
        context: a context that we want to find the response for
        responses: list of candidate responses containing the actual response
        
    Returns:
        List of response indices sorted in descending order by cosine similarity with context
    '''
    # Infer topic distribution and vectorize
    context_vector = get_lda_feature_vector(model[dictionary.doc2bow(context.split())], model.num_topics)
    sims = []
    
    for response in responses:
        # Calculate cosine similarity between the lda feature vectors of response and context
        response_vector = get_lda_feature_vector(model[dictionary.doc2bow(response.split())], model.num_topics)
        sim = cosine_similarity(context_vector.reshape(1, -1), response_vector.reshape(1, -1))[0][0]
        sims.append(sim)
        
    return np.argsort(sims, axis=0)[::-1]

# load saved dictionary
dictionary = gensim.corpora.dictionary.Dictionary.load("dict/ubuntu.dict")

test = pd.read_csv("data/test.csv")

# Evaluate model performance with lda model
y = [lda_predict(lda, dictionary, test.Context[x], test.iloc[x,1:].values) for x in range(10)]
for k in [1, 2, 5, 10]:
    print("Recall @ {}, 10 total choices: {:g}".format(k, recall_at_k(y, k)))