def recall_at_k(y_pred, k=1):
	"""
	recall@k evaluation metric

	Args:
		y_pred: list of predicted rankings for each of 10 possible responses (descending order)
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

# TODO: Move into separate testing file
assert(recall_at_k([[0,1,2,3,4], [1,2,3,4]]) == 0.5)
assert(recall_at_k([[0,1,2,3,4]]) == 1)