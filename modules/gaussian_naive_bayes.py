import math
import pandas as pd
import numpy as np

class GaussianNaiveBayes():
    '''
    Implementation of Gaussian Naive Bayes

    Args:
        data:   pandas dataframe, first column is class labels, rest of columns features
                We use doc2vec embeddings as feature vectors
                Train on this data
        
        num_classes: number of classes
        alpha: Laplace smoothing parameter (default 1.0)
    '''

    def __init__(self, data, num_classes):
        # Get number of features 
        # First column is not a feature, but the class label instead
        self.num_features = data.shape[1] - 1
        self.num_classes = num_classes

        # Compute prior probabilities for each class
        #   number of occurrences of class / total number of rows
        self.priors = data.groupby(data.columns[0])[data.columns[0]].count() / data.shape[0]
        # Compute means for each feature for each class
        self.means = data.groupby(data.columns[0])[data.columns[range(1, data.shape[1])]].mean()
        # Compute standard deviation for each feature for each class
        self.stds = data.groupby(data.columns[0])[data.columns[range(1, data.shape[1])]].std()
        # Check for NaN standard deviations, and set them to 1
        if self.stds.isnull().values.any():
            self.stds = self.stds.fillna(1.0)
            print "Warning: NaN standard deviations found. Filling them in with 1.0 standard deviation instead."


    def predict_proba(self, obs):
        '''
        Compute probability estimates for each of the classes for a new observation

        Args:
          obs: a feature vector representing a document (here we'll use a doc2vec vector)
        
        Returns:
          The probabilities for each class for use as a feature vector
        '''
        # Feature vecture is of length number of classes
        prob_vector = np.zeros(self.num_classes)

        # Calculate posterior probability for each class
        # P(k|X) = P(X_1|k)*...*P(X_n|k)*P(k)
        for c in range(self.num_classes):
            prob_vector[c] = self.priors[c] * np.prod([
                gaussian_pdf(
                    obs[f], 
                    self.means.ix[c][f], 
                    self.stds.ix[c][f]
                ) 
                for f in range(len(obs))])

        return prob_vector

# Calculate the Gaussian probability at a given value, given mean and standard deviation
def gaussian_pdf(value, mean, std):
    return (1 / (math.sqrt(2 * math.pi) * std)) * math.exp(-((value - mean) ** 2)/(2 * std ** 2))

# test_df = pd.DataFrame({'class': [0, 1, 0],
#                     'feature1': [2., 3., 4.],
#                     'feature2': [5.,6.,7.]
#                     })

# GaussianNaiveBayes(test_df, 2).predict_proba([1,3])

assert(gaussian_pdf(71.5, 73, 6.2) - 0.0624896575937 < 0.00001)
assert(gaussian_pdf(0, 0, 1) - 0.39894228 < 0.00001)
        