from math import log

class NaiveBayes():
    '''Input:
        Data - pandas dataframe, first column class, rest of columns features
        This should be bag of words, but might also be Doc2Vec vectorization
        Trains on this data
        N - number of classes
    '''
    def __init__(self, data, num_classes):
        self.num_words = data.shape[1]-1
        # store the number of words within each utterance
        self.counts = [[0 for j in range(num_words)] for i in range(num_classes)]
        # store the number of utterances for each class
        self.nclass = [0 for i in range(num_classes)]
        # model to be fit
        self.f_vals = [[0 for i in range(num_words)] for j in range(num_classes)]
        self.alpha = 1
        self.num_classes = num_classes
        # iterate through dataframe
        for x in range(len(data)):
            # list of values, first index is class, rest of indices represent counts of features
            row = data.iloc[x].values
            class_num = row[0]
            self.nclass[class_num] += 1
            # iterate through the row
            for y in range(1, len(row)):
                # increase counts 
                self.counts[class_num][y] += row[y]
        # for each class
        for z in range(num_classes):
            total_num_words = sum(self.counts[z])
            for index in range(num_words):
                # store the negative log of the cond prob
                self.f_vals[z][index] = -log(float(counts[z][index] + self.alpha)/(total_num_words + (self.alpha * self.num_words)))
            
    def vectorize(self, obs):
        # get negative log class probabilities given words
        row = obs
        probs = [0 for i in range(num_classes)]
        # obs is a pandas dataframe
        for y in range(self.num_classes):
            # compute prior
            prior = -log(float(self.nclass[y])/sum(self.nclass))
            probs[y] += prior
            for z in range(len(row)):
                probs[y] += (self.f_vals[y][z] * row[z])
        return probs
                    
        
