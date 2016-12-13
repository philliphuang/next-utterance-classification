# import modules
import pandas as pd
import numpy as np
import editdistance
import nltk

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

def bleu_pred_trigrams_only(obs):
    
    context = obs[0].split()
    
    # list of scores
    dists = [
        nltk.translate.bleu_score.sentence_bleu(
            obs[x].split(), 
            context, 
            weights=[0,0,3] # trigrams
        ) 
        for x in range(1,11)
    ]
    
    # dataframe for sorting
    sort_df = pd.DataFrame({'choices': list(range(0,10)), 'distances': dists})
    sort_df.sort_values(by='distances', inplace=True, ascending=False)
    return sort_df['choices'].tolist()

answers = []
for index, row in test.iterrows():
    if index % 1000 == 0:
        print str(index) + 'th row reached'
    answers.append(bleu_pred_trigrams_only(row.tolist()))

for k in [1, 2, 5, 10]:
    print("Recall @ {}, 10 total choices: {:g}".format(k, recall_at_k(answers, k)))