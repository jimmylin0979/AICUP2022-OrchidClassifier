#
import pandas as pd
import numpy as np

# 
# register_models = ['model_stnvit_mixs', 'model_swin_mixs_tmax101', 'model_convnext_384_tmax101']

# # Register confidence score dataframe from register models
# confidence_scores = []
# for model_name in register_models:
#     # Read condifence score table for each model
#     path = f'./saved/{model_name}/prediction-Confidence-best.csv'
#     confidence_scores.append(pd.read_csv(path))

register_models = ['predict', 'model_stnvit_mixs', 'model_swin', 'model_swin_mixs_tmax101', 'model_convnext_384_tmax101']

# Register confidence score dataframe from register models
confidence_scores = []
for model_name in register_models:
    # Read condifence score table for each model
    path = f'./rearrange_prediction_{model_name}.csv'
    confidence_scores.append(pd.read_csv(path))

'''
validset_index, topN, Ground_truth, 1, prob1, 2, prob2, 3, prob3, 4, prob4, 5, prob5
0, 5, 0, 0, 0.7134590148925781, 49, 0.01988092251121998, 178, 0.009589494206011295, 200, 0.00530626904219389, 174, 0.005254499148577452
'''

#
with open('./ensemble_predict.csv', 'w') as f:
    # 

    n_correct = 0

    N = len(confidence_scores[0])
    for idx in range(N):   
        # the ensemble result from each model
        # the format is [label1 : total confidence score, label2 : total confidence score, ...]
        ensemble = {}

        # Add top 5 confidence score of each predicted label into ensemble
        for confidence in confidence_scores:
            for i in range(3, 8, 2):
                if confidence.iloc[idx, i] in ensemble:
                    ensemble[confidence.iloc[idx, i]] += confidence.iloc[idx, i+1]
                else:
                    ensemble[confidence.iloc[idx, i]] = confidence.iloc[idx, i+1]
        
        # Sort the ensemble result
        ensemble = sorted(ensemble.items(), key=lambda x: x[1], reverse=True)
        # print(ensemble)
        # break

        ensemble_label = ensemble[0][0]

        if ensemble_label == int(idx / 2):
            n_correct += 1

        # And then write the ensemble result into csv file
        f.write(f'{idx},{ensemble_label}\n')
    
    print(f'Accuracy : {n_correct/N}')
