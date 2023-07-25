#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 14:25:10 2023

@author: ptruong
"""

import pandas as pd
import xgboost
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from xgboost import plot_importance
import matplotlib.pyplot as plt

import torch
from tape import ProteinBertModel, TAPETokenizer

model = ProteinBertModel.from_pretrained('bert-base')
tokenizer = TAPETokenizer(vocab='iupac')  # iupac is the vocab for TAPE models, use unirep for the UniRep model

def encode_seqeuence_BERT(sequence, model, tokenizer):
    # Pfam Family: Hexapep, Clan: CL0536
    sequence = 'GCTVEDRCLIGMGAILLNGCVIGSGSLVAAGALITQ'
    
    token_ids = torch.tensor([tokenizer.encode(sequence)])
    output = model(token_ids)
    sequence_output = output[0]
    pooled_output = output[1]

    # NOTE: pooled_output is *not* trained for the transformer, do not use
    # w/o fine-tuning. A better option for now is to simply take a mean of
    # the sequence output

    return sequence_output.mean() # this is the mean of the sequence output as mentioned above...

df = pd.read_csv("summary.tsv", sep="\t")
df["BERT_encoded_sequences"]=df.peptide_charge.map(lambda x:encode_seqeuence_BERT(x.split("_")[0], model, tokenizer))
df = df.drop(["peptide_charge", 'preceding_ratio_increase', 'preceding_log2_increase',
              'succeeding_ratio_increase', 'succeeding_log2_increase', 'succeeding_log2_ratio_increase', 
              'group'], axis=1)
df = df[['Gravy Score', 'ClogP', 'charge', 'BB Index', 'unlabeled_rt',
       'labeled_rt', 'length', 'BERT_encoded_sequences', 'preceding_log2_ratio_increase']]

dataset = df.values

# split data into X and y

X = dataset[:, 0:8]
Y = dataset[:, 8]

# CV model
model = xgboost.XGBRegressor()
kfold = KFold(n_splits=20, random_state=7, shuffle=True)
results = cross_val_score(model, X, Y, cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

# Train XGBoost model
model.fit(X, Y)

# Get feature importances
importances = model.feature_importances_

# Create a dictionary to store feature weights
feature_weights = {}
for i, importance in enumerate(importances):
    #feature_weights[f'Feature {i+1}'] = importance
    feature_weights[df.columns[i]] = importance



# Sort feature weights in descending order
sorted_weights = sorted(feature_weights.items(), key=lambda x: x[1], reverse=True)

# Print feature weights
print("Feature Weights:")
for feature, weight in sorted_weights:
    print(f"{feature}: {weight}")

# Plot feature importance
plot_importance(model)
plt.show()



# ToDo
"""
Redo this with random forest
can we redo this example with previously processed data and have a look, 
and maybe investigate what is going on and see if the peptide sequences have any effect.

Include the one-hot-encoded features.
"""