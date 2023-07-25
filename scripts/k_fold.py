#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 10:09:58 2023

@author: ptruong
"""

from numpy import loadtxt
import xgboost
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


datasets = ["C1", "C2", "C3", "C4", "C5", "C6"]

dfs = []
for i, dataset in enumerate(datasets):
    df = pd.read_csv(f"{dataset}.tsv", sep="\t", index_col=0, usecols=[0, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
#    df = pd.read_csv(f"{dataset}.tsv", sep="\t", index_col=0)
    dfs.append(df)

df = pd.concat(dfs)

df = df.drop("peptide", axis = 1)

dataset = df.values

# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# CV model
model = xgboost.XGBRegressor()
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, Y, cv=kfold)

#scores = cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error')

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))













