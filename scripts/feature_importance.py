#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 12:15:10 2023

@author: ptruong
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression


df = pd.read_csv("C1.tsv", sep = "\t", index_col = 0, usecols=[0,4,5,11,12,13,14,15,16,17,18,19,20,21])
#df = pd.read_csv("C1.tsv", sep = "\t", index_col = 0)


feature_names = df.drop("log10_lu_ratio", axis = 1).columns

X = df.drop("log10_lu_ratio", axis = 1).values
y = df["log10_lu_ratio"].values

# Create a random forest regressor
forest = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the random forest regressor
forest.fit(X, y)

# Calculate feature importances
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Define the index names
#feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5',
#                 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']

# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. {feature_names[indices[f]]} ({importances[indices[f]]})")

# Plot the feature importances
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices])  # Set the index names
plt.xlim([-1, X.shape[1]])
plt.show()





