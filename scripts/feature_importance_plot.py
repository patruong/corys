#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:33:23 2023

@author: ptruong
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

def plot_feature_importance(df, title="Feature importances", subplot_index=111):
    feature_names = df.drop("log10_lu_ratio", axis=1).columns

    X = df.drop("log10_lu_ratio", axis=1).values
    y = df["log10_lu_ratio"].values

    # Create a random forest regressor
    forest = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the random forest regressor
    forest.fit(X, y)

    # Calculate feature importances
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print(f"Feature ranking ({title}):")
    for f in range(X.shape[1]):
        print(f"{f + 1}. {feature_names[indices[f]]} ({importances[indices[f]]})")

    # Plot the feature importances
    ax = plt.subplot(subplot_index)
    ax.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    ax.set_title(title)
    ax.set_xticks(range(X.shape[1]))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation='vertical')
    ax.set_xlim([-1, X.shape[1]])


# Create a 2x3 grid for subplots
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

datasets = ["C1", "C2", "C3", "C4", "C5", "C6"]

for i, dataset in enumerate(datasets):
    df = pd.read_csv(f"{dataset}.tsv", sep="\t", index_col=0, usecols=[0, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    subplot_index = 231 + i  # Calculate subplot index
    plot_feature_importance(df, title=dataset, subplot_index=subplot_index)

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()