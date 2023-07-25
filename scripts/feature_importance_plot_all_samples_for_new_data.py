 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 18:02:25 2023

@author: ptruong
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import xgboost as xgb
import lightgbm as lgb

def plot_feature_importance(df, title="Feature importances", subplot_index=111, peptide_sequence_encoding=False, model = "randomForest"):

    if type(peptide_sequence_encoding) == pd.core.frame.DataFrame:
        peptide_sequence_encoding = peptide_sequence_encoding.reset_index().drop("index", axis = 1)
        X = df.drop(["preceding_log2_ratio_increase", "peptide_charge", "group"], axis=1)#.values
        X = X.reset_index().drop("index", axis = 1)
        X = pd.concat([X, peptide_sequence_encoding], axis = 1)
        feature_names = X.columns
        X = X.values
        y = df["preceding_log2_ratio_increase", "preceding_ratio_increase"].values
    else:
        X = df.drop(["preceding_log2_ratio_increase", "preceding_ratio_increase", "peptide_charge", "group"], axis=1).values
        feature_names = df.drop("preceding_log2_ratio_increase", axis=1).columns
        y = df["preceding_log2_ratio_increase"].values


    # RANDOM FOREST
    if model == "randomForest":
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


    if model == "xgboost":
        # Train the XGBoost regression model
        model = xgb.XGBRegressor(n_estimators=50)
        model.fit(X, y)

        # Get feature importances
        importances = model.feature_importances_

        indices = np.argsort(importances)[::-1]
        
        # Print the feature ranking
        print(f"Feature ranking ({title}):")
        for f in range(X.shape[1]):
            print(f"{f + 1}. {feature_names[indices[f]]} ({importances[indices[f]]})")

        # Plot the feature importances
        ax = plt.subplot(subplot_index)
        ax.bar(range(X.shape[1]), importances[indices], color="r", align="center")
        ax.set_title(title)
        ax.set_xticks(range(X.shape[1]))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation='vertical')
        ax.set_xlim([-1, X.shape[1]])
        
    if model == "lightgbm":
        # Train the LightGBM regression model
        model = lgb.LGBMRegressor(n_estimators=100)
        model.fit(X, y)
        
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Print the feature ranking
        print(f"Feature ranking ({title}):")
        for f in range(X.shape[1]):
            print(f"{f + 1}. {feature_names[indices[f]]} ({importances[indices[f]]})")

        # Plot the feature importances
        ax = plt.subplot(subplot_index)
        ax.bar(range(X.shape[1]), importances[indices], color="r", align="center")
        ax.set_title(title)
        ax.set_xticks(range(X.shape[1]))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation='vertical')
        ax.set_xlim([-1, X.shape[1]])



# Print feature importance scores
for feature, importance in zip(features.columns, importances):
    print(f"{feature}: {importance}")

# Create a 2x3 grid for subplots
fig, axs = plt.subplots(1, 1, figsize=(12, 8))

datasets = ["C1", "C2", "C3", "C4", "C5", "C6"]

dfs = []
for i, dataset in enumerate(datasets):
    df = pd.read_csv(f"{dataset}.tsv", sep="\t", index_col=0, usecols=[0, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
#    df = pd.read_csv(f"{dataset}.tsv", sep="\t", index_col=0)
    dfs.append(df)

df = pd.concat(dfs)

df = pd.read_csv("summary.tsv", sep = "\t", usecols = [0,1,2,3,4,5,6,7,8,9,10,14])
#df = pd.read_csv("summary.tsv", sep = "\t", usecols = [0,1,2,3,4,5,6,7,8,9,10,14])3
#df = pd.read_csv("summary.tsv", sep = "\t", usecols = [0,1,2,3,4,5,6,7,8,9,10,14])
df.isna().sum()


# one-hot-encoded-sequences.
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def one_hot_encoding_aminoacid(df):
    one_hot_encoded_sequences = pd.get_dummies(df.peptide_charge, prefix="seq")
    return one_hot_encoded_sequences
    
def one_hot_encoding_count(df):
    return pd.DataFrame([ProteinAnalysis(i).count_amino_acids() for i in df["peptide_charge"]])

def one_hot_encoding_positional(df):
    byposition = df['peptide_charge'].apply(lambda x:pd.Series(list(x)))
    return pd.get_dummies(byposition)


by_count = one_hot_encoding_count(df)
by_position = one_hot_encoding_positional(df)
one_hot_encoded_sequences = one_hot_encoding_aminoacid(df)



####

plot_feature_importance(df, title="All C", subplot_index=111, peptide_sequence_encoding=False)
# Adjust the spacing between subplots
plt.tight_layout()
# Show the plot
plt.show()

# peptide_sequence by count
plot_feature_importance(df, title="All C", subplot_index=111, peptide_sequence_encoding=by_count)
# Adjust the spacing between subplots
plt.tight_layout()
# Show the plot
plt.show()

# peptide sequence by aminoacid
plot_feature_importance(df, title="All C", subplot_index=111, peptide_sequence_encoding=one_hot_encoded_sequences)
# Adjust the spacing between subplots
plt.tight_layout()
# Show the plot
plt.show()

# peptide sequence by position
plot_feature_importance(df, title="All C", subplot_index=111, peptide_sequence_encoding=by_position)
# Adjust the spacing between subplots
plt.tight_layout()
# Show the plot
plt.show()

# peptide sequence by position
plot_feature_importance(df, title="All C", subplot_index=111, peptide_sequence_encoding=False, model = "xgboost")
# Adjust the spacing between subplots
plt.tight_layout()
# Show the plot
plt.show()


# peptide sequence by position
plot_feature_importance(df, title="All C", subplot_index=111, peptide_sequence_encoding=False, model = "lightgbm")
# Adjust the spacing between subplots
plt.tight_layout()
# Show the plot
plt.show()



