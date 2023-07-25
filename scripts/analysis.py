#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 15:40:40 2023

@author: ptruong
"""

import pandas as pd 
import numpy as np 
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


def compute_eval_metrics(X, y, n_estimators=50):
    # Assuming you have your data and target variables in X and y
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # CV model
    model = xgb.XGBRegressor()
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    results = cross_val_score(model, X_train, y_train, cv=kfold)
    
    
    print("Cross-Validation Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    
    # Fit the model on the entire training set
    model = xgb.XGBRegressor(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    
    # Evaluate on the test set
    test_accuracy = model.score(X_test, y_test)
    print("Test Set Accuracy (R-squared): %.2f%%" % (test_accuracy * 100))
    
    cv_accuracy, cv_std, test_accuracy = results.mean()*100, results.std()*100,(test_accuracy * 100)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Compute evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
    r2 = r2_score(y_test, y_pred)
    
    print()
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R-squared:", r2)
    
    return cv_accuracy, cv_std, test_accuracy

def grid_search_CV(X, y):
    # Assuming you have your data and target variables in X and y
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # CV model with n_estimators as a hyperparameter to tune
    model = xgb.XGBRegressor()
    
    # Define a range of values for n_estimators to search over
    param_grid = {
        'n_estimators': [10, 20, 30, 40, 50, 100, 150, 200, 250, 300]
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Get the best value of n_estimators from the grid search
    best_n_estimators = grid_search.best_params_['n_estimators']
    
    print("Best n_estimators:", best_n_estimators)
    
    # Fit the XGBoost model with the best n_estimators on the entire training set
    model = xgb.XGBRegressor(n_estimators=best_n_estimators)
    model.fit(X_train, y_train)
    
    # Evaluate on the test set
    test_accuracy = model.score(X_test, y_test)
    print("Test Set Accuracy with Best n_estimators: %.2f%%" % (test_accuracy * 100))
    return best_n_estimators
    





df = pd.read_csv("summary_derived.tsv", sep = "\t")
df.drop(["peptide_charge", "group"], axis = 1)
# Note we need to exclude all the ratios used to directly compute the signal enhancement such as
#        'deriv_labeled_intensity_preceding_HEK_EColi_ratio',
#        'deriv_labeled_intensity_preceding_HEK_EColi_log2_ratio',
#        'deriv_unlabeled_intensity_preceding_HEK_EColi_ratio',
#        'deriv_unlabeled_intensity_preceding_HEK_EColi_log2_ratio',

preceding_cols = [ 'preceding_log2_ratio_increase',
        'BB Index', 'ClogP', 'Gravy Score', 'charge',
        'labeled_#_of_carbon_added', 'labeled_label_frequency',  'labeled_log2_intensity',
        'labeled_mass', 'labeled_mz', 'labeled_preceding_ecoli_log2_intensity',
        'labeled_preceding_ecoli_rt', 'labeled_rt', 'length',
        'unlabeled_log2_intensity', 'unlabeled_mass', 'unlabeled_mz',
        'unlabeled_preceding_ecoli_log2_intensity', 'unlabeled_preceding_ecoli_rt',
        'unlabeled_rt', 
        'deriv_within_unlabeled_preceding_delta_rt',
        'deriv_within_unlabeled_preceding_rt_ratio',
        'deriv_preceding_delta_rt',
        'deriv_preceding_ratio_rt',
        'deriv_delta_rt',
        'deriv_ratio_rt',
        'deriv_ratio_mass',
        'deriv_delta_mass',
        'deriv_ratio_mz',
        'deriv_delta_mz']        


succeeding_cols = [i.replace('preceding', 'succeeding') for i in preceding_cols]
      
preceding_df = df[preceding_cols]
succeeding_df = df[succeeding_cols]

df = preceding_df

# split data into X (features) and y (target)
X = df.values[:,1:]
y = df.values[:,0]

















cv_accuracy, cv_std, test_accuracy, mae, mse, rmse, r2 = compute_eval_metrics(X,y)

model = xgb.XGBRegressor(n_estimators=50)
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(len(preceding_df.columns)-1):
    print(f"{f + 1}. {df.columns[indices[f]]} ({importances[indices[f]]})")

title = "test"

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Plot the feature importances
ax.bar(range(X.shape[1]), importances[indices], color="r", align="center")
ax.set_title(title + (" Accuracy: %.2f%% (std : %.2f%%)" % (results.mean()*100, results.std()*100)))
ax.set_xticks(range(X.shape[1]))
ax.set_xticklabels([df.columns[i] for i in indices], rotation='vertical')
ax.set_xlim([-1, X.shape[1]])



