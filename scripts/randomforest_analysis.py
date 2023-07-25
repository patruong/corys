#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 19:32:06 2023

@author: ptruong
"""
import os
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


def compute_eval_metrics(X, y, n_estimators=50):
    # Assuming you have your data and target variables in X and y
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # CV model
    model = RandomForestRegressor()
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    results = cross_val_score(model, X_train, y_train, cv=kfold)
    
    
    print("Cross-Validation Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    
    # Fit the model on the entire training set
    model = RandomForestRegressor(n_estimators=n_estimators)
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
    
    return cv_accuracy, cv_std, test_accuracy, mae, mse, rmse, r2

def grid_search_CV(X, y, n_estimators = [100, 150, 200, 250, 300]):
    # Assuming you have your data and target variables in X and y
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # CV model with n_estimators as a hyperparameter to tune
    model = RandomForestRegressor()
    
    # Define a range of values for n_estimators to search over
    param_grid = {
        'n_estimators': n_estimators
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Get the best value of n_estimators from the grid search
    best_n_estimators = grid_search.best_params_['n_estimators']
    
    print("Best n_estimators:", best_n_estimators)
    
    # Fit the RandomForest model with the best n_estimators on the entire training set
    model = RandomForestRegressor(n_estimators=best_n_estimators)
    model.fit(X_train, y_train)
    
    # Evaluate on the test set
    test_accuracy = model.score(X_test, y_test)
    print("Test Set Accuracy with Best n_estimators: %.2f%%" % (test_accuracy * 100))
    return best_n_estimators

def scatter_actual_vs_prediction(X, y, n_estimators, title ='RandomForest Regression - Actual vs. Predicted log2 signal enhancement',
                                 save_path = "RandomForest_regression.png"):
    
    cv_accuracy, cv_std, test_accuracy, mae, mse, rmse, r2 = compute_eval_metrics(X,y,n_estimators=n_estimators)

    # Assuming you have your data and target variables in X and y
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit the RandomForest model
    model = RandomForestRegressor(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Compute evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
    r2 = r2_score(y_test, y_pred)
    
    # Create a matplotlib plot
    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('Actual Values', fontsize=18, fontweight='bold')
    plt.ylabel('Predicted Values', fontsize=18, fontweight='bold')
    plt.title(title, fontsize=24, fontweight='bold')
    plt.grid(True)
    
    # Add regression line
    regression_line = np.polyfit(y_test, y_pred, 1)
    plt.plot(y_test, np.polyval(regression_line, y_test), color='red')
    
    
    # Add textbox with the evaluation metrics
    metrics_text = f"""
    n_estimators: {n_estimators}
    Cross-Validation Accuracy: {cv_accuracy/100:.2%} ({cv_std/100:.2%})
    Test Set Accuracy (R-squared): {r2:.2%}
    
    Mean Absolute Error: {mae:.5f}
    Mean Squared Error: {mse:.5f}
    Root Mean Squared Error: {rmse:.5f}
    R-squared: {r2:.5f}"""
    
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Increase tick label font size
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.tight_layout()  # Adjust subplot parameters to fit the plot elements within the figure

    plt.savefig(save_path, dpi=300)  # Save the plot as a PNG file
    plt.show()
    
    
def plot_feature_importance_bar_plot(df, X,y,n_estimators, title = 'RandomForest Regression - Preceding feature importance',
                                     save_path = "RandomForest_feature_importance.png"):
    model = RandomForestRegressor(n_estimators=n_estimators)
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    
    indices = np.argsort(importances)[::-1]
    
    rank_series = []
    feature_series = []
    importance_series = []
    for f in range(len(df.iloc[:,1:].columns)):
        print(f"{f + 1}. {df.columns[indices[f]]} ({importances[indices[f]]})")
        rank = f + 1
        feature = df.iloc[:,1:].columns[indices[f]]
        weight = importances[indices[f]]
        rank_series.append(rank)
        feature_series.append(feature)
        importance_series.append(weight)
    weight_df = pd.DataFrame([rank_series, feature_series, importance_series], index = ["rank", "feature", "weight"]).T

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot the feature importances
    ax.bar(range(X.shape[1]), importances[indices], color="r", align="center")
    ax.set_title(title, fontsize=24, fontweight='bold')
    ax.set_xticks(range(X.shape[1]))
    ax.set_xticklabels([df.columns[i+1] for i in indices], rotation='vertical', fontsize=14, fontweight='bold')
    ax.set_xlim([-1, X.shape[1]])
    
    # Increase tick label font size
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.tight_layout()  # Adjust subplot parameters to fit the plot elements within the figure

    plt.savefig(save_path, dpi=300)  # Save the plot as a PNG file
    plt.show()  # Show the plot (optional, you can remove this line if you only want to save the plot without displaying it)
    weight_df.to_csv(save_path+".tsv", sep = "\t", index = False)

def generate_randomforest_plots(df,
                                scatter_title='RandomForest Regression - Actual vs. Predicted log2 signal enhancement',
                                bar_title='RandomForest Regression - Preceding feature importance',
                                scatter_save_path="RandomForest_regression.png",
                                bar_save_path="RandomForest_feature_importance.png"):
    # split data into X (features) and y (target)
    X = df.values[:, 1:]
    y = df.values[:, 0]
    
    best_n_estimator = grid_search_CV(X, y)  # Finds the best n_estimator using gridSearchCV
    cv_accuracy, cv_std, test_accuracy, mae, mse, rmse, r2 = compute_eval_metrics(X, y, n_estimators=best_n_estimator)
    scatter_actual_vs_prediction(X, y, n_estimators=best_n_estimator, title=scatter_title, save_path=scatter_save_path)
    plot_feature_importance_bar_plot(df, X, y, n_estimators=best_n_estimator, title=bar_title,
                                     save_path=bar_save_path)

# Note we need to exclude all the ratios used to directly compute the signal enhancement such as
#        'deriv_labeled_intensity_preceding_HEK_EColi_ratio',
#        'deriv_labeled_intensity_preceding_HEK_EColi_log2_ratio',
#        'deriv_unlabeled_intensity_preceding_HEK_EColi_ratio',
#        'deriv_unlabeled_intensity_preceding_HEK_EColi_log2_ratio',
#        'unlabeled_preceding_ecoli_log2_intensity', 
#        'unlabeled_log2_intensity', 
#        'labeled_preceding_ecoli_log2_intensity',
#        'labeled_log2_intensity',

preceding_cols = ['preceding_log2_ratio_increase',
                  'BB Index', 'ClogP', 'Gravy Score', 'charge',
                  'labeled_#_of_carbon_added', 'labeled_label_frequency',
                  'labeled_mass', 'labeled_mz',
                  'labeled_preceding_ecoli_rt', 'labeled_rt', 'length',
                  'unlabeled_mass', 'unlabeled_mz',
                  'unlabeled_preceding_ecoli_rt',
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


if __name__ == "__main__":
    df = pd.read_csv("summary_derived.tsv", sep="\t")
    df.drop(["peptide_charge", "group"], axis=1)
    
    succeeding_cols = [i.replace('preceding', 'succeeding') for i in preceding_cols]
    preceding_cols_no_deriv = [col for col in preceding_cols if not col.startswith("deriv_")]
    succeeding_cols_no_deriv = [col for col in succeeding_cols if not col.startswith("deriv_")]
      
    preceding_df = df[preceding_cols]
    succeeding_df = df[succeeding_cols].dropna()
    
    preceding_df_no_deriv = df[preceding_cols_no_deriv]
    succeeding_df_no_deriv = df[succeeding_cols_no_deriv].dropna()
    
    # generate plots from preceding ratios
    generate_randomforest_plots(df=preceding_df,
                                scatter_title='RandomForest Regression - Preceding signal enhancement',
                                bar_title='RandomForest Regression - Preceding feature importance',
                                scatter_save_path="randomforest/RandomForest_regression_preceding.png",
                                bar_save_path="randomforest/RandomForest_feature_importance_preceding.png")
    
    generate_randomforest_plots(df=preceding_df_no_deriv,
                                scatter_title='RandomForest Regression - Preceding signal enhancement (no deriv)',
                                bar_title='RandomForest Regression - Preceding feature importance (no deriv)',
                                scatter_save_path="randomforest/RandomForest_regression_preceding(no_deriv).png",
                                bar_save_path="randomforest/RandomForest_feature_importance_preceding(no_deriv).png")
    
    # generate plots from succeeding ratios
    generate_randomforest_plots(df=succeeding_df,
                                scatter_title='RandomForest Regression - Succeeding signal enhancement',
                                bar_title='RandomForest Regression - Succeeding feature importance',
                                scatter_save_path="randomforest/RandomForest_regression_succeeding.png",
                                bar_save_path="randomforest/RandomForest_feature_importance_succeeding.png")
    
    generate_randomforest_plots(df=succeeding_df_no_deriv,
                                scatter_title='RandomForest Regression - Succeeding signal enhancement (no deriv)',
                                bar_title='RandomForest Regression - Succeeding feature importance (no deriv)',
                                scatter_save_path="randomforest/RandomForest_regression_succeeding(no_deriv).png",
                                bar_save_path="randomforest/RandomForest_feature_importance_succeeding(no_deriv).png")
