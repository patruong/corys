#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:59:11 2023

@author: ptruong
"""

import pandas as pd 
import numpy as np 
import seaborn as sns



df = pd.read_csv("summary_derived.tsv", sep = "\t")
df.columns
df = df.drop(["peptide_charge", "group"], axis = 1)

# Assuming your DataFrame is named `df`

# Get the list of features excluding 'preceding_log2_increase'
features = [col for col in df.columns if col != 'preceding_log2_increase']

# Set the figure size for the plots
plt.figure(figsize=(12, 8))


# Loop through each feature and create regplot with 'preceding_log2_increase'
for feature in features:
    sns.regplot(x='preceding_log2_increase', y=feature, data=df, scatter_kws={'s': 10}, line_kws={'color': 'red'})
    plt.title(f"Regression plot for 'preceding_log2_increase' and '{feature}'")
    plt.xlabel("preceding_log2_increase")
    plt.ylabel(feature)
    
    # Save the plot as a high-resolution PNG file
    plt.savefig(f"regplot_{feature}.png", dpi=300, bbox_inches='tight')
    
    # Clear the current plot to prepare for the next one
    plt.clf()



