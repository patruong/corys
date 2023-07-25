#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 16:15:06 2023

@author: ptruong
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def correlation_heatmap(df):
    # simple correlations
    corr_matrix = df.corr()
    # Plot the heatmap
    plt = sns.heatmap(data=corr_matrix, annot=True, cmap='coolwarm')
    
        
    # Angle the x-axis labels by 45 degrees
    plt.set_xticklabels(plt.get_xticklabels(), rotation=20)
    
    # Angle the y-axis labels by 45 degrees
    plt.set_yticklabels(plt.get_yticklabels(), rotation=0)
    
    # Set plot title
    plt.title('Correlation Matrix Heatmap')
    plt.savefig('correlation_heatmap.png')
    # Display the plot
    plt.show()
    
def correlation_heatmap_group(df, group = "C1"):
    # simple correlations
    corr_matrix = df[df["group"] == "C1"].corr()
    # Plot the heatmap
    sns.heatmap(data=corr_matrix, annot=True, cmap='coolwarm')
    # Set plot title
    plt.title('Correlation Matrix Heatmap')
    # Display the plot
    plt.show()


df = pd.read_csv("summary.tsv", sep = "\t")
#df.set_index("peptide_charge", inplace = True)

# plot histogram
sns.kdeplot(data=df, x = "preceding_log2_ratio_increase", hue="group")
plt.title('Preceding_log2_ratio_increase')
plt.savefig('preceding_log2_ratio_increase.png')

sns.kdeplot(data=df, x = "preceding_log2_increase", hue="group")
plt.title('Preceding_log2_increase')
plt.savefig('preceding_log2_increase.png')

sns.kdeplot(data=df, x = "succeeding_log2_ratio_increase", hue="group")
plt.title('succeeding_log2_ratio_increase')
plt.savefig('succeeding_log2_ratio_increase.png')


sns.kdeplot(data=df, x = "succeeding_log2_increase", hue="group")
plt.title('Succeeding_log2_increase')
plt.savefig('succeeding_log2_increase.png')

correlation_heatmap(df)


corr












