#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 10:26:04 2023

@author: ptruong
"""

import pandas as pd 
import numpy as np



def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def add_peptide_charge(df):
    df["peptide_charge"] = df["peptide"] + "_" + df["charge"].astype(str)
    return df

def filter_on_unique_peptide_charge(df):
    df["peptide_charge"] = df["peptide"] + "_" + df["charge"].astype(str)
    unique_peptides = df.groupby(by = "peptide_charge").count().peptide == 1
    df = df[df["peptide_charge"].isin(unique_peptides.index)]
    return df

def find_unique_peptides(df):
    unique_peptides = (df.groupby(by = "peptide_charge").count().peptide == 1)
    unique_peptides = unique_peptides[unique_peptides == True]
    return unique_peptides.index

def remove_duplicated_peptide_charge(df):
    unique_peptides = find_unique_peptides(df)
    df = df[df["peptide_charge"].isin(unique_peptides)]
    return df

def get_intersecting_unlabelled_labelled_df(labeled_df, unlabeled_df):
    # Make peptide matching columns
    labeled_df = add_peptide_charge(labeled_df)
    unlabeled_df = add_peptide_charge(unlabeled_df)
    # take only intersecting peptides
    labeled_df = remove_duplicated_peptide_charge(labeled_df)
    unlabeled_df = remove_duplicated_peptide_charge(unlabeled_df)
    
    intersecting_peptides = intersection(unlabeled_df.peptide_charge, labeled_df.peptide_charge)
    
    unlabeled = unlabeled_df[unlabeled_df["peptide_charge"].isin(intersecting_peptides)].reset_index().drop("index", axis = 1)
    labeled = labeled_df[labeled_df["peptide_charge"].isin(intersecting_peptides)].reset_index().drop("index", axis = 1)

    return labeled, unlabeled

def add_peptide_charge_ecoli(ecoli_std):
    ecoli_std["peptide_charge"] = ecoli_std.Peptide + "_" + ecoli_std.Charge.astype(str)
    ecoli_std["modified_peptide_charge"] = ecoli_std["Modified Peptide"] + "_" + ecoli_std.Charge.astype(str)
    return ecoli_std

def merge_with_ecoli_predecing(df, ecoli_std):
    return df.merge(ecoli_std, left_on="preceding Ecoli peptide ID", right_on="ecoli_std_modified_peptide_charge")

def merge_with_ecoli_succeeding(df, ecoli_std):
    return df.merge(ecoli_std, left_on="succeeding Ecoli peptide ID", right_on="ecoli_std_modified_peptide_charge")

def qc(labeled_df, unlabeled_df):
    check1 = sum(labeled_df["Gravy Score"] == unlabeled_df["Gravy Score"]) == len(labeled_df)
    check2 = sum(labeled_df["ClogP"] == unlabeled_df["ClogP"])== len(labeled_df)
    check3 = sum(labeled_df["charge"] == unlabeled_df["charge"])== len(labeled_df)
    check4 = sum(labeled_df["BB Index"] == unlabeled_df["BB Index"])== len(labeled_df)
    return check1+check2+check3+check4

def log2_values(df):
    df["log2_intensity"] = np.log2(df["intensity"])
    df["log2_preceding_Ecoli_int"] = np.log2(df["preceding Ecoli int"])
    df["log2_succeeding_Ecoli_int"] = np.log2(df["succeeding Ecoli int"])
    df["ecoli_std_log2_intensity"] = np.log2(df["ecoli_std_Intensity"])
    return df


def preprocess(labeled_df, unlabeled_df, ecoli_std):
    ecoli_std = add_peptide_charge_ecoli(ecoli_std).add_prefix("ecoli_std_")
    labeled_df, unlabeled_df = get_intersecting_unlabelled_labelled_df(labeled_df, unlabeled_df)

    labeled_df = merge_with_ecoli_predecing(labeled_df, ecoli_std).set_index("peptide_charge").sort_index()
    unlabeled_df = merge_with_ecoli_predecing(unlabeled_df, ecoli_std).set_index("peptide_charge").sort_index()

    labeled_df = log2_values(labeled_df)
    unlabeled_df = log2_values(unlabeled_df)
    return labeled_df, unlabeled_df


def compute_ratio_increase_preceding(labeled_df, unlabeled_df):
    return labeled_df["intensity"] / labeled_df["preceding Ecoli int"] * \
    labeled_df["ecoli_std_Intensity"] / unlabeled_df["ecoli_std_Intensity"] * \
    unlabeled_df["preceding Ecoli int"] / unlabeled_df["intensity"]

def compute_log2_increase_preceding(labeled_df, unlabeled_df):
    return labeled_df["log2_intensity"] - labeled_df["log2_preceding_Ecoli_int"] + \
    labeled_df["ecoli_std_log2_intensity"] - unlabeled_df["ecoli_std_log2_intensity"] + \
    unlabeled_df["log2_preceding_Ecoli_int"] - unlabeled_df["log2_intensity"]

def compute_log2_ratio_increase_preceding(labeled_df, unlabeled_df):
    return labeled_df["log2_intensity"] / labeled_df["log2_preceding_Ecoli_int"] * \
    labeled_df["ecoli_std_log2_intensity"] / unlabeled_df["ecoli_std_log2_intensity"] * \
    unlabeled_df["log2_preceding_Ecoli_int"] / unlabeled_df["log2_intensity"]


def compute_ratio_increase_succeeding(labeled_df, unlabeled_df):
    return labeled_df["intensity"] / labeled_df["succeeding Ecoli int"] * \
    labeled_df["ecoli_std_Intensity"] / unlabeled_df["ecoli_std_Intensity"] * \
    unlabeled_df["succeeding Ecoli int"] / unlabeled_df["intensity"]

def compute_log2_increase_succeeding(labeled_df, unlabeled_df):
    return labeled_df["log2_intensity"] - labeled_df["log2_succeeding_Ecoli_int"] + \
    labeled_df["ecoli_std_log2_intensity"] - unlabeled_df["ecoli_std_log2_intensity"] + \
    unlabeled_df["log2_succeeding_Ecoli_int"] - unlabeled_df["log2_intensity"]

def compute_log2_ratio_increase_succeeding(labeled_df, unlabeled_df):
    return labeled_df["log2_intensity"] / labeled_df["log2_succeeding_Ecoli_int"] * \
    labeled_df["ecoli_std_log2_intensity"] / unlabeled_df["ecoli_std_log2_intensity"] * \
    unlabeled_df["log2_succeeding_Ecoli_int"] / unlabeled_df["log2_intensity"]


def compute_signal_enhancement(labeled_df, unlabeled_df):
    df = pd.DataFrame()
    # Check so that index are the same!!!
    if qc(labeled_df, unlabeled_df):
        df["Gravy Score"] = labeled_df["Gravy Score"]
        df["ClogP"] = labeled_df["ClogP"]
        df["charge"] = labeled_df["charge"]
        df["BB Index"] = labeled_df["BB Index"]
        df["unlabeled_rt"] = unlabeled_df["rt"]
        df["labeled_rt"] = labeled_df["rt"]
        df["charge"] = labeled_df["charge"]
        df["length"] = labeled_df.peptide.map(lambda x:len(x))
        
        df["preceding_ratio_increase"] = compute_ratio_increase_preceding(labeled_df, unlabeled_df)
        df["preceding_log2_increase"] = compute_log2_increase_preceding(labeled_df, unlabeled_df)
        df["preceding_log2_ratio_increase"] = compute_log2_ratio_increase_preceding(labeled_df, unlabeled_df)
        
        
        df["succeeding_ratio_increase"] = compute_ratio_increase_succeeding(labeled_df, unlabeled_df)
        df["succeeding_log2_increase"] = compute_log2_increase_succeeding(labeled_df, unlabeled_df)
        df["succeeding_log2_ratio_increase"] = compute_log2_ratio_increase_succeeding(labeled_df, unlabeled_df)
        return df
    else:
        return 0    



files = ["C1", "C2", "C3", "C4", "C5", "C6"]
ecoli_std = pd.read_csv("Processed_files/Control_files/ecoli_std.tsv", sep = "\t", index_col = 0)

dfs = []
for file in files:
    labeled_df = pd.read_csv(f"Processed_files/Labeled_files/{file}_Labeled_Hek.tsv", sep = "\t", index_col = 0)
    unlabeled_df = pd.read_csv(f"Processed_files/Unlabeled_files/{file}_Unlabeled_Hek.tsv", sep = "\t", index_col = 0)
    print(len(labeled_df))
    print(len(unlabeled_df))
    print(len(intersection(unlabeled_df.peptide, labeled_df.peptide)))
    print()
    
    labeled_df, unlabeled_df = preprocess(labeled_df, unlabeled_df, ecoli_std)
    df = compute_signal_enhancement(labeled_df, unlabeled_df)
    df["group"] = file
    dfs.append(df)

df = pd.concat(dfs)


df.to_csv("summary.tsv", sep = "\t")








