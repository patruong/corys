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
    check5 = sum(unlabeled_df["positive peptides"] == labeled_df["positive peptides"]) == len(labeled_df)
    check6 = sum(unlabeled_df["negative peptides"] == labeled_df["negative peptides"]) == len(labeled_df)
    #check7 = sum(unlabeled_df["# of carbon added"] == labeled_df["# of carbon added"]) == len(labeled_df)
    #check7 = sum(unlabeled_df["label_frequency"] == labeled_df["label_frequency"]) == len(labeled_df)

    return check1+check2+check3+check4+check5+check6

def log2_values(df):
    df["log2_intensity"] = np.log2(df["intensity"])
    df["log2_preceding_Ecoli_int"] = np.log2(df["preceding Ecoli int"])
    df["log2_succeeding_Ecoli_int"] = np.log2(df["succeeding Ecoli int"])
    df["ecoli_std_log2_intensity"] = np.log2(df["ecoli_std_Intensity"])
    return df


def preprocess(labeled_df, unlabeled_df, ecoli_std):
    ecoli_std = add_peptide_charge_ecoli(ecoli_std).add_prefix("ecoli_std_")
    #labeled_df, unlabeled_df = get_intersecting_unlabelled_labelled_df(labeled_df, unlabeled_df)

    labeled_df = merge_with_ecoli_predecing(labeled_df, ecoli_std).set_index("peptide+charge").sort_index()
    unlabeled_df = merge_with_ecoli_predecing(unlabeled_df, ecoli_std).set_index("peptide+charge").sort_index()

    labeled_df = log2_values(labeled_df).reset_index()
    unlabeled_df = log2_values(unlabeled_df).reset_index()
    return labeled_df, unlabeled_df


# CHANGE ALL THESE TO FOR-LOOPS ####

def compute_ratio_increase_preceding(labeled_df, unlabeled_df):
    
    signal_enhancements = []
    for peptide_charge in unlabeled_df["peptide+charge"].unique():
        if len(labeled_df[labeled_df["peptide+charge"] == peptide_charge]) > 1:
            for i in range(len(unlabeled_df[unlabeled_df["peptide+charge"] == peptide_charge])):
                if not unlabeled_df[unlabeled_df["peptide+charge"] == peptide_charge].empty:
                    if np.shape(unlabeled_df[unlabeled_df["peptide+charge"] == peptide_charge])[0] == 1:
                        val_unlabeled = unlabeled_df[unlabeled_df["peptide+charge"] == peptide_charge].iloc[0].copy() # the unlabeled always has only one value!
                        val_labeled = labeled_df[labeled_df["peptide+charge"] == peptide_charge].iloc[i].copy() 
                        
                        val = val_labeled["intensity"] / val_labeled["preceding Ecoli int"] * \
                              val_labeled["ecoli_std_Intensity"] / val_unlabeled["ecoli_std_Intensity"] * \
                              val_unlabeled["preceding Ecoli int"] / val_unlabeled["intensity"]
                        signal_enhancements.append(val)
                    elif np.shape(unlabeled_df[unlabeled_df["peptide+charge"] == peptide_charge])[0] > 1:
                        raise Exception("More than 1 peptide in the unlabeled file")
                    else:
                        raise Exception(f"Error check unlabeled_df[unlabeled_df['peptide+charge'] == peptide_charge] for {peptide_charge}")
    labeled_df
                


#    unlabeled_df[unlabeled_df.index == peptide_charge].
    
    
    
#    return labeled_df["intensity"] / labeled_df["preceding Ecoli int"] * \
#    labeled_df["ecoli_std_Intensity"] / unlabeled_df["ecoli_std_Intensity"] * \
#    unlabeled_df["preceding Ecoli int"] / unlabeled_df["intensity"]

def compute_log2_increase_preceding(labeled_df, unlabeled_df):
#    return labeled_df["log2_intensity"] - labeled_df["log2_preceding_Ecoli_int"] + \
#    labeled_df["ecoli_std_log2_intensity"] - unlabeled_df["ecoli_std_log2_intensity"] + \
#    unlabeled_df["log2_preceding_Ecoli_int"] - unlabeled_df["log2_intensity"]

def compute_log2_ratio_increase_preceding(labeled_df, unlabeled_df):
#    return labeled_df["log2_intensity"] / labeled_df["log2_preceding_Ecoli_int"] * \
#    labeled_df["ecoli_std_log2_intensity"] / unlabeled_df["ecoli_std_log2_intensity"] * \
#    unlabeled_df["log2_preceding_Ecoli_int"] / unlabeled_df["log2_intensity"]


def compute_ratio_increase_succeeding(labeled_df, unlabeled_df):
#    return labeled_df["intensity"] / labeled_df["succeeding Ecoli int"] * \
#    labeled_df["ecoli_std_Intensity"] / unlabeled_df["ecoli_std_Intensity"] * \
#    unlabeled_df["succeeding Ecoli int"] / unlabeled_df["intensity"]

def compute_log2_increase_succeeding(labeled_df, unlabeled_df):
#    return labeled_df["log2_intensity"] - labeled_df["log2_succeeding_Ecoli_int"] + \
#    labeled_df["ecoli_std_log2_intensity"] - unlabeled_df["ecoli_std_log2_intensity"] + \
#    unlabeled_df["log2_succeeding_Ecoli_int"] - unlabeled_df["log2_intensity"]

def compute_log2_ratio_increase_succeeding(labeled_df, unlabeled_df):
#    return labeled_df["log2_intensity"] / labeled_df["log2_succeeding_Ecoli_int"] * \
#    labeled_df["ecoli_std_log2_intensity"] / unlabeled_df["ecoli_std_log2_intensity"] * \
#    unlabeled_df["log2_succeeding_Ecoli_int"] / unlabeled_df["log2_intensity"]

#########################

def compute_signal_enhancement(labeled_df, unlabeled_df):
    df = pd.DataFrame()
    # Check so that index are the same!!!
    if qc(labeled_df, unlabeled_df) == 6:
        df["Gravy Score"] = labeled_df["Gravy Score"]
        df["ClogP"] = labeled_df["ClogP"]
        df["charge"] = labeled_df["charge"]
        df["BB Index"] = labeled_df["BB Index"]
        
        # rt
        df["unlabeled_rt"] = unlabeled_df["rt"]
        df["labeled_rt"] = labeled_df["rt"]
        
        df["unlabeled_preceding_ecoli_rt"] = unlabeled_df["preceding Ecoli rt"]
        df["labeled_preceding_ecoli_rt"] = labeled_df["preceding Ecoli rt"]
        
        df["unlabeled_succeeding_ecoli_rt"] = unlabeled_df["succeeding Ecoli rt"]
        df["labeled_succeeding_ecoli_rt"] = labeled_df["succeeding Ecoli rt"]
        
        # features from labeled_df
        df["labeled_label_frequency"] = labeled_df["label_frequency"]
        df["labeled_label_frequency"] = df.labeled_label_frequency.map(lambda x:float(x.replace(",",".")))

        df["labeled_#_of_carbon_added"] = labeled_df["# of carbon added"]

        # mass and mz
        df["unlabeled_mass"] = unlabeled_df["mass"]
        df["labeled_mass"] = labeled_df["mass"]
        
        df["unlabeled_mz"] = unlabeled_df["mz"]
        df["labeled_mz"] = labeled_df["mz"]
        
        # intensity
        df["unlabeled_intensity"] = unlabeled_df["intensity"]
        df["labeled_intensity"] = labeled_df["intensity"]
        
        df["unlabeled_log2_intensity"] = unlabeled_df["log2_intensity"]
        df["labeled_log2_intensity"] = labeled_df["log2_intensity"]
         
        df["unlabeled_preceding_ecoli_log2_intensity"] = unlabeled_df["log2_preceding_Ecoli_int"]
        df["labeled_preceding_ecoli_log2_intensity"] = labeled_df["log2_preceding_Ecoli_int"]
        
        df["unlabeled_succeeding_ecoli_log2_intensity"] = unlabeled_df["log2_succeeding_Ecoli_int"]
        df["labeled_succeeding_ecoli_log2_intensity"] = labeled_df["log2_succeeding_Ecoli_int"]        
        
        df["unlabeled_preceding_ecoli_intensity"] = unlabeled_df["preceding Ecoli int"]
        df["labeled_preceding_ecoli_intensity"] = labeled_df["preceding Ecoli int"]
 
        df["unlabeled_succeeding_ecoli_intensity"] = unlabeled_df["succeeding Ecoli int"]
        df["labeled_succeeding_ecoli_intensity"] = labeled_df["succeeding Ecoli int"]
 
    
        df["charge"] = labeled_df["charge"]
        df["length"] = labeled_df.peptide.map(lambda x:len(x))
        
        
        # below here we need to modify
        df["preceding_ratio_increase"] = compute_ratio_increase_preceding(labeled_df, unlabeled_df)
        df["preceding_log2_increase"] = compute_log2_increase_preceding(labeled_df, unlabeled_df)
        df["preceding_log2_ratio_increase"] = compute_log2_ratio_increase_preceding(labeled_df, unlabeled_df)
        
        
        df["succeeding_ratio_increase"] = compute_ratio_increase_succeeding(labeled_df, unlabeled_df)
        df["succeeding_log2_increase"] = compute_log2_increase_succeeding(labeled_df, unlabeled_df)
        df["succeeding_log2_ratio_increase"] = compute_log2_ratio_increase_succeeding(labeled_df, unlabeled_df)
        return df
    else:
        return 0    


def add_derived_ratios(df):
    # derived ratios 
    
    # intensity within
    # labeled HEK / E.Coli
    df["deriv_labeled_intensity_preceding_HEK_EColi_ratio"] = df["labeled_intensity"] / df["labeled_preceding_ecoli_intensity"]
    df["deriv_labeled_intensity_succeeding_HEK_EColi_ratio"] = df["labeled_intensity"] / df["labeled_succeeding_ecoli_intensity"]
    df["deriv_labeled_intensity_preceding_HEK_EColi_log2_ratio"] = df["labeled_log2_intensity"] / df["labeled_preceding_ecoli_log2_intensity"]
    df["deriv_labeled_intensity_succeeding_HEK_EColi_log2_ratio"] = df["labeled_log2_intensity"] / df["labeled_succeeding_ecoli_log2_intensity"]
    
    # unlabeled E.Coli / HEK
    df["deriv_unlabeled_intensity_preceding_HEK_EColi_ratio"] = df["unlabeled_intensity"] / df["unlabeled_preceding_ecoli_intensity"]
    df["deriv_unlabeled_intensity_succeeding_HEK_EColi_ratio"] = df["unlabeled_intensity"] / df["unlabeled_succeeding_ecoli_intensity"]
    df["deriv_unlabeled_intensity_preceding_HEK_EColi_log2_ratio"] = df["unlabeled_log2_intensity"] / df["unlabeled_preceding_ecoli_log2_intensity"]
    df["deriv_unlabeled_intensity_succeeding_HEK_EColi_log2_ratio"] = df["unlabeled_log2_intensity"] / df["unlabeled_succeeding_ecoli_log2_intensity"]
    
    # rt within
    df["deriv_within_unlabeled_preceding_delta_rt"] = df["unlabeled_rt"] - df["unlabeled_preceding_ecoli_rt"]
    df["deriv_within_unlabeled_preceding_rt_ratio"] = df["unlabeled_rt"] / df["unlabeled_preceding_ecoli_rt"]
    
    df["deriv_within_unlabeled_succeeding_delta_rt"] = df["unlabeled_rt"] - df["unlabeled_succeeding_ecoli_rt"]
    df["deriv_within_unlabeled_succeeding_rt_ratio"] = df["unlabeled_rt"] / df["unlabeled_succeeding_ecoli_rt"]
    
    df["deriv_within_unlabeled_preceding_delta_rt"] = df["labeled_rt"] - df["labeled_preceding_ecoli_rt"]
    df["deriv_within_unlabeled_preceding_rt_ratio"] = df["labeled_rt"] / df["labeled_preceding_ecoli_rt"]
    
    df["deriv_within_unlabeled_succeeding_delta_rt"] = df["labeled_rt"] - df["labeled_succeeding_ecoli_rt"]
    df["deriv_within_unlabeled_succeeding_rt_ratio"] = df["labeled_rt"] / df["labeled_succeeding_ecoli_rt"]
    
    # rt between
    df["deriv_preceding_delta_rt"] = df["labeled_preceding_ecoli_rt"] - df["unlabeled_preceding_ecoli_rt"]
    df["deriv_preceding_ratio_rt"] = df["labeled_preceding_ecoli_rt"] / df["unlabeled_preceding_ecoli_rt"]
    
    df["deriv_succeeding_delta_rt"] = df["labeled_succeeding_ecoli_rt"] - df["unlabeled_succeeding_ecoli_rt"]
    df["deriv_succeeding_ratio_rt"] = df["labeled_succeeding_ecoli_rt"] / df["unlabeled_succeeding_ecoli_rt"]
    
    df["deriv_delta_rt"] = df["labeled_rt"] - df["unlabeled_rt"]
    df["deriv_ratio_rt"] = df["labeled_rt"] / df["unlabeled_rt"]
   
    # mass
    df["deriv_ratio_mass"] = df["labeled_mass"] / df["unlabeled_mass"]
    df["deriv_delta_mass"] = df["labeled_mass"] - df["unlabeled_mass"]

    # mz
    df["deriv_ratio_mz"] = df["labeled_mz"] / df["unlabeled_mz"]
    df["deriv_delta_mz"] = df["labeled_mz"] - df["unlabeled_mz"]
    
    
    return df

def count_overlapping_peptides(labeled_df, unlabeled_df):
    df = pd.concat([unlabeled_df, labeled_df], axis = 0)
    cols = ["peptide+charge", "labeled_df_count", "unlabeled_df_count"]
    peptide_charge_row = []
    labeled_df_count_row = []
    unlabeled_df_count_row = []
    for peptide_charge in df["peptide+charge"].unique():
        peptide_charge_row.append(peptide_charge)
        labeled_df_count_row.append(len(labeled_df[labeled_df["peptide+charge"] == peptide_charge]))
        unlabeled_df_count_row.append(len(unlabeled_df[unlabeled_df["peptide+charge"] == peptide_charge]))
    res = pd.DataFrame([peptide_charge_row, labeled_df_count_row, unlabeled_df_count_row], index = cols).T
    res["overlap"] = (res["labeled_df_count"] > 0)*(res["unlabeled_df_count"] > 0)
    return res


files = ["C1", "C2", "C3", "C4", "C5", "C6"]
ecoli_std = pd.read_csv("Processed_files/Control_files/ecoli_std.tsv", sep = "\t", index_col = 0)

dfs = []
overlap_dfs = []
for file in files:
    labeled_df = pd.read_csv(f"Processed_files/Labeled_files/{file}_Labeled_Hek.tsv", sep = "\t", index_col = 0)
    unlabeled_df = pd.read_csv(f"Processed_files/Unlabeled_files/{file}_Unlabeled_Hek.tsv", sep = "\t", index_col = 0)
    print(len(labeled_df))
    print(len(unlabeled_df))
    print(len(intersection(unlabeled_df.peptide, labeled_df.peptide)))
    print()
    
    
    count_overlap_df = count_overlapping_peptides(labeled_df, unlabeled_df)
    count_overlap_df["group"] = file 
    overlap_dfs.append(count_overlap_df)
    
    labeled_df, unlabeled_df = preprocess(labeled_df, unlabeled_df, ecoli_std)
    
    
    
    df = compute_signal_enhancement(labeled_df, unlabeled_df) # work here
    df["group"] = file
    dfs.append(df)


### new code for merging unlabeled and labeled df

for peptide_charge in unlabeled_df["peptide+charge"].unique():
    print(len(labeled_df[labeled_df["peptide+charge"] == peptide_charge]))


### make one dataframe
### adapt the rest of the code to that one dataframe
### run the analysis scripts



df = pd.concat(dfs)
df = df.reindex(sorted(df.columns), axis=1)

overlap_dfs = pd.concat(overlap_dfs)
overlap_dfs.to_csv("overlapping_peptides.tsv", sep = "\t")



df.to_csv("summary.tsv", sep = "\t")
df = add_derived_ratios(df)
df.to_csv("summary_derived.tsv", sep = "\t")









