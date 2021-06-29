# -*- coding: utf-8 -*-
"""
Runs 1 run of the specified ML training and evaluation

"""
from sklearn.model_selection import KFold

import pandas as pd

def RunRegRun(model, X_path, y_path, runs):

    # Read in the data
    X = pd.read_csv(X_path)

    for r in range(runs):
        print(f"Run {r}")
        # Establish 10 fold crossvalidation splits
        kf = KFold(10, shuffle = True)
        for train_index, valid_index in kf.split(X):

            print(" ")
            # fit the model 
    


            # Make our predictions


        # Avg the scores across the 10 folds

    # average the scores across the r runs and get standard deviations



    # Write text file, or output dataframes to appropriate folders