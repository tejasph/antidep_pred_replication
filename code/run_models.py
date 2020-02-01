# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:20:32 2020

@author: jjnun
"""
import os
from utility import subsample
from utility import featureSelectionChi, featureSelectionELAS, drawROC, featureSelectionAgglo
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
import datetime
import numpy as np
from randomForrests_cv import RandomForrestEnsemble

startTime = datetime.datetime.now()


# Simplified imputation
    ##pathData = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\532M_project\data\teyden-git\code\data-cleaning\final_datasets\to_run_experiment_simple_imput\X_lvl2_rem_qids01__final_simple_imputation.csv'
    ##pathLabel = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\532M_project\data\teyden-git\code\data-cleaning\final_datasets\to_run_experiment_simple_imput\y_lvl2_rem_qids01__final_simple_imputation.csv'
# Full features
pathData = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\teyden-git\data\final_datasets\to_run_20201016\1_Replication\X_lvl2_rem_qids01__final.csv'
pathLabel = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\teyden-git\data\final_datasets\to_run_20201016\1_Replication\y_lvl2_rem_qids01__final.csv'
pathResults = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\teyden-git\results'    
runs = 2
model = "rf" #many others
f_select =  "full" #chi, elas
#f_select = "elas"


# Create numpy arrays to store all the results
accus = np.zeros(runs)
bal_accus = np.zeros(runs)
aucs = np.zeros(runs)
senss = np.zeros(runs)
specs = np.zeros(runs)
precs = np.zeros(runs)
f1s = np.zeros(runs)
feats = np.zeros(runs) # Average number of the average number of features used per classifier trained


for i in range(runs):
        if model == "rf":    
            accus[i], bal_accus[i], aucs[i], senss[i], specs[i], precs[i], f1s[i], feats[i] = RandomForrestEnsemble(pathData, pathLabel, f_select)
        print("Finished run: " + str(i + 1) + " of " + str(runs) + "\n")

filename = "{}_{}_{}_{}.txt".format(model, runs, f_select, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
f = open(os.path.join(pathResults, filename), 'w')

f.write("MODEL RESULTS for run at: " + filename + "\n\n")
f.write("Model Parameters:-----------------------------------\n")
f.write("Model: " + model + "\n")
f.write("Feature selection: " + f_select + "\n")
f.write("X is: " + pathData + "\n")
f.write("y is: " + pathLabel + "\n")
f.write(str(runs) +" runs of 10-fold CV\n\n")
f.write("Summary of Results:------------------------------------\n")
f.write("Mean accuracy is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(accus), np.std(accus)))
f.write("Mean balanced accuracy is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(bal_accus), np.std(bal_accus)))
f.write("Mean sensitivty is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(senss), np.std(senss)))
f.write("Mean specificty is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(specs), np.std(specs)))
f.write("Mean precision is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(precs), np.std(precs)))
f.write("Mean f1 is: {:.4f}, with Standard Deviation: {:.4f}\n".format(np.mean(f1s), np.std(f1s)))
f.write("Mean number of features used is: {:.4f}, with Standard Deviation: {:.4f}\n\n".format(np.mean(feats), np.std(feats)))
f.write("Raw results:----------------------------------------\n")
f.write("Accuracies\n")
f.write(str(accus) + "\n")
f.write("Balanced Accuracies\n")
f.write(str(bal_accus) + "\n")
f.write("Sensitivites\n")
f.write(str(senss) + "\n")
f.write("Specificities\n")
f.write(str(specs) + "\n")
f.write("Precisions\n")
f.write(str(precs) + "\n")
f.write("F1s\n")
f.write(str(f1s) + "\n")

f.close()


print("Completed after seconds: \n")
print(datetime.datetime.now() - startTime)

        
