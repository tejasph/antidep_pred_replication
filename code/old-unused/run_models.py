# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:20:32 2020

@author: jjnun
"""
import os
import re
##from utility import subsample
##from utility import featureSelectionChi, featureSelectionELAS, drawROC, featureSelectionAgglo
##from sklearn.ensemble import RandomForestClassifier
##from sklearn.metrics import confusion_matrix
##from sklearn.model_selection import train_test_split
##from sklearn.metrics import balanced_accuracy_score
##from sklearn.model_secv import RandomForrestEnsemble
from scipy.stats import ttest_1samp
import datetime
import numpy as np
##from randomForrests_cv import RandomForrestEnsemble
from run_model import RunModel
#from joblib import dump, load
import _pickle as cPickle
import pickle
import bz2
import lzma
import pickle

startTime = datetime.datetime.now()

# Simplified imputation
    ##data_path = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\532M_project\data\teyden-git\code\data-cleaning\final_datasets\to_run_experiment_simple_imput\X_lvl2_rem_qids01__final_simple_imputation.csv'
    ##label_path = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\532M_project\data\teyden-git\code\data-cleaning\final_datasets\to_run_experiment_simple_imput\y_lvl2_rem_qids01__final_simple_imputation.csv'

# Parameters
##result_root = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\results'    
##runs = 100

# Evaluation
## evl = "cv"
#evl = "extval"

# Model
## model = "rf"
##model = "elnet"
##model = "gbdt"
##model = "l2logreg"
##model = "xgbt"


#f_select =  "all" #chi, elas
#f_select = "elas"
#data = "X_full_trd"
#data = "X_ovlap_resp" # For external validation
#data = "X_top30_trd"
#data = "X_top10_trd"
#data = "X_top30_resp"
#data = "X_top10_resp"
#data = "X_top30_resp_ovlap"
#data = "X_top10_resp_ovlap"
#data = "X_top30_resp_ovlap_fromovlap"
#data = "X_top30_ovlap_trd"
#data = "X_top10_ovlap_trd"
#data = "ovlap_trd"
#data = "X_full_resp"
#data = "X_ful_resp_trdcrit"

#label = "y_all_trd"
#label = "y_ovlap_resp" # Keep the old name, it's just the y for the training data for ext val
#label = "y_ful_resp_trdcrit" # Y for response, but using the TRD subject selection criteria 

def RunModels(runs, evl, model,f_select, data, label, table=""):    
    # My directories
    result_root = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\results'    
    result_models_root = r'F:\ml_paper_models' # I needed a seperate dir due to file size
    data_dir = r"C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\for_paper_20201012\\"
    
    
    result_path = os.path.join(result_root, table)
    result_models_path = os.path.join(result_models_root, table)
    data_path = os.path.join(data_dir, data + ".csv")
    label_path = os.path.join(data_dir, label + ".csv")
# =============================================================================
#     
#     if data == "X_full_trd":
#         # Full features, y is TRD, replicating Nie et al
#         data_path = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_20200311\1_Replication\X_lvl2_rem_qids01__final.csv'
#     elif (data == "X_ovlap_resp") or (data == "X_ovlap_tillwk4_sr"):
#         # Only features overlapping with CANBIND, with the week 4 subject selection criteria, having qids sr values in week0
#         data_path = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_20200809\X_train_stard_extval.csv'
#     elif data == "X_ovlap_trd":
#         # Only features overlapping with CANBIND, with the TRD subject selection criteria
#         data_path = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_overlapping_for_trd\X_overlap_trd.csv'
#     elif data == "X_top30_trd":
#         data_path = r"C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_top30\X_top30_trd.csv"
#     elif data == "X_top10_trd":    
#         data_path = r"C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_top30\X_top10_trd.csv"
#     elif data == "X_top30_ovlap_trd":
#         data_path = r"C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_top30\X_top30_ovlap_trd.csv"
#     elif data == "X_top10_ovlap_trd":
#         data_path = r"C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_top30\X_top10_ovlap_trd.csv"
#     elif data == "X_top30_resp":
#         data_path = r"C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_top30\X_top30_resp.csv"
#     elif data == "X_top10_resp":    
#         data_path = r"C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_top30\X_top10_resp.csv"
#     elif data == "X_top30_resp_ovlap":
#         data_path = r"C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_top30\X_top30_resp_ovlap.csv"
#     elif data == "X_top10_resp_ovlap":    
#         data_path = r"C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_top30\X_top10_resp_ovlap.csv"
#     elif data == "X_top30_resp_ovlap_fromovlap":
#         # Same ish as the X_top30_resp_ovlap, but drawn from the overlapping and not full STAR*D dataset, so it's been through some conversions
#         data_path = r"C:/Users/jjnun/Documents/Sync/Research/1_CANBIND_Replication/teyden-git/data/final_datasets/to_run_top30/X_top30_resp_ovlap_fromovlap.csv"
#     
#     
#     elif data == "X_full_resp":
#         # Full features, for all those in the response y (n~3000)
#         data_path = r"C:/Users/jjnun/Documents/Sync/Research/1_CANBIND_Replication/teyden-git/data/final_datasets/to_run_20200311/1a_ReplicationWithResponse/X_wk8_response_qids01__final.csv"    
#     elif data == "X_ful_resp_trdcrit":
#         # Full features, response subject selection, used for the TRD y label, but would be same as the full, response subject selection above.  
#         data_path = r"C:/Users/jjnun/Documents/Sync/Research/1_CANBIND_Replication/teyden-git/data/final_datasets/to_run_overlapping_for_trd/X_full_resp_trdcrit.csv"    
#     elif data == "X_tillwk4_qids_c":
#         # Full features, all subjects who last until week 4, and have baseline QIDS-C. (For remissions/responses at week 8 according to qids-c)
#         data_path = r"C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_20200809\X_tillwk4_qids_c__final.csv"
#     elif data == "X_tillwk4_qids_sr":
#         # Full features, all subjects who last until week 4, and have baseline QIDS-SR. (For remissions/responses at week 8 according to qids-sr)
#         data_path = r"C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_20200809\X_tillwk4_qids_sr__final.csv"
#     elif data == "X_nolvl1drop_qids_sr":
#         # X for TRD, no level 1 drop, and requires initial QIDS-SR
#         data_path = r"C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_20200809\X_nolvl1drop_qids_sr__final.csv"
#     elif data == "X_nolvl1drop_qids_c__final":
#         # X for TRD, no level 1 drop, and requires initial QIDS-C, same as the very one hehe
#         data_path = r"C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_20200809\X_nolvl1drop_qids_c__final.csv"
#     
#     elif data == "X_overlap_stard_and_cb":
#         # Combined X for week 8, have initial qids-sr values, with both canbind and stard
#         data_path = r"C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_20200809\X_overlap_stard_and_cb.csv"
#     
#     
#     if label == "y_all_trd":
#         # All TRD y in the full STAR*D
#         label_path = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_20200311\1_Replication\y_lvl2_rem_qids01__final.csv'
#     elif label == "y_ovlap_trd":
#         # STAR*D subjects who are in the overlapping dataset (so their data can be used) but pulled from the TRD Y matrix, has six less than the full y matrix
#         label_path = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_overlapping_for_trd\y_overlap_trd.csv'
#     elif label == 'y_ovlap_resp':
#         # Y matrix for the overlapping dataset, with y as response as same in CANBIND    
#         label_path = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_20200311\2_ExternalValidation\y_train_stard_extval.csv'
#     elif label == "y_ful_resp_trdcrit":
#         label_path = r"C:/Users/jjnun/Documents/Sync/Research/1_CANBIND_Replication/teyden-git/data/final_datasets/to_run_overlapping_for_trd/y_full_resp_trdcrit.csv"    
#     elif label == "y_full_rem_qids_sr":
#         # Targets for remissions at week 8 according to qids-sr
#         label_path = r"C:/Users/jjnun\Documents/Sync\Research/1_CANBIND_Replication/teyden-git/data/final_datasets/to_run_20200808/1_Replication/y_wk8_rem_qids_sr__final.csv"
#     elif label == "y_full_rem_qids_c":
#         # Targets for remissions at week 8 according to qids-c
#         label_path = r"C:/Users/jjnun\Documents/Sync\Research/1_CANBIND_Replication/teyden-git/data/final_datasets/to_run_20200808/1_Replication/y_wk8_rem_qids_c__final.csv"
#     elif label == "y_wk8_rem_qids_c":
#         # Targets for remissions at week 8 according to qids-c
#         label_path = r"C:/Users/jjnun\Documents/Sync\Research/1_CANBIND_Replication/teyden-git/data/final_datasets/to_run_20200809/y_wk8_rem_qids_c__final.csv"
#     elif label == "y_wk8_rem_qids_sr":
#         # Targets for remissions at week 8 according to qids-sr
#         label_path = r"C:/Users/jjnun\Documents/Sync\Research/1_CANBIND_Replication/teyden-git/data/final_datasets/to_run_20200809/y_wk8_rem_qids_sr__final.csv"
#     elif label == "y_wk8_resp_qids_c":
#         # Targets for response at week 8 according to qids-c
#         label_path = r"C:/Users/jjnun\Documents/Sync\Research/1_CANBIND_Replication/teyden-git/data/final_datasets/to_run_20200809/y_wk8_resp_qids_c__final.csv"
#     elif label == "y_wk8_resp_qids_sr":
#         # Targets for response at week 8 according to qids-sr
#         label_path = r"C:/Users/jjnun\Documents/Sync\Research/1_CANBIND_Replication/teyden-git/data/final_datasets/to_run_20200809/y_wk8_resp_qids_sr__final.csv"
#     elif label == "y_lvl2_rem_qids_sr__final":
#         # Targets for TRD according to QIDS-SR
#         label_path = r"C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_20200809\y_lvl2_rem_qids_sr__final.csv"
#     elif label == "y_lvl2_rem_qids_c_tillwk4__final":
#         # Targets for QIDS-C TRD but for all subjects in study until week4 (marks level 1 dropouts as TRD)
#         label_path = data_dir + "to_run_20200809\y_lvl2_rem_qids_c_tillwk4__final.csv"
#     elif label == "y_wk8_resp_qids_sr_stard_and_cb":
#         # Combined y for week 8, have initial qids-sr values, with both canbind and stard, for response
#         label_path = r"C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_20200809\y_wk8_resp_qids_sr_stard_and_cb.csv"
#     elif label == "y_wk8_rem_qids_sr_stard_and_cb":    
#         # Combined y for week 8, have initial qids-sr values, with both canbind and stard, for remonse
#         label_path = r"C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_20200809\y_wk8_rem_qids_sr_stard_and_cb.csv"
#     elif label == "y_wk8_resp_qids_sr_nolvl1drop":    
#         # qids sr response, but only for subjects within the nolvl1drop dataset
#         label_path = r"C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_20200809\y_wk8_resp_qids_sr_nolvl1drop.csv"
#     elif label == "y_wk8_resp_qids_c_nolvl1drop":    
#         # qids c response, but only for subjects within the nolvl1drop dataset
#         label_path = r"C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_20200809\y_wk8_resp_qids_c_nolvl1drop.csv"
#     
# =============================================================================
    
    # Set n_splits, how many fold the cross-validation used for traiinng should be, as well as ensemble_n,
    # the number of models to train to use as an ensemble for prediction. Set to Nie et al's values.
    ensemble_n=30
    n_splits=10
    
    # Create numpy arrays to store all the results
    accus = np.zeros(runs)
    bal_accus = np.zeros(runs)
    aucs = np.zeros(runs)
    senss = np.zeros(runs)
    specs = np.zeros(runs)
    precs = np.zeros(runs)
    f1s = np.zeros(runs)
    tps = np.zeros(runs)
    fps = np.zeros(runs)
    tns = np.zeros(runs)
    fns = np.zeros(runs)
    feats = np.zeros(runs) # Average number of the average number of features used per classifier trained
    ##clfs = np.zeros(shape=(runs, n_splits, ensemble_n)) # Entry entry of this is an array of the 10 classifiers used for each cross-validation in a model run
    ##clfs = [None]*runs 
    
    # Create filename based on parameters
    filename = "{}_{}_{}_{}_{}_{}_{}".format(evl, model, runs,data, label, f_select, datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    # Make a dir for each result
    result_dir = os.path.join(result_models_path, filename)
    os.mkdir(result_dir)
    
    
    for i in range(runs):
            
            accus[i], bal_accus[i], aucs[i], senss[i], specs[i], precs[i], f1s[i], feats[i], impt, confus_mat, run_clfs = RunModel(data_path, label_path, f_select, model, evl, ensemble_n, n_splits)
       
            tps[i] = confus_mat['tp']
            fps[i] = confus_mat['fp']
            tns[i] = confus_mat['tn']
            fns[i] = confus_mat['fn']
    
            if i == 0:
                # Initialize impts now as number of features can change
                impts = np.empty([runs,np.size(impt)], dtype=float)
                
            impts[i,:] = impt
            
            # Save this run's models into a cPickle with bz2
            ##cpicklt = datetime.datetime.now()
            models_filename = os.path.join(result_dir, f"run_{i}") + '.pbz2'
            with bz2.BZ2File(models_filename, 'w') as f2: 
                cPickle.dump(run_clfs, f2)
            ##print(f'{datetime.datetime.now() - cpicklt} for cPickle bz2')
            ##with open(os.path.join(result_dir, f"run_{i}") + '.pickle', 'wb') as f2:
            ##    pickle.dump(run_clfs, f2)
            ##f2.close()
            
                
            print("Finished run: " + str(i + 1) + " of " + str(runs) + "\n")
            
            
    
    # Process feature importance
    avg_impts = np.mean(impts, axis=0)
    std_impts = np.std(impts, axis=0)
    
    #print("Shape of avg_impts", np.shape(avg_impts))
    sorted_features = np.argsort(avg_impts)[::-1]
    top_31_features = sorted_features[0:31] #In descending importance, first is most important
    with open(data_path) as f:
        feature_names = f.readline().split(',')
    
    # Write output file
    ##filename = "{}_{}_{}_{}_{}_{}_{}".format(evl, model, runs,data, label, f_select, datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    f = open(os.path.join(result_path, filename + '.txt'), 'w')
    
    f.write("MODEL RESULTS for run at: " + filename + "\n\n")
    
    f.write("Model Parameters:-----------------------------------\n")
    f.write("Evaluation: " + evl + "\n")
    f.write("Model: " + model + "\n")
    f.write("Feature selection: " + f_select + "\n")
    f.write("X is: " + data_path + "\n")
    f.write("y is: " + label_path + "\n")
    f.write(str(runs) +" runs of 10-fold CV\n\n")
    
    f.write("Summary of Results:------------------------------------\n")
    f.write("Mean accuracy is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(accus), np.std(accus)))
    f.write("Mean balanced accuracy is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(bal_accus), np.std(bal_accus)))
    f.write("Mean AUC is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(aucs), np.std(aucs)))
    f.write("Mean sensitivty is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(senss), np.std(senss)))
    f.write("Mean specificty is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(specs), np.std(specs)))
    f.write("Mean precision is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(precs), np.std(precs)))
    f.write("Mean f1 is: {:.4f}, with Standard Deviation: {:.4f}\n".format(np.mean(f1s), np.std(f1s)))
    
    f.write("Mean true positive is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(tps), np.std(tps)))
    f.write("Mean false positive is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(fps), np.std(fps)))
    f.write("Mean true negative is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(tns), np.std(tns)))
    f.write("Mean false negative is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(fns), np.std(fns)))
    
    f.write("Mean number of features used is: {:.4f} of {:d}, with Standard Deviation: {:.4f}\n\n".format(np.mean(feats), np.size(avg_impts), np.std(feats)))
    
    
    f.write("Feature Importance And Use:---------------------------\n")
    f.write("Top 31 Features by importance, in descending order (1st most important):\n")
    f.write("By position in data matrix, 1 added to skip index=0 \n")
    #print("Here are the top 30 features...")
    #print(top_31_features + 1)
    ##print(f'here are the importance of features: {avg_impts}')
    
    if np.sum(avg_impts) != 0:
        f.write(str(top_31_features + 1) + "\n")
        for i in range(len(top_31_features)): f.write(feature_names[top_31_features[i] + 1] + "\n")
        f.write("\n")
    else:
        f.write("Code does not support feature for this model at this time\n")
    ##f.write(str(feature_names[top_31_features + 1]) + "\n")
    
    f.write("Statistical Significance:----------------------------\n")
    if (data == "full_trd" or data =="ovlap_trd") and model == "rf_cv" and f_select == "all":
      _,acc_pvalue = ttest_1samp(accus, 0.70)
      f.write("P-value from one sided t-test vs Nie et al's 0.70 Accuracy: {:.6f}\n".format(acc_pvalue))
      _,bal_pvalue = ttest_1samp(bal_accus, 0.70)
      f.write("P-value from one sided t-test vs Nie et al's 0.70 Balanced Accuracy: {:.6f}\n".format(bal_pvalue))
      _,auc_pvalue = ttest_1samp(aucs, 0.78)
      f.write("P-value from one sided t-test vs Nie et al's 0.78 AUC: {:.6f}\n".format(auc_pvalue))
      _,senss_pvalue = ttest_1samp(senss, 0.69)
      f.write("P-value from one sided t-test vs Nie et al's 0.69 Sensitivity: {:.6f}\n".format(auc_pvalue))
      _,specs_pvalue = ttest_1samp(specs, 0.71)
      f.write("P-value from one sided t-test vs Nie et al's 0.71 Specificity: {:.6f}\n\n".format(auc_pvalue))  
        
    f.write("Raw results:----------------------------------------\n")
    f.write("Accuracies\n")
    f.write(re.sub(r"\s+",r",",str(accus)) + "\n")
    f.write("Balanced Accuracies\n")
    f.write(re.sub(r"\s+",r",",str(bal_accus)) + "\n")
    f.write("AUCs\n")
    f.write(re.sub(r"\s+",r",",str(aucs)) + "\n")
    f.write("Sensitivites\n")
    f.write(re.sub(r"\s+",r",",str(senss)) + "\n")
    f.write("Specificities\n")
    f.write(re.sub(r"\s+",r",",str(specs)) + "\n")
    f.write("Precisions\n")
    f.write(re.sub(r"\s+",r",",str(precs)) + "\n")
    f.write("F1s\n")
    f.write(re.sub(r"\s+",r",",str(f1s)) + "\n")
    f.write("Number of features used\n")
    f.write(re.sub(r"\s+",r",",str(feats)) + "\n")
    f.write("Mean Feature importances Across Runs\n")
    f.write(re.sub(r" +",r",",np.array_str(avg_impts,precision=4,max_line_width=100)) + "\n")
    f.write("Mean Feature importances std. deviation Across Runs\n")
    f.write(re.sub(r" +",r",",np.array_str(std_impts,precision=4,max_line_width=100)) + "\n")
    
    f.close()
    
    #joblib with bz2 to pickle the array of classifiers
    ##bz2t = datetime.datetime.now()
    ##dump(clfs, os.path.join(result_models_path, filename + '.bz2'), compress='bz2') 
    ##print(f'{datetime.datetime.now() - bz2t} for bz2')
    
    ##lzmat = datetime.datetime.now()
    ##dump(clfs, os.path.join(result_path, filename + '.lzma'), compress='lzma') 
    ##print(f'{datetime.datetime.now() - lzmat} for lzma')
    
    """
    cpicklt = datetime.datetime.now()
    models_filename = os.path.join(result_models_path, filename) + '.pbz2'
    with bz2.BZ2File(models_filename, 'w') as f2: 
        cPickle.dump(clfs, f2)
    print(f'{datetime.datetime.now() - cpicklt} for cPickle bz2')
    f2.close()
    """
    
    
    
    clfs = None
    ##cpickltlzma = datetime.datetime.now()
    ##with lzma.LZMAFile(os.path.join(result_path, filename) + '.plzma', 'w') as f: 
    ##    cPickle.dump(clfs, f)
    ##print(f'{datetime.datetime.now() - cpickltlzma} for cPickle lzma')  
    
    ##pt = datetime.datetime.now()
    ##pickle.dump(clfs, open(os.path.join(result_path, filename) + '.pickle', "wb" )) 
    ##print(f'{datetime.datetime.now() - pt} for pickle') 
    
    print("Completed after seconds: \n")
    print(datetime.datetime.now() - startTime)
    
    #test
    ##testclfs = load(os.path.join(result_path, filename + '.joblib'))
    ##print(len(testclfs))
    ##print(len(testclfs[0]))
    ##print(len(testclfs[0][0]))
    ##print(testclfs[0][0])
        
