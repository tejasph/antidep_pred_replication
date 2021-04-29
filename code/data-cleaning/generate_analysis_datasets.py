# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 10:43:13 2020

Generates datasets used for additional analysis, not really used in the paper

@author: jjnun
"""

import pandas as pd


# Select subjects with corresponding y values
def generate_analysis_datasets_overlapping_trd():
    """
    Generates dataset of the overlapping features, but for the subjects with TRD y
    
    """
    output_dir = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\teyden-git\data\final_datasets\to_run_overlapping_for_trd'   
    full_trd_X = pd.read_csv(r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\teyden-git\data\final_datasets\to_run_20201016\1_Replication\X_lvl2_rem_qids01__final.csv')
    full_trd_y = pd.read_csv(r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\teyden-git\data\final_datasets\to_run_20201016\1_Replication\y_lvl2_rem_qids01__final.csv')
    overlap_resp_X = pd.read_csv(r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\teyden-git\data\final_datasets\to_run_20201016\2_ExternalValidation\X_train_stard_extval.csv')
    overlap_resp_y = pd.read_csv(r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\teyden-git\data\final_datasets\to_run_20201016\2_ExternalValidation\y_train_stard_extval.csv')

    # Make new dataset with the columns from the overlapping dataset, but with only the subjects that meet criteria for the TRD experiment
    overlap_trd_X = overlap_resp_X[overlap_resp_X["SUBJLABEL:::subjectkey"].isin(full_trd_X["subjectkey"])]
    # Then filter the trd y to only include the subjects that are in the above X
    overlap_trd_y = full_trd_y[full_trd_y["subjectkey"].isin(overlap_trd_X["SUBJLABEL:::subjectkey"])]
    
    
    # Sort and output
    overlap_trd_X = overlap_trd_X.reset_index(drop=True)
    overlap_trd_X = overlap_trd_X.sort_index(axis=1) # Newly added, sorts columns alphabetically so same for both matrices
    overlap_trd_X = overlap_trd_X.set_index(['SUBJLABEL:::subjectkey'])

    overlap_trd_y = overlap_trd_y.reset_index(drop=True)
    overlap_trd_y = overlap_trd_y.sort_index(axis=1) # Newly added, sorts columns alphabetically so same for both matrices
    overlap_trd_y = overlap_trd_y.set_index(['subjectkey'])
    
    overlap_trd_X.to_csv(output_dir + "/X_overlap_trd.csv",index=True)
    overlap_trd_y.to_csv(output_dir + "/y_overlap_trd.csv",index=True)
    
def generate_analysis_datasets_resp_with_trdcriteria():
    """
    Generates X and y with all features, with only the subjects included in the TRD anaylses,
    but with the antidepressant response y values (requires to be in both)
    """
    output_dir = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\teyden-git\data\final_datasets\to_run_overlapping_for_trd'   

    
    full_resp_X = pd.read_csv(r"C:/Users/jjnun/Documents/Sync/Research/1_CANBIND Replication/teyden-git/data/final_datasets/to_run_20201016/1a_ReplicationWithResponse/X_wk8_response_qids01__final.csv")    
    full_resp_y = pd.read_csv(r"C:/Users/jjnun/Documents/Sync/Research/1_CANBIND Replication/teyden-git/data/final_datasets/to_run_20201016/1a_ReplicationWithResponse/y_wk8_response_qids01__final.csv")    

    full_trd_y = pd.read_csv(r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\teyden-git\data\final_datasets\to_run_20201016\1_Replication\y_lvl2_rem_qids01__final.csv')


    full_resp_trdcrit_X = full_resp_X[full_resp_X['subjectkey'].isin(full_trd_y['subjectkey'])]
    full_resp_trdcrit_y = full_resp_y[full_resp_y['subjectkey'].isin(full_trd_y['subjectkey'])]

    # Sort and output
    full_resp_trdcrit_X = full_resp_trdcrit_X.reset_index(drop=True)
    full_resp_trdcrit_X = full_resp_trdcrit_X.sort_index(axis=1) 
    full_resp_trdcrit_X = full_resp_trdcrit_X.set_index(['subjectkey'])
    
    full_resp_trdcrit_y = full_resp_trdcrit_y.reset_index(drop=True)
    full_resp_trdcrit_y = full_resp_trdcrit_y.sort_index(axis=1) 
    full_resp_trdcrit_y = full_resp_trdcrit_y.set_index(['subjectkey'])

    
    full_resp_trdcrit_X.to_csv(output_dir + "/X_full_resp_trdcrit.csv",index=True)
    full_resp_trdcrit_y.to_csv(output_dir + "/y_full_resp_trdcrit.csv",index=True)



def generate_analysis_datasets_top30and10_trd():
    """
    Generate an X with only the top 30 features as determined by a 100 CV RF run on full_trd_rf_cv_100_all_20200203-014658.txt
    """
    top30 = ["subjectkey","qids01_w2c__qstot","imput_idsc5w2","imput_qidscpccg","qids01_w2sr__qstot","imput_idsc5pccg","qids01_w2c__vmdsd","qlesq01__totqlesq","sfhs01__pcs12","wsas01__totwsas","qids01_w2sr__vmdsd","qids01_w2c__vengy","phx01__episode_date","wsas01__wsas03","qids01_w0sr__qstot","interview_age","qids01_w0c__qstot","hrsd01__hdtot_r","sfhs01__mcs12","wsas01__wsas01","wsas01__wsas04","dm01_w0__totincom","qids01_w2c__vintr","phx01__dage","qids01_w2c__vcntr","qids01_w2c__vsoin","dm01_enroll__educat","dm01_enroll__resm","qids01_w2sr__vvwsf","dm01_w0__mempl","imput_gibbons"]    
    top10 = top30[0:11]
    output_dir = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\teyden-git\data\final_datasets\to_run_top30'   
    full_trd_X = pd.read_csv(r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\teyden-git\data\final_datasets\to_run_20201016\1_Replication\X_lvl2_rem_qids01__final.csv')

    top30_trd_X = full_trd_X[top30]
    top10_trd_X = full_trd_X[top10]
    
    top30_trd_X.reset_index(drop=True)
    top30_trd_X = top30_trd_X.sort_index(axis=1) # Newly added, sorts columns alphabetically so same for both matrices
    top30_trd_X = top30_trd_X.set_index(['subjectkey'])
    top30_trd_X.to_csv(output_dir + "/X_top30_trd.csv")
    
    top10_trd_X.reset_index(drop=True)
    top10_trd_X = top10_trd_X.sort_index(axis=1) # Newly added, sorts columns alphabetically so same for both matrices
    top10_trd_X = top10_trd_X.set_index(['subjectkey'])
    top10_trd_X.to_csv(output_dir + "/X_top10_trd.csv")
    
def generate_analysis_datasets_top30and10_resp():
    """
    Generate an X with only the top 30 features as determined by a 10 CV RF run on rf_cv_10_X_full_resp_y_ovlap_resp_all_20200205-1421.txt
    """
    top30 = ["subjectkey","qids01_w2sr__qstot","imput_qidscpccg","imput_idsc5pccg","qids01_w2c__qstot","sfhs01__pcs12","imput_idsc5w2","qids01_w0sr__qstot","interview_age","qids01_w2c__vmdsd","sfhs01__mcs12","phx01__episode_date","wsas01__totwsas","qlesq01__totqlesq","qids01_w2sr__vintr","phx01__dage","dm01_w0__totincom","qids01_w2sr__vmdsd","qids01_w2sr__vagit","qids01_w2sr__vslow","dm01_enroll__resm","qids01_w2sr__vengy","hrsd01__hdtot_r","dm01_w0__mempl","dm01_enroll__educat","wsas01__wsas01","wpai01__wpai04","qids01_w2sr__vvwsf","qids01_w0c__qstot","imput_gibbons","wpai01__wpai_totalhrs"]  
    top10 = top30[0:11]
    output_dir = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\teyden-git\data\final_datasets\to_run_top30'   
    full_resp_X = pd.read_csv(r"C:/Users/jjnun/Documents/Sync/Research/1_CANBIND Replication/teyden-git/data/final_datasets/to_run_20201016/1a_ReplicationWithResponse/X_wk8_response_qids01__final.csv")


    top30_resp_X = full_resp_X[top30]
    top10_resp_X = full_resp_X[top10]
    
    top30_resp_X.reset_index(drop=True)
    top30_resp_X = top30_resp_X.sort_index(axis=1) # Newly added, sorts columns alphabetically so same for both matrices
    top30_resp_X = top30_resp_X.set_index(['subjectkey'])
    top30_resp_X.to_csv(output_dir + "/X_top30_resp.csv")
    
    top10_resp_X.reset_index(drop=True)
    top10_resp_X = top10_resp_X.sort_index(axis=1) # Newly added, sorts columns alphabetically so same for both matrices
    top10_resp_X = top10_resp_X.set_index(['subjectkey'])
    top10_resp_X.to_csv(output_dir + "/X_top10_resp.csv")    

 
    
#generate_analysis_datasets_overlapping_trd()    
#generate_analysis_datasets_top30and10_trd()
#generate_analysis_datasets_resp_with_trdcriteria()
generate_analysis_datasets_top30and10_resp()

