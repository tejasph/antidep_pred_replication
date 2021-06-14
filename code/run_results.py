from run_result import RunResult
from run_experiment import RunExperiment

"""
Top level script to produce results from our paper

Does not take arguments. Adjust data and results directory in 
run_globals.py

Adjust number of runs below
"""

if __name__ == "__main__":
    runs = 1

    # Table 3: Replication 
    if False:
        table = 'table3_replication'
    
        X_matrix = "X_nolvl1drop_qids_c" # STAR*D full feature data matrix, with subjects who do not drop in level according to having QIDS-C scores
        y_labels = "y_nolvl1drop_trdrem_qids_c"# STAR*D targets for QIDS-C TRD as defined by remission, for subjects who do not drop in level 1 according to having QIDS-C scores
        
        for model in ["l2logreg","rf"]: ## 'rf','gbdt',"xgbt",
            for f_select in ["all", "chi", "elas"]: 
                RunResult(runs, "cv", model, f_select, X_matrix, y_labels, table)

                # Try with MinMax Scaling
                RunResult(runs, "cv", model, f_select, X_matrix, y_labels, table, f_scaling = "norm")
                
        RunResult(runs, "cv", 'elnet', 'all', X_matrix, y_labels, table)
        RunResult(runs, "cv", 'elnet', 'all', X_matrix, y_labels, table, f_scaling = "norm")
    
    # Table 4: External Validation
    if False:
        table = 'table4_externalvalidation'
    
        X_matrix = "X_overlap_tillwk4_qids_sr" # STAR*D dataset, only overlapping features with CAN-BIND, subjects who have qids-sr until at least week 4
    
        # QIDS-SR Response
        y_labels = "y_tillwk4_wk8_resp_qids_sr"# STAR*D targets for training external validation, subjects who have qids-sr until at least week 4, targeting week 8 qids sr response
        for model in ['rf','gbdt',"xgbt", "l2logreg", "elnet"]:
            RunResult(runs, "extval_resp", model, 'all', X_matrix, y_labels, table)
    
        # QIDS-SR Remission
        y_labels = "y_tillwk4_wk8_rem_qids_sr"# STAR*D targets for training external validation, subjects who have qids-sr until at least week 4, targeting week 8 qids sr remission
        for model in ['rf','gbdt',"xgbt", "l2logreg", "elnet"]:
            RunResult(runs, "extval_rem", model, 'all', X_matrix, y_labels, table)
            
        
    
    # Table 5: Comparisons
    if False:
        table = 'table5_comparisons'
        # Results in order they appear in table
        
        # TRD with QIDS-C and QIDS-R Remission, all STAR*D features, must not drop in lvl 1, as in Nie et al
        ##RunResult(runs, "cv", 'rf', 'all', 'X_nolvl1drop_qids_c', 'y_nolvl1drop_trdrem_qids_c', table)
        ##RunResult(runs, "cv", 'rf', 'all', 'X_nolvl1drop_qids_sr', 'y_nolvl1drop_trdrem_qids_sr', table)

        # QIDS-C and -SR Remission, cross-validated on STAR*D, using all features of subjects who have QIDS- until week 4, and then with only the features overlapping with CANBIND
        ##RunResult(runs, "cv", 'rf', 'all', 'X_tillwk4_qids_c', 'y_wk8_rem_qids_c', table)
        ##RunResult(runs, "cv", 'rf', 'all', 'X_tillwk4_qids_sr', 'y_wk8_rem_qids_sr', table)
        RunResult(runs, "cv", 'rf', 'all', 'X_overlap_tillwk4_qids_sr', 'y_wk8_rem_qids_sr', table)

        # QIDS-C and -SR Response, cross-validated on STAR*D, with subjects who have at least week 4 of QIDS-. Varied features, including overlapping and with feature selection
        RunResult(runs, "cv", 'rf', 'all', 'X_tillwk4_qids_c', 'y_wk8_resp_qids_c', table)
        RunResult(runs, "cv", 'rf', 'all', 'X_tillwk4_qids_sr', 'y_wk8_resp_qids_sr', table)
        RunResult(runs, "cv", 'rf', 'chi', 'X_tillwk4_qids_sr', 'y_wk8_resp_qids_sr', table)
        RunResult(runs, "cv", 'rf', 'elas', 'X_tillwk4_qids_sr', 'y_wk8_resp_qids_sr', table)
        RunResult(runs, "cv", 'rf', 'all', 'X_overlap_tillwk4_qids_sr', 'y_wk8_resp_qids_sr', table)
        
        # External Validation with QIDS-SR Remission and Response on CANBIND
        #RunResult(runs, "extval_rem", 'rf', 'all', 'X_overlap_tillwk4_qids_sr', 'y_tillwk4_wk8_rem_qids_sr', table)
       # RunResult(runs, "extval_resp", 'rf', 'all', 'X_overlap_tillwk4_qids_sr', 'y_tillwk4_wk8_resp_qids_sr', table)

    if True:

        table = 'Tejas_experiements'
        X_matrix = "X_nolvl1drop_qids_c" # STAR*D full feature data matrix, with subjects who do not drop in level according to having QIDS-C scores
        y_labels = "y_nolvl1drop_trdrem_qids_c"# STAR*D targets for QIDS-C TRD as defined by remission, for subjects who do not drop in level 1 according to having QIDS-C scores

       
            

        # Try with MinMax Scaling
        RunResult(runs, "cv", "mlp", 'all', X_matrix, y_labels, table, f_scaling = "norm")
        RunResult(runs, "cv","l2logreg", 'all', X_matrix, y_labels, table, f_scaling = "norm")
        RunResult(runs, "cv","svc", 'all', X_matrix, y_labels, table, f_scaling = "norm")
        RunResult(runs, "cv","knn", 'all', X_matrix, y_labels, table, f_scaling = "norm")

            
    print("Ran all succesfully!")
    
    
    
    