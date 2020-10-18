from run_result import RunResult

"""
Top level function to produce results from our paper
"""

if __name__ == "__main__":
    runs = 1
    
    
    # Table for investigating decreased performance of external validation
    ## RunResult(runs, "cv", "rf", "all", "X_full_resp", "y_ovlap_resp")    

    # Output for the model change table, changings to week 8 qids c and sr remission
    ## RunResult(runs, "cv", "rf", "all", "X_full_rem_qids_c", "y_full_rem_qids_c")  
    ## RunResult(runs, "cv", "rf", "all", "X_full_rem_qids_sr", "y_full_rem_qids_sr")  
    
    # Output for the model change table, changings to week 8 qids c and sr response
    # RunResult(runs, "cv", "rf", "all", "X_tillwk4_qids_c", "y_wk8_resp_qids_c")  
    # RunResult(runs, "cv", "rf", "all", "X_tillwk4_qids_sr", "y_wk8_resp_qids_sr") 
    
    
    # Output for the model change table, Replcation with QIDS SR
    ## RunResult(runs, "cv", "rf", "all", "X_nolvl1drop_qids_sr__final", "y_lvl2_rem_qids_sr__final")
    # Output for the model change table, Replcation changing inclusion criteria
    ## RunResult(runs, "cv", "rf", "all", "X_tillwk4_qids_c__final", "y_lvl2_rem_qids_c_tillwk4__final")  
    # Output for the model change table, until week 4 subject inclusion, qids c and sr response
    ## RunResult(runs, "cv", "rf", "all", "X_tillwk4_qids_c__final", "y_wk8_resp_qids_c__final")  
    ## RunResult(runs, "cv", "rf", "all", "X_tillwk4_qids_sr__final", "y_wk8_resp_qids_sr__final") 
    
    # Output for the model change table, more QIDS-SR remission
    ## RunResult(runs, "cv", "rf", "all", "X_ovlap_resp", "y_wk8_rem_qids_sr") 
    ## RunResult(runs, "cv", "rf", "all", "X_tillwk4_qids_sr", "y_wk8_rem_qids_sr") 
    
    ##RunResult(runs, "cv", "rf", "all", "X_ovlap_resp", "y_wk8_resp_qids_sr") 
    ##RunResult(runs, "cv", "rf", "all", "X_tillwk4_qids_sr", "y_wk8_resp_qids_sr") 
    
    ## Table 6
    ##RunResult(runs, "cv", "rf", "all", "X_tillwk4_qids_sr", "y_wk8_resp_qids_sr") 
    ##RunResult(runs, "cv", "rf", "all", "X_tillwk4_qids_c", "y_wk8_resp_qids_c")  
    ##RunResult(runs, "cv", "rf", "all", "X_nolvl1drop_qids_sr", "y_wk8_resp_qids_sr_nolvl1drop")
    ##RunResult(runs, "cv", "rf", "all", "X_ovlap_resp", "y_wk8_resp_qids_sr") 
    ##RunResult(runs, "cv", "rf", "all", "X_overlap_stard_and_cb", "y_wk8_resp_qids_sr_stard_and_cb") 

    ## ? Extra for Table 6
    ##RunResult(runs, "cv", "rf", "chi", "X_tillwk4_qids_sr", "y_wk8_resp_qids_sr") 
    ## RunResult(runs, "cv", "rf", "elas", "X_tillwk4_qids_sr", "y_wk8_resp_qids_sr") 
    ##RunResult(runs, "cv", "elnet", "all", "X_tillwk4_qids_sr", "y_wk8_resp_qids_sr")
    ##RunResult(runs, "cv", "l2logreg", "elas", "X_tillwk4_qids_sr", "y_wk8_resp_qids_sr")
    
    ##RunResult(runs, "cv", "rf", "chi", "X_tillwk4_qids_sr", "y_wk8_rem_qids_sr") 
    ##RunResult(runs, "cv", "rf", "elas", "X_tillwk4_qids_sr", "y_wk8_rem_qids_sr") 
    
    
    ## ? New version of Table 6
    ##RunResult(runs, "cv", "rf", "chi", "X_tillwk4_qids_sr", "y_wk8_resp_qids_sr") 
    ##RunResult(runs, "cv", "rf", "elas", "X_tillwk4_qids_sr", "y_wk8_resp_qids_sr") 
    ##RunResult(runs, "cv", "elnet", "all", "X_tillwk4_qids_sr", "y_wk8_resp_qids_sr")

    
    # Output to try cv on combined stard and canbind datasets
    ## RunResult(runs, "cv", "rf", "all", "X_overlap_stard_and_cb", "y_wk8_resp_qids_sr_stard_and_cb") 
    ## RunResult(runs, "cv", "rf", "all", "X_overlap_stard_and_cb", "y_wk8_rem_qids_sr_stard_and_cb") 
    
    # Output for the external validations
    ## RunResult(runs, "extval_resp", "rf", "all", "X_ovlap_resp", "y_wk8_resp_qids_sr")  # X_ovlap_resp is misnamed, simply the subjects who stay till 4 weeks and have qids sr at baseline
    ## RunResult(runs, "extval_rem", "rf", "all", "X_ovlap_resp", "y_wk8_rem_qids_sr")  
    
    ##RunResult(runs, "extval_rem", "gbdt", "all", "X_ovlap_resp", "y_wk8_rem_qids_sr")  
    ##RunResult(runs, "extval_rem", "xgbt", "all", "X_ovlap_resp", "y_wk8_rem_qids_sr")  
    
    ##RunResult(runs, "extval_resp", "gbdt", "all", "X_ovlap_resp", "y_wk8_resp_qids_sr")  
    ##RunResult(runs, "extval_resp", "xgbt", "all", "X_ovlap_resp", "y_wk8_resp_qids_sr")  
    
    # Run to double check out extval_rem results are robust, uses a ext val target file that is scrambled
    ## RunResult(runs, "extval_rem_randomized", "rf", "all", "X_ovlap_resp", "y_wk8_rem_qids_sr") 
    
    # Run to check about response performance using the nolvl1drop datasets.
    ## RunResult(runs, "cv", "rf", "all", "X_nolvl1drop_qids_sr__final", "y_wk8_resp_qids_sr_nolvl1drop")
    ## RunResult(runs, "cv", "rf", "all", "X_nolvl1drop_qids_c__final", "y_wk8_resp_qids_c_nolvl1drop")
    ## RunResult(runs, "cv", "rf", "all", "X_ful_resp_trdcrit", "y_wk8_resp_qids_sr_nolvl1drop")
    ## RunResult(runs, "cv", "rf", "all", "X_full_trd", "y_wk8_resp_qids_c_nolvl1drop")

    # Missing xgbt with elas for some reason

    ##RunResult(runs, "cv", 'xgbt', 'all', "X_full_trd", "y_all_trd")
    ##RunResult(runs, "cv", 'gbdt', 'all', "X_full_trd", "y_all_trd")
    
    ## RunResult(runs, "cv", 'xgbt', 'chi', "X_full_trd", "y_all_trd")
    ## RunResult(runs, "cv", 'xgbt', 'elas', "X_full_trd", "y_all_trd")
    
    # Elastic Net model 
    #RunResult(runs, "cv", 'elnet', 'all', "X_full_trd", "y_all_trd")
    
    #l2logreg model
    ##RunResult(runs, "cv", 'l2logreg', 'elas', "X_full_trd", "y_all_trd")
    ##RunResult(runs, "cv", 'l2logreg', 'chi', "X_full_trd", "y_all_trd")
    ##RunResult(runs, "cv", 'l2logreg', 'all', "X_full_trd", "y_all_trd")
    
    
    
    
    # External Validation table
    # for model in ["l2logreg",'elnet']: # already done: 'gbdt','rf',"xgbt"]
    #     for evl in ["extval_rem", "extval_resp"]: # for l2logreg "all",
    #         if evl == "extval_rem":
    #             target = 'y_wk8_rem_qids_sr'
    #         elif evl == "extval_resp":
    #             target = 'y_wk8_resp_qids_sr'
    #         RunResult(runs, evl, model, "all", "X_ovlap_tillwk4_sr", target)
    
    
    # Table 3: Replication 
    if False:
        table = 'table3_replication'
    
        X_matrix = "X_nolvl1drop_qids_c" # STAR*D full feature data matrix, with subjects who do not drop in level according to having QIDS-C scores
        y_labels = "y_nolvl1drop_trdrem_qids_c"# STAR*D targets for QIDS-C TRD as defined by remission, for subjects who do not drop in level 1 according to having QIDS-C scores
        
        for model in ['rf','gbdt',"xgbt", "l2logreg"]:
            for f_select in ["all", "chi", "elas"]: # for l2logreg "all",
                RunResult(runs, "cv", model, f_select, X_matrix, y_labels, table)
        RunResult(runs, "cv", 'elnet', 'all', X_matrix, y_labels, table)
    
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
    if True:
        table = 'table5_comparisons'
        
        
        # Results in order they appear in table
        
        # TRD with QIDS-C and QIDS-R Remission, all STAR*D features, must not drop in lvl 1, as in Nie et al
        RunResult(runs, "cv", 'rf', 'all', 'X_nolvl1drop_qids_c', 'y_nolvl1drop_trdrem_qids_c', table)
        RunResult(runs, "cv", 'rf', 'all', 'X_nolvl1drop_qids_sr', 'y_nolvl1drop_trdrem_qids_sr', table)

        # QIDS-C and -SR Remission, cross-validated on STAR*D, using all features of subjects who have QIDS- until week 4, and then with only the features overlapping with CANBIND
        RunResult(runs, "cv", 'rf', 'all', 'X_tillwk4_qids_c', 'y_wk8_rem_qids_c', table)
        RunResult(runs, "cv", 'rf', 'all', 'X_tillwk4_qids_sr', 'y_wk8_rem_qids_sr', table)
        RunResult(runs, "cv", 'rf', 'all', 'X_overlap_tillwk4_qids_sr', 'y_wk8_rem_qids_sr', table)

        # QIDS-C and -SR Response, cross-validated on STAR*D, with subjects who have at least week 4 of QIDS-. Varied features, including overlapping and with feature selection
        RunResult(runs, "cv", 'rf', 'all', 'X_tillwk4_qids_c', 'y_wk8_resp_qids_c', table)
        RunResult(runs, "cv", 'rf', 'all', 'X_tillwk4_qids_sr', 'y_wk8_resp_qids_sr', table)
        RunResult(runs, "cv", 'rf', 'chi', 'X_tillwk4_qids_sr', 'y_wk8_resp_qids_sr', table)
        RunResult(runs, "cv", 'rf', 'elas', 'X_tillwk4_qids_sr', 'y_wk8_resp_qids_sr', table)
        RunResult(runs, "cv", 'rf', 'all', 'X_overlap_tillwk4_qids_sr', 'y_wk8_resp_qids_sr', table)
        
        # External Validation with QIDS-SR Remission and Response on CANBIND
        RunResult(runs, "extval_rem", 'rf', 'all', 'X_overlap_tillwk4_qids_sr', 'y_tillwk4_wk8_rem_qids_sr', table)
        RunResult(runs, "extval_rem", 'rf', 'all', 'X_overlap_tillwk4_qids_sr', 'y_tillwk4_wk8_resp_qids_sr', table)
        
    print("Ran all succesfully!")
    
    
    
    