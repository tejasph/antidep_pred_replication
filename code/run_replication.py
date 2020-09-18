from run_models import RunModels

"""
Top level function to run models. 
Was not used for all results
Was thinking of making it parralel, but the ML models already run in parralel so 
there is no point
"""

if __name__ == "__main__":
    runs = 100
    
    
    # Table for investigating decreased performance of external validation
    # Output for the model change table, third row only changing the target
    ## RunModels(runs, "cv", "rf", "all", "X_ful_resp_trdcrit", "y_ful_resp_trdcrit")
    # Output for the model change table, fourth changing the target and inclusion criteria
    ## RunModels(runs, "cv", "rf", "all", "X_full_resp", "y_ovlap_resp")    

    # Output for the model change table, changings to week 8 qids c and sr remission
    ## RunModels(runs, "cv", "rf", "all", "X_full_rem_qids_c", "y_full_rem_qids_c")  
    ## RunModels(runs, "cv", "rf", "all", "X_full_rem_qids_sr", "y_full_rem_qids_sr")  
    
    # Output for the model change table, changings to week 8 qids c and sr response
    # RunModels(runs, "cv", "rf", "all", "X_tillwk4_qids_c", "y_wk8_resp_qids_c")  
    # RunModels(runs, "cv", "rf", "all", "X_tillwk4_qids_sr", "y_wk8_resp_qids_sr") 
    
    
    # Output for the model change table, Replcation with QIDS SR
    ## RunModels(runs, "cv", "rf", "all", "X_nolvl1drop_qids_sr__final", "y_lvl2_rem_qids_sr__final")
    # Output for the model change table, Replcation changing inclusion criteria
    ## RunModels(runs, "cv", "rf", "all", "X_tillwk4_qids_c__final", "y_lvl2_rem_qids_c_tillwk4__final")  
    # Output for the model change table, until week 4 subject inclusion, qids c and sr response
    ## RunModels(runs, "cv", "rf", "all", "X_tillwk4_qids_c__final", "y_wk8_resp_qids_c__final")  
    ## RunModels(runs, "cv", "rf", "all", "X_tillwk4_qids_sr__final", "y_wk8_resp_qids_sr__final") 
    
    # Output for the model change table, more QIDS-SR remission
    ## RunModels(runs, "cv", "rf", "all", "X_ovlap_resp", "y_wk8_rem_qids_sr") 
    ## RunModels(runs, "cv", "rf", "all", "X_tillwk4_qids_sr", "y_wk8_rem_qids_sr") 

    # Output to try cv on combind stard and canbind datasets
    RunModels(runs, "cv", "rf", "all", "X_overlap_stard_and_cb", "y_wk8_resp_qids_sr_stard_and_cb") 
    RunModels(runs, "cv", "rf", "all", "X_overlap_stard_and_cb", "y_wk8_rem_qids_sr_stard_and_cb") 

    # Output for the external validations
    ## RunModels(runs, "extval_resp", "rf", "all", "X_ovlap_resp", "y_wk8_resp_qids_sr")  # X_ovlap_resp is misnamed, simply the subjects who stay till 4 weeks and have qids sr at baseline
    ## RunModels(runs, "extval_rem", "rf", "all", "X_ovlap_resp", "y_wk8_rem_qids_sr")  
    
    ## RunModels(runs, "extval_rem", "gbdt", "all", "X_ovlap_resp", "y_wk8_rem_qids_sr")  
    ## RunModels(runs, "extval_rem", "xgbt", "all", "X_ovlap_resp", "y_wk8_rem_qids_sr")  
    
    ## RunModels(runs, "extval_resp", "gbdt", "all", "X_ovlap_resp", "y_wk8_resp_qids_sr")  
    ## RunModels(runs, "extval_resp", "xgbt", "all", "X_ovlap_resp", "y_wk8_resp_qids_sr")  
    
    # Run to double check out extval_rem results are robust, uses a ext val target file that is scrambled
    ## RunModels(runs, "extval_rem_randomized", "rf", "all", "X_ovlap_resp", "y_wk8_rem_qids_sr") 
    
    # Run to check about response performance using the nolvl1drop datasets.
    ## RunModels(runs, "cv", "rf", "all", "X_nolvl1drop_qids_sr__final", "y_wk8_resp_qids_sr_nolvl1drop")
    ## RunModels(runs, "cv", "rf", "all", "X_nolvl1drop_qids_c__final", "y_wk8_resp_qids_c_nolvl1drop")
    ## RunModels(runs, "cv", "rf", "all", "X_ful_resp_trdcrit", "y_wk8_resp_qids_sr_nolvl1drop")
    ## RunModels(runs, "cv", "rf", "all", "X_full_trd", "y_wk8_resp_qids_c_nolvl1drop")

    
    
    # Replication table
    ##for model in ["l2logreg"]: #'gbdt','rf',"xgbt" (already done)| 
    ##    for f_select in ["all", "chi", "elas"]: # for l2logreg "all",
    ##        RunModels(runs, "cv", model, f_select, "X_full_trd", "y_all_trd")
     
    # External Validation table
    # for model in ["l2logreg",'elnet']: # already done: 'gbdt','rf',"xgbt"]
    #     for evl in ["extval_rem", "extval_resp"]: # for l2logreg "all",
    #         if evl == "extval_rem":
    #             target = 'y_wk8_rem_qids_sr'
    #         elif evl == "extval_resp":
    #             target = 'y_wk8_resp_qids_sr'
    #         RunModels(runs, evl, model, "all", "X_ovlap_tillwk4_sr", target)
    
    print("Ran all succesfully!")
    
    
    
    