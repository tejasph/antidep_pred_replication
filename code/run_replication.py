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
    RunModels(2, "cv", "rf", "all", "X_ful_resp_trdcrit", "y_ful_resp_trdcrit")
    # Output for the model change table, fourth changing the target and inclusion criteria
    RunModels(2, "cv", "rf", "all", "X_full_resp", "y_ovlap_resp")    

    
    # Replication table
    for model in ['rf', 'gbdt', "l2logreg", "xgbt"]:
        for f_select in ["all","chi", "elas"]:
            RunModels(runs, "cv", model, f_select, "X_full_trd", "y_all_trd")
        
    


    
    
    
    
    print("Ran all succesfully!")
    
    
    
    