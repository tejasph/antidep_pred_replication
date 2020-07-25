from run_models import run_models
from multiprocessing import Process

"""
Top level function to run models. 
Uses multiprocessing to speed things up. 
Was not used for all results
"""





if __name__ == "__main__":
    # Process for the model change table, third row only changing the target
    p1 = Process(target=run_models, args=(2, "cv", "rf", "all", "X_ful_resp_trdcrit", "y_ful_resp_trdcrit"))

    # Process for the model change table, fourth changing the target and inclusion criteria
    p2 = Process(target=run_models, args=(2, "cv", "rf", "all", "X_full_resp", "y_ovlap_resp"))
    p3 = Process(target=print, args=("hello",))
    procs = []
    procs.append(p1)
    procs.append(p2)
    procs.append(p3)
    
    for proc in procs:
        print("Starting a process")
        proc.start()
    
    for proc in procs:
        proc.join()
        
    print("Ran all succesfully!")
    
    
    
    