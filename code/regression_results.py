#Regression Results
from run_regression import RunRegRun
from run_globals import REG_RESULTS_DIR
import os

"""
Run Experiment script

Change variables and experiment name using this script. 
"""

if __name__ == "__main__":

    exp_name = "test_paths1"
    out_path = os.path.join(REG_RESULTS_DIR, exp_name)

    # Makes sure not to overwrite any files
    if os.path.isdir(out_path):
        raise Exception("Name already exists")
    else:
        os.mkdir(out_path + "/")

    runs = 10
    regressors = ["rf"]
    X_paths = ["X_train_norm_over"]
    y = "y_train"
    y_proxies = ["score_change"]
    test_data = False

    for regressor in regressors:
        for y_proxy in y_proxies:
            for X_path in X_paths:
                RunRegRun(regressor, X_path, y, y_proxy, out_path,  runs , test_data)

