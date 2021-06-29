#Regression Results
from run_regression import RunRegRun

"""
Temporary
"""

if __name__ == "__main__":
    runs = 1
    regressor = "rf"
    X_path = "data/modelling_data/X_train_norm.csv"
    y = "data/modelling_data/y_train.csv"

    RunRegRun(regressor, X_path, y, runs)