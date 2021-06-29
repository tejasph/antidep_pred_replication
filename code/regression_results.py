#Regression Results
from run_regression import RunRegRun

"""
Temporary
"""

if __name__ == "__main__":
    runs = 1
    regressor = "rf"
    X_path = "X_train_norm"
    y = "y_train"

    RunRegRun(regressor, X_path, y, runs)