#Regression Results
from run_regression import RunRegRun

"""
Temporary
"""

if __name__ == "__main__":
    runs = 10
    model = "rf"
    X_path = "data/modelling_data/X_train_norm.csv"
    y = "y_df"

    RunRegRun(model, X_path, y, runs)