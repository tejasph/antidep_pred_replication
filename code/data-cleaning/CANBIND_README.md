## Explanation of CAN-BIND Data Processing

#### Processing was broken up into 2 major steps, in this following order, in addirtion to the y-generation:

1. Aggregation and cleaning (canbind_preprocessing_manager.py, will call remaining two)
	- Aggregation of data from the various raw data files
	- Removing unused columns, and those from unused time points
	- Selects only valid subjects
	--> canbind_clean_aggregated.csv
	
2. Imputation (canbind_imputer.py)
	- Imputes blank values 
	--> canbind_imputed.csv

3. y-generation (canbind_ygen.py)
    - Makes the targets using the raw data 
	--> canbind_targets.csv
	