## Explanation of the STAR*D processed data files

#### Processing was broken up into 5 major steps, in this following order:

1. Row selection (row_selected_scales)
	- Rows were selected based on column conditions
	- Rows were selected to retrieve unique subjects per scale
	- Filenames are prefixed with "rs__"

2. Column selection (column_selected_scales)
	- Selecting for columns of interest per scale
	- Dropping empty columns
	- Filenames are prefixed with "rs__cs__"

3. One-hot encoding (one_hot_encoded_scales)
	- Converting values in some columns so that they can be one-hot encoded
	- Creating dummy variables from some categorical columns 
	- Filenames are prefixed with "rs__cs__ohe__"

4. Value conversion (values_converted_scales)
	- Converting values for some columns 
	- Filenames are prefixed with "rs__cs__ohe__vc__"

5. Row aggregation (aggregated_rows_scales)
	- Combining all the scales from the previous step, joined by subjectkey
	- Filenames are prefixed with "rs__cs__ohe__vc__ag__"

6. Imputation (imputed_scales)

7. Y-matrix generation (y_matrix)

8. Subject selection (final_xy_data_matrices)

The final data matrix can be found in aggregated_rows_scales (step 5). If debugging data values, I recommend to search in reverse, opening up the files from the previous step.

#### Naming scheme

- Column names in the final data matrix are prefixed with the "<scale name>_<row selected state if any>"
- Filenames are verbose, for easy debugging reasons to inform of the order processing. E.g. if the value conversion 
file has suspicious values, given that it has the prefix "rs__cs__ohe__vc__", one can check the level up, which is 
"ohe" and stands for one-hot encoding.