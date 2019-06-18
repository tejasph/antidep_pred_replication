## Explanation of the processed data files

#### Processing was broken up into 5 major steps, in this following order:

1. Row selection (row_selected_scales)
	- Rows were selected based on column conditions
	- Rows were selected to retrieve unique subjects per scale

2. Column selection (column_selected_scales)
	- Selecting for columns of interest per scale
	- Dropping empty columns

3. One-hot encoding (one_hot_encoded_scales)
	- Converting values in some columns so that they can be one-hot encoded
	- Creating dummy variables from some categorical columns 

4. Value conversion (values_converted_scales)
	- Converting values for some columns 

5. Row aggregation (aggregated_rows_scales)
	- Combining all the scales from the previous step, joined by subjectkey

The final data matrix can be found in aggregated_rows_scales (step 5). If debugging data values, I recommend to search in reverse, opening up the files from the previous step.


