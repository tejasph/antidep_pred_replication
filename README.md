# Replication of Machine Learning Methods to Predict Treatment Outcome with Antidepressant Medications in Patients with Major Depressive Disorder from STAR*D and CAN-BIND-1
## Accepted for publication in PLOS One May 27, 2021

README by John-Jose Nunez, jjnunez11@gmail.com

Thank you for your interest in our project! Our aim is to make it as reproducible as possible, please do not hesitate to reach out. 

## Data Availability
The raw clinical data from STAR*D is found [in our NIMH NDA project collection](http://dx.doi.org/10.15154/1503299)

You can download the data freshly from NDA via the measures tab, or can download a zip of the exact data we used going to the "Data Analysis" tab
and downloading the file [Version Of Raw Data Used with One Modification] https://nda.nih.gov/study.html?tab=result&id=640. 

The raw clinical data from CAN-BIND-1 can obtained from [Brian-CODE](https://braininstitute.ca/research-data-sharing/brain-code)

The processed STAR*D datasets used directly for machine learning are obtainable through our NIMH NDA project, in the data analysis tab,
and is named [Processed STARD Datasets used for ML](https://nda.nih.gov/study.html?tab=result&id=640)

The processed CAN-BIND datasets used directly for machine learning are obtainable through [Brian-CODE](https://braininstitute.ca/research-data-sharing/brain-code).
We may update the NDA collection with them when possible. 

The results used in our paper are located within zips for each table in the [results folder](./results)

## Data Processing

Please find our data cleaning and processing code in [data-cleaning](./code/data-cleaning/)

The raw STAR*D and CAN-BIND data should be placed in separate folders within [data](./data/)

### Run order
1. Generate STAR*D dataset with [stard_preprocessing_manager.py](./code/data-cleaning/stard_preprocessing_manager.py)
2. Generate initial CAN-BIND dataset with  [canbind_preprocessing_manager.py](./code/data-cleaning/canbind_preprocessing_manager.py)
3. Generate overlapping datasets with [generate_overlapping_features](./code/data-cleaning/generate_overlapping_features.py)

## Running Machine Learning Analysis
1. Update the path of your data and result directories in [run_globals](./code/run_globals.py)
2. Place your .csv files with the X and y matrices you want to run (or ours as downloaded) into the data directory above
3. Run the machine learning with [run_results](./code/run_results.py) script. This script calls [run_result](./code/run_results.py), assuming the filename of the 
dataset csv's will be passed along, minus the '.csv' ending. 

## Running Regression Aspect of the Project
1. Generate STAR*D dataset with [stard_preprocessing_manager.py](./code/data-cleaning/stard_preprocessing_manager.py). The script writes the processed data to the same folder that contains the raw data
    Usage example: `python antidep_pred_replication/code/data-cleaning/stard_preprocessing_manager.py STARD_Raw_Data_Used -a` 
2. Generate initial CAN-BIND dataset with  [canbind_preprocessing_manager.py](./code/data-cleaning/canbind_preprocessing_manager.py). Running this also creates overlapping dataset.
    Usage example: `C:\Users\Tejas\Documents\star_project\antidep_pred_replication>python -v code/data-cleaning/canbind_preprocessing_manager.py`

note: before running this, go into canbind_preprocessing_manager.py and alter pathData to a correct pathing for the raw data (see below)
```
    elif len(sys.argv) == 1:
        pathData = r'C:\Users\Tejas\Documents\star_project\antidep_pred_replication\data\canbind_raw_data'
        aggregate_and_clean(pathData, verbose=False)
        ygen(pathData)
        impute(pathData)
        convert_canbind_to_overlapping(pathData)
```

3. Create overlapping STARD dataset with [generate_overlapping_features.py](./code/data-cleaning/generate_overlapping_features.py).
    Usage example: `C:\Users\Tejas\Documents\star_project\antidep_pred_replication>python code/data-cleaning/generate_overlapping_features.py -sd data`

    
1. Run all steps in "Run Order" to generate the processed STARD and CAN-BIND data
2. Prepare the data beforehand using `python code/prepare_reg_data.py` 