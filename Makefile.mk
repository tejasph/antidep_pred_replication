# Looking at qids score change
# author: Tejas Phaterpekar

all: data/modelling_data

data/modelling_data/*: src/split_data.py data/X_tillwk4_qids_sr__final.csv data/y_wk8_resp_mag_qids_sr__final.csv
    python src/split_data.py --X=data/X_tillwk4_qids_sr__final.csv --y=data/y_wk8_resp_mag_qids_sr__final.csv