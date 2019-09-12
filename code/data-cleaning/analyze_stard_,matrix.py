import os
import sys
import pandas as pd
import numpy as np

from utils import *
from overlapping_globals import STARD_OVERLAPPING_VALUE_CONVERSION_MAP, STARD_TO_DROP


stard_dir = './stard_data'
#canbind_dir = './canbind_data'

stard_df = pd.read_csv(stard_dir + "/X_lvl2_rem_qids01__stringent.csv")
#canbind_df = pd.read_csv(canbind_dir + "/canbind-overlapping-X-data.csv")

stard_headers = set(list(stard_df.columns.values))
#canbind_headers = set(list(canbind_df.columns.values))

#for s_header, c_header in zip(list(stard_df.columns.values), list(canbind_df.columns.values)):
#    if not s_header == c_header:
#        print()

# Check if any duplicates

if (len(stard_headers) == len(list(stard_df.columns.values))):
    print("No duplicate columns found in stard matrix")
else:
    print("Warning, duplicate columns found in the stard matrix")

f = open("analyze_stard_output.csv","w")
f.write("Output entailing various statistics of columns in STAR*D not found in the overlapping matrix to find bugs,\n")
f.write(",STARD,\n")

# Only look at columns that aren't in the overlapping matrix, as those are checked elsewhere
non_overlap_stard_headers = list(stard_headers) 
for overlap_header in STARD_OVERLAPPING_VALUE_CONVERSION_MAP['whitelist']: # Remove anything that was whitelisted
    non_overlap_stard_headers.remove(overlap_header)
non_overlap_stard_headers = non_overlap_stard_headers + STARD_TO_DROP # Add back in features that were dropped as these weren't analyzed

# Write mean, median, mode, min, max to file for analysis
for col in sorted(non_overlap_stard_headers): # Sort headers alphabetically to make things easier
    f.write(col + ",\n")
    s_mean = stard_df.loc[:,col].mean()
    s_median = stard_df.loc[:,col].median()
    s_mode = stard_df.mode()[col][0]
    s_min = stard_df.loc[:,col].min()
    s_max = stard_df.loc[:,col].max()

    f.write('Mean, '   + str(s_mean) + ",\n")
    f.write('Median, ' + str(s_median) + ",\n")
    f.write('Mode, '   + str(s_mode) + ",\n")
    f.write('Min, '    + str(s_min) + ",\n")
    f.write('Max, '    + str(s_max) + ",\n")

f.close()