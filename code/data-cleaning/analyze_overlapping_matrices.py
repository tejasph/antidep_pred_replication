import os
import sys
import pandas as pd
import numpy as np

from utils import *

"""
Script to used to analyze some of our data, mostly for double checking
"""


stard_dir = './stard_data'
canbind_dir = './canbind_data'

stard_df = pd.read_csv(stard_dir + "/stard-overlapping-X-data.csv")
canbind_df = pd.read_csv(canbind_dir + "/canbind-overlapping-X-data.csv")

stard_headers = set(list(stard_df.columns.values))
canbind_headers = set(list(canbind_df.columns.values))

#for s_header, c_header in zip(list(stard_df.columns.values), list(canbind_df.columns.values)):
#    if not s_header == c_header:
#        print()

# Check if any duplicates, and if headers only in one matrix or the other:
only_stard = (stard_headers^canbind_headers)&stard_headers
only_canbind = (stard_headers^canbind_headers)&canbind_headers

if (len(stard_headers) == len(list(stard_df.columns.values))):
    print("No duplicate columns found in stard matrix")
else:
    print("Warning, duplicate columns found in the stard matrix")
    

if (len(canbind_headers) == len(list(canbind_df.columns.values))):
    print("No duplicate columns found in canbind matrix")
else:
    print("Warning, duplicate columns found in the canbind matrix")


print("Following columns are only found in stard*d overlapping data matrix:")
if len(only_stard) == 0:
    print("All columns in stard matrix also found in canbind!")
else:
    print(only_stard)

print("Following columns are only found in canbind overlapping data matrix:")
if len(only_canbind) ==0:
    print("All columns in canbind matrix also found in stard!")
else:
    print(only_canbind)

f = open("analyze_overlapping_output.csv","w")
f.write("Output entailing various statistics of columns in both STARD and CANBIND overlapping matrices to compare and find bugs,\n")
f.write(",CANBIND,STARD,\n")

# Write mean, median, mode, min, max to file for analysis
for col in sorted(list(stard_headers)): # Sort headers alphabetically to make things easier
    f.write(col + ",\n")
    s_mean = stard_df.loc[:,col].mean()
    s_median = stard_df.loc[:,col].median()
    s_mode = stard_df.mode()[col][0]
    s_min = stard_df.loc[:,col].min()
    s_max = stard_df.loc[:,col].max()
    
    c_mean = canbind_df.loc[:,col].mean()
    c_median = canbind_df.loc[:,col].median()
    c_mode = canbind_df.mode()[col][0]
    c_min = canbind_df.loc[:,col].min()
    c_max = canbind_df.loc[:,col].max()

    f.write('Mean, ' + str(c_mean) + ", " + str(s_mean) + ",\n")
    f.write('Median, ' + str(c_median) + ", " + str(s_median) + ",\n")
    f.write('Mode, ' + str(c_mode) + ", " + str(s_mode) + ",\n")
    f.write('Min, ' + str(c_min) + ", " + str(s_min) + ",\n")
    f.write('Max, ' + str(c_max) + ", " + str(s_max) + ",\n")

f.close()