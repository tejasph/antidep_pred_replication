# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 12:48:28 2020

@author: jjnun
"""

import os
import re

##output_dir = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\Paper Submission\DataForFigures\Table3_Replication/'
##output_dir = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\Paper Submission\DataForFigures\Table4_ExternalValidation/'
output_dir = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\Paper Submission\DataForFigures\Table5_Comparing/'

mean_re = re.compile('is: 0\.\d{1,4}')
mean_re = re.compile('a')
sd_re = re.compile('Deviation: 0\.\d{1,4}')

metrics = ['Mean accuracy', 'Mean balanced accuracy', 'Mean AUC', 'Mean sensitivty', 'Mean specificty', 'Mean precision', 'Mean f1','Mean true positive','Mean false positive','Mean true negative', 'Mean false negative'] 

w = open(output_dir + 'parsed_outputs.txt', 'w+')

for filename in os.listdir(output_dir):
    if filename.endswith('.txt'):
        w.write(filename + '\n')
        out_str = ''
        f = open(output_dir + '/' + filename, 'r')
        for line in f.readlines():
            for metric in metrics:
                if metric in line:
                    mean_match = re.search(r'is: (0\.\d{1,6})', line)
                    sd_match = re.search(r'Deviation: (0\.\d{1,8})', line)
                    out_str = out_str + mean_match.group(1) + "\t" + sd_match.group(1) + "\t"
                    
        f.close()
        w.write(out_str + '\n')
        
w.close()