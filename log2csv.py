import pandas as pd
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help="name of input file.", default='record', type=str)
parser.add_argument('-o', '--out', help="directory of output.", default='record', type=str)

args = parser.parse_args()

file_name = args.file
out_dir = args.out

table = pd.DataFrame()

index = list()

AP_all_list = list()
AP_5_list = list()
AP_75_list = list()
AP_small_list = list()
AP_medium_list = list()
AP_large_list = list()

AR_1 = list()
AR_10 = list() 
AR_100 = list()
AR_s = list()
AR_m = list()
AR_l = list()

with open(file_name+'.log', 'r') as target:
    line = target.readlines()[-1].strip('\n').split(',')[-1].split(' ')
    AP_all_list.append(line[-6])
    AP_5_list.append(line[-5])
    AP_75_list.append(line[-4])
    AP_small_list.append(line[-3])
    AP_medium_list.append(line[-2])
    AP_large_list.append(line[-1])

    table['AP'] =  AP_all_list
    table['AP.5'] = AP_5_list
    table['AP.75'] = AP_75_list
    table['AP_s'] = AP_small_list
    table['AP_m'] = AP_medium_list
    
    table.index = [out_dir]
    
    table.to_csv(out_dir+'.csv')