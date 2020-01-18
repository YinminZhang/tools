"""
convert the result of coco to csv file.
"""
import pandas as pd
import json
import os

file_name = 'record_faster'
out_dir = 'faster'

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

with open(file_name+'.txt', 'r') as target:
    for line in target.readlines():
        if '.pkl' in line:
            index.append(line.split('/')[-1][:-5])
        if 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]' in line:
            AP_all_list.append(line.strip('\n')[-5:])
        if 'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]' in line:
            AP_5_list.append(line.strip('\n')[-5:])
        if 'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]' in line:
            AP_75_list.append(line.strip('\n')[-5:])
        if 'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]' in line:
            AP_small_list.append(line.strip('\n')[-5:])
        if 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]' in line:
            AP_medium_list.append(line.strip('\n')[-5:])
        if 'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]' in line:
            AP_large_list.append(line.strip('\n')[-5:])
            
        if 'AR' in line and 'maxDets=  1' in line:
            AR_1.append(line.strip('\n')[-5:])
        if 'AR' in line and 'maxDets= 10' in line:
            AR_10.append(line.strip('\n')[-5:])
        if 'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]' in line:
            AR_100.append(line.strip('\n')[-5:])
        if 'AR' in line and 'small' in line:
            AR_s.append(line.strip('\n')[-5:])
        if 'AR' in line and 'medium' in line:
            AR_m.append(line.strip('\n')[-5:])
        if 'AR' in line and 'large' in line:
            AR_l.append(line.strip('\n')[-5:])
    
    table['AP'] =  AP_5_list
    table['AP.5'] = AP_5_list
    table['AP.75'] = AP_75_list
    table['AP_s'] = AP_small_list
    table['AP_m'] = AP_medium_list
    table['AP_l'] = AP_large_list
    table['AR_1'] = AR_1
    table['AR_10'] = AR_10
    table['AR_100'] = AR_100
    table['AR_s'] = AR_s
    table['AR_m'] = AR_m
    table['AR_l'] = AR_l
    table.index = index
    
    table.to_csv(file_name+'.csv')
